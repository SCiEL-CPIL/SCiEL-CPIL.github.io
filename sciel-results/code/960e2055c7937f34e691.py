"""Pack 50 equal circles in a unit circle; maximize the common radius."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
DEADLINE = 110.0  # safety margin before 120s kill


def _score(pts):
    """Compute the packing radius for a set of centers."""
    if len(pts) != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # Check all centers are inside unit circle
    dists = np.hypot(pts[:, 0], pts[:, 1])
    if np.any(dists > 1.0):
        return 0.0
    # Minimum distance to boundary
    r_boundary = 1.0 - np.max(dists)
    # Minimum pairwise distance / 2
    min_pair = np.inf
    for i in range(N):
        for j in range(i + 1, N):
            d = np.hypot(pts[i, 0] - pts[j, 0], pts[i, 1] - pts[j, 1])
            if d < min_pair:
                min_pair = d
    r_pair = min_pair / 2.0
    return min(r_boundary, r_pair)


def _soft_min_objective(x, beta):
    """
    Smooth surrogate: maximize soft-min of (boundary distances, pairwise half-distances).
    We minimize the negative soft-min.
    x: flat array of 2N coordinates.
    """
    pts = x.reshape(N, 2)
    # Boundary terms: 1 - |c_i|
    dists = np.hypot(pts[:, 0], pts[:, 1])
    boundary = 1.0 - dists
    # Pairwise half-distances
    pair_dists = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            d = np.hypot(pts[i, 0] - pts[j, 0], pts[i, 1] - pts[j, 1])
            pair_dists[i, j] = d / 2.0
            pair_dists[j, i] = d / 2.0
    # Collect all terms
    all_terms = np.concatenate([boundary, pair_dists[np.triu_indices(N, k=1)]])
    # Soft-min via log-sum-exp
    m = np.min(all_terms)
    log_sum = m + np.log(np.sum(np.exp(all_terms - m)))
    soft_min = m - log_sum  # This is -log-sum-exp, which approximates -max... 
    # Actually soft-min = -logsumexp(-terms) = min approx
    # Let's use: soft_min = -np.log(np.sum(np.exp(-all_terms)))  -- no, that's not right either
    # soft-min(x) ≈ (1/beta) * log(sum(exp(-beta * x_i))) with sign... 
    # Let's just use: soft_min = -np.log(np.sum(np.exp(-beta * all_terms))) / beta
    # This converges to min(all_terms) as beta -> inf
    exp_terms = np.exp(-beta * all_terms)
    s = np.sum(exp_terms)
    if s <= 0:
        return 0.0
    soft_min_val = -np.log(s) / beta
    # We want to maximize soft_min, so minimize -soft_min
    return -soft_min_val


def _ring_init(n, inner_r=0.0, outer_r=1.0):
    """Generate N points in concentric rings."""
    pts = np.zeros((N, 2))
    if N == 1:
        return pts
    # Use a few rings
    # Center point + rings
    pts[0] = [0.0, 0.0]
    remaining = N - 1
    ring_idx = 1
    r = 0.2
    while remaining > 0:
        # How many fit on this ring?
        circumference = 2 * np.pi * r
        # Approximate max points on ring with spacing ~ r
        count = max(1, int(circumference / (r * 1.1)))
        if count > remaining:
            count = remaining
        angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
        start = ring_idx
        end = start + count
        pts[start:end, 0] = r * np.cos(angles)
        pts[start:end, 1] = r * np.sin(angles)
        remaining -= count
        ring_idx += count
        r += 0.15
    return pts[:N]


def _random_init():
    """Random initialization within unit circle."""
    angles = np.random.uniform(0, 2 * np.pi, N)
    radii = np.sqrt(np.random.uniform(0, 1, N))
    pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return pts


def _gradient_ascent_step(pts, step_size=0.01, iterations=500):
    """
    Simple gradient ascent on the soft-min objective using numerical gradients.
    """
    beta = 50.0
    for it in range(iterations):
        x = pts.flatten()
        f0 = _soft_min_objective(x, beta)
        # Numerical gradient
        grad = np.zeros_like(x)
        eps = 1e-6
        for i in range(len(x)):
            xp = x.copy()
            xp[i] += eps
            fp = _soft_min_objective(xp, beta)
            xm = x.copy()
            xm[i] -= eps
            fm = _soft_min_objective(xm, beta)
            grad[i] = (fp - fm) / (2 * eps)
        # Gradient ascent
        pts = (pts + step_size * grad.reshape(N, 2))
        # Project back: keep inside unit circle
        dists = np.hypot(pts[:, 0], pts[:, 1])
        mask = dists > 1.0
        if np.any(mask):
            pts[mask] = pts[mask] / dists[mask][:, None] * 0.999
    return pts


def construct():
    start_time = time.time()
    
    # Best solution found so far
    best_pts = None
    best_score = -np.inf
    
    def update_best(pts):
        nonlocal best_pts, best_score
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()
    
    # Strategy 1: Binary search on radius with L-BFGS-B feasibility
    # We'll try to find the maximum r such that there exist N points with
    # all pairwise distances >= 2r and all within distance 1-r from origin.
    
    # Start with a reasonable guess
    # Known best is ~0.1258, so let's search around there
    r_low = 0.10
    r_high = 0.15
    
    # First, get an initial feasible solution
    init_pts = _ring_init(N)
    update_best(init_pts)
    
    # Try several random restarts
    for restart in range(10):
        if time.time() - start_time > DEADLINE - 30:
            break
        
        if restart == 0:
            pts = _ring_init(N)
        else:
            pts = _random_init()
        
        # Use L-BFGS-B to maximize a smooth objective
        # Objective: minimize sum of violations of constraints
        # Constraint 1: |c_i| <= 1 - r
        # Constraint 2: |c_i - c_j| >= 2r
        
        def objective_and_grad(x):
            pts_x = x.reshape(N, 2)
            # Boundary violations
            dists = np.hypot(pts_x[:, 0], pts_x[:, 1])
            boundary_viol = np.maximum(0, dists - (1.0 - r_target))
            # Pairwise violations
            pair_viol = 0.0
            pair_grad = np.zeros_like(x)
            for i in range(N):
                for j in range(i + 1, N):
                    d = np.hypot(pts_x[i, 0] - pts_x[j, 0], pts_x[i, 1] - pts_x[j, 1])
                    if d < 2 * r_target:
                        viol = 2 * r_target - d
                        pair_viol += viol ** 2
                        # Gradient w.r.t. pts_i and pts_j
                        if d > 1e-10:
                            dx = (pts_x[i, 0] - pts_x[j, 0]) / d
                            dy = (pts_x[i, 1] - pts_x[j, 1]) / d
                            pair_grad[2*i] += 2 * viol * dx
                            pair_grad[2*i+1] += 2 * viol * dy
                            pair_grad[2*j] -= 2 * viol * dx
                            pair_grad[2*j+1] -= 2 * viol * dy
            # Boundary gradient
            boundary_grad = np.zeros_like(x)
            for i in range(N):
                if dists[i] > (1.0 - r_target) and dists[i] > 1e-10:
                    boundary_grad[2*i] = 2 * boundary_viol[i] * pts_x[i, 0] / dists[i]
                    boundary_grad[2*i+1] = 2 * boundary_viol[i] * pts_x[i, 1] / dists[i]
            total = np.sum(boundary_viol ** 2) + pair_viol
            total_grad = boundary_grad + pair_grad
            return total, total_grad
        
        # Binary search on r
        for _ in range(20):
            if time.time() - start_time > DEADLINE - 30:
                break
            r_target = (r_low + r_high) / 2.0
            
            # Optimize to minimize violations
            x0 = pts.flatten()
            try:
                res = minimize(objective_and_grad, x0, jac=True, method='L-BFGS-B',
                               options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
                opt_pts = res.x.reshape(N, 2)
                # Check feasibility
                dists = np.hypot(opt_pts[:, 0], opt_pts[:, 1])
                feasible = np.all(dists <= 1.0 - r_target + 1e-9)
                for i in range(N):
                    for j in range(i+1, N):
                        d = np.hypot(opt_pts[i,0]-opt_pts[j,0], opt_pts[i,1]-opt_pts[j,1])
                        if d < 2*r_target - 1e-9:
                            feasible = False
                            break
                    if not feasible:
                        break
                
                if feasible:
                    r_low = r_target
                    pts = opt_pts
                    update_best(pts)
                else:
                    r_high = r_target
            except Exception:
                r_high = r_target
        
        # After binary search, do a final optimization at r_low
        r_target = r_low
        x0 = pts.flatten()
        try:
            res = minimize(objective_and_grad, x0, jac=True, method='L-BFGS-B',
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
            opt_pts = res.x.reshape(N, 2)
            update_best(opt_pts)
            pts = opt_pts
        except Exception:
            pass
    
    # Strategy 2: Direct optimization of soft-min with various betas
    if time.time() - start_time < DEADLINE - 20:
        for beta in [20, 50, 100]:
            if time.time() - start_time > DEADLINE - 20:
                break
            x0 = pts.flatten()
            try:
                def obj(x):
                    return _soft_min_objective(x, beta)
                res = minimize(obj, x0, method='L-BFGS-B',
                               options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
                opt_pts = res.x.reshape(N, 2)
                # Project inside
                dists = np.hypot(opt_pts[:, 0], opt_pts[:, 1])
                mask = dists > 1.0
                if np.any(mask):
                    opt_pts[mask] = opt_pts[mask] / dists[mask][:, None] * 0.9999
                update_best(opt_pts)
            except Exception:
                pass
    
    # Final polish: local refinement
    if time.time() - start_time < DEADLINE - 5 and best_pts is not None:
        # Small perturbations and re-optimize
        for _ in range(20):
            if time.time() - start_time > DEADLINE - 5:
                break
            perturbed = best_pts + np.random.normal(0, 0.001, (N, 2))
            # Project inside
            dists = np.hypot(perturbed[:, 0], perturbed[:, 1])
            mask = dists > 1.0
            if np.any(mask):
                perturbed[mask] = perturbed[mask] / dists[mask][:, None] * 0.9999
            
            # Quick L-BFGS-B
            r_target = _score(perturbed)
            if r_target <= 0:
                continue
                
            def obj_final(x):
                pts_x = x.reshape(N, 2)
                dists_x = np.hypot(pts_x[:, 0], pts_x[:, 1])
                boundary_viol = np.maximum(0, dists_x - (1.0 - r_target))
                pair_viol = 0.0
                for i in range(N):
                    for j in range(i + 1, N):
                        d = np.hypot(pts_x[i, 0] - pts_x[j, 0], pts_x[i, 1] - pts_x[j, 1])
                        if d < 2 * r_target:
                            pair_viol += (2 * r_target - d) ** 2
                return np.sum(boundary_viol ** 2) + pair_viol
            
            x0 = perturbed.flatten()
            try:
                res = minimize(obj_final, x0, method='L-BFGS-B',
                               options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10})
                opt_pts = res.x.reshape(N, 2)
                dists_f = np.hypot(opt_pts[:, 0], opt_pts[:, 1])
                mask_f = dists_f > 1.0
                if np.any(mask_f):
                    opt_pts[mask_f] = opt_pts[mask_f] / dists_f[mask_f][:, None] * 0.99999
                update_best(opt_pts)
            except Exception:
                pass
    
    if best_pts is None:
        # Fallback: return ring init
        best_pts = _ring_init(N)
    
    return best_pts
