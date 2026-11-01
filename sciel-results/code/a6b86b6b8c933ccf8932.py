import numpy as np
import time
from scipy.optimize import minimize

N = 14
D = 3

def get_pairwise_sq_dists(points):
    """Returns flat array of squared pairwise distances (upper triangle)."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    iu = np.triu_indices(N, k=1)
    return sq_dists[iu]

def raw_score(points):
    """Compute (max_dist / min_dist)^2 = max_sq / min_sq."""
    sq_dists = get_pairwise_sq_dists(points)
    mn = np.min(sq_dists)
    mx = np.max(sq_dists)
    if mn < 1e-14:
        return np.inf
    return mx / mn

def smooth_min_max(sq_dists, tau):
    """
    Compute smooth approximations of min and max of sq_dists.
    Uses log-sum-exp tricks for stability.
    """
    # soft_min = -1/tau * log(sum(exp(-sq_dists/tau)))
    # To avoid overflow/underflow, shift by min and max
    m_min = np.min(sq_dists)
    m_max = np.max(sq_dists)
    
    # Soft min
    # exp(-sq/tau) is large for small sq.
    # Shift: exp(-(sq - m_min)/tau)
    exp_terms_min = np.exp(-(sq_dists - m_min) / tau)
    soft_min = m_min - tau * np.log(np.sum(exp_terms_min))
    
    # Soft max
    # exp(sq/tau) is large for large sq.
    # Shift: exp((sq - m_max)/tau)
    exp_terms_max = np.exp((sq_dists - m_max) / tau)
    soft_max = m_max + tau * np.log(np.sum(exp_terms_max))
    
    return soft_min, soft_max

def objective_qcqp(x, tau, lam):
    """
    Objective function for the QCQP-like formulation.
    Minimizes -soft_min + lam * max(0, soft_max - 1)^2
    
    This encourages soft_max <= 1 and soft_min as large as possible.
    """
    points = x.reshape(N, D)
    
    # Center the points to remove translation invariance
    points = points - np.mean(points, axis=0)
    
    # Optional: Scale points to prevent numerical issues if they drift too far
    # The penalty term handles the scale, but if points collapse, sq_dists -> 0
    # If points explode, sq_dists -> inf.
    # Let's check norms
    norms = np.linalg.norm(points, axis=1)
    if np.any(norms < 1e-6):
        return 1e10 # Collapsed
    
    sq_dists = get_pairwise_sq_dists(points)
    
    # Ensure positive
    sq_dists = np.maximum(sq_dists, 1e-12)
    
    soft_min, soft_max = smooth_min_max(sq_dists, tau)
    
    # Penalty for exceeding upper bound 1.0
    excess = np.maximum(0.0, soft_max - 1.0)
    penalty = lam * excess**2
    
    # We want to maximize soft_min, so minimize -soft_min
    obj = -soft_min + penalty
    
    return obj

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin from 150s limit
    
    best_score = np.inf
    best_points = None
    
    def update_best(pts):
        nonlocal best_score, best_points
        s = raw_score(pts)
        if s < best_score:
            best_score = s
            best_points = pts.copy()
            
    # Generate diverse initial candidates
    candidates = []
    rng = np.random.default_rng(42)
    
    # 1. Grid-based start
    points_grid = []
    for x in range(3):
        for y in range(3):
            for z in range(2):
                points_grid.append((x, y, z))
    to_remove = [(0,0,0), (0,0,1), (0,2,0), (0,2,1)]
    final_points_grid = [p for p in points_grid if p not in to_remove]
    init_grid = np.array(final_points_grid, dtype=float)
    init_grid = init_grid - np.mean(init_grid, axis=0)
    # Normalize initial grid to have max dist ~ 1
    sq_dists_grid = get_pairwise_sq_dists(init_grid)
    max_d_grid = np.max(sq_dists_grid)
    if max_d_grid > 0:
        init_grid = init_grid * (1.0 / np.sqrt(max_d_grid))
    candidates.append(init_grid.flatten())
    
    # 2. Random starts
    for trial in range(40):
        if time.time() - start_time > deadline - 10:
            break
            
        if trial % 3 == 0:
            pts = rng.uniform(-1, 1, size=(N, D))
        elif trial % 3 == 1:
            pts = rng.normal(size=(N, D))
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            pts = pts / np.maximum(norms, 1e-6) * 2.0
        else:
            # Spherical shell with some randomness
            pts = rng.normal(size=(N, D))
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            pts = pts / np.maximum(norms, 1e-6) * (1.0 + 0.5 * rng.uniform(0, 1, size=(N, 1)))
            
        pts = pts - np.mean(pts, axis=0)
        # Normalize to max dist ~ 1
        sq_d = get_pairwise_sq_dists(pts)
        mx = np.max(sq_d)
        if mx > 1e-10:
            pts = pts * (1.0 / np.sqrt(mx))
        candidates.append(pts.flatten())
        
    # 3. Icosahedron-like start (12 points) + 2 extra
    phi = (1 + np.sqrt(5)) / 2
    icos_verts = np.array([
        [0, 1, phi], [0, -1, phi],
        [1, phi, 0], [-1, phi, 0],
        [phi, 0, 1], [phi, 0, -1],
        [0, 1, -phi], [0, -1, -phi],
        [-1, -phi, 0], [1, -phi, 0],
        [-phi, 0, -1], [-phi, 0, 1]
    ])
    icos_verts = icos_verts - np.mean(icos_verts, axis=0)
    sq_d_ico = get_pairwise_sq_dists(icos_verts)
    mx_ico = np.max(sq_d_ico)
    icos_verts = icos_verts * (1.0 / np.sqrt(mx_ico))
    
    # Add 2 points near the center or on the surface
    extra1 = rng.normal(size=(1, D)) * 0.1
    extra2 = rng.normal(size=(1, D)) * 0.1
    full_ico = np.vstack([icos_verts, extra1, extra2])
    full_ico = full_ico - np.mean(full_ico, axis=0)
    sq_d_full = get_pairwise_sq_dists(full_ico)
    mx_full = np.max(sq_d_full)
    if mx_full > 1e-10:
        full_ico = full_ico * (1.0 / np.sqrt(mx_full))
    candidates.append(full_ico.flatten())

    # Optimization loop
    # Tau schedule: start smooth, end sharp
    # Lambda schedule: start low, end high to strictly enforce constraint
    
    tau_schedule = [5.0, 2.0, 1.0, 0.5, 0.2, 0.1]
    lam_schedule = [10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0]
    
    for i, init_x in enumerate(candidates):
        if time.time() - start_time > deadline - 5:
            break
            
        x_curr = init_x.copy()
        
        # Annealing optimization
        for idx, (tau, lam) in enumerate(zip(tau_schedule, lam_schedule)):
            if time.time() - start_time > deadline - 3:
                break
                
            try:
                res = minimize(
                    objective_qcqp,
                    x_curr,
                    args=(tau, lam),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12, 'maxls': 50}
                )
                
                if res.x is not None and res.fun < 1e9:
                    x_curr = res.x
            except Exception:
                pass
        
        # Final polish with very high lambda and low tau
        if time.time() - start_time < deadline - 2:
            try:
                res_final = minimize(
                    objective_qcqp,
                    x_curr,
                    args=(0.05, 10000.0),
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-14}
                )
                if res_final.x is not None:
                    x_curr = res_final.x
            except Exception:
                pass
        
        # Evaluate and update best
        pts = x_curr.reshape(N, D)
        update_best(pts)
        
        # Local perturbation refinement on the best point found so far?
        # Or just perturb the current candidate?
        if time.time() - start_time < deadline - 1 and i % 5 == 0:
            for _ in range(2):
                x_pert = x_curr + rng.normal(0, 0.05, size=x_curr.shape)
                try:
                    res_p = minimize(
                        objective_qcqp,
                        x_pert,
                        args=(0.1, 1000.0),
                        method='L-BFGS-B',
                        options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-12}
                    )
                    if res_p.x is not None:
                        pts_p = res_p.x.reshape(N, D)
                        update_best(pts_p)
                except Exception:
                    pass

    # Final fallback
    if best_points is None:
        best_points = init_grid
        best_score = raw_score(best_points)
    
    return best_points

if __name__ == "__main__":
    pts = construct()
    sq_dists = get_pairwise_sq_dists(pts)
    score = (np.max(sq_dists) / np.min(sq_dists))
    print(f"Score: {score}")
