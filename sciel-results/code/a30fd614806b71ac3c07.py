import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14

def _hull_area(pts):
    """Compute convex hull area."""
    if len(pts) < 3:
        return 0.0
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by 3 points."""
    n = len(pts)
    min_area = float('inf')
    x = pts[:, 0]
    y = pts[:, 1]
    
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            dx_ij = x[j] - x[i]
            dy_ij = y[j] - y[i]
            dx_ik = x[j+1:] - x[i]
            dy_ik = y[j+1:] - y[i]
            cross = dx_ij * dy_ik - dy_ij * dx_ik
            areas = 0.5 * np.abs(cross)
            current_min = np.min(areas)
            if current_min < min_area:
                min_area = current_min
                if min_area < 1e-12:
                    return min_area
    return min_area

def _score(pts):
    """Compute the score: min triangle area / hull area."""
    hull_a = _hull_area(pts)
    if hull_a <= 1e-10:
        return 0.0
    min_a = _min_triangle_area(pts)
    return min_a / hull_a

def _all_triangle_areas(pts):
    """Return array of all triangle areas (n choose 3)."""
    n = len(pts)
    x = pts[:, 0]
    y = pts[:, 1]
    areas = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                cross = (x[j] - x[i]) * (y[k] - y[i]) - (y[j] - y[i]) * (x[k] - x[i])
                areas.append(0.5 * abs(cross))
    return np.array(areas)

def _soft_min_obj(pts_flat, beta):
    """Smooth soft-min of triangle areas divided by hull area.
    Returns negative value for minimization."""
    pts = pts_flat.reshape(-1, 2)
    if np.any(~np.isfinite(pts)):
        return 1e10
    hull_a = _hull_area(pts)
    if hull_a <= 1e-10:
        return 1e10
    areas = _all_triangle_areas(pts)
    if len(areas) == 0:
        return 1e10
    # Soft min: -log(sum(exp(-beta * a))) / beta
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min_val = -log_sum_exp / beta
    return -soft_min_val / hull_a

def _soft_min_obj_and_grad(pts_flat, beta):
    """Compute soft-min objective and its gradient w.r.t. points."""
    pts = pts_flat.reshape(-1, 2)
    n = len(pts)
    x = pts[:, 0].copy()
    y = pts[:, 1].copy()
    
    hull_a = _hull_area(pts)
    if hull_a <= 1e-10:
        return 1e10, np.zeros_like(pts_flat)
    
    # Compute all triangle areas and their gradients
    # Triangle (i,j,k) area = 0.5 * |cross((j-i), (k-i))|
    # For the soft-min, we need d(soft_min)/d(pts)
    # soft_min = -(1/beta) * log(sum(exp(-beta * a_t)))
    # d(soft_min)/d(pts_p) = (1/beta) * sum_t [exp(-beta*a_t) * (-beta * d(a_t)/d(pts_p))] / sum_t exp(-beta*a_t)
    #                    = -sum_t [w_t * d(a_t)/d(pts_p)] where w_t = exp(-beta*a_t) / sum(exp(-beta*a_t))
    
    # Collect all triangles
    tri_idx = []
    tri_areas = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                cross = (x[j] - x[i]) * (y[k] - y[i]) - (y[j] - y[i]) * (x[k] - x[i])
                area = 0.5 * abs(cross)
                tri_idx.append((i, j, k, cross))
                tri_areas.append(area)
    
    tri_areas = np.array(tri_areas)
    if len(tri_areas) == 0:
        return 1e10, np.zeros_like(pts_flat)
    
    # Compute weights
    scaled = -beta * tri_areas
    max_scaled = np.max(scaled)
    exp_vals = np.exp(scaled - max_scaled)
    w = exp_vals / np.sum(exp_vals)
    
    # Gradient of each triangle area w.r.t. each point
    # Area of triangle (i,j,k) = 0.5 * |cross|
    # d(0.5*|cross|)/d(pts) = sign(cross) * 0.5 * d(cross)/d(pts)
    # cross = (x_j - x_i)(y_k - y_i) - (y_j - y_i)(x_k - x_i)
    # d(cross)/dx_i = -(y_k - y_i)
    # d(cross)/dy_i = (x_k - x_i)
    # d(cross)/dx_j = (y_k - y_i)
    # d(cross)/dy_j = -(x_k - x_i)
    # d(cross)/dx_k = -(y_j - y_i)
    # d(cross)/dy_k = (x_j - x_i)
    
    grad = np.zeros((n, 2))
    for idx, (i, j, k, cross) in enumerate(tri_idx):
        wt = w[idx]
        s = np.sign(cross) if abs(cross) > 1e-15 else 1.0
        coeff = 0.5 * s * wt
        # d(area)/d(x_i) = coeff * (-(y_k - y_i))
        grad[i, 0] += coeff * (-(y[k] - y[i]))
        grad[i, 1] += coeff * ((x[k] - x[i]))
        # d(area)/d(x_j) = coeff * (y_k - y_i)
        grad[j, 0] += coeff * ((y[k] - y[i]))
        grad[j, 1] += coeff * (-(x[k] - x[i]))
        # d(area)/d(x_k) = coeff * (-(y_j - y_i))
        grad[k, 0] += coeff * (-(y[j] - y[i]))
        grad[k, 1] += coeff * ((x[j] - x[i]))
    
    # soft_min = -(1/beta) * log(sum(exp(-beta * a)))
    # d(soft_min)/d(pts) = sum_t w_t * d(a_t)/d(pts)  (the negative signs cancel)
    # Wait: let me re-derive.
    # f = -(1/beta) * log(sum_t exp(-beta * a_t))
    # df/d(pts_p) = -(1/beta) * [1/sum(exp(-beta*a_t))] * sum_t exp(-beta*a_t) * (-beta * da_t/d(pts_p))
    #             = sum_t [exp(-beta*a_t)/sum(exp(-beta*a_t))] * da_t/d(pts_p)
    #             = sum_t w_t * da_t/d(pts_p)
    # So grad of soft_min w.r.t. pts is just sum_t w_t * grad(a_t)
    
    # We want to maximize soft_min / hull_area
    # Objective for minimizer: -(soft_min / hull_area)
    # d(-soft_min/hull_a)/d(pts) = -(1/hull_a) * d(soft_min)/d(pts)
    # (ignoring derivative of hull_area w.r.t. pts for simplicity, or we can include it)
    
    obj_val = -np.sum(w * tri_areas) / hull_a
    grad_flat = -(grad / hull_a).flatten()
    
    return obj_val, grad_flat

def _neg_score(pts_flat):
    """Objective function for minimization (hard objective)."""
    pts = pts_flat.reshape(-1, 2)
    if np.any(~np.isfinite(pts)):
        return 1e10
    return -_score(pts)

def _initial_regular_polygon(n):
    """Generate points of a regular n-gon."""
    angles = 2 * np.pi * np.arange(n) / n
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    return pts

def _initial_hexagonal(n):
    """Generate points in a hexagonal lattice pattern."""
    pts = []
    # Use a grid and select n points
    rows = 5
    cols = 5
    all_pts = []
    for r in range(rows):
        for c in range(cols):
            px = c + 0.5 * (r % 2)
            py = r * np.sqrt(3) / 2
            all_pts.append([px, py])
    all_pts = np.array(all_pts)
    # Center and scale
    all_pts -= all_pts.mean(axis=0)
    scale = np.sqrt(np.sum(all_pts**2, axis=1)).max()
    if scale > 0:
        all_pts /= scale
    # Select n points: try to pick evenly spaced
    rng = np.random.RandomState(123)
    perm = rng.permutation(len(all_pts))
    selected = []
    for idx in perm:
        if len(selected) >= n:
            break
        p = all_pts[idx]
        if all(np.linalg.norm(p - s) > 0.15 for s in selected):
            selected.append(p)
    # If not enough, fill with remaining
    while len(selected) < n:
        for idx in perm:
            p = all_pts[idx]
            if p not in selected:
                selected.append(p)
                break
        if len(selected) >= n:
            break
    return np.array(selected[:n])

def _initial_mixed(n, seed=42):
    """Mix of regular polygon and random perturbation."""
    base = _initial_regular_polygon(n)
    rng = np.random.RandomState(seed)
    noise = rng.normal(0, 0.05, size=base.shape)
    return base + noise

def _anneal_lbfgs(init_pts, start_time, deadline, beta_start=10.0, beta_end=500.0, n_anneal_steps=15):
    """Run L-BFGS-B with simulated annealing on beta."""
    best_pts = init_pts.copy()
    best_hard_score = _score(init_pts)
    
    n_beta_steps = n_anneal_steps
    betas = np.linspace(beta_start, beta_end, n_beta_steps)
    
    for beta in betas:
        if time.time() - start_time > deadline - 5:
            break
        
        x0 = best_pts.flatten()
        
        def obj_and_grad(x):
            return _soft_min_obj_and_grad(x, beta)
        
        try:
            res = minimize(
                obj_and_grad, x0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 500, 'gtol': 1e-6, 'maxfun': 1000}
            )
            if res.x is not None:
                cand = res.x.reshape(-1, 2)
                if np.all(np.isfinite(cand)):
                    s = _score(cand)
                    if s > best_hard_score:
                        best_hard_score = s
                        best_pts = cand
        except Exception:
            pass
    
    return best_pts, best_hard_score

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s kill
    
    best_pts = None
    best_score = 0.0
    
    # Generate diverse initializations
    inits = []
    inits.append(_initial_regular_polygon(N))
    inits.append(_initial_hexagonal(N))
    inits.append(_initial_mixed(N, seed=42))
    inits.append(_initial_mixed(N, seed=123))
    
    rng = np.random.RandomState(7)
    for i in range(3):
        noise = rng.normal(0, 0.03, size=(N, 2))
        inits.append(_initial_regular_polygon(N) + noise)
    
    # Phase 1: Annealed L-BFGS-B from each initialization
    for init in inits:
        if time.time() - start_time > DEADLINE - 10:
            break
        
        try:
            cand, s = _anneal_lbfgs(init, start_time, DEADLINE,
                                     beta_start=5.0, beta_end=300.0,
                                     n_anneal_steps=12)
            if s > best_score:
                best_score = s
                best_pts = cand
        except Exception:
            pass
    
    # Phase 2: Hard-objective polish with Nelder-Mead on the best candidate
    if best_pts is not None and time.time() - start_time < DEADLINE - 8:
        x0 = best_pts.flatten()
        try:
            res = minimize(_neg_score, x0, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            if res.x is not None:
                cand = res.x.reshape(-1, 2)
                s = _score(cand)
                if s > best_score:
                    best_score = s
                    best_pts = cand
        except Exception:
            pass
    
    # Phase 3: If time remains, do another round of annealed L-BFGS-B from the best
    if best_pts is not None and time.time() - start_time < DEADLINE - 15:
        try:
            cand, s = _anneal_lbfgs(best_pts, start_time, DEADLINE,
                                     beta_start=50.0, beta_end=500.0,
                                     n_anneal_steps=10)
            if s > best_score:
                best_score = s
                best_pts = cand
        except Exception:
            pass
    
    # Phase 4: Final hard polish
    if best_pts is not None and time.time() - start_time < DEADLINE - 5:
        x0 = best_pts.flatten()
        try:
            res = minimize(_neg_score, x0, method='Nelder-Mead',
                           options={'maxiter': 1000, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True})
            if res.x is not None:
                cand = res.x.reshape(-1, 2)
                s = _score(cand)
                if s > best_score:
                    best_score = s
                    best_pts = cand
        except Exception:
            pass
    
    if best_pts is None:
        best_pts = _initial_regular_polygon(N)
    
    return best_pts
