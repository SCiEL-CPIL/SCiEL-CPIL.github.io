import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 16
D = 2

def _pairwise_distances(pts):
    """Compute all pairwise Euclidean distances."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=2))

def _score(pts):
    """Compute (max/min)^2 for pairwise distances."""
    d = _pairwise_distances(pts)
    np.fill_diagonal(d, np.inf)
    dmin = d.min()
    dmax = d.max()
    if dmin <= 1e-12:
        return np.inf
    return (dmax / dmin) ** 2

def _smooth_objective(pts_flat, tau):
    """
    Smooth approximation of the log-ratio of max to min distance.
    Minimizing this is equivalent to minimizing the ratio.
    We use log-sum-exp for max and min of log-distances.
    """
    pts = pts_flat.reshape(N, 2)
    d = _pairwise_distances(pts)
    np.fill_diagonal(d, 1.0) # Avoid log(0)
    
    # Mask diagonal
    mask = ~np.eye(N, dtype=bool)
    d_valid = d[mask]
    
    if np.any(d_valid < 1e-12):
        return 1e10
        
    log_d = np.log(d_valid)
    
    # Softmax for max: (1/tau) * log(sum(exp(tau * x)))
    # Softmin for min: -(1/tau) * log(sum(exp(-tau * x)))
    # We want to minimize (max - min).
    
    # Numerical stability: subtract max/min
    max_log_d = np.max(log_d)
    min_log_d = np.min(log_d)
    
    # Softmax of log_d
    exp_pos = np.exp(tau * (log_d - max_log_d))
    soft_max = max_log_d + np.log(np.sum(exp_pos)) / tau
    
    # Softmin of log_d
    exp_neg = np.exp(-tau * (log_d - min_log_d))
    soft_min = min_log_d - np.log(np.sum(exp_neg)) / tau
    
    # Objective: soft_max - soft_min
    # This approximates log(max_d) - log(min_d) = log(max_d/min_d)
    # Minimizing this is equivalent to minimizing the ratio.
    # The squared ratio is (max/min)^2 = exp(2 * (log(max) - log(min)))
    # Minimizing the log-ratio is sufficient and smoother.
    return soft_max - soft_min

def _make_hexagonal_cluster(n, spread=6):
    """Generate a hexagonal lattice cluster of n points centered at origin."""
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, np.sqrt(3) / 2])

    pts = []
    for i in range(-spread, spread + 1):
        for j in range(-spread, spread + 1):
            p = i * v1 + j * v2
            pts.append(p)
    pts = np.array(pts)

    # Center the cluster
    pts -= pts.mean(axis=0)

    # Sort by distance to center and pick the closest n
    dists = np.linalg.norm(pts, axis=1)
    order = np.argsort(dists)
    return pts[order[:n]]

def _make_circular_config(n, radius=1.0):
    """Points equally spaced on a circle, plus center if odd, or two concentric rings."""
    if n == 16:
        # 1 on center + 7 on inner ring + 8 on outer ring
        pts = [np.array([0.0, 0.0])]
        r1 = 1.0
        r2 = 2.0
        for k in range(7):
            theta = 2 * np.pi * k / 7
            pts.append(r1 * np.array([np.cos(theta), np.sin(theta)]))
        for k in range(8):
            theta = 2 * np.pi * k / 8 + np.pi / 8
            pts.append(r2 * np.array([np.cos(theta), np.sin(theta)]))
        return np.array(pts)
    # Generic fallback
    angles = 2 * np.pi * np.arange(n) / n
    return radius * np.column_stack([np.cos(angles), np.sin(angles)])

def _make_grid_config(n):
    """4x4 grid, centered."""
    x, y = np.meshgrid(np.arange(4), np.arange(4))
    pts = np.column_stack([x.ravel(), y.ravel()]).astype(float)
    pts -= pts.mean(axis=0)
    return pts

def _make_random_config(n, seed=0):
    """Random points in unit square, centered."""
    rng = np.random.default_rng(seed)
    pts = rng.random((n, 2))
    pts -= pts.mean(axis=0)
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit

    best_pts = None
    best_score = np.inf

    def update_best(pts, score):
        nonlocal best_pts, best_score
        if score < best_score:
            best_score = score
            best_pts = pts.copy()

    # 1. Generate diverse initial candidates
    candidates = [
        _make_hexagonal_cluster(N),
        _make_circular_config(N),
        _make_grid_config(N),
        _make_random_config(N, seed=1),
        _make_random_config(N, seed=2),
        _make_random_config(N, seed=3),
    ]

    # Evaluate all candidates
    for c in candidates:
        s = _score(c)
        update_best(c, s)

    # 2. Global Search using Differential Evolution
    # We use a smooth objective for DE to help it navigate, but evaluate raw score for tracking.
    # However, DE works on the flat vector.
    def de_objective(pts_flat):
        # Use a moderate tau for DE to keep it smooth but representative
        return _smooth_objective(pts_flat, tau=10.0)

    if time.time() < deadline - 60:
        try:
            bounds = [(-5, 5)] * (N * 2)
            result = differential_evolution(
                de_objective,
                bounds,
                seed=42,
                maxiter=100,
                popsize=20,
                tol=1e-6,
                polish=False,
                init='sobol',
            )
            if result.fun < np.inf:
                new_pts = result.x.reshape(N, 2)
                new_score = _score(new_pts)
                update_best(new_pts, new_score)
        except Exception:
            pass

    # 3. Local Refinement using L-BFGS-B on the smooth objective
    # We will refine the best candidates found so far.
    
    def local_refine(pts_init, time_limit):
        """Refine a configuration using L-BFGS-B with annealing tau."""
        current_pts = pts_init.copy()
        current_score = _score(current_pts)
        
        # Annealing schedule: start with low tau (smooth), increase to high tau (sharp)
        taus = [5.0, 10.0, 20.0, 50.0, 100.0, 200.0]
        
        for tau in taus:
            if time.time() > time_limit:
                break
                
            # Define the objective for this specific tau
            def obj_flat(x):
                return _smooth_objective(x, tau)
                
            try:
                res = minimize(
                    obj_flat,
                    current_pts.ravel(),
                    method='L-BFGS-B',
                    options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-15}
                )
                new_pts = res.x.reshape(N, 2)
                new_score = _score(new_pts)
                
                if new_score < current_score:
                    current_pts = new_pts
                    current_score = new_score
            except Exception:
                pass
        
        return current_pts, current_score

    # Refine top candidates from initial set and DE result
    candidates_to_refine = [c for c in candidates]
    if best_pts is not None:
        candidates_to_refine.append(best_pts)
    
    # Sort by raw score to prioritize best ones
    candidates_to_refine.sort(key=lambda x: _score(x))
    
    # Refine the top 5 distinct candidates
    refined_count = 0
    for c in candidates_to_refine:
        if time.time() > deadline - 20:
            break
        if refined_count >= 5:
            break
            
        # Avoid refining exact duplicates
        is_dup = False
        for r in candidates_to_refine[:refined_count]:
            if np.allclose(c, r, atol=1e-5):
                is_dup = True
                break
        if is_dup:
            continue
            
        try:
            refined_pts, refined_score = local_refine(c, deadline - 15)
            update_best(refined_pts, refined_score)
            refined_count += 1
        except Exception:
            pass

    # 4. Final Polish with Powell on the raw objective (non-smooth)
    # This can help escape slight local minima of the smooth approx if the smooth approx is too soft.
    if best_pts is not None and time.time() < deadline - 5:
        try:
            def raw_obj(x):
                pts = x.reshape(N, 2)
                return _score(pts)
            
            res = minimize(
                raw_obj,
                best_pts.ravel(),
                method='Powell',
                options={'maxiter': 2000, 'maxfev': 50000, 'xtol': 1e-12, 'ftol': 1e-12}
            )
            if res.fun < best_score:
                new_pts = res.x.reshape(N, 2)
                new_score = _score(new_pts)
                update_best(new_pts, new_score)
        except Exception:
            pass

    if best_pts is None:
        return _make_hexagonal_cluster(N)

    return best_pts
