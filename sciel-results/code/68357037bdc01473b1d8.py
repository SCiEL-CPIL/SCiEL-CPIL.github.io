# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution, minimize

def _compute_score(points):
    """Compute R = (d_min/d_max)^2 for a set of points."""
    if points.shape[0] < 2:
        return 0.0
    d = pdist(points)
    if len(d) == 0:
        return 0.0
    d_min = np.min(d)
    d_max = np.max(d)
    if d_max == 0:
        return 0.0
    if d_min == 0:
        return 0.0
    return (d_min / d_max) ** 2

def _neg_smooth_objective(x, p_min=8.0, p_max=8.0):
    """
    Smooth surrogate objective for minimization.
    Minimizes M_max^p_max - M_min^p_min, which approximates maximizing min/max.
    Actually, we want to maximize min/max.
    Let's minimize: 1/(M_min^p) + M_max^q ? No.
    
    Better: minimize -log(M_min) + log(M_max) is non-smooth.
    
    Let's use: M_max^p - M_min^p. Minimizing this encourages M_min to be large and M_max to be small.
    Since M_min <= min(d) and M_max >= max(d), this is a valid surrogate.
    Wait, if M_min is small, -M_min^p is large negative, so we maximize M_min^p.
    If M_max is large, M_max^p is large positive, so we minimize M_max^p.
    So minimizing M_max^p - M_min^p works.
    """
    pts = x.reshape(16, 2)
    d = pdist(pts)
    if len(d) == 0:
        return 1e10
    
    # Power means
    # M_p = (mean(d^p))^(1/p)
    # For p -> infinity, M_p -> max
    # For p -> -infinity, M_p -> min
    # But we want to maximize min and minimize max.
    
    # Let's try minimizing: (1/p) * log(mean(exp(p*d))) for max (soft max)
    # and minimizing: -(1/p) * log(mean(exp(-p*d))) for min (soft min)
    
    # Objective: soft_max(d) - soft_min(d)
    # This is smooth and differentiable.
    
    # Use a temperature parameter p
    p = 20.0
    
    # Soft max: (1/p) * log(sum(exp(p*d)))
    # Shift to avoid overflow
    d_max_val = np.max(d)
    soft_max = (1.0 / p) * np.log(np.sum(np.exp(p * (d - d_max_val)))) + d_max_val
    
    # Soft min: -(1/p) * log(sum(exp(-p*d)))
    d_min_val = np.min(d)
    soft_min = -(1.0 / p) * np.log(np.sum(np.exp(-p * (d - d_min_val)))) + d_min_val
    
    return soft_max - soft_min

def _refine(x0, deadline):
    """Refine a candidate solution using L-BFGS-B on the smooth objective."""
    try:
        res = minimize(_neg_smooth_objective, x0, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-8})
        if res.success or res.fun < 1e6:
            pts = res.x.reshape(16, 2)
            s = _compute_score(pts)
            return pts, s
    except Exception:
        pass
    return None, 0.0

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    """
    n = 16
    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    
    best_points = None
    best_score = -1.0
    
    # Initial guesses
    initial_guesses = []
    
    # 1. Hexagonal lattice
    hex_pts = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            s = -q - r
            if abs(q) <= 3 and abs(r) <= 3 and abs(s) <= 3:
                x = 1.5 * q
                y = np.sqrt(3) * (r + q / 2)
                hex_pts.append([x, y])
    hex_pts = np.array(hex_pts)
    hex_pts -= np.mean(hex_pts, axis=0)
    dists_to_origin = np.sum(hex_pts**2, axis=1)
    idx = np.argsort(dists_to_origin)[:16]
    hex_pts = hex_pts[idx]
    hex_pts /= np.sqrt(np.sum(hex_pts**2) / n)
    initial_guesses.append(hex_pts.flatten())
    
    # 2. Grid
    x = np.linspace(-1.5, 1.5, 4)
    y = np.linspace(-1.5, 1.5, 4)
    xx, yy = np.meshgrid(x, y)
    grid_pts = np.column_stack((xx.ravel(), yy.ravel()))
    grid_pts -= np.mean(grid_pts, axis=0)
    grid_pts /= np.sqrt(np.sum(grid_pts**2) / n)
    initial_guesses.append(grid_pts.flatten())
    
    # 3. Ring
    angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
    ring_pts = np.column_stack((np.cos(angles), np.sin(angles)))
    initial_guesses.append(ring_pts.flatten())
    
    # 4. Random
    np.random.seed(42)
    rand_pts = np.random.randn(n, 2)
    rand_pts -= np.mean(rand_pts, axis=0)
    rand_pts /= np.sqrt(np.sum(rand_pts**2) / n)
    initial_guesses.append(rand_pts.flatten())
    
    # Refine initial guesses
    for x0 in initial_guesses:
        if time.time() > deadline - 10:
            break
        pts, s = _refine(x0, deadline)
        if pts is not None and s > best_score:
            best_score = s
            best_points = pts.copy()
    
    # Differential Evolution
    if time.time() < deadline - 60:
        try:
            # Bounds: assume points are in [-2, 2]
            bounds = [(-2, 2)] * (n * 2)
            
            # Callback to update best
            def callback(xk, convergence):
                nonlocal best_points, best_score
                pts = xk.reshape(n, 2)
                s = _compute_score(pts)
                if s > best_score:
                    best_score = s
                    best_points = pts.copy()
                
                # Check time
                if time.time() > deadline - 30:
                    raise ValueError("Time limit reached")
            
            res = differential_evolution(
                _neg_smooth_objective,
                bounds,
                maxiter=50,
                popsize=15,
                tol=1e-5,
                mutation=(0.5, 1.5),
                recombination=0.9,
                seed=42,
                callback=callback,
                polish=False  # We will polish later
            )
            
            # Refine the best result from DE
            if res.fun < 1e6:
                pts, s = _refine(res.x, deadline)
                if pts is not None and s > best_score:
                    best_score = s
                    best_points = pts.copy()
                
        except Exception as e:
            # Continue even if DE fails
            pass
    
    # Final refinement loop
    if time.time() < deadline - 10 and best_points is not None:
        # Try a few more refinements with perturbations
        for _ in range(5):
            if time.time() > deadline - 5:
                break
            x0 = best_points.flatten() + 0.1 * np.random.randn(n * 2)
            pts, s = _refine(x0, deadline)
            if pts is not None and s > best_score:
                best_score = s
                best_points = pts.copy()
    
    if best_points is None:
        # Fallback
        best_points = initial_guesses[0].reshape(n, 2)
    
    return best_points

# EVOLVE-BLOCK-END
