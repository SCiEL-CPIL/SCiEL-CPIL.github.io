"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
D = 2

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def _pairwise_dists(pts):
    """Compute all pairwise distances for N points in 2D."""
    n = len(pts)
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            diff = pts[i] - pts[j]
            d[i, j] = d[j, i] = np.sqrt(np.dot(diff, diff))
    return d

def _objective(pts):
    """Smooth surrogate: variance of log pairwise distances."""
    d = _pairwise_dists(pts)
    # Get upper triangle distances
    iu = np.triu_indices(N, 1)
    dists = d[iu]
    # Filter out any zero or near-zero distances
    dists = dists[dists > 1e-12]
    if len(dists) == 0:
        return 1e10
    log_dists = np.log(dists)
    return np.var(log_dists)

def _actual_score(pts):
    """Compute the actual score: (max/min)^2."""
    d = _pairwise_dists(pts)
    iu = np.triu_indices(N, 1)
    dists = d[iu]
    dists = dists[dists > 1e-12]
    if len(dists) == 0:
        return 1e10
    return (np.max(dists) / np.min(dists)) ** 2

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit
    
    best_score = 1e10
    best_pts = None
    
    # Try multiple initial configurations
    rng = np.random.RandomState(42)
    
    initial_configs = []
    
    # 1. Points on a circle
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    circle_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    initial_configs.append(circle_pts)
    
    # 2. Perturbed grid
    side = 4
    grid = list(itertools.product(range(side), repeat=2))[:N]
    grid_pts = np.array(grid, dtype=float)
    # Scale and center
    grid_pts = grid_pts - grid_pts.mean(axis=0)
    grid_pts = grid_pts / np.max(np.abs(grid_pts))
    initial_configs.append(grid_pts)
    
    # 3. Random points in unit square
    for _ in range(10):
        rand_pts = rng.random((N, 2))
        rand_pts = rand_pts - rand_pts.mean(axis=0)
        rand_pts = rand_pts / (np.max(np.abs(rand_pts)) + 1e-12)
        initial_configs.append(rand_pts)
    
    # 4. Points on two concentric circles
    n1 = 8
    n2 = 8
    ang1 = np.linspace(0, 2*np.pi, n1, endpoint=False)
    ang2 = np.linspace(0, 2*np.pi, n2, endpoint=False) + np.pi/n2
    c1 = np.column_stack([np.cos(ang1), np.sin(ang1)])
    c2 = np.column_stack([1.5*np.cos(ang2), 1.5*np.sin(ang2)])
    two_circle = np.vstack([c1, c2])
    initial_configs.append(two_circle)
    
    from scipy.optimize import minimize
    
    for init_idx, init_pts in enumerate(initial_configs):
        if time.time() > deadline - 5.0:
            break
        
        # Try a few optimization methods
        for method in ['L-BFGS-B', 'Powell']:
            if time.time() > deadline - 3.0:
                break
            
            try:
                result = minimize(
                    _objective,
                    init_pts.flatten(),
                    method=method,
                    options={'maxiter': 2000, 'maxfev': 5000}
                )
                
                if result.success or result.fun < best_score:
                    pts = result.x.reshape(N, 2)
                    score = _actual_score(pts)
                    if score < best_score:
                        best_score = score
                        best_pts = pts.copy()
                        
            except Exception:
                pass
        
        # Also try with different scaling of initial config
        if time.time() > deadline - 5.0:
            break
        
        for scale in [0.5, 1.0, 2.0]:
            if time.time() > deadline - 3.0:
                break
            try:
                scaled_init = init_pts * scale
                result = minimize(
                    _objective,
                    scaled_init.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 1000, 'maxfev': 3000}
                )
                if result.success or result.fun < best_score:
                    pts = result.x.reshape(N, 2)
                    score = _actual_score(pts)
                    if score < best_score:
                        best_score = score
                        best_pts = pts.copy()
            except Exception:
                pass
    
    # Final polish: use Nelder-Mead on the best found
    if best_pts is not None and time.time() < deadline - 2.0:
        try:
            result = minimize(
                _objective,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if result.fun < best_score:
                pts = result.x.reshape(N, 2)
                score = _actual_score(pts)
                if score < best_score:
                    best_score = score
                    best_pts = pts.copy()
        except Exception:
            pass
    
    # If no optimization succeeded, return the best initial config by score
    if best_pts is None:
        for init_pts in initial_configs:
            score = _actual_score(init_pts)
            if score < best_score:
                best_score = score
                best_pts = init_pts.copy()
    
    if best_pts is None:
        # Fallback: simple grid
        side = 4
        grid = list(itertools.product(range(side), repeat=2))[:N]
        best_pts = np.array(grid, dtype=float)
    
    return best_pts
# EVOLVE-BLOCK-END
