from scipy.optimize import minimize
import numpy as np
import time

N = 16

def _pairwise_dists_sq(P):
    """Compute squared pairwise distances for points P of shape (n, 2)."""
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    d_sq = np.sum(diff ** 2, axis=-1)
    # Zero out diagonal
    np.fill_diagonal(d_sq, 0.0)
    return d_sq

def _exact_objective(params):
    """
    Exact objective: (max pairwise dist / min pairwise dist)^2
    params: flattened array of shape (28,) representing points 2..15.
    Point 0 is (0,0), Point 1 is (1,0).
    """
    P = np.zeros((N, 2))
    P[0] = [0.0, 0.0]
    P[1] = [1.0, 0.0]
    P[2:] = params.reshape(-1, 2)
    
    d_sq = _pairwise_dists_sq(P)
    # Get non-zero distances
    mask = d_sq > 1e-12
    if not np.any(mask):
        return 1e10
    dists_sq = d_sq[mask]
    min_d_sq = np.min(dists_sq)
    max_d_sq = np.max(dists_sq)
    
    if min_d_sq < 1e-12:
        return 1e10
        
    return (max_d_sq / min_d_sq)

def _soft_objective(params, beta):
    """
    Soft min/max surrogate for gradient-based optimization in log-space.
    params: flattened array of shape (28,) representing points 2..15.
    Point 0 is (0,0), Point 1 is (1,0).
    """
    P = np.zeros((N, 2))
    P[0] = [0.0, 0.0]
    P[1] = [1.0, 0.0]
    P[2:] = params.reshape(-1, 2)
    
    d_sq = _pairwise_dists_sq(P)
    # Ensure positivity for log
    d_sq = np.maximum(d_sq, 1e-12)
    
    log_d_sq = np.log(d_sq)
    
    # Soft max of log(d_sq)
    max_log = np.max(log_d_sq)
    soft_max_log = max_log + (1.0 / beta) * np.log(np.sum(np.exp(beta * (log_d_sq - max_log))))
    
    # Soft min of log(d_sq)
    min_log = np.min(log_d_sq)
    soft_min_log = min_log - (1.0 / beta) * np.log(np.sum(np.exp(-beta * (log_d_sq - min_log))))
    
    # Objective is soft_max_log - soft_min_log
    # This is equivalent to log(soft_max / soft_min)
    return soft_max_log - soft_min_log

def _local_optimize(params, maxiter_nm=300, maxiter_lbfgs=500, deadline=None):
    """
    Two-phase local optimization:
    1. Powell on exact objective (robust for non-smooth)
    2. L-BFGS-B on soft objective with annealing
    Returns best params and exact score.
    """
    best_params = params.copy()
    best_score = _exact_objective(params)
    
    # Phase 1: Powell
    try:
        if deadline and time.time() > deadline:
            return best_params, best_score
        res_powell = minimize(
            _exact_objective,
            best_params,
            method='Powell',
            options={'maxiter': maxiter_nm, 'xtol': 1e-8, 'ftol': 1e-8, 'disp': False}
        )
        if res_powell.fun < best_score:
            best_score = res_powell.fun
            best_params = res_powell.x.copy()
    except Exception:
        pass
    
    # Phase 2: L-BFGS-B with annealing on soft objective
    betas = [10.0, 50.0, 200.0, 1000.0]
    for beta in betas:
        if deadline and time.time() > deadline:
            return best_params, best_score
        try:
            res_lbfgs = minimize(
                lambda p: _soft_objective(p, beta),
                best_params,
                method='L-BFGS-B',
                options={'maxiter': maxiter_lbfgs, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
            )
            exact_s = _exact_objective(res_lbfgs.x)
            if exact_s < best_score:
                best_score = exact_s
                best_params = res_lbfgs.x.copy()
        except Exception:
            pass
    
    return best_params, best_score

def construct():
    start_time = time.time()
    # Deadline: 120s limit, use 110s for safety
    deadline = start_time + 110.0
    
    best_params = None
    best_score = float('inf')
    
    # Multi-start strategy based on knowledge: rings/spirals
    # We fix P[0]=(0,0) and P[1]=(1,0). We optimize P[2..15].
    # Initializations:
    inits = []
    
    # 1. Random
    inits.append(np.random.randn(14, 2) * 2.0)
    
    # 2. Ring around origin
    angles = np.linspace(0, 2*np.pi, 14, endpoint=False)
    radii = np.ones(14) * 1.0
    ring = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    inits.append(ring.flatten())
    
    # 3. Spiral
    t = np.linspace(0, 3*np.pi, 14)
    r = np.linspace(0.5, 2.0, 14)
    spiral = np.column_stack([r * np.cos(t), r * np.sin(t)])
    inits.append(spiral.flatten())
    
    # 4. Grid-like
    grid_x = np.linspace(-1, 2, 14)
    grid_y = np.zeros(14)
    grid = np.column_stack([grid_x, grid_y])
    inits.append(grid.flatten())
    
    # 5. More randoms
    for _ in range(5):
        inits.append(np.random.randn(14, 2) * 1.5)
        
    for init in inits:
        if time.time() > deadline:
            break
        # Ensure init is valid shape
        if init.shape != (28,):
            continue
        p, s = _local_optimize(init, deadline=deadline)
        if s < best_score:
            best_score = s
            best_params = p
            
    if best_params is None:
        # Fallback to random if all failed
        best_params = np.random.randn(14, 2) * 1.0
        best_score = _exact_objective(best_params)
        
    # Final check
    if time.time() > deadline:
        pass # Just return what we have
        
    # Reconstruct points
    P = np.zeros((N, 2))
    P[0] = [0.0, 0.0]
    P[1] = [1.0, 0.0]
    P[2:] = best_params.reshape(-1, 2)
    
    return P
