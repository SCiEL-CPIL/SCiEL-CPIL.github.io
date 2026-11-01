import numpy as np
import time
import itertools
from scipy.optimize import minimize

N = 14
D = 3

def _pairwise_sq_dists(X):
    """
    Compute squared pairwise distances for N points in D dimensions.
    Returns a flat array of length N*(N-1)/2.
    """
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=-1)
    iu = np.triu_indices(N, k=1)
    return sq_dists[iu]

def _objective_smooth(X_flat, tau=50.0):
    """
    Smooth approximation of the log ratio of max to min squared distance.
    Uses Log-Sum-Exp for soft max and soft min.
    This objective is differentiable, allowing gradient-based methods like L-BFGS-B
    to navigate the landscape more effectively than derivative-free methods like Powell.
    """
    X = X_flat.reshape(N, D)
    
    # Normalize to prevent scale invariance issues (fixes scale to unit variance/radius)
    # We center and normalize to unit mean squared norm
    X = X - np.mean(X, axis=0)
    norm_sq = np.sum(X**2)
    if norm_sq < 1e-12:
        return 1e10
    X = X / np.sqrt(norm_sq / N) # Scale such that mean sq norm is 1
    
    sq_dists = _pairwise_sq_dists(X)
    sq_dists = np.maximum(sq_dists, 1e-12)
    
    # Soft max
    m_max = np.max(sq_dists)
    exp_max = np.exp(tau * (sq_dists - m_max))
    log_sum_exp_max = m_max + np.log(np.sum(exp_max)) / tau
    
    # Soft min
    m_min = np.min(sq_dists)
    exp_min = np.exp(-tau * (sq_dists - m_min))
    log_sum_exp_min = m_min - np.log(np.sum(exp_min)) / tau
    
    # Minimize log(soft_max) - log(soft_min)
    obj_val = np.log(log_sum_exp_max) - np.log(log_sum_exp_min)
    
    return obj_val

def _objective_raw(X_flat):
    """
    Direct non-smooth objective for final polishing or evaluation.
    Minimizes (max/min)^2.
    """
    X = X_flat.reshape(N, D)
    sq_dists = _pairwise_sq_dists(X)
    if np.min(sq_dists) < 1e-12:
        return 1e10
    ratio_sq = np.max(sq_dists) / np.min(sq_dists)
    return ratio_sq

def construct():
    start_time = time.time()
    # Use 135 seconds as deadline to be safe against 150s limit
    deadline = start_time + 135.0
    
    best_score = float('inf')
    best_X = None
    
    def update_best(X_candidate):
        nonlocal best_score, best_X
        if X_candidate is None:
            return
        X_candidate = np.asarray(X_candidate)
        if X_candidate.shape != (N, D):
            return
        
        # Compute raw score
        score = _objective_raw(X_candidate.flatten())
        
        if score < best_score:
            best_score = score
            best_X = X_candidate.copy()
            
    # Generate initial candidates
    candidates = []
    
    # 1. Fibonacci sphere
    # This is a good deterministic start
    golden_angle = np.pi * (3 - np.sqrt(5))
    fib_pts = np.zeros((N, D))
    for i in range(N):
        z = 1 - 2 * i / (N - 1)
        r = np.sqrt(max(0, 1 - z**2))
        theta = golden_angle * i
        fib_pts[i, 0] = r * np.cos(theta)
        fib_pts[i, 1] = r * np.sin(theta)
        fib_pts[i, 2] = z
    candidates.append(fib_pts)
    
    # 2. Icosahedron + Poles (Symmetric structure often helps)
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    p_top = np.array([0, 0, 1.0])
    p_bot = np.array([0, 0, -1.0])
    pts_sym = np.vstack([ico, p_top.reshape(1, D), p_bot.reshape(1, D)])
    candidates.append(pts_sym)
    
    # 3. Random starts
    rng = np.random.default_rng(42)
    for _ in range(20):
        # Random on sphere
        pts = rng.normal(size=(N, D))
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        candidates.append(pts)
        
    # 4. Random in cube (centered)
    for _ in range(20):
        pts = rng.rand(N, D)
        pts -= np.mean(pts, axis=0)
        candidates.append(pts)
        
    # Optimization Strategy:
    # Phase 1: Smooth optimization using L-BFGS-B on the log-sum-exp objective.
    # This is fast and handles the non-differentiability of min/max via smoothing.
    # Phase 2: Polish with Nelder-Mead on the raw objective to remove smoothing bias.
    
    num_restarts = len(candidates)
    
    for i in range(num_restarts):
        if time.time() > deadline - 20: # Leave time for final polish
            break
            
        X0 = candidates[i]
        
        # Try L-BFGS-B on smooth objective
        try:
            res = minimize(_objective_smooth, X0.flatten(), method='L-BFGS-B',
                           options={'maxiter': 200, 'maxfun': 10000})
            
            X_opt = res.x.reshape(N, D)
            update_best(X_opt)
            
            # If we have time, do a local polish with Nelder-Mead on the raw objective
            # This helps snap to the exact local minimum of the non-smooth function
            if time.time() < deadline - 10:
                res2 = minimize(_objective_raw, X_opt.flatten(), method='Nelder-Mead',
                                options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10})
                X_polished = res2.x.reshape(N, D)
                update_best(X_polished)
                
        except Exception:
            continue
            
    # If we have time left, do a final polish on the absolute best found
    if best_X is not None and time.time() < deadline - 5:
        try:
            # A final, more aggressive polish
            res_final = minimize(_objective_raw, best_X.flatten(), method='Nelder-Mead',
                                 options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12})
            X_final = res_final.x.reshape(N, D)
            update_best(X_final)
        except Exception:
            pass

    # Fallback
    if best_X is None:
        side = int(np.ceil(N ** (1 / D)))
        grid = list(itertools.product(range(side), repeat=D))[:N]
        best_X = np.array(grid, dtype=float)
        
    return best_X
