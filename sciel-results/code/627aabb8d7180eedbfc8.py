import numpy as np
import scipy.optimize as so
import time
import itertools

N = 14
D = 3
SEED = 42

def _pairwise_dist2(X):
    """Compute squared pairwise distances matrix for X (N x D)."""
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    return np.sum(diff**2, axis=2)

def _raw_score(X):
    """
    Calculate the actual score: (max_dist / min_dist)^2.
    Returns 1e10 if degenerate.
    """
    if X.shape != (N, D):
        return 1e10
    dist2 = _pairwise_dist2(X)
    mask = ~np.eye(N, dtype=bool)
    d2 = dist2[mask]
    if len(d2) == 0:
        return 1e10
    d_min = np.min(d2)
    d_max = np.max(d2)
    if d_min < 1e-12:
        return 1e10
    return (d_max / d_min)

def _objective_smooth(X_flat, tau):
    """
    Smooth objective function for optimization.
    Approximates (max d / min d)^2 using softmax of log distances.
    log_d is the log of squared distances.
    max_d ~ exp(soft_max(log_d))
    min_d ~ exp(soft_min(log_d))
    ratio = max_d / min_d
    score = ratio^2
    """
    X = X_flat.reshape(N, D)
    dist2 = _pairwise_dist2(X)
    mask = ~np.eye(N, dtype=bool)
    d2 = dist2[mask]
    
    # Filter out very small distances to avoid log(0) issues
    # We add a small epsilon
    d2_safe = np.maximum(d2, 1e-10)
    log_d2 = np.log(d2_safe)
    
    # Softmax for max and min
    # To avoid overflow in exp, we subtract max/min first?
    # Standard softmax: exp(x - max(x)) / sum(exp(x - max(x)))
    # But we want the "value" of the soft max, which is log(sum(exp(x)))
    # Actually, soft_max(x) = log(sum(exp(x)))
    # soft_min(x) = -log(sum(exp(-x)))
    
    # We want to minimize (max/min)^2.
    # Let L = log(max/min) = log(max) - log(min).
    # score = exp(2*L).
    # So we can minimize 2 * (soft_max(log_d) - soft_min(log_d)).
    # This is smoother and avoids the square until the end?
    # Minimizing log(score) is equivalent to minimizing score (monotonic).
    # Let's return log_score to keep values bounded.
    
    max_log = np.max(log_d2)
    min_log = np.min(log_d2)
    
    # Soft max
    exp_d2_max = np.exp(log_d2 - max_log)
    soft_max_log = max_log + np.log(np.sum(exp_d2_max) / tau) # Wait, standard softmax is 1/tau * log(sum(exp(tau*x)))?
    # Let's stick to standard:
    # soft_max(x, tau) = (1/tau) * log(sum(exp(tau*x)))
    # soft_min(x, tau) = -(1/tau) * log(sum(exp(-tau*x)))
    
    # To prevent overflow:
    # soft_max = (1/tau) * (tau*max_log + log(sum(exp(tau*(log_d2 - max_log)))))
    # soft_min = -(1/tau) * (tau*min_log + log(sum(exp(-tau*(log_d2 - min_log)))))
    
    if tau > 0:
        term_max = tau * (log_d2 - max_log)
        sum_exp_max = np.sum(np.exp(term_max))
        soft_max_val = max_log + (1/tau) * np.log(sum_exp_max)
        
        term_min = -tau * (log_d2 - min_log)
        sum_exp_min = np.sum(np.exp(term_min))
        soft_min_val = min_log - (1/tau) * np.log(sum_exp_min)
    else:
        soft_max_val = max_log
        soft_min_val = min_log
        
    log_ratio = soft_max_val - soft_min_val
    return 2.0 * log_ratio # This is log(score)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s safety margin
    
    best_score = 1e10
    best_X = None
    
    np.random.seed(SEED)
    
    # Generate diverse initial points
    inits = []
    
    # 1. Fibonacci sphere
    golden_angle = np.pi * (3. - np.sqrt(5.))
    X_fib = np.zeros((N, 3))
    for i in range(N):
        z = 1. - (i / (N - 1.)) * 2.
        radius = np.sqrt(1. - z**2)
        theta = golden_angle * i
        x = np.cos(theta) * radius
        y = np.sin(theta) * radius
        X_fib[i] = [x, y, z]
    inits.append(X_fib)
    
    # 2. Random points on sphere
    for _ in range(5):
        pts = np.random.randn(N, 3)
        norms = np.linalg.norm(pts, axis=1)
        pts /= norms[:, np.newaxis]
        inits.append(pts)
        
    # 3. Random points in ball
    for _ in range(5):
        pts = np.random.randn(N, 3)
        # Scale to fit in unit ball
        max_norm = np.max(np.linalg.norm(pts, axis=1))
        if max_norm > 0:
            pts /= max_norm
        inits.append(pts)
        
    # 4. Lattice-like
    side = int(np.ceil(N ** (1/3))) + 1
    grid = list(itertools.product(range(side), repeat=3))[:N]
    X_lat = np.array(grid, dtype=float)
    X_lat -= np.mean(X_lat, axis=0)
    inits.append(X_lat)
    
    # Optimization Strategy:
    # For each init, run a multi-stage tau annealing with L-BFGS-B.
    # Then polish with Nelder-Mead or L-BFGS-B with high tau.
    
    tau_schedule = [10, 50, 200, 500]
    
    for init_idx, X_init in enumerate(inits):
        if time.time() > deadline:
            break
            
        x_current = X_init.flatten()
        
        # Center and scale to improve conditioning
        x_c = x_current.reshape(N, D) - np.mean(X_init, axis=0)
        # Scale to unit min distance?
        # Just centering is usually enough for sphere/ball inits.
        # For lattice, scaling helps.
        x_current = x_c.flatten()
        
        for tau in tau_schedule:
            if time.time() > deadline - 5:
                break
                
            try:
                res = so.minimize(
                    _objective_smooth, 
                    x_current, 
                    args=(tau,),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                if res.x is not None:
                    x_current = res.x
            except Exception:
                pass
        
        # Evaluate current best
        pts = x_current.reshape(N, D)
        score = _raw_score(pts)
        if score < best_score:
            best_score = score
            best_X = pts.copy()
            
        # Final Polish:
        # Use a high tau L-BFGS-B step, then Nelder-Mead on the raw objective?
        # Nelder-Mead on non-differentiable min/max is slow but robust for local polishing.
        if time.time() < deadline - 2:
             try:
                 # High tau L-BFGS-B
                 res_polish = so.minimize(
                     _objective_smooth, 
                     x_current, 
                     args=(1000.0,),
                     method='L-BFGS-B',
                     options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
                 )
                 if res_polish.x is not None:
                     x_polish = res_polish.x
                     pts_p = x_polish.reshape(N, D)
                     score_p = _raw_score(pts_p)
                     if score_p < best_score:
                         best_score = score_p
                         best_X = pts_p.copy()
                         x_current = x_polish
             except Exception:
                 pass

    if best_X is None:
        # Fallback
        side = int(np.ceil(N ** (1 / D)))
        grid = list(itertools.product(range(side), repeat=D))[:N]
        return np.array(grid, dtype=float)
        
    return best_X
