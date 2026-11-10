import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def _compute_P(support, weights):
    """
    Compute P[X1 + X2 + X3 < 2*X4] for a discrete measure.
    support: 1D array of shape (n,)
    weights: 1D array of shape (n,)
    """
    n = len(support)
    if n == 0:
        return 0.0
    
    s = np.asarray(support, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    
    # Ensure normalized
    total_w = w.sum()
    if total_w <= 1e-12:
        return 0.0
    w = w / total_w
    
    # 1. Pair sums S_ij = s_i + s_j
    S = s[:, None] + s[None, :]
    # 2. Thresholds T_kl = 2*s_l - s_k
    T = 2.0 * s[None, :] - s[:, None]
    
    # Weights matrix W = w w^T
    W = w[:, None] * w[None, :]
    
    # --- Compute F_kl ---
    # F_kl = sum_{i,j: S_ij < T_kl} W_ij
    S_flat = S.flatten()
    T_flat = T.flatten()
    W_flat = W.flatten()
    
    order_S = np.argsort(S_flat, kind='stable')
    S_sorted = S_flat[order_S]
    W_sorted_S = W_flat[order_S]
    cum_W_S = np.cumsum(W_sorted_S)
    
    idxs_F = np.searchsorted(S_sorted, T_flat, side='left')
    padded_cum_W_S = np.concatenate(([0.0], cum_W_S))
    F_flat = padded_cum_W_S[idxs_F]
    F = F_flat.reshape(n, n)
    
    # --- Compute P ---
    # P = sum_{k,l} w_k w_l F_kl
    P = np.sum(W * F)
    
    return float(P)

def _compute_P_grid(w, grid):
    """
    Compute P for a fixed grid and weights w.
    grid: 1D array of shape (m,)
    w: 1D array of shape (m,)
    """
    m = len(grid)
    if m == 0:
        return 0.0
    
    s = grid
    w = np.asarray(w, dtype=np.float64)
    total_w = w.sum()
    if total_w <= 1e-12:
        return 0.0
    w = w / total_w
    
    # 1. Pair sums S_ij = s_i + s_j
    S = s[:, None] + s[None, :]
    # 2. Thresholds T_kl = 2*s_l - s_k
    T = 2.0 * s[None, :] - s[:, None]
    
    # Weights matrix W = w w^T
    W = w[:, None] * w[None, :]
    
    # --- Compute F_kl ---
    # F_kl = sum_{i,j: S_ij < T_kl} W_ij
    S_flat = S.flatten()
    T_flat = T.flatten()
    W_flat = W.flatten()
    
    order_S = np.argsort(S_flat, kind='stable')
    S_sorted = S_flat[order_S]
    W_sorted_S = W_flat[order_S]
    cum_W_S = np.cumsum(W_sorted_S)
    
    idxs_F = np.searchsorted(S_sorted, T_flat, side='left')
    padded_cum_W_S = np.concatenate(([0.0], cum_W_S))
    F_flat = padded_cum_W_S[idxs_F]
    F = F_flat.reshape(m, m)
    
    # --- Compute P ---
    # P = sum_{k,l} w_k w_l F_kl
    P = np.sum(W * F)
    
    return float(P)

def _objective_de(x, n):
    """
    Objective function for Differential Evolution.
    x is flattened [s_0, ..., s_{n-1}, w_0, ..., w_{n-1}]
    Returns -P (to minimize)
    """
    s = x[:n]
    w = x[n:]
    
    # Clip weights to non-negative
    w = np.maximum(w, 0)
    if w.sum() <= 1e-12:
        return 1.0 # Penalty
    
    # Normalize weights for calculation
    w_norm = w / w.sum()
    
    # Clip support to [0, 1] just in case DE goes out of bounds slightly
    s = np.clip(s, 0, 1)
    
    P = _compute_P(s, w_norm)
    return -P

def _objective_grid(w, grid):
    """
    Objective function for grid optimization.
    w: weights on grid
    grid: fixed grid points
    Returns -P
    """
    w = np.maximum(w, 0)
    if w.sum() <= 1e-12:
        return 1.0
    w = w / w.sum()
    P = _compute_P_grid(w, grid)
    return -P

def _project_simplex(w):
    """Project w onto the probability simplex."""
    n = len(w)
    w = np.maximum(w, 0)
    s = w.sum()
    if s <= 1e-12:
        return np.ones(n) / n
    return w / s

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    start_time = time.time()
    deadline = start_time + 590.0 # Safety margin
    
    best_support = None
    best_weights = None
    best_score = -1.0
    
    # Phase 1: Optimize weights on a fine grid
    grid_size = 256
    grid = np.linspace(0, 1, grid_size)
    
    # Initialize weights with a distribution that favors the condition
    # We want X1+X2+X3 < 2*X4, so we want X4 to be large and X1,X2,X3 to be small
    # So we put more weight on the right side of the grid
    w_grid = np.linspace(0.1, 1.0, grid_size)
    w_grid = w_grid / w_grid.sum()
    
    # Gradient ascent on the simplex
    step_size = 0.05
    for it in range(500):
        if time.time() > deadline - 120.0:
            break
            
        # Compute gradient numerically (central difference)
        grad = np.zeros(grid_size)
        eps = 1e-5
        for i in range(grid_size):
            # Forward
            w_plus = w_grid.copy()
            w_plus[i] += eps
            w_plus = _project_simplex(w_plus)
            f_plus = _compute_P_grid(w_plus, grid)
            
            # Backward
            w_minus = w_grid.copy()
            w_minus[i] -= eps
            w_minus = _project_simplex(w_minus)
            f_minus = _compute_P_grid(w_minus, grid)
            
            grad[i] = (f_plus - f_minus) / (2 * eps)
        
        # Update
        w_grid = w_grid + step_size * grad
        w_grid = _project_simplex(w_grid)
        
        # Decrease step size
        step_size *= 0.99
    
    # Extract support and weights from grid
    mask = w_grid > 1e-4
    support_grid = grid[mask]
    weights_grid = w_grid[mask]
    weights_grid = weights_grid / weights_grid.sum()
    
    score_grid = _compute_P(support_grid, weights_grid)
    if score_grid > best_score:
        best_score = score_grid
        best_support = support_grid.copy()
        best_weights = weights_grid.copy()
    
    # Phase 2: Optimize support and weights jointly using DE
    n_values = [10, 15, 20]
    
    for n in n_values:
        if time.time() > deadline - 60.0:
            break
            
        # Bounds for s: [0, 1], for w: [0, 1]
        bounds = [(0, 1)] * n + [(0, 1)] * n
        
        try:
            res = differential_evolution(
                _objective_de,
                bounds,
                args=(n,),
                maxiter=150,
                popsize=20,
                tol=1e-5,
                seed=42,
                polish=False
            )
            
            if time.time() > deadline - 30.0:
                break
                
            x_opt = res.x
            s_opt = x_opt[:n]
            w_opt = x_opt[n:]
            
            w_opt = np.maximum(w_opt, 0)
            if w_opt.sum() > 0:
                w_opt = w_opt / w_opt.sum()
            else:
                continue
                
            idx = np.argsort(s_opt)
            s_opt = s_opt[idx]
            w_opt = w_opt[idx]
            
            score = _compute_P(s_opt, w_opt)
            
            if score > best_score:
                best_score = score
                best_support = s_opt.copy()
                best_weights = w_opt.copy()
                
        except Exception as e:
            print(f"DE failed for n={n}: {e}")
            continue
    
    # Fallback
    if best_support is None:
        best_support = np.array([0.0, 1.0])
        best_weights = np.array([0.5, 0.5])
        
    # Prune small weights
    mask = best_weights > 1e-6
    best_support = best_support[mask]
    best_weights = best_weights[mask]
    
    if len(best_support) == 0:
        best_support = np.array([0.0, 1.0])
        best_weights = np.array([0.5, 0.5])
    else:
        best_weights = best_weights / best_weights.sum()
        
    return best_support.astype(np.float64), best_weights.astype(np.float64)
