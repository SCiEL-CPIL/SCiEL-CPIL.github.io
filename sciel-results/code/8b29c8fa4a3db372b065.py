# EVOLVE-BLOCK-START
import numpy as np

def _entropy_of_projection(P, r):
    """Shannon entropy (bits) of the projected distribution x + r*y."""
    n = P.shape[0]
    xs = np.arange(n, dtype=np.float64)
    ys = np.arange(n, dtype=np.float64)
    vals = xs[None, :] + r * ys[:, None]
    uniq, inv = np.unique(vals, return_inverse=True)
    pooled = np.bincount(inv.ravel(), weights=P.ravel(), minlength=len(uniq))
    pooled = pooled[pooled > 0.0]
    if len(pooled) == 0:
        return 0.0
    return float(-(pooled * np.log2(pooled)).sum())

def _ratio(P):
    D = _entropy_of_projection(P, -1.0)
    M = max(_entropy_of_projection(P, 0.0),
            _entropy_of_projection(P, 1.0),
            _entropy_of_projection(P, 2.0))
    if M <= 1e-9:
        return -np.inf
    return D / M

def _evaluate_weights(weights):
    """Evaluate the ratio for integer weights."""
    P = weights.astype(np.float64)
    s = P.sum()
    if s < 1e-12:
        return -np.inf
    P = P / s
    return _ratio(P)

def _anneal_search():
    n = 10
    max_iter = 20000
    T_start = 1.0
    T_end = 0.01
    alpha = (T_end / T_start) ** (1.0 / max_iter)
    
    # Initial random integer weights
    # Bias towards non-zero to avoid empty states
    np.random.seed(42)
    w = np.random.randint(0, 5, (n, n))
    w = np.maximum(w, 1) # Ensure non-zero start
    
    current_score = _evaluate_weights(w)
    if not np.isfinite(current_score):
        w = np.ones((n, n))
        current_score = _evaluate_weights(w)
        
    best_w = w.copy()
    best_score = current_score
    
    for i in range(max_iter):
        T = T_start * (alpha ** i)
        
        # Pick a random cell to modify
        r, c = np.random.randint(0, n, 2)
        
        # Propose change: add or subtract a small integer amount
        # Weights should stay non-negative
        delta = np.random.choice([-2, -1, 1, 2])
        if w[r, c] + delta < 0:
            continue
            
        w_new = w.copy()
        w_new[r, c] += delta
        
        new_score = _evaluate_weights(w_new)
        
        if new_score > current_score:
            w = w_new
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_w = w.copy()
        else:
            # Metropolis criterion
            if np.random.rand() < np.exp((new_score - current_score) / T):
                w = w_new
                current_score = new_score
                
    return best_w

def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    # Run multiple restarts with different seeds/strategies
    best_P = None
    best_score = -np.inf
    
    strategies = [
        lambda: _anneal_search(),
        # Add another variant: start with a specific pattern and anneal
        lambda: _anneal_search_from_pattern(np.eye(n) * 5),
        lambda: _anneal_search_from_pattern(np.ones((n, n)))
    ]
    
    for strat in strategies:
        try:
            w_best = strat()
            P = w_best.astype(np.float64)
            s = P.sum()
            if s > 1e-9:
                P = P / s
                score = _ratio(P)
                if score > best_score:
                    best_score = score
                    best_P = P
        except Exception:
            continue
            
    # Fallback if all failed
    if best_P is None or not np.isfinite(best_score):
        best_P = np.ones((n, n)) / (n * n)
        
    # Final safety check
    P = np.clip(best_P, 0.0, None)
    s = P.sum()
    if s <= 1e-9:
        P = np.ones((n, n)) / (n * n)
    else:
        P = P / s
        
    return P

def _anneal_search_from_pattern(pattern):
    n = 10
    max_iter = 10000
    T_start = 0.5
    T_end = 0.01
    alpha = (T_end / T_start) ** (1.0 / max_iter)
    
    np.random.seed(123)
    w = pattern.astype(np.float64)
    # Discretize to integers for the search space, scaling up
    w_int = (w * 10).astype(int)
    w_int = np.maximum(w_int, 0)
    
    current_score = _evaluate_weights(w_int)
    if not np.isfinite(current_score):
        w_int = np.ones((n, n))
        current_score = _evaluate_weights(w_int)
        
    best_w = w_int.copy()
    best_score = current_score
    
    for i in range(max_iter):
        T = T_start * (alpha ** i)
        r, c = np.random.randint(0, n, 2)
        delta = np.random.choice([-2, -1, 1, 2])
        if w_int[r, c] + delta < 0:
            continue
            
        w_new = w_int.copy()
        w_new[r, c] += delta
        
        new_score = _evaluate_weights(w_new)
        
        if new_score > current_score:
            w_int = w_new
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_w = w_int.copy()
        else:
            if np.random.rand() < np.exp((new_score - current_score) / T):
                w_int = w_new
                current_score = new_score
                
    return best_w
# EVOLVE-BLOCK-END
