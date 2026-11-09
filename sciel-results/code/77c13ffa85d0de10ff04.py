import numpy as np
from scipy.optimize import differential_evolution

def _compute_P_vec(s, w):
    """
    Compute P[X1+X2+X3 < 2*X4] for discrete measure defined by support s and weights w.
    """
    n = len(s)
    S = s[:, None] + s[None, :]
    W = w[:, None] * w[None, :]
    
    P = 0.0
    for l in range(n):
        for k in range(n):
            threshold = 2.0 * s[l] - s[k]
            inner_sum = np.sum(W[S < threshold])
            P += w[l] * w[k] * inner_sum
    return P

def _objective(params, n):
    s = params[:n]
    w_raw = params[n:]
    
    w = np.maximum(w_raw, 1e-12)
    w_sum = np.sum(w)
    if w_sum < 1e-12:
        return 1e10
    w = w / w_sum
    
    s = np.maximum(s, 0.0)
    
    sort_idx = np.argsort(s)
    s = s[sort_idx]
    w = w[sort_idx]
    
    P = _compute_P_vec(s, w)
    return -P

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    best_P = -1
    best_support = None
    best_weights = None
    
    for n in [2, 3, 4, 5, 6]:
        bounds = [(0, 100) for _ in range(n)] + [(0, 1) for _ in range(n)]
        
        res = differential_evolution(
            _objective, 
            bounds, 
            args=(n,),
            maxiter=150, 
            popsize=15, 
            tol=1e-8, 
            seed=42,
            polish=False
        )
        
        if res.fun < -1e-9:
            params = res.x
            s = np.maximum(params[:n], 0.0)
            w_raw = np.maximum(params[n:], 1e-12)
            w = w_raw / np.sum(w_raw)
            
            sort_idx = np.argsort(s)
            s = s[sort_idx]
            w = w[sort_idx]
            
            P = _compute_P_vec(s, w)
            
            if P > best_P:
                best_P = P
                best_support = s.copy()
                best_weights = w.copy()
                
    if best_support is None:
        q_2pt = (1 + np.sqrt(33)) / 16
        p_2pt = 1 - q_2pt
        best_support = np.array([0.0, 1.0])
        best_weights = np.array([p_2pt, q_2pt])
        
    support = np.asarray(best_support, dtype=np.float64).flatten()
    weights = np.asarray(best_weights, dtype=np.float64).flatten()
    
    if len(support) != len(weights) or len(support) < 1:
        support = np.array([0.0, 1.0])
        weights = np.array([0.5, 0.5])
        
    if np.any(support < -1e-9) or np.any(weights < -1e-9):
        support = np.array([0.0, 1.0])
        weights = np.array([0.5, 0.5])
        
    if weights.sum() <= 1e-9:
        support = np.array([0.0, 1.0])
        weights = np.array([0.5, 0.5])
        
    return support, weights
