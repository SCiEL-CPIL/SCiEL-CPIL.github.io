# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _entropy_bits(p):
    """Shannon entropy in bits, ignoring zero probabilities."""
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 1e-15]
    if len(p) == 0:
        return 0.0
    return float(-np.sum(p * np.log2(p)))

def _projection_entropy(P, slope):
    """Compute entropy of projection x + slope*y."""
    n = P.shape[0]
    # P[y, x], so y is row, x is col
    ys = np.arange(n)
    xs = np.arange(n)
    Y, X = np.meshgrid(ys, xs, indexing='ij')
    values = X + slope * Y
    unique_vals, inverse = np.unique(values, return_inverse=True)
    counts = np.zeros(len(unique_vals))
    np.add.at(counts, inverse.ravel(), P.ravel())
    return _entropy_bits(counts)

def _objective(params):
    """Objective: minimize -D/M (equivalently maximize D/M).
    params: 100 weights, will be softmaxed and normalized."""
    w = np.exp(params - np.max(params))  # stable softmax
    w = w / w.sum()
    P = w.reshape(10, 10)
    
    D = _projection_entropy(P, -1.0)
    H0 = _projection_entropy(P, 0.0)
    H1 = _projection_entropy(P, 1.0)
    H2 = _projection_entropy(P, 2.0)
    M = max(H0, H1, H2)
    
    if M < 1e-9:
        return 1e10  # invalid
    return -D / M

def _diagonal_initializer(diag_offset, spread=0.0, noise_scale=0.1):
    """Initialize weights concentrated on diagonal x-y = diag_offset."""
    n = 10
    params = np.zeros(n * n)
    for y in range(n):
        for x in range(n):
            if x - y == diag_offset:
                params[y * n + x] = 5.0  # high weight on diagonal
            else:
                params[y * n + x] = -5.0
    # Add small noise to break symmetry
    params += noise_scale * np.random.randn(n * n)
    return params

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya entropy ratio optimization on a 10x10 grid.
    Maximizes D/M where D = H_{x-y} and M = max(H_x, H_{x+y}, H_{x+2y}).
    """
    start_time = time.time()
    deadline = start_time + 600.0
    safety_margin = 10.0
    
    n = 10
    best_score = 0.0
    best_P = np.ones((n, n)) / (n * n)
    
    # Try multiple diagonal initializations
    diag_offsets = list(range(-9, 10))
    
    # Shuffle to explore different diagonals
    np.random.seed(42)
    np.random.shuffle(diag_offsets)
    
    # Also try some mixed initializations
    for attempt, diag in enumerate(diag_offsets):
        if time.time() > deadline - safety_margin:
            break
        
        # Multiple restarts per diagonal with different noise
        for restart in range(3):
            if time.time() > deadline - safety_margin:
                break
            
            noise_scale = 0.1 + 0.05 * restart
            params0 = _diagonal_initializer(diag, noise_scale=noise_scale)
            
            # Use Nelder-Mead for non-convex optimization
            try:
                result = minimize(
                    _objective,
                    params0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 2000,
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
                
                if result.fun < -best_score:  # we're minimizing -D/M
                    w = np.exp(result.x - np.max(result.x))
                    w = w / w.sum()
                    P_candidate = w.reshape(n, n)
                    
                    # Validate
                    if (P_candidate.shape == (n, n) and 
                        np.all(np.isfinite(P_candidate)) and
                        np.all(P_candidate >= -1e-9) and
                        np.sum(P_candidate) > 1e-9):
                        P_candidate = np.clip(P_candidate, 0, None)
                        P_candidate = P_candidate / P_candidate.sum()
                        
                        D = _projection_entropy(P_candidate, -1.0)
                        H0 = _projection_entropy(P_candidate, 0.0)
                        H1 = _projection_entropy(P_candidate, 1.0)
                        H2 = _projection_entropy(P_candidate, 2.0)
                        M = max(H0, H1, H2)
                        
                        if M > 1e-9:
                            score = D / M
                            if score > best_score:
                                best_score = score
                                best_P = P_candidate.copy()
                                
            except Exception:
                continue
    
    # Final polish: try L-BFGS-B from the best found point
    if best_score > 0 and time.time() < deadline - safety_margin:
        w = best_P.ravel()
        # Convert to log-softmax params
        params0 = np.log(w + 1e-15)
        
        try:
            result = minimize(
                _objective,
                params0,
                method='L-BFGS-B',
                options={
                    'maxiter': 5000,
                    'ftol': 1e-12,
                    'gtol': 1e-8,
                    'maxfun': 10000
                }
            )
            
            if result.fun < -best_score:
                w = np.exp(result.x - np.max(result.x))
                w = w / w.sum()
                P_candidate = w.reshape(n, n)
                
                if (P_candidate.shape == (n, n) and 
                    np.all(np.isfinite(P_candidate)) and
                    np.all(P_candidate >= -1e-9) and
                    np.sum(P_candidate) > 1e-9):
                    P_candidate = np.clip(P_candidate, 0, None)
                    P_candidate = P_candidate / P_candidate.sum()
                    
                    D = _projection_entropy(P_candidate, -1.0)
                    H0 = _projection_entropy(P_candidate, 0.0)
                    H1 = _projection_entropy(P_candidate, 1.0)
                    H2 = _projection_entropy(P_candidate, 2.0)
                    M = max(H0, H1, H2)
                    
                    if M > 1e-9:
                        score = D / M
                        if score > best_score:
                            best_score = score
                            best_P = P_candidate.copy()
                            
        except Exception:
            pass
    
    # Ensure validity
    best_P = np.clip(best_P, 0, None)
    total = best_P.sum()
    if total > 1e-9:
        best_P = best_P / total
    
    return best_P

# EVOLVE-BLOCK-END
