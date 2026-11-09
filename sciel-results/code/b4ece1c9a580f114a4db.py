# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya entropy ratio optimization on 10x10 grid.
    Maximizes H(x-y) / max(H(x), H(x+y), H(x+2y)).
    """
    n = 10
    start_time = time.time()
    deadline = start_time + 600.0  # 600 second deadline
    
    # Precompute projection indices
    x = np.arange(n)
    y = np.arange(n)
    X, Y = np.meshgrid(x, y, indexing='xy')  # X[y,x] = x, Y[y,x] = y
    
    proj_minus1 = X - Y      # x - y, range -9..9
    proj_0 = X               # x, range 0..9
    proj_1 = X + Y           # x + y, range 0..18
    proj_2 = X + 2 * Y       # x + 2y, range 0..27
    
    def compute_entropy(proj, P):
        """Compute Shannon entropy in bits of projection distribution."""
        # Get unique values and their probabilities
        vals, counts = np.unique(proj, return_inverse=True)
        prob = np.zeros(len(vals))
        for i, v in enumerate(vals):
            mask = (proj == v)
            prob[i] = np.sum(P[mask])
        # Only consider non-zero probabilities
        prob = prob[prob > 1e-15]
        if len(prob) == 0:
            return 0.0
        return -np.sum(prob * np.log2(prob))
    
    def objective(params):
        """Negative ratio to minimize (we want to maximize ratio)."""
        # params are log-weights for each cell
        log_w = params.reshape((n, n))
        # Softmax to get probabilities
        log_w -= np.max(log_w)  # for numerical stability
        w = np.exp(log_w)
        P = w / np.sum(w)
        
        D = compute_entropy(proj_minus1, P)
        H0 = compute_entropy(proj_0, P)
        H1 = compute_entropy(proj_1, P)
        H2 = compute_entropy(proj_2, P)
        M = max(H0, H1, H2)
        
        if M < 1e-9:
            return 1e9  # penalty
        
        ratio = D / M
        return -ratio
    
    # Initialize with uniform distribution (log weights all zero)
    best_params = np.zeros(n * n)
    best_score = 0.0
    best_P = np.ones((n, n)) / (n * n)
    
    # Try several initializations and local optimizations
    initializations = [
        np.zeros(n * n),  # uniform
    ]
    
    # Add some structured initializations
    # Bias toward making x-y more uniform
    for init_idx in range(20):
        if init_idx == 0:
            init = np.zeros(n * n)
        elif init_idx == 1:
            # Weight by diagonal to flatten x-y
            init = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    d = i - j
                    init[i, j] = 1.0 / (10 - abs(d))
            init = init - np.mean(init)
        elif init_idx == 2:
            # Random
            init = np.random.randn(n * n) * 0.5
        elif init_idx == 3:
            # Concentrate on x=0 column
            init = np.zeros((n, n))
            init[:, 0] = 1.0
        elif init_idx == 4:
            # Concentrate on diagonal x=y
            init = np.zeros((n, n))
            for i in range(n):
                init[i, i] = 1.0
        elif init_idx == 5:
            # Weight to make x+y concentrated
            init = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    s = i + j
                    init[i, j] = 1.0 if s <= 5 else 0.1
        elif init_idx == 6:
            # Random with larger scale
            init = np.random.randn(n * n) * 2.0
        elif init_idx == 7:
            # Anti-diagonal concentration
            init = np.zeros((n, n))
            for i in range(n):
                init[i, n-1-i] = 1.0
        elif init_idx == 8:
            # Corner concentration
            init = np.zeros((n, n))
            init[0, 0] = 1.0
            init[0, -1] = 1.0
            init[-1, 0] = 1.0
            init[-1, -1] = 1.0
        else:
            init = np.random.randn(n * n) * (0.5 + init_idx * 0.1)
        
        # Check time
        if time.time() > deadline - 30:
            break
        
        # Try Nelder-Mead optimization
        try:
            result = minimize(objective, init, method='Nelder-Mead',
                             options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})
            
            if result.fun < -best_score:  # since we're minimizing negative ratio
                best_score = -result.fun
                best_params = result.x.copy()
                # Compute P from best_params
                log_w = best_params.reshape((n, n))
                log_w -= np.max(log_w)
                w = np.exp(log_w)
                best_P = w / np.sum(w)
        except Exception:
            pass
        
        # Also try L-BFGS-B
        if time.time() > deadline - 60:
            break
        try:
            result2 = minimize(objective, init, method='L-BFGS-B',
                              options={'maxiter': 3000, 'ftol': 1e-10})
            
            if result2.fun < -best_score:
                best_score = -result2.fun
                best_params = result2.x.copy()
                log_w = best_params.reshape((n, n))
                log_w -= np.max(log_w)
                w = np.exp(log_w)
                best_P = w / np.sum(w)
        except Exception:
            pass
    
    # Final refinement with more iterations if time allows
    if time.time() < deadline - 120 and best_score > 0:
        try:
            result3 = minimize(objective, best_params, method='Nelder-Mead',
                             options={'maxiter': 20000, 'xatol': 1e-10, 'fatol': 1e-10})
            if result3.fun < -best_score:
                best_score = -result3.fun
                best_params = result3.x.copy()
                log_w = best_params.reshape((n, n))
                log_w -= np.max(log_w)
                w = np.exp(log_w)
                best_P = w / np.sum(w)
        except Exception:
            pass
    
    # Ensure valid output
    best_P = np.clip(best_P, 0, None)
    total = np.sum(best_P)
    if total <= 1e-9:
        best_P = np.ones((n, n)) / (n * n)
    else:
        best_P = best_P / total
    
    return best_P


# EVOLVE-BLOCK-END
