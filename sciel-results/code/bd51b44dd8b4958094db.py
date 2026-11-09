# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt

def calc_prob(support, weights):
    """
    Calculate P[X1 + X2 + X3 < 2*X4]
    support: 1D array of shape (n,)
    weights: 1D array of shape (n,) (normalized)
    """
    n = len(support)
    w_sum = np.sum(weights)
    if w_sum <= 1e-9:
        return 0.0
    w = weights / w_sum
    
    p_val = 0.0
    for k in range(n):
        for l in range(n):
            threshold = 2.0 * support[l] - support[k]
            inner_sum = 0.0
            for i in range(n):
                for j in range(n):
                    if support[i] + support[j] < threshold:
                        inner_sum += w[i] * w[j]
            p_val += w[k] * w[l] * inner_sum
    return p_val

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    start_time = time.time()
    deadline = start_time + 550  # Leave 50s margin
    
    best_score = -1.0
    best_support = None
    best_weights = None
    
    def _prob_3point(M, p0, p1, p2):
        s = np.array([0.0, 1.0, M])
        w = np.array([p0, p1, p2])
        p_val = 0.0
        for k in range(3):
            for l in range(3):
                threshold = 2.0 * s[l] - s[k]
                inner = 0.0
                for i in range(3):
                    for j in range(3):
                        if s[i] + s[j] < threshold:
                            inner += w[i] * w[j]
                p_val += w[k] * w[l] * inner
        return p_val
    
    # 2D grid search over M and p1, with 1D search over p2
    M_coarse = np.linspace(1.0, 4.0, 80)
    p1_coarse = np.linspace(0.05, 0.95, 60)
    
    for M in M_coarse:
        if time.time() > deadline:
            break
        
        for p1 in p1_coarse:
            p2_grid = np.linspace(0.01, 1.0 - p1 - 0.01, 40)
            for p2 in p2_grid:
                p0 = 1.0 - p1 - p2
                if p0 <= 0.01:
                    continue
                
                p_val = _prob_3point(M, p0, p1, p2)
                if p_val > best_score:
                    best_score = p_val
                    best_support = np.array([0.0, 1.0, M])
                    best_weights = np.array([p0, p1, p2])
    
    # Refine around best point
    if best_support is not None and time.time() < deadline:
        M_best = best_support[2]
        p1_best = best_weights[1]
        p2_best = best_weights[2]
        
        M_refine = np.linspace(max(1.0, M_best - 0.3), min(4.0, M_best + 0.3), 150)
        p1_refine = np.linspace(max(0.05, p1_best - 0.1), min(0.95, p1_best + 0.1), 80)
        
        for M in M_refine:
            if time.time() > deadline:
                break
            
            for p1 in p1_refine:
                p2_min = max(0.01, p2_best - 0.1)
                p2_max = min(1.0 - p1 - 0.01, p2_best + 0.1)
                if p2_max <= p2_min:
                    continue
                p2_refine = np.linspace(p2_min, p2_max, 60)
                for p2 in p2_refine:
                    p0 = 1.0 - p1 - p2
                    if p0 <= 0.01:
                        continue
                    
                    p_val = _prob_3point(M, p0, p1, p2)
                    if p_val > best_score:
                        best_score = p_val
                        best_support = np.array([0.0, 1.0, M])
                        best_weights = np.array([p0, p1, p2])
    
    # Fallback
    if best_support is None:
        n = 32
        best_support = np.arange(n, dtype=np.float64)
        best_weights = np.ones(n, dtype=np.float64) / n
    
    return best_support, best_weights

# EVOLVE-BLOCK-END
