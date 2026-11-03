import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _score_raw(h):
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return l2_sq / (l1 * autoconv_max)

def _find_max_sidon_set(n):
    """
    Find the largest Sidon set in {0, ..., n-1} using backtracking.
    """
    best_set = []
    
    def backtrack(current_set, start):
        nonlocal best_set
        if len(current_set) > len(best_set):
            best_set = current_set[:]
            
        # Pruning: if remaining elements cannot beat best_set size
        remaining = n - start
        if len(current_set) + remaining <= len(best_set):
            return

        for i in range(start, n):
            # Check if i can be added
            is_valid = True
            for j in range(len(current_set)):
                s1 = current_set[j] + i
                for k in range(j+1, len(current_set)):
                    s2 = current_set[j] + current_set[k]
                    if s1 == s2:
                        is_valid = False
                        break
                if not is_valid:
                    break
            # Also check i+i against existing sums?
            # Sidon condition: all sums a+b (a<=b) are distinct.
            # So we need to check i+i and i+current_set[j] against all existing sums.
            
            if is_valid:
                # Generate all new sums involving i
                new_sums = set()
                for j in range(len(current_set)):
                    new_sums.add(current_set[j] + i)
                new_sums.add(2*i)
                
                # Check against existing sums
                existing_sums = set()
                for j in range(len(current_set)):
                    for k in range(j+1, len(current_set)):
                        existing_sums.add(current_set[j] + current_set[k])
                    existing_sums.add(2*current_set[j])
                
                if not (new_sums & existing_sums):
                    backtrack(current_set + [i], i+1)

    backtrack([0], 1)
    return best_set

def construct():
    start_time = time.time()
    deadline = 140.0
    
    # 1. Find max Sidon set
    sidon_set = _find_max_sidon_set(N)
    K = len(sidon_set)
    
    # 2. Initialize uniform weights
    h = np.zeros(N)
    h[sidon_set] = 1.0 / K
    
    best_score = _score_raw(h)
    
    # 3. Optimize weights on Sidon support
    # Maximize sum(h^2) / max(conv) subject to sum(h)=1, h>=0
    # Variables: h[sidon_set]
    
    def neg_obj(x):
        # x are the weights on sidon_set
        h_full = np.zeros(N)
        h_full[sidon_set] = x
        return -_score_raw(h_full)
    
    # Initial guess: uniform
    x0 = np.ones(K) / K
    
    # Constraints
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0})
    bounds = [(0, 1) for _ in range(K)]
    
    if time.time() < deadline - 10:
        try:
            res = minimize(neg_obj, x0, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 1000, 'ftol': 1e-9})
            if res.success:
                h_opt = np.zeros(N)
                h_opt[sidon_set] = res.x
                score_opt = _score_raw(h_opt)
                if score_opt > best_score:
                    best_score = score_opt
                    h = h_opt
        except Exception:
            pass

    return list(h)
