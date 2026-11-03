import numpy as np
import time
import itertools

N = 50
DELTA = 0.5 / N

def _score_raw(h):
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    h_norm = h / l1
    l2_sq = np.sum(h_norm**2)
    h_conv = np.convolve(h_norm, h_norm, mode='full')
    autoconv_max = np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return l2_sq / autoconv_max

def _find_sidon_set(n, max_size=10):
    """
    Greedy search for a large Sidon set in [0, n-1].
    A Sidon set has all pairwise sums distinct.
    """
    S = []
    sums = set()
    
    # Try to add elements in order, or random order?
    # Random order with restarts is better.
    best_S = []
    best_score = 0
    
    for _ in range(100):
        S = []
        sums = set()
        candidates = list(range(n))
        np.random.shuffle(candidates)
        
        for i in candidates:
            # Check if i can be added
            valid = True
            new_sums = set()
            
            # Check sums with existing elements
            for j in S:
                s1 = i + j
                s2 = j + i # same
                if s1 in sums:
                    valid = False
                    break
                new_sums.add(s1)
            
            if not valid:
                continue
                
            # Check sum with itself
            s_self = 2 * i
            if s_self in sums or s_self in new_sums:
                continue
                
            # If valid, add
            S.append(i)
            sums.update(new_sums)
            sums.add(s_self)
            
            if len(S) > len(best_S):
                best_S = S[:]
                if len(best_S) >= max_size:
                    break
                    
    return best_S

def construct():
    start_time = time.time()
    deadline = 145.0 # Safety margin
    
    # 1. Find good sparse supports
    best_h = np.ones(N) / N
    best_score = _score_raw(best_h)
    
    # Try to find a large Sidon set
    sidon_set = _find_sidon_set(N, max_size=8)
    
    if len(sidon_set) > 0:
        h_sparse = np.zeros(N)
        h_sparse[sidon_set] = 1.0 / len(sidon_set)
        score_sparse = _score_raw(h_sparse)
        
        if score_sparse > best_score:
            best_score = score_sparse
            best_h = h_sparse
    
    # 2. Try random sparse sets with optimization
    # We can also try to optimize weights on the Sidon set
    if len(sidon_set) > 0:
        # Simple local optimization of weights
        # Maximize sum(w^2) / max_conv(w)
        # Subject to sum(w)=1, w>=0
        
        # Use SLSQP on the sparse weights
        from scipy.optimize import minimize
        
        w = np.ones(len(sidon_set)) / len(sidon_set)
        
        def neg_obj(w):
            h = np.zeros(N)
            h[sidon_set] = w
            # Normalize
            l1 = np.sum(h)
            if l1 == 0: return 0
            h_n = h / l1
            l2_sq = np.sum(h_n**2)
            h_conv = np.convolve(h_n, h_n, mode='full')
            autoconv_max = np.max(h_conv)
            if autoconv_max == 0: return 0
            return -(l2_sq / autoconv_max)
        
        bounds = [(0, 1)] * len(sidon_set)
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        
        try:
            res = minimize(neg_obj, w, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 100})
            if res.success:
                w_opt = res.x
                h_opt = np.zeros(N)
                h_opt[sidon_set] = w_opt
                score_opt = _score_raw(h_opt)
                if score_opt > best_score:
                    best_score = score_opt
                    best_h = h_opt
        except Exception:
            pass

    # 3. Try a few random sparse sets of various sizes
    for size in range(3, 10):
        if time.time() > deadline:
            break
        for _ in range(50):
            if time.time() > deadline:
                break
            idx = np.random.choice(N, size, replace=False)
            h_r = np.zeros(N)
            h_r[idx] = 1.0 / size
            score_r = _score_raw(h_r)
            if score_r > best_score:
                best_score = score_r
                best_h = h_r
                
    return list(best_h)
