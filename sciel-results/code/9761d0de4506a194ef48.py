import numpy as np
from scipy.optimize import minimize

def construct():
    n = 50
    
    def calc_score(h):
        if np.sum(h) == 0:
            return 0.0
        l1 = np.sum(h)
        l2_sq = np.sum(h**2)
        c = np.convolve(h, h, mode='full')
        c_max = np.max(c)
        if c_max == 0:
            return 0.0
        return (l2_sq / (l1 * c_max))

    def find_maximal_sidon_set(max_trials=500):
        best_set = []
        for _ in range(max_trials):
            indices = list(range(n))
            np.random.shuffle(indices)
            
            current_set = []
            used_sums = set()
            
            for i in indices:
                valid = True
                new_sums = []
                for j in current_set:
                    s = i + j
                    if s in used_sums:
                        valid = False
                        break
                    new_sums.append(s)
                
                if valid:
                    s_ii = i + i
                    if s_ii in used_sums:
                        valid = False
                    else:
                        new_sums.append(s_ii)
                
                if valid:
                    current_set.append(i)
                    used_sums.update(new_sums)
            
            if len(current_set) > len(best_set):
                best_set = current_set
                if len(best_set) >= 13: # Theoretical max
                    break
        return best_set

    # 1. Find a good support
    support = find_maximal_sidon_set()
    
    if not support:
        h = np.zeros(n)
        h[0] = 1.0
        return list(h)

    k = len(support)
    
    # 2. Optimize weights on this support
    w0 = np.ones(k) / k
    
    def objective_penalized(w):
        if np.any(w < 0):
            return 100.0 + np.sum(w[w<0]**2)
        if np.sum(w) == 0:
            return 100.0
        w_norm = w / np.sum(w)
        h = np.zeros(n)
        for idx, wi in zip(support, w_norm):
            h[idx] = wi
        l2_sq = np.sum(h**2)
        c = np.convolve(h, h, mode='full')
        c_max = np.max(c)
        if c_max == 0:
            return 100.0
        return -(l2_sq / c_max)

    res = minimize(objective_penalized, w0, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
    
    best_w = res.x
    best_w = np.maximum(best_w, 0)
    if np.sum(best_w) == 0:
        best_w = np.ones(k) / k
    else:
        best_w = best_w / np.sum(best_w)
        
    h = np.zeros(n)
    for idx, wi in zip(support, best_w):
        h[idx] = wi
        
    # Verify score
    score = calc_score(h)
    
    # If optimization failed or score is low, fallback to uniform
    if score < k/2 * 0.9: # Should be at least close to k/2
        h = np.zeros(n)
        for idx in support:
            h[idx] = 1.0 / k
            
    return list(h)
