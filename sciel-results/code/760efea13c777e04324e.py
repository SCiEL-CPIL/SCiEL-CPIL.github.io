import numpy as np
from scipy.optimize import minimize

def construct():
    n = 50
    best_h = np.zeros(n)
    best_score = 0.0
    
    def calc_score(h):
        if np.sum(h) == 0:
            return 0.0
        l1 = np.sum(h)
        l2_sq = np.sum(h**2)
        # Autoconvolution
        c = np.convolve(h, h, mode='full')
        # Supremum norm of c
        c_max = np.max(c)
        if c_max == 0:
            return 0.0
        return (l2_sq / (l1 * c_max))

    def optimize_weights(support):
        k = len(support)
        if k == 0:
            return np.zeros(n)
        
        def objective(w):
            # w is the vector of weights for the support
            # Normalize to unit L2 norm to fix scale
            norm = np.linalg.norm(w)
            if norm == 0:
                return 1e10
            w_norm = w / norm
            h = np.zeros(n)
            for i, idx in enumerate(support):
                h[idx] = w_norm[i]
            c = np.convolve(h, h, mode='full')
            return np.max(c)
            
        w0 = np.ones(k)
        # Nelder-Mead is robust for small non-smooth problems
        try:
            res = minimize(objective, w0, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-5})
            w_opt = res.x
        except:
            w_opt = w0
            
        w_opt = np.maximum(w_opt, 0)
        norm = np.linalg.norm(w_opt)
        if norm > 0:
            w_opt = w_opt / norm
        else:
            w_opt = np.ones(k) / k
            
        h = np.zeros(n)
        for i, idx in enumerate(support):
            h[idx] = w_opt[i]
        return h

    # Try many random orderings to find a good Sidon set
    for _ in range(2000):
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
        
        if len(current_set) < 3:
            continue
            
        # Optimize weights for this support
        h_opt = optimize_weights(current_set)
        
        score = calc_score(h_opt)
        if score > best_score:
            best_score = score
            best_h = h_opt.copy()
            
    # If no good set found, fallback to single spike
    if best_score == 0.0:
        best_h[0] = 1.0
        
    return list(best_h)
