import numpy as np

def construct():
    n = 50
    best_h = np.zeros(n)
    best_score = 0.0
    
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

    def optimize_weights(indices):
        # Start with uniform weights
        k = len(indices)
        h = np.zeros(n)
        for idx in indices:
            h[idx] = 1.0 / k
            
        # Coordinate descent on weights
        # We want to maximize score.
        # Score is scale invariant, so we can fix sum(h)=1.
        # We adjust h[i] and h[j] to keep sum=1.
        # Simple perturbation: move small mass from i to j.
        
        current_score = calc_score(h)
        
        for _iter in range(50):
            improved = False
            # Try moving mass between pairs of indices in the set
            for i_idx in range(k):
                for j_idx in range(i_idx + 1, k):
                    i = indices[i_idx]
                    j = indices[j_idx]
                    
                    # Try several step sizes
                    for step in [0.01, 0.05, 0.1]:
                        h_trial = h.copy()
                        # Move step from i to j
                        if h_trial[i] > step:
                            h_trial[i] -= step
                            h_trial[j] += step
                            s = calc_score(h_trial)
                            if s > current_score:
                                h = h_trial
                                current_score = s
                                improved = True
                                break
                        # Move step from j to i
                        if h_trial[j] > step:
                            h_trial[j] -= step
                            h_trial[i] += step
                            s = calc_score(h_trial)
                            if s > current_score:
                                h = h_trial
                                current_score = s
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
                
        return h, current_score

    max_k_found = 0
    best_set = None
    
    # Search for Sidon sets
    for _ in range(1000):
        current_set = []
        used_sums = set()
        
        candidates = list(range(n))
        np.random.shuffle(candidates)
        
        for i in candidates:
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
                
        if len(current_set) > max_k_found:
            max_k_found = len(current_set)
            best_set = current_set
            if max_k_found >= 8:
                break

    if best_set is not None and len(best_set) > 0:
        h_opt, score_opt = optimize_weights(best_set)
        if score_opt > best_score:
            best_score = score_opt
            best_h = h_opt

    if best_score == 0.0:
        best_h[0] = 1.0
        
    return list(best_h)
