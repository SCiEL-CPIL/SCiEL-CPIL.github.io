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

    # Backtracking search for a Sidon set of size 13
    target_size = 13
    
    def find_sidon_set(target_k):
        current_set = []
        used_sums = set()
        
        def backtrack(start_idx):
            if len(current_set) == target_k:
                return True
            
            for i in range(start_idx, n):
                valid = True
                new_sums = []
                for j in current_set:
                    s = i + j
                    if s in used_sums:
                        valid = False
                        break
                    new_sums.append(s)
                
                if valid:
                    s_ii = 2 * i
                    if s_ii in used_sums:
                        valid = False
                    else:
                        new_sums.append(s_ii)
                
                if valid:
                    current_set.append(i)
                    used_sums.update(new_sums)
                    
                    if backtrack(i + 1):
                        return True
                    
                    current_set.pop()
                    for s in new_sums:
                        used_sums.remove(s)
            
            return False
        
        if backtrack(0):
            return current_set
        return None

    sidon_set = find_sidon_set(target_size)
    
    if sidon_set:
        h = np.zeros(n)
        k = len(sidon_set)
        for idx in sidon_set:
            h[idx] = 1.0 / k
        best_h = h
        best_score = calc_score(h)
    else:
        # Fallback to greedy if backtracking fails (should not happen for 13)
        for _ in range(1000):
            h = np.zeros(n)
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
                    s_ii = 2 * i
                    if s_ii in used_sums:
                        valid = False
                    else:
                        new_sums.append(s_ii)
                if valid:
                    current_set.append(i)
                    used_sums.update(new_sums)
            
            if len(current_set) > 0:
                k = len(current_set)
                for idx in current_set:
                    h[idx] = 1.0 / k
                score = calc_score(h)
                if score > best_score:
                    best_score = score
                    best_h = h.copy()
    
    return list(best_h)
