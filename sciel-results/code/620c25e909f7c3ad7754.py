import numpy as np
import random

def construct():
    def calc_score(U):
        U = sorted(set(U))
        if not U or 0 not in U:
            return -np.inf
        max_u = max(U)
        if max_u == 0:
            return 1.0
        
        size = 2 * max_u + 1
        n_fft = 1
        while n_fft < size:
            n_fft *= 2
            
        vec = np.zeros(n_fft, dtype=np.float64)
        for x in U:
            vec[x] = 1.0
            
        fft_vec = np.fft.fft(vec)
        
        # Sumset
        sum_poly = np.fft.ifft(fft_vec * fft_vec).real
        sum_counts = np.round(sum_poly).astype(int)
        size_sum = np.sum(sum_counts[:2*max_u+1] > 0)
        
        # Diffset
        corr = np.fft.ifft(fft_vec * np.conj(fft_vec)).real
        diff_counts = np.round(corr).astype(int)
        
        indices = np.concatenate([np.arange(0, max_u+1), np.arange(n_fft-max_u, n_fft)])
        size_diff = np.sum(diff_counts[indices] > 0)
        
        if size_sum == 0 or size_diff == 0:
            return -np.inf
            
        ratio = size_diff / size_sum
        if ratio <= 0:
            return -np.inf
            
        score = 1 + np.log(ratio) / np.log(2 * max_u + 1)
        return score

    random.seed(42)
    
    # Initialize with a geometric progression
    # Powers of 2 or 3
    base = 3
    max_elements = 15
    U = [0]
    val = 1
    for _ in range(max_elements):
        U.append(val)
        val *= base
    
    best_set = U.copy()
    best_score = calc_score(best_set)
    
    # Local Search
    # Use Simulated Annealing or simple Hill Climbing
    # Given time limit 120s, we can do many iterations.
    # FFT is fast.
    
    current_set = best_set.copy()
    current_score = best_score
    
    max_iters = 5000
    for i in range(max_iters):
        # Mutation
        op = random.choice(['add', 'remove', 'replace'])
        new_set = current_set.copy()
        
        if op == 'add' and len(new_set) < 50:
            # Add a random element
            max_u = max(new_set)
            # Try adding something larger to increase range/diffs
            # Or smaller to create collisions
            new_val = random.randint(0, max_u + 100)
            if new_val not in new_set:
                new_set.append(new_val)
            else:
                continue
        elif op == 'remove' and len(new_set) > 1:
            idx = random.randint(1, len(new_set)-1)
            new_set.pop(idx)
        elif op == 'replace':
            if len(new_set) > 1:
                idx = random.randint(1, len(new_set)-1)
                max_u = max(new_set)
                new_val = random.randint(0, max_u + 100)
                if new_val not in new_set:
                    new_set[idx] = new_val
                else:
                    continue
        
        if 0 not in new_set:
            continue
            
        new_score = calc_score(new_set)
        
        # Hill Climbing: accept only if better
        # Or Simulated Annealing: accept with prob exp(-delta/T)
        # Let's stick to Hill Climbing with restarts or just pure HC
        if new_score > current_score:
            current_score = new_score
            current_set = new_set
            
            if current_score > best_score:
                best_score = current_score
                best_set = current_set.copy()
        else:
            # Occasionally revert to best to avoid local minima?
            # Or just keep current if it's not too bad?
            # Simple HC: keep current.
            pass
            
    return sorted(best_set)
