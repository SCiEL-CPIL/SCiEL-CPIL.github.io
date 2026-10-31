import numpy as np
import random
import time

def construct():
    def calc_score(U):
        U = sorted(set(U))
        if not U or 0 not in U:
            return -np.inf
        max_u = max(U)
        if max_u == 0:
            return 1.0
        
        # FFT size
        n_fft = 1
        target_size = 2 * max_u + 1
        while n_fft < target_size:
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
        
        # Lags 0..max_u are at indices 0..max_u
        # Lags -max_u..-1 are at indices n_fft-max_u .. n_fft-1
        size_diff = np.sum(diff_counts[:max_u+1] > 0) + np.sum(diff_counts[n_fft - max_u:] > 0)
        
        if size_sum == 0 or size_diff == 0:
            return -np.inf
            
        ratio = size_diff / size_sum
        if ratio <= 0:
            return -np.inf
            
        score = 1 + np.log(ratio) / np.log(2 * max_u + 1)
        return score

    # Initialization
    candidates = []
    
    # Candidate 1: Small arithmetic progression
    candidates.append(list(range(10)))
    
    # Candidate 2: Sparse powers of 2
    candidates.append([0] + [2**i for i in range(1, 10)])
    
    # Candidate 3: Random dense set
    random.seed(42)
    U_rand = [0] + random.sample(range(1, 50), 20)
    candidates.append(U_rand)
    
    # Candidate 4: Structured "blocks"
    U_blocks = [0, 1, 2, 5, 6, 7, 10, 11, 12]
    candidates.append(U_blocks)
    
    best_init = max(candidates, key=lambda u: calc_score(u))
    current_set = sorted(best_init)
    current_score = calc_score(current_set)
    
    best_set = current_set.copy()
    best_score = current_score
    
    # Simulated Annealing
    start_time = time.time()
    time_limit = 115.0 # Leave 5s buffer
    T = 1.0
    cooling = 0.999
    max_iter = 100000
    
    # Define max range for new elements
    # If max_u is large, adding even larger is bad due to denominator.
    # We should probably cap the search space or let it grow slowly.
    
    for i in range(max_iter):
        if time.time() - start_time > time_limit:
            break
            
        U = current_set.copy()
        max_u = max(U)
        
        # Choose mutation
        op = random.random()
        
        if op < 0.4:
            # Add element
            # Bias towards existing range or slightly larger
            # If max_u is small, allow growth.
            upper_bound = max(50, int(1.5 * max_u))
            new_val = random.randint(0, upper_bound)
            if new_val not in U:
                U.append(new_val)
        elif op < 0.7:
            # Remove element
            if len(U) > 1:
                # Don't remove 0
                idx = random.randint(1, len(U)-1)
                U.pop(idx)
        else:
            # Move element
            if len(U) > 1:
                idx = random.randint(1, len(U)-1)
                old_val = U[idx]
                upper_bound = max(50, int(1.5 * max_u))
                new_val = random.randint(0, upper_bound)
                if new_val not in U or new_val == old_val:
                    U[idx] = new_val
                else:
                    # If collision, try another or skip
                    pass
                    
        # Ensure 0 is present
        if 0 not in U:
            # If 0 was removed (shouldn't happen with logic above, but just in case)
            # Replace a random non-zero with 0
            if len(U) > 1:
                idx = random.randint(0, len(U)-1)
                U[idx] = 0
            else:
                U = [0]
            U = list(set(U))
            
        new_score = calc_score(U)
        
        # Acceptance
        if new_score > current_score:
            current_set = U
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_set = U
        else:
            # Accept with probability exp((new - curr) / T)
            # Scores are around 1.x. Differences are small.
            delta = new_score - current_score
            if delta < 0:
                prob = np.exp(delta / T)
                if random.random() < prob:
                    current_set = U
                    current_score = new_score
            # If delta >= 0, it's an improvement, handled above.
            
        T *= cooling
        if T < 0.01:
            T = 0.01 # Floor for temperature to allow some exploration
            
    return sorted(best_set)
