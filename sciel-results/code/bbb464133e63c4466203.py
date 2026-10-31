import time
import random
import math
import numpy as np

_START = time.time()
_DEADLINE = _START + 115.0

def _score_from_set(U):
    if not U:
        return 0.0
    max_val = max(U)
    if max_val < 1:
        return 0.0
    
    L = max_val + 1
    fft_len = 1
    while fft_len < 2 * L - 1:
        fft_len *= 2
        
    vec = np.zeros(fft_len, dtype=np.float64)
    for u in U:
        vec[u] = 1.0
        
    f = np.fft.fft(vec)
    
    conv_sum = np.fft.ifft(f * f).real
    sums_count = np.sum(conv_sum > 0.5)
    
    conv_diff = np.fft.ifft(np.conj(f) * f).real
    diffs_count = np.sum(np.abs(conv_diff) > 0.5)
    
    if sums_count == 0:
        return 0.0
        
    ratio = diffs_count / float(sums_count)
    if ratio <= 0:
        return 0.0
        
    return 1.0 + math.log(ratio) / math.log(2 * max_val + 1)

def _generate_golomb(n):
    """Generate a greedy Golomb ruler of size n."""
    U = [0]
    diffs = {0}
    current_max = 0
    for _ in range(n - 1):
        candidate = current_max + 1
        while True:
            valid = True
            for u in U:
                d = candidate - u
                if d in diffs:
                    valid = False
                    break
            if valid:
                U.append(candidate)
                for u in U[:-1]:
                    diffs.add(candidate - u)
                current_max = candidate
                break
            else:
                candidate += 1
    return U

def construct():
    global _DEADLINE
    best_score = 0.0
    best_set = [0, 1]
    
    random.seed(42)
    
    # Try different sizes
    for n in [5, 8, 10, 12, 15, 20]:
        if time.time() > _DEADLINE - 10.0:
            break
            
        # Initialize with Golomb ruler
        U_init = _generate_golomb(n)
        current_U = U_init[:]
        current_score = _score_from_set(current_U)
        
        if current_score > best_score:
            best_score = current_score
            best_set = current_U[:]
            
        # Simulated Annealing
        T = 1.0
        T_min = 0.01
        cooling = 0.99
        max_val_limit = 2000  # Keep M bounded
        
        start_time = time.time()
        time_limit = 10.0  # 10s per size
        
        while time.time() - start_time < time_limit and T > T_min:
            # Pick a random element to move
            idx = random.randint(1, len(current_U) - 1) # Keep 0 fixed
            old_val = current_U[idx]
            
            # Generate new value
            # Bias towards keeping M small?
            # Or just random in [0, max_val_limit]
            # Better: random in [0, max(current_U) + 10]
            new_val = random.randint(0, max(current_U) + 10)
            
            if new_val == old_val:
                continue
            
            # Check uniqueness
            if new_val in current_U:
                continue
                
            # Apply move
            current_U[idx] = new_val
            current_U.sort()
            
            new_score = _score_from_set(current_U)
            
            # Acceptance
            if new_score > current_score:
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_set = current_U[:]
            else:
                # Metropolis criterion
                delta = new_score - current_score
                prob = math.exp(delta / T)
                if random.random() < prob:
                    current_score = new_score
                else:
                    # Revert
                    current_U[idx] = old_val
                    current_U.sort()
            
            T *= cooling
            
    # Final polish
    if time.time() < _DEADLINE - 5.0:
        # Run a tighter local search on the best set found
        # ... (optional)
        pass
        
    best_set = sorted(set(int(x) for x in best_set))
    if 0 not in best_set:
        best_set.append(0)
    if len(best_set) < 2:
        best_set = [0, 1]
        
    return best_set
