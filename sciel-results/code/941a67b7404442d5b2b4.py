import time
import random
import math
import numpy as np

# EVOLVE-BLOCK-START
_START = time.time()
_DEADLINE = _START + 115.0  # Safety margin vs 150 s kill

def _score_from_vector(vec, n, max_val):
    """
    Compute the score for a set represented by a binary indicator vector.
    Score = 1 + log(|U-U|/|U+U|) / log(2*max(U) + 1)
    
    Args:
        vec: Binary numpy array of length L.
        n: Number of elements in the set.
        max_val: Maximum value in the set.
    """
    if n < 2 or max_val < 1:
        return 0.0
    
    # Use FFT to compute sumset and differenceset sizes
    # Sumset: |U+U| is the number of non-zero coefficients in the convolution of U with U.
    # Differenceset: |U-U| is the number of non-zero coefficients in the cross-correlation of U with U.
    
    # Length of the result for convolution/correlation is 2*L - 1.
    # We need sufficient length to avoid circular convolution artifacts.
    # Since max_val is bounded by the index of the last 1 in vec, let's ensure vec length is sufficient.
    # Actually, vec is passed with a specific length. We should ensure it's large enough for the support.
    # The caller ensures vec length >= 2 * (max_val + 1) or similar? 
    # Let's just use the provided vec length. If vec is too short, FFT wraps around.
    # To be safe, we should pad vec to a power of 2 at least 2 * (max_val + 1).
    
    # However, for speed, we assume the caller provides a vec of length L such that L >= 2 * (max_val + 1).
    # Let's check if we need to pad.
    required_len = 2 * (max_val + 1)
    L = len(vec)
    if L < required_len:
        # Pad with zeros
        vec = np.pad(vec, (0, required_len - L), mode='constant')
    
    # Find next power of 2 for FFT efficiency
    fft_len = 1
    while fft_len < 2 * L - 1:
        fft_len *= 2
        
    # FFT
    f = np.fft.fft(vec, n=fft_len)
    
    # Sumset: |U+U|
    # Convolution: IFFT(f * f)
    conv = np.fft.ifft(f * f)
    # Real part, threshold for numerical stability
    sums = np.abs(conv.real) > 1e-9
    num_sums = np.sum(sums)
    
    # Differenceset: |U-U|
    # Cross-correlation: IFFT(conj(f) * f)
    corr = np.fft.ifft(np.conj(f) * f)
    diffs = np.abs(corr.real) > 1e-9
    num_diffs = np.sum(diffs)
    
    if num_sums == 0:
        return 0.0
        
    ratio = num_diffs / float(num_sums)
    if ratio <= 0:
        return 0.0
        
    return 1.0 + math.log(ratio) / math.log(2 * max_val + 1)

def _vec_to_list(vec):
    """Convert binary vector to list of indices where value is 1."""
    return np.nonzero(vec)[0].tolist()

def _list_to_vec(lst, length):
    """Convert list of indices to binary vector of given length."""
    vec = np.zeros(length, dtype=np.float64)
    for i in lst:
        if i < length:
            vec[i] = 1.0
    return vec

def construct():
    global _DEADLINE
    
    random.seed(42)
    np.random.seed(42)
    
    best_score = 0.0
    best_set = [0, 1]
    
    # We will explore different set sizes n and maximum values M.
    # The score depends on the ratio of unique diffs to unique sums, normalized by log(M).
    # Larger M allows for more distinct differences, but increases the denominator.
    # We want to find a sweet spot.
    
    # Strategy:
    # 1. Initialize with a set of random integers in [0, M].
    # 2. Use Simulated Annealing to optimize the set.
    #    Moves:
    #    a) Add a random element.
    #    b) Remove a random element.
    #    c) Move a random element to a new random position.
    
    # We need to choose M and n.
    # Let's try a few configurations.
    
    configs = [
        # (n_target, M_max, vec_len)
        (50, 2000, 4000),
        (100, 5000, 10000),
        (200, 10000, 20000),
        (300, 20000, 40000),
    ]
    
    # For each config, run SA for a certain time.
    # Total time budget is ~115s.
    # Let's allocate time proportionally or just iterate until deadline.
    
    for n_target, M_max, vec_len in configs:
        if time.time() > _DEADLINE - 20.0:
            break
            
        # Initialize set
        # Ensure 0 is in the set.
        # Pick n_target - 1 random numbers from 1 to M_max.
        # To avoid duplicates, we can use a set.
        initial_set = {0}
        while len(initial_set) < n_target:
            val = random.randint(1, M_max)
            initial_set.add(val)
            
        current_set = list(initial_set)
        current_set.sort()
        
        # Convert to vector for fast scoring
        current_vec = _list_to_vec(current_set, vec_len)
        current_max = max(current_set)
        current_score = _score_from_vector(current_vec, len(current_set), current_max)
        
        best_set_local = current_set[:]
        best_score_local = current_score
        
        # Simulated Annealing
        T = 1.0
        cooling_rate = 0.99
        min_T = 0.01
        
        # Determine number of steps based on time? 
        # Or just run until time limit for this config.
        # Let's run for a fixed number of steps, checking time periodically.
        
        while T > min_T:
            if time.time() > _DEADLINE - 10.0:
                break
                
            # Generate neighbor
            new_set = current_set[:]
            move_type = random.choice(['add', 'remove', 'move'])
            
            if move_type == 'add':
                if len(new_set) < 500: # Cap size to avoid huge FFTs
                    val = random.randint(0, M_max)
                    if val not in new_set:
                        new_set.append(val)
                        new_set.sort()
                    else:
                        continue # Skip if duplicate
            elif move_type == 'remove':
                if len(new_set) > 2:
                    # Don't remove 0
                    if len(new_set) > 1 and new_set[0] == 0:
                        idx = random.randint(1, len(new_set) - 1)
                    else:
                        idx = random.randint(0, len(new_set) - 1)
                    new_set.pop(idx)
            else: # move
                if len(new_set) > 1:
                    idx = random.randint(0, len(new_set) - 1)
                    if new_set[idx] == 0:
                        continue # Don't move 0
                    old_val = new_set[idx]
                    new_val = random.randint(0, M_max)
                    if new_val in new_set:
                        continue
                    new_set[idx] = new_val
                    new_set.sort()
            
            # Evaluate new set
            new_max = max(new_set)
            new_vec = _list_to_vec(new_set, vec_len)
            new_score = _score_from_vector(new_vec, len(new_set), new_max)
            
            # Acceptance probability
            delta = new_score - current_score
            if delta > 0:
                current_set = new_set
                current_score = new_score
                if new_score > best_score_local:
                    best_score_local = new_score
                    best_set_local = new_set[:]
            else:
                # Boltzmann factor
                if random.random() < math.exp(delta / T):
                    current_set = new_set
                    current_score = new_score
            
            T *= cooling_rate
            
        # Update global best
        if best_score_local > best_score:
            best_score = best_score_local
            best_set = best_set_local[:]
            
    # Final sanity check
    best_set = sorted(set(int(x) for x in best_set))
    if 0 not in best_set:
        best_set.append(0)
    if len(best_set) < 2:
        best_set = [0, 1]
        
    return best_set
# EVOLVE-BLOCK-END
