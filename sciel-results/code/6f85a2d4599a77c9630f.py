import time
import random
import math
import numpy as np

# EVOLVE-BLOCK-START
_START = time.time()
_DEADLINE = _START + 115.0  # Safety margin vs 150 s kill

def _score_from_vector(v):
    """
    Compute the score for a binary vector v representing the set U.
    v[i] = 1 if i in U, else 0.
    Score = 1 + log(|U-U|/|U+U|) / log(2*max(U) + 1)
    """
    n = len(v)
    if n < 2:
        return 0.0
    
    # Find max element
    # v is 1-indexed effectively? No, v[0] corresponds to 0.
    # max_val is the largest index i where v[i] == 1
    indices = np.nonzero(v)[0]
    if len(indices) < 2:
        return 0.0
    max_val = indices[-1]
    
    if max_val < 1:
        return 0.0
        
    # FFT length needs to be at least 2*max_val + 1 for linear convolution
    # We pad to next power of 2
    L = max_val + 1
    fft_len = 1
    while fft_len < 2 * L - 1:
        fft_len *= 2
        
    # Ensure v is long enough or pad it
    # v is already length n, but n might be > L. 
    # Actually, v represents the set. If v has length n, max_val <= n-1.
    # We need to create a vector of length fft_len for FFT.
    vec = np.zeros(fft_len, dtype=np.float64)
    # Copy v into vec. v might be shorter or longer?
    # v is the indicator vector. Its length should be max_val + 1 or larger.
    # In our representation, v is fixed length M+1 where M is the current max.
    # So len(v) == max_val + 1.
    vec[:len(v)] = v
    
    f = np.fft.fft(vec)
    
    # Sumset: |U+U|
    # Coefficients of (sum v[i] x^i)^2
    conv_sum = np.fft.ifft(f * f).real
    # Count non-zero. Threshold for numerical stability.
    # Since inputs are 0/1, coefficients are integers. 
    # However, floating point errors exist. 
    # A coefficient > 0.5 implies >= 1.
    sums_count = np.sum(conv_sum > 0.5)
    
    # Difference set: |U-U|
    # Coefficients of (sum v[i] x^i) * (sum v[j] x^{-j})
    # This is cross-correlation.
    # IFFT(conj(f) * f) gives circular cross-correlation.
    # Since we padded to >= 2*max+1, circular = linear.
    # The result is centered at 0.
    # Indices 0..max_val are positive diffs (and 0).
    # Indices fft_len-max_val..fft_len-1 are negative diffs.
    # We just count all non-zero entries in the entire array, 
    # because the support of the true correlation is within [-max, max] 
    # and we padded sufficiently so no wrap-around artifacts exist outside this range 
    # (actually, the linear correlation is zero outside [-max, max], 
    # and the circular correlation equals linear correlation in the valid range 
    # and is zero elsewhere if padding is sufficient).
    # Wait, if padding is sufficient (fft_len >= 2*max+1), then the circular 
    # correlation is exactly the linear correlation padded with zeros.
    # So counting non-zeros in the whole array gives |U-U|.
    conv_diff = np.fft.ifft(np.conj(f) * f).real
    diffs_count = np.sum(np.abs(conv_diff) > 0.5)
    
    if sums_count == 0:
        return 0.0
        
    ratio = diffs_count / float(sums_count)
    if ratio <= 0:
        return 0.0
        
    return 1.0 + math.log(ratio) / math.log(2 * max_val + 1)

def _vec_to_list(v):
    return [int(i) for i in range(len(v)) if v[i] == 1]

def _list_to_vec(U, length):
    v = np.zeros(length, dtype=np.float64)
    for u in U:
        if u < length:
            v[u] = 1.0
    return v

def _random_initial_set(n_target, max_span):
    """
    Generate a random set of size ~n_target within [0, max_span].
    """
    U = {0}
    # Add n_target - 1 random distinct integers
    while len(U) < n_target:
        U.add(random.randint(1, max_span))
    return sorted(list(U))

def construct():
    global _DEADLINE
    
    best_score = 0.0
    best_set = [0, 1]
    
    # Strategy:
    # 1. Initialize with several diverse random sets of different sizes/spans.
    # 2. Perform local search: Add, Remove, Move elements.
    # 3. Use FFT-based scoring for O(M log M) evaluation.
    
    # Define search parameters
    # We want to explore a range of sizes n and spans M.
    # The score is 1 + log(R)/log(2M+1).
    # R = |U-U|/|U+U|.
    # For random sets, |U-U| ~ n^2, |U+U| ~ n^2. R ~ 1. Score ~ 1.
    # For Golomb, R ~ 2. Score > 1.
    # We want to find sets where R is large but M is small.
    
    # Initial configurations:
    # Try sizes from 5 to 50.
    # Try spans from 10 to 200.
    
    configs = []
    for n in [5, 10, 20, 30, 50]:
        for M in [2*n, 5*n, 10*n, 20*n]:
            if M > 500: # Cap span to keep FFT fast
                continue
            configs.append((n, M))
            
    # Shuffle configs to explore different regions early
    random.seed(42)
    random.shuffle(configs)
    
    # Also include some structured starts: Golomb rulers
    # Quick greedy Golomb for small n
    def _greedy_golomb(num_elements):
        U = [0, 1]
        diffs = {1}
        current_max = 1
        for _ in range(num_elements - 2):
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

    for n in [5, 10, 20]:
        try:
            U_g = _greedy_golomb(n)
            configs.append((n, max(U_g) + 10))
        except:
            pass
            
    # Local Search
    def _local_search(U, time_limit):
        """
        Local search on a set U.
        Moves:
        1. Add a random element.
        2. Remove a random element (not 0).
        3. Move a random element to a new random location.
        """
        current_U = sorted(U)
        current_score = _score_from_set(current_U)
        best_U = current_U[:]
        best_score_local = current_score
        
        start_time = time.time()
        
        # Determine max span for moves. 
        # Start with current max * 1.5, cap at 500.
        max_span = max(50, int(max(current_U) * 1.5))
        
        # Precompute score for current
        # We will recompute score after each move.
        
        no_improvement_count = 0
        max_no_improvement = 1000
        
        while time.time() < start_time + time_limit:
            if no_improvement_count > max_no_improvement:
                break
                
            move_type = random.choice(['add', 'remove', 'move'])
            new_U = current_U[:]
            
            if move_type == 'add':
                # Add a random element
                if len(new_U) < 100: # Cap size
                    new_val = random.randint(1, max_span)
                    if new_val not in new_U:
                        new_U.append(new_val)
                        new_U.sort()
                        if len(new_U) == len(current_U): # No change
                            continue
                    else:
                        continue
            elif move_type == 'remove':
                if len(new_U) > 2:
                    # Remove a random element (not 0)
                    idx = random.randint(1, len(new_U) - 1)
                    new_U.pop(idx)
            elif move_type == 'move':
                if len(new_U) > 2:
                    idx = random.randint(1, len(new_U) - 1)
                    old_val = new_U[idx]
                    new_val = random.randint(1, max_span)
                    if new_val not in new_U:
                        new_U[idx] = new_val
                        new_U.sort()
                    else:
                        continue
            
            # Evaluate new_U
            new_score = _score_from_set(new_U)
            
            if new_score > current_score:
                current_U = new_U
                current_score = new_score
                no_improvement_count = 0
                
                if current_score > best_score_local:
                    best_score_local = current_score
                    best_U = current_U[:]
                    
                # Occasionally expand max_span if we are hitting the boundary
                if max(current_U) > max_span * 0.9:
                    max_span = int(max_span * 1.2)
                    if max_span > 500:
                        max_span = 500
            else:
                no_improvement_count += 1
                # Simulated annealing style: accept worse moves with small prob?
                # For now, strict hill climbing to be safe and fast.
                # But strict hill climbing gets stuck.
                # Let's accept if new_score is close to current_score?
                # Or just restart if stuck?
                
        return best_U, best_score_local

    # Helper to score a list
    def _score_from_set(U):
        if not U:
            return 0.0
        max_val = max(U)
        if max_val < 1:
            return 0.0
        v = _list_to_vec(U, max_val + 1)
        return _score_from_vector(v)

    # Main Loop
    for n, M in configs:
        if time.time() > _DEADLINE - 5.0:
            break
            
        # Generate initial set
        U_init = _random_initial_set(n, M)
        
        # Allocate time for this config
        remaining = _DEADLINE - time.time()
        # Distribute time among remaining configs
        # Simple heuristic: 2 seconds per config, min 0.5s
        time_per_config = max(0.5, min(2.0, remaining / len(configs)))
        
        U_opt, score_opt = _local_search(U_init, time_per_config)
        
        if score_opt > best_score:
            best_score = score_opt
            best_set = U_opt[:]
            
    # If we have time left, do a final polish on the best set found
    if time.time() < _DEADLINE - 10.0 and len(best_set) > 1:
        remaining = _DEADLINE - time.time() - 10.0
        # Polish with more time
        U_polished, score_polished = _local_search(best_set, remaining)
        if score_polished > best_score:
            best_score = score_polished
            best_set = U_polished[:]

    # Final sanity check
    best_set = sorted(set(int(x) for x in best_set))
    if 0 not in best_set:
        best_set.append(0)
    if len(best_set) < 2:
        best_set = [0, 1]
        
    return best_set
# EVOLVE-BLOCK-END
