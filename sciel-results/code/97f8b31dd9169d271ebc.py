import time
import random
import math
import numpy as np

# EVOLVE-BLOCK-START
_START = time.time()
_DEADLINE = _START + 115.0  # Safety margin vs 150 s kill

def _score_from_vector(vec, max_val):
    """
    Compute the score for a set represented by a binary indicator vector.
    Score = 1 + log(|U-U|/|U+U|) / log(2*max(U) + 1)
    
    Args:
        vec: Binary numpy array of length L.
        max_val: Maximum value in the set.
    """
    n = np.sum(vec)
    if n < 2 or max_val < 1:
        return 0.0
    
    # Use FFT to compute sumset and differenceset sizes
    # Sumset: |U+U| is the number of non-zero coefficients in the convolution of U with U.
    # Differenceset: |U-U| is the number of non-zero coefficients in the cross-correlation of U with U.
    
    L = len(vec)
    # Ensure vector is long enough to avoid circular convolution artifacts
    # The support of the convolution is [0, 2*max_val]
    # The support of the correlation is [-max_val, max_val], shifted to [0, 2*max_val]
    required_len = 2 * (max_val + 1)
    if L < required_len:
        vec = np.pad(vec, (0, required_len - L), mode='constant')
        L = len(vec)
        
    # Find next power of 2 for FFT efficiency
    fft_len = 1
    while fft_len < 2 * L - 1:
        fft_len *= 2
        
    # FFT
    f = np.fft.fft(vec, n=fft_len)
    
    # Sumset: |U+U|
    conv = np.fft.ifft(f * f)
    # Real part, threshold for numerical stability
    sums = np.abs(conv.real) > 1e-9
    num_sums = np.sum(sums)
    
    # Differenceset: |U-U|
    corr = np.fft.ifft(np.conj(f) * f)
    diffs = np.abs(corr.real) > 1e-9
    num_diffs = np.sum(diffs)
    
    if num_sums == 0:
        return 0.0
        
    ratio = num_diffs / float(num_sums)
    if ratio <= 0:
        return 0.0
        
    return 1.0 + math.log(ratio) / math.log(2 * max_val + 1)

def _list_to_vec(lst, length):
    """Convert list of indices to binary vector of given length."""
    vec = np.zeros(length, dtype=np.float64)
    for i in lst:
        if i < length:
            vec[i] = 1.0
    return vec

def _generate_exponential_set(base, n):
    """Generate a set with exponential growth."""
    u = [0]
    current = 1
    for _ in range(n - 1):
        u.append(current)
        current = int(current * base)
    return u

def _generate_fibonacci_set(n):
    """Generate a set based on Fibonacci numbers."""
    u = [0, 1]
    a, b = 0, 1
    for _ in range(n - 2):
        a, b = b, a + b
        u.append(b)
    return u

def _generate_random_sorted(n, max_val):
    """Generate a random sorted set."""
    if n == 1:
        return [0]
    # Ensure 0 is included
    others = random.sample(range(1, max_val + 1), n - 1)
    u = [0] + sorted(others)
    return u

def _local_search_greedy(u, max_val_limit, time_limit):
    """
    Perform a greedy local search to improve the score.
    Moves: Move one element to a new position.
    """
    current_u = sorted(u)
    current_max = max(current_u)
    vec_len = 2 * (current_max + 1) + 10
    current_vec = _list_to_vec(current_u, vec_len)
    current_score = _score_from_vector(current_vec, current_max)
    
    best_score = current_score
    best_u = current_u[:]
    
    # If max_val changes significantly, we need to recompute vec_len
    # For simplicity in local search, we fix the "universe" size to max_val_limit
    # This might be suboptimal if we can reduce max_val, but moving elements
    # to reduce max_val is a specific move type.
    
    # Let's allow max_val to change but keep vec_len fixed to a reasonable upper bound
    # to avoid expensive re-allocation.
    fixed_vec_len = 2 * (max_val_limit + 1)
    
    current_vec = _list_to_vec(current_u, fixed_vec_len)
    current_score = _score_from_vector(current_vec, current_max)
    best_score = current_score
    best_u = current_u[:]
    
    n_iter = 0
    start_time = time.time()
    
    # Greedy hill climbing with random restarts within the local search
    # We will try to move each element to a better position.
    
    # To speed up, we don't check every possible position, but sample them.
    # Or, we can just do a random walk that accepts improvements.
    
    while time.time() < time_limit:
        n_iter += 1
        # Pick a random element to move (not 0)
        if len(current_u) <= 1:
            break
            
        idx = random.randint(1, len(current_u) - 1)
        old_val = current_u[idx]
        
        # Pick a new value
        # Bias towards values that create larger gaps or fit the structure
        # Simple random within [0, max_val_limit]
        new_val = random.randint(0, max_val_limit)
        
        if new_val == old_val:
            continue
            
        # Check for collision
        if new_val in current_u:
            continue
            
        # Create new set
        new_u = current_u[:]
        new_u[idx] = new_val
        new_u.sort()
        new_max = max(new_u)
        
        new_vec = _list_to_vec(new_u, fixed_vec_len)
        new_score = _score_from_vector(new_vec, new_max)
        
        if new_score > current_score:
            current_u = new_u
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_u = new_u[:]
        else:
            # With some probability, accept worse moves to escape local minima
            # (Simulated Annealing style, but simple)
            if random.random() < 0.1:
                current_u = new_u
                current_score = new_score
                
        # Check time
        if time.time() > time_limit:
            break
            
    return best_u, best_score

def construct():
    global _DEADLINE
    
    random.seed(42)
    np.random.seed(42)
    
    best_score = 0.0
    best_set = [0, 1]
    
    # Strategy:
    # 1. Generate diverse initial sets using different structural heuristics.
    # 2. Refine each using fast FFT-based local search.
    # 3. Keep the best result.
    
    # We want to explore different sizes n and max values M.
    # The score is 1 + log(|U-U|/|U+U|) / log(2M+1).
    # To maximize, we want large |U-U|/|U+U| and small M.
    # However, |U-U| is bounded by 2M+1.
    # If |U-U| is close to 2M+1, the ratio is ~ (2M+1)/|U+U|.
    # |U+U| is at least 2M+1 (if U is an interval) but can be smaller?
    # No, |U+U| is at least 2|U|-1.
    # For sparse sets, |U-U| can be much larger than |U+U|.
    # Example: U = {0, 1, 3}. U+U={0,1,2,3,4,6} (6). U-U={-3,-2,-1,0,1,2,3} (7). Ratio 7/6.
    # U = {0, 1, 4}. U+U={0,1,2,4,5,8} (6). U-U={-4,-3,-2,-1,0,1,2,3,4} (9). Ratio 9/6 = 1.5.
    # U = {0, 1, 3, 7}. U+U size?
    
    # Let's try different n and M.
    # M should be large enough to allow sparse structure, but not too large to dilute the log term.
    
    configs = [
        # (n, M_limit, description)
        (10, 1000, "small_n"),
        (20, 5000, "med_n"),
        (50, 20000, "large_n"),
        (100, 100000, "very_large_n"),
    ]
    
    for n, M_limit, desc in configs:
        if time.time() > _DEADLINE - 10.0:
            break
            
        # Generate multiple initial candidates for this config
        candidates = []
        
        # 1. Exponential
        try:
            u_exp = _generate_exponential_set(2.0, n)
            if max(u_exp) <= M_limit:
                candidates.append(u_exp)
            else:
                # Scale down
                scale = M_limit / max(u_exp)
                u_exp_scaled = [int(x * scale) for x in u_exp]
                # Remove duplicates and ensure 0
                u_exp_scaled = sorted(set(u_exp_scaled))
                if 0 not in u_exp_scaled:
                    u_exp_scaled.append(0)
                if len(u_exp_scaled) >= 2:
                    candidates.append(u_exp_scaled[:n]) # Truncate if needed
        except:
            pass
            
        # 2. Fibonacci
        try:
            u_fib = _generate_fibonacci_set(n)
            if max(u_fib) <= M_limit:
                candidates.append(u_fib)
            else:
                scale = M_limit / max(u_fib)
                u_fib_scaled = [int(x * scale) for x in u_fib]
                u_fib_scaled = sorted(set(u_fib_scaled))
                if 0 not in u_fib_scaled:
                    u_fib_scaled.append(0)
                if len(u_fib_scaled) >= 2:
                    candidates.append(u_fib_scaled[:n])
        except:
            pass
            
        # 3. Random Sorted
        for _ in range(3):
            u_rand = _generate_random_sorted(n, M_limit)
            candidates.append(u_rand)
            
        # 4. "Golomb-like" Greedy Construction
        # Start with 0, 1. Add smallest number that doesn't create duplicate differences.
        # This is slow, so we do a simplified version.
        u_golomb = [0, 1]
        diffs = {0, 1, -1}
        next_val = 2
        while len(u_golomb) < n and next_val <= M_limit:
            # Check if next_val creates new differences
            valid = True
            for x in u_golomb:
                d1 = next_val - x
                d2 = x - next_val
                if d1 in diffs or d2 in diffs:
                    valid = False
                    break
            if valid:
                u_golomb.append(next_val)
                for x in u_golomb[:-1]:
                    diffs.add(next_val - x)
                    diffs.add(x - next_val)
            next_val += 1
        if len(u_golomb) >= 2:
            candidates.append(u_golomb)
            
        # Refine each candidate
        time_per_candidate = max(0.5, (115.0 - (time.time() - _START)) / (len(candidates) + 1))
        
        for u_init in candidates:
            if time.time() > _DEADLINE - 5.0:
                break
                
            # Ensure 0 is in the set
            if 0 not in u_init:
                u_init = [0] + u_init
            u_init = sorted(set(u_init))
            
            # Limit size for FFT efficiency in local search
            if len(u_init) > 200:
                # Keep 0 and randomly sample others
                others = [x for x in u_init if x != 0]
                random.shuffle(others)
                u_init = [0] + others[:200]
                
            u_opt, score_opt = _local_search_greedy(u_init, M_limit, time_per_candidate)
            
            if score_opt > best_score:
                best_score = score_opt
                best_set = u_opt
                
    # Final sanity check
    best_set = sorted(set(int(x) for x in best_set))
    if 0 not in best_set:
        best_set.append(0)
    if len(best_set) < 2:
        best_set = [0, 1]
        
    return best_set
# EVOLVE-BLOCK-END
