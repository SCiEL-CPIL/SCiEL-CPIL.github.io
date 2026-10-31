import time
import random
import math
import numpy as np

_START = time.time()
_DEADLINE = _START + 145.0  # Safety margin for evaluator kill

def _compute_scores_from_vector(vec):
    """
    Compute |U+U| and |U-U| from a binary vector representing U.
    vec is a numpy array of 0s and 1s.
    Returns (n_sums, n_diffs, max_val).
    """
    # Find max value with 1
    indices = np.nonzero(vec)[0]
    if len(indices) == 0:
        return 0, 0, 0
    max_val = int(indices[-1])
    
    # If max_val is 0, set is {0}
    if max_val == 0:
        return 1, 1, 0
        
    # FFT size needs to be at least 2*max_val + 1
    size = 2 * max_val + 1
    fft_size = 1
    while fft_size < size:
        fft_size *= 2
    
    # Prepare vector for FFT (zero padded)
    fft_vec = np.zeros(fft_size, dtype=np.float64)
    fft_vec[:len(vec)] = vec
    
    # Sumset
    fft_sum = np.fft.fft(fft_vec)
    sum_conv = np.fft.ifft(fft_sum * fft_sum).real
    # Count non-zero entries in [0, 2*max_val]
    # Note: due to floating point, use a threshold
    n_sums = int(np.sum(sum_conv[:size] > 0.5))
    
    # Difference set (Autocorrelation)
    # diff[k] corresponds to sum_x vec[x] * vec[x+k]
    # In FFT: ifft(fft(vec) * conj(fft(vec)))
    # The result is circular autocorrelation.
    # For linear autocorrelation of a signal of length L padded to N:
    # c[k] for k=0..N-1.
    # The linear autocorrelation r[k] for k in [-L+1, L-1] is:
    # r[k] = c[k] for k >= 0
    # r[k] = c[N-k] for k < 0
    # We want k in [-max_val, max_val].
    
    fft_diff = np.fft.fft(fft_vec)
    diff_conv = np.fft.ifft(fft_diff * np.conj(fft_diff)).real
    
    # Indices for k = 0 to max_val
    # diff_conv[0] is k=0
    # diff_conv[1] is k=1
    # ...
    # diff_conv[max_val] is k=max_val
    
    # Indices for k = -1 to -max_val
    # diff_conv[N-1] is k=-1
    # diff_conv[N-2] is k=-2
    # ...
    # diff_conv[N-max_val] is k=-max_val
    
    # Count non-zeros
    # k=0
    count = 0
    if diff_conv[0] > 0.5:
        count += 1
        
    # k > 0
    if max_val > 0:
        count += int(np.sum(diff_conv[1:max_val+1] > 0.5))
        # k < 0
        count += int(np.sum(diff_conv[fft_size-max_val:fft_size] > 0.5))
        
    n_diffs = count
    
    return n_sums, n_diffs, max_val

def _score_from_vector(vec):
    n_sums, n_diffs, max_val = _compute_scores_from_vector(vec)
    if n_sums == 0 or max_val == 0:
        return 0.0
    ratio = n_diffs / float(n_sums)
    if ratio <= 0:
        return 0.0
    return 1.0 + math.log(ratio) / math.log(2 * max_val + 1)

def _vector_to_set(vec):
    return [int(x) for x in np.nonzero(vec)[0]]

def construct():
    random.seed(42)
    
    best_score = 0.0
    best_vec = None
    
    # We will search over different window sizes M (max value).
    # The score is 1 + log(R) / log(2M+1).
    # To maximize this, we want R large and M small.
    # However, R is bounded by (2M+1) / (|U+U|).
    # |U+U| >= 2|U|-1.
    # So R <= (2M+1) / (2|U|-1).
    # If we pick |U| small, R can be large.
    # If we pick M large, log(2M+1) grows, reducing the score.
    
    # Let's try a range of M.
    # Small M: M=10, 20, 50, 100, 200, 500, 1000.
    
    # For a given M, we have a binary vector of length M+1.
    # We want to choose a subset U containing 0.
    
    # Strategy:
    # 1. Start with a random sparse set.
    # 2. Use simulated annealing on the binary vector.
    #    Moves: Flip a bit (add/remove element).
    #    Constraint: 0 must always be present.
    
    # To handle different M's, we can fix a maximum M_max and allow the "active" max to vary?
    # No, the score depends on max(U). If we flip a high bit to 1, max increases.
    # It is better to search each M separately or use a dynamic M.
    # Given the time limit, searching specific M's is safer.
    
    # Let's evaluate a few promising M values.
    # Based on theory, optimal |U| for a given M is roughly sqrt(M) or similar?
    # Actually, for random sets, |U+U| ~ M^2 / 2^k, |U-U| ~ M^2 / 2^k.
    # Ratio ~ 1.
    # We want Ratio > 1.
    # Sets with many differences and few sums.
    # Sidon sets have |U+U| ~ |U|^2, |U-U| ~ |U|^2. Ratio ~ 1.
    # But we can do better?
    # If U is an arithmetic progression, |U+U| ~ 2|U|, |U-U| ~ 2|U|. Ratio ~ 1.
    # If U is a geometric progression, |U+U| ~ |U|, |U-U| ~ |U|. Ratio ~ 1.
    
    # Wait, the score is 1 + log(R)/log(2M+1).
    # If R=1, score = 1.
    # If R=2, score = 1 + log(2)/log(2M+1).
    # For M=10, log(21) ~ 3.04. log(2) ~ 0.69. Score ~ 1.22.
    # For M=100, log(201) ~ 5.3. Score ~ 1.13.
    # So smaller M is better IF we can maintain R>1.
    
    # Can we get R >> 1?
    # |U-U| can be as large as 2M+1.
    # |U+U| can be as small as 2|U|-1.
    # Max R = (2M+1) / (2|U|-1).
    # If |U|=2, {0, M}, U+U={0, M, 2M}, |U+U|=3. U-U={-M, 0, M}, |U-U|=3. R=1.
    # If |U|=3, {0, a, b}.
    # Let's just search.
    
    # We will run SA for a few M values.
    # M values: 10, 20, 50, 100, 200, 500, 1000, 2000.
    # Time budget per M: ~10s.
    
    m_values = [10, 20, 50, 100, 200, 500, 1000, 2000]
    
    # Sort by potential? Smaller M has higher ceiling.
    # Let's try smaller M first.
    
    for M in m_values:
        if time.time() > _DEADLINE - 10:
            break
            
        # Initialize vector
        # Start with a sparse random set
        # Density p such that |U| ~ sqrt(M)?
        # Let's try |U| ~ 10 to 20 for small M.
        
        n_elements = min(M, 15) # Cap number of elements
        
        vec = np.zeros(M + 1, dtype=np.float64)
        vec[0] = 1.0
        
        # Randomly select other elements
        if n_elements > 1:
            candidates = list(range(1, M + 1))
            random.shuffle(candidates)
            for i in range(n_elements - 1):
                if i < len(candidates):
                    vec[candidates[i]] = 1.0
        
        # Simulated Annealing
        current_score = _score_from_vector(vec)
        best_local_score = current_score
        best_local_vec = vec.copy()
        
        T = 0.5
        T_min = 0.001
        steps = 2000
        
        start_time = time.time()
        
        for step in range(steps):
            if time.time() > _DEADLINE - 5:
                break
            if time.time() > start_time + 8.0: # Max 8s per M
                break
                
            # Pick a random index to flip
            # Don't flip 0
            idx = random.randint(1, M)
            
            old_val = vec[idx]
            vec[idx] = 1.0 - old_val
            
            new_score = _score_from_vector(vec)
            
            delta = new_score - current_score
            
            # Metropolis
            if delta > 0 or (T > T_min and random.random() < math.exp(delta / T)):
                current_score = new_score
                if new_score > best_local_score:
                    best_local_score = new_score
                    best_local_vec = vec.copy()
            else:
                # Revert
                vec[idx] = old_val
            
            # Cool down
            T *= 0.995
            
        if best_local_score > best_score:
            best_score = best_local_score
            best_vec = best_local_vec.copy()
            
    # If we found a good vector, convert to list
    if best_vec is not None:
        return _vector_to_set(best_vec)
    
    # Fallback: simple set
    return [0, 1]

# Test locally
if __name__ == "__main__":
    start = time.time()
    u = construct()
    end = time.time()
    print(f"Time: {end-start:.2f}s")
    print(f"Set: {u}")
    print(f"Max: {max(u)}")
    # Manually score
    sums = set()
    diffs = set()
    for a in u:
        for b in u:
            sums.add(a+b)
            diffs.add(a-b)
    m = max(u)
    score = 1.0 + math.log(len(diffs)/len(sums)) / math.log(2*m+1)
    print(f"Score: {score}")
