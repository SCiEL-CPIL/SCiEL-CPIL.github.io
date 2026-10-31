import time
import random
import math
import numpy as np

# EVOLVE-BLOCK-START
_START = time.time()
_DEADLINE = _START + 115.0  # Safety margin vs 150 s kill

def _score_from_set_fft(u):
    """
    Compute the score for a set of integers u using FFT for speed.
    Score = 1 + log(|U-U|/|U+U|) / log(2*max(U) + 1)
    
    This is O(M log M) where M is max(U), much faster than O(n^2) for large n
    if M is not excessively large relative to n.
    """
    if not u or len(u) < 2:
        return 0.0
    m = max(u)
    if m < 1:
        return 0.0
    
    u_arr = np.array(u, dtype=np.int64)
    n = len(u_arr)
    
    # We need to compute |U+U| and |U-U|.
    # Let f(x) be the indicator function of U.
    # The number of representations of s in U+U is (f*f)(s).
    # The number of representations of d in U-U is (f*f_neg)(d) where f_neg(x) = f(-x).
    
    # Range of sums: [0, 2m]
    # Range of diffs: [-m, m]
    
    # For FFT, we need a size that accommodates the convolution without wrap-around.
    # Max sum index is 2m. Min diff index is -m.
    # We can compute sums and diffs separately or use a single larger array.
    
    # Let's compute sums first.
    # Size needs to be at least 2m + 1.
    # Next power of 2 for FFT efficiency.
    size = 1
    while size < 2 * m + 1:
        size *= 2
        
    # Indicator vector for U
    f = np.zeros(size, dtype=np.float64)
    f[u_arr] = 1.0
    
    # Convolution for sums
    F = np.fft.rfft(f)
    conv_sums = np.fft.irfft(F * F, n=size)
    
    # Unique sums are indices where conv_sums > 0
    # Note: numerical noise might make small non-zeros appear, so threshold > 0.5
    unique_sums = np.sum(conv_sums > 0.5)
    
    # For differences, we can use the autocorrelation property or construct a negative vector.
    # Let g(x) = 1 if x in U, 0 otherwise.
    # We want h(d) = sum_{x} g(x) g(x-d). This is the cross-correlation.
    # Alternatively, define v where v[x] = 1 if x in U, 0 else.
    # And w where w[-x] = 1 if x in U.
    # In numpy, we can just shift the array or use a trick.
    
    # Easier: Construct array for differences directly.
    # Range [-m, m]. Shift by m to make indices [0, 2m].
    # Let's use a separate FFT for clarity and to avoid aliasing issues with the sum FFT.
    # Size needs to be at least 2m + 1 for the range of differences [-m, m] shifted to [0, 2m].
    
    # Actually, we can reuse the same size if it's large enough.
    # The difference d = u_i - u_j. Max d = m, Min d = -m.
    # If we define array A of size size, with A[u] = 1.
    # And array B of size size, with B[size - u] = 1 (representing -u modulo size).
    # Then convolution C = A * B.
    # C[k] = sum A[i] B[k-i].
    # B[j] is 1 if j = size - u for some u. So k-i = size - u => i - u = k - size.
    # So C[k] counts pairs (i, u) such that i - u = k - size.
    # Let d = i - u. Then k = d + size.
    # So C[size + d] is the count of differences d.
    
    B = np.zeros(size, dtype=np.float64)
    # B[size - u] = 1. Note: if u=0, B[size]=0 (index 0). 
    # Wait, index size is 0 in modulo. So B[0] should be 1 if 0 is in U.
    # Let's be careful.
    # We want B[j] = 1 if (size - j) % size is in U.
    # So for each u in U, we set B[(size - u) % size] = 1.
    indices = (size - u_arr) % size
    B[indices] = 1.0
    
    GB = np.fft.rfft(B)
    conv_diffs = np.fft.irfft(F * GB, n=size)
    
    # The difference d is represented at index (size + d) % size.
    # Since d is in [-m, m], and size > 2m, there is no wrap-around collision for the main lobe.
    # Specifically, indices of interest are [size-m, size+m].
    # Note: index 0 corresponds to d=0 (since size+0=size % size = 0).
    # So we just need to count non-zeros in the relevant range.
    
    # The range of valid difference indices in the FFT output [0, size-1]
    # corresponds to d in [-(size//2), size//2 - 1] roughly, but we know d in [-m, m].
    # Let's just check the slice [size-m, size+m+1].
    # Since size is power of 2 and size > 2m, size+m < size.
    # So the slice is simply conv_diffs[size-m : size+m+1].
    
    diff_slice = conv_diffs[size-m : size+m+1]
    unique_diffs = np.sum(diff_slice > 0.5)
    
    if unique_sums == 0 or unique_diffs == 0:
        return 0.0
        
    ratio = unique_diffs / float(unique_sums)
    if ratio <= 0:
        return 0.0
        
    return 1.0 + math.log(ratio) / math.log(2 * m + 1)

def _generate_structured_set(n, type_id):
    """
    Generate a candidate set of size n using different structural templates.
    """
    u = {0}
    if type_id == 0:
        # Arithmetic progression (low sumset, low diffset)
        step = random.randint(1, 5)
        for i in range(1, n):
            u.add(i * step)
    elif type_id == 1:
        # Geometric progression (high max, sparse)
        ratio = random.uniform(1.1, 1.5)
        curr = 1
        for i in range(1, n):
            curr = int(curr * ratio)
            u.add(curr)
    elif type_id == 2:
        # "Bounded" random: Random points in [0, N] with N ~ n^1.5
        N = int(n ** 1.5) + 10
        while len(u) < n:
            u.add(random.randint(0, N))
    elif type_id == 3:
        # "Clustered": Groups of close numbers
        num_clusters = max(1, n // 10)
        points_per_cluster = n // num_clusters
        curr_max = 0
        for _ in range(num_clusters):
            start = curr_max + random.randint(5, 20)
            for _ in range(points_per_cluster):
                u.add(start + random.randint(0, 5))
            curr_max = max(u)
    elif type_id == 4:
        # "Golomb-like" greedy but with random restarts to avoid local minima
        # This is just a random set that we will optimize heavily
        N = n * n * 2
        while len(u) < n:
            u.add(random.randint(0, N))
    else:
        # Fallback: random
        N = n * 10
        while len(u) < n:
            u.add(random.randint(0, N))
            
    return sorted(list(u))

def _local_search_fft(u, timeout=10.0):
    """
    Local search using FFT-based scoring.
    Moves: Add, Remove, Replace, Shift.
    """
    u = sorted(set(u))
    if len(u) < 2:
        return [0, 1], 0.0
        
    current_score = _score_from_set_fft(u)
    best_u = u[:]
    best_score = current_score
    start_time = time.time()
    
    n = len(u)
    m = max(u)
    
    # Heuristic: If n is small, allow larger relative moves.
    # If n is large, moves should be smaller to maintain structure.
    
    for step in range(10000):
        if time.time() > start_time + timeout or time.time() > _DEADLINE - 5.0:
            break
            
        # Random move selection
        move_type = random.random()
        
        if move_type < 0.3:
            # Replace an element
            if n < 2: continue
            idx = random.randint(1, n - 1) # Don't replace 0
            old_val = u[idx]
            # Perturb
            # Range depends on density
            # If dense, small range. If sparse, larger range.
            # Estimate density: n / (m+1)
            density = n / (m + 1)
            # If density is high, step small.
            max_delta = max(1, int(10 / density))
            # Also allow larger jumps occasionally
            if random.random() < 0.1:
                max_delta = int(m * 0.1)
            
            new_val = old_val + random.randint(-max_delta, max_delta)
            if new_val < 0 or new_val in u:
                continue
                
            new_u = u[:]
            new_u[idx] = new_val
            new_u = sorted(set(new_u))
            if len(new_u) != n:
                continue
                
        elif move_type < 0.6:
            # Add an element
            if n > 200: # Cap size to keep FFT fast
                continue
            # Try adding in a gap or near existing
            if n > 1:
                idx = random.randint(0, n - 1)
                base = u[idx]
                # Try nearby
                new_val = base + random.randint(1, max(1, int(m*0.05)))
                if new_val in u:
                    # Try random
                    new_val = random.randint(0, m + 10)
                    if new_val in u:
                        continue
            else:
                new_val = 1
                
            new_u = sorted(u + [new_val])
            
        elif move_type < 0.8:
            # Remove an element
            if n <= 5:
                continue
            idx = random.randint(1, n - 1)
            new_u = u[:]
            new_u.pop(idx)
            
        else:
            # Shift a subset? Or just a large random jump for one element
            if n < 2: continue
            idx = random.randint(1, n - 1)
            old_val = u[idx]
            # Large jump
            new_val = random.randint(0, m + 50)
            if new_val in u:
                continue
            new_u = u[:]
            new_u[idx] = new_val
            new_u = sorted(set(new_u))
            if len(new_u) != n:
                continue

        s = _score_from_set_fft(new_u)
        
        if s > current_score:
            u = new_u
            current_score = s
            n = len(u)
            m = max(u)
            if current_score > best_score:
                best_score = current_score
                best_u = u[:]
            # Reset step counter implicitly by continuing, but maybe accept worse moves (simulated annealing)?
            # For now, pure hill climbing.
        else:
            # Occasionally accept worse moves to escape local optima?
            # Given time constraints, stick to greedy for safety, 
            # but the diverse initialization helps.
            pass

    return best_u, best_score

def construct():
    global _DEADLINE
    
    random.seed(42)
    np.random.seed(42)
    
    best_u = [0, 1]
    best_s = 0.0
    
    # Strategy:
    # 1. Try a variety of initial structures (AP, Geo, Random, Clustered).
    # 2. Use FFT-based local search to refine them.
    # 3. The FFT method allows us to handle n ~ 100-200 efficiently.
    
    # Sizes to try: Small to Medium.
    # Theoretical optimum might be at intermediate sizes where |D|/|S| is maximized
    # relative to the log(max U) penalty.
    
    sizes_to_try = [10, 20, 30, 50, 80, 120, 150]
    types_to_try = [0, 1, 2, 3, 4]
    
    # We have ~115 seconds.
    # FFT for m=10000 is fast. For m=100000, still fast.
    # Local search step is O(1) FFT calls? No, each step is 1 FFT call.
    # FFT of size 2^17 (131072) is ~ms.
    # So we can do thousands of steps.
    
    for size in sizes_to_try:
        if time.time() > _DEADLINE - 10.0:
            break
            
        for t in types_to_try:
            if time.time() > _DEADLINE - 5.0:
                break
                
            # Generate initial set
            u_init = _generate_structured_set(size, t)
            
            # Ensure 0 is in set
            if 0 not in u_init:
                u_init = sorted(set(u_init) | {0})
            
            # Local search
            # Allocate more time for larger sizes if they look promising?
            # Just use a fixed budget per candidate.
            # If we have 7 sizes * 5 types = 35 candidates.
            # 110s / 35 ~ 3s per candidate.
            timeout_per_candidate = 3.0
            
            u_opt, s_opt = _local_search_fft(u_init, timeout=timeout_per_candidate)
            
            if s_opt > best_s:
                best_s = s_opt
                best_u = u_opt[:]
                
    # Final polish on the global best with more time
    if time.time() < _DEADLINE - 2.0:
        remaining_time = _DEADLINE - time.time() - 2.0
        u_final, s_final = _local_search_fft(best_u, timeout=remaining_time)
        if s_final > best_s:
            best_s = s_final
            best_u = u_final[:]
            
    # Sanity check
    best_u = sorted(set(int(x) for x in best_u))
    if 0 not in best_u:
        best_u.append(0)
    if len(best_u) < 2:
        best_u = [0, 1]
        
    return best_u
# EVOLVE-BLOCK-END
