import numpy as np
import time

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # 1. Precompute all arithmetic progressions
    # An AP is defined by start a, step s, length k.
    # Indices: a, a+s, ..., a+(k-1)s < n
    # We store them as arrays of indices.
    progs = []
    for a in range(n):
        for s in range(1, n):
            idx = list(range(a, n, s))
            if len(idx) >= 1:
                progs.append(np.array(idx, dtype=np.int32))
    
    Np = len(progs)
    
    # 2. Build incidence matrix M (n x Np) where M[i, p] = 1 if i in prog p
    # Using float32 for speed and memory efficiency in dot products
    M = np.zeros((n, Np), dtype=np.float32)
    for pid, idx in enumerate(progs):
        M[idx, pid] = 1.0
    
    # M_T = M.T (Np x n). sums = M_T @ signs
    M_T = M.T
    
    # 3. Identify "Critical" progressions
    # The max absolute sum is determined by a few "bottleneck" progressions.
    # Random signs will have sums roughly N(0, sqrt(k)).
    # Progressions with larger k have higher potential max sums.
    # However, short progressions (k=1) are trivial (sum is +/-1).
    # Let's identify progressions with length >= 2 and potentially high variance.
    # Actually, to be safe, we should consider all, but for optimization speed,
    # we can focus on a subset if we assume the global max is driven by a few.
    # But for correctness of the "max" evaluation, we must check all.
    # For the *search heuristic*, we can use a subset to estimate the objective faster?
    # No, the objective is the max over ALL. We must compute max over all to be sure.
    # However, computing max over all 256*~25000 ~ 6.5M elements per step is slow in pure Python loops,
    # but fast in NumPy.
    
    # Let's check the cost of max_abs_sum.
    # sums is size Np. np.max(np.abs(sums)) is O(Np). Np is approx 256 * 256 / 2 ~ 32k?
    # Let's estimate Np.
    # For each s, number of starts is n. Total ~ n^2.
    # Actually, for s=1, n progressions. s=2, n/2 + n/2... 
    # Sum_{s=1}^{n} n = n^2 = 65536.
    # So Np is around 65k.
    # np.max on 65k floats is very fast (~microseconds).
    
    def get_max_abs_sum(sums):
        return np.max(np.abs(sums))
    
    # 4. Initial random assignment
    rng = np.random.RandomState(42)
    signs = rng.choice([-1, 1], size=n).astype(np.float32)
    sums = M_T @ signs
    current_score = get_max_abs_sum(sums)
    best_signs = signs.copy()
    best_score = current_score
    
    # 5. Block Coordinate Descent (BCD)
    # Instead of flipping one sign, flip a block of contiguous signs.
    # Why? Flipping one sign changes sums by -2*sign_i*M[i,:].
    # Flipping a block B changes sums by -2 * (signs_B @ M[B,:]).
    # This allows larger jumps in the solution space, potentially escaping local minima
    # that single flips cannot.
    
    block_sizes = [1, 2, 4, 8, 16]
    
    # We will iterate:
    # 1. Try all blocks of size 1 (standard local search)
    # 2. Try random blocks of size 2, 4, 8, 16
    
    # Optimization: Precompute M_blocks? No, M is dense enough.
    # M[B, :] is a view/slice.
    
    def try_block_flip(signs, sums, score, block_start, block_len, M_T, M, rng, deadline):
        # block_start, block_len
        # indices: block_start to block_start + block_len - 1
        # Must be within [0, n)
        if block_start + block_len > n:
            return signs, sums, score, False
        
        # Extract block signs
        block_signs = signs[block_start:block_start+block_len]
        
        # Calculate delta in sums
        # delta_sums = -2 * (M[block_indices, :] @ block_signs)
        # M[block_indices, :] is (block_len, Np)
        # block_signs is (block_len,)
        # Result is (Np,)
        
        # This matrix-vector product is (block_len * Np) ops.
        # For block_len=1, it's just M[i,:] which we can precompute or access.
        # For larger blocks, it's a small mat-vec.
        
        delta_sums = -2.0 * (M[block_start:block_start+block_len, :] @ block_signs)
        new_sums = sums + delta_sums
        new_score = np.max(np.abs(new_sums))
        
        if new_score < score:
            # Accept
            signs[block_start:block_start+block_len] = -block_signs
            return signs, new_sums, new_score, True
        return signs, sums, score, False

    # Phase 1: Deterministic Greedy Block Descent
    # Iterate through blocks, accepting any improvement.
    # We'll cycle through block sizes.
    
    # To speed up, we can check if a flip is *likely* to improve by looking at
    # the current max sum APs. But for now, let's just do the calculation.
    # The calculation cost for block_len=1 is Np additions/mults.
    # For 256 positions, that's 256 * Np ops. Np~65k. 256*65k ~ 16M ops.
    # This is fast in NumPy (vectorized).
    
    # Let's refine the single flip check to be vectorized over ALL positions?
    # No, we need to check if ANY single flip improves.
    # We can compute the new score for ALL possible single flips in parallel?
    # new_sums_i = sums - 2*signs[i]*M[i,:]
    # We need max_abs(new_sums_i) for each i.
    # This requires storing Np * n sums? No, we can do it iteratively or use a trick.
    
    # Trick:
    # Let S = sums.
    # If we flip i, S' = S - 2*s_i*M_i.
    # We want to know if max(|S'|) < max(|S|).
    # This is hard to vectorize fully without O(n*Np) memory.
    # However, Np is ~65k, n=256. 256*65k*4 bytes = 64MB. Feasible.
    
    # Let's try to vectorize the "best single flip" search.
    # For each position i, compute new_sums_i.
    # new_sums = S[None, :] - 2.0 * signs[:, None] * M  # (n, Np)
    # max_abs_new = np.max(np.abs(new_sums), axis=1) # (n,)
    # best_i = argmin(max_abs_new)
    # If max_abs_new[best_i] < current_score, flip it.
    
    # This is O(n * Np) per iteration.
    # 256 * 65k = 16.6M ops. Very fast in NumPy.
    
    def greedy_single_flip_pass(signs, sums, score):
        # Vectorized check of all single flips
        # signs: (n,), M: (n, Np), sums: (Np,)
        # new_sums_matrix: (n, Np)
        # Memory: 256 * 65536 * 4 bytes = 67MB. Acceptable.
        
        # To save memory, we can do it in chunks or just accept the 67MB.
        # Let's do it.
        
        # Optimization: We only care about the max.
        # Let's compute the new max abs sum for each potential flip.
        
        # S'_{i,p} = S_p - 2 * s_i * M_{i,p}
        
        # We can compute this as:
        # term = 2.0 * signs[:, None] * M  # (n, Np)
        # new_sums = sums[None, :] - term
        
        # This creates a large temporary array.
        # Let's try to be smarter.
        # The max of |S_p - delta_{i,p}|.
        
        # For now, let's just do the standard loop but optimized with NumPy slices.
        # The vectorized approach is O(n*Np).
        
        best_i = -1
        best_new_score = score
        
        # We can parallelize this check if we use the large matrix.
        # Let's use the matrix approach as it's likely faster than a Python loop of 256 numpy ops.
        
        # Constructing the (n, Np) matrix:
        # signs[:, None] is (n, 1)
        # M is (n, Np)
        # product is (n, Np)
        
        # Let's check memory limit. 67MB is fine.
        
        # However, creating this matrix every iteration might be slow due to allocation.
        # Pre-allocating?
        
        # Let's stick to a faster heuristic:
        # Only check flips that affect the current "worst" progressions.
        
        # Identify top K progressions with highest |sum|
        K = 10
        abs_sums = np.abs(sums)
        worst_indices = np.argpartition(abs_sums, -K)[-K:]
        worst_sums = sums[worst_indices]
        
        # For a flip to improve the max, it must reduce the magnitude of the current max,
        # OR it must not increase any sum beyond the current max.
        
        # This is complex. Let's go back to the vectorized full check, but only if time permits.
        # Given the 90s budget, and that 16M ops is ~10ms in NumPy, we can do ~1000 passes.
        
        # Let's implement the vectorized full check.
        
        # To avoid 67MB allocation overhead in tight loops, we can reuse a buffer?
        # Or just accept it.
        
        # Actually, a simpler vectorized approach:
        # We want to find i that minimizes max_p |S_p - 2 s_i M_{i,p}|
        
        # Let's just use the loop with NumPy vectorized inner step for the "try" part.
        # The loop is 256 iterations. Each iteration does a (Np,) vector op.
        # 256 * (Np ops) = 16M ops.
        # A Python loop of 256 iterations calling NumPy is slower than one big NumPy op.
        
        # Let's try the big NumPy op.
        
        # signs: (256,)
        # M: (256, Np)
        
        # delta = 2.0 * signs[:, None] * M  # (256, Np)
        # new_sums = sums[None, :] - delta   # (256, Np)
        # max_abs = np.max(np.abs(new_sums), axis=1) # (256,)
        
        # This allocates 256*Np floats.
        
        # Let's try it.
        
        try:
            # Use float32 to save memory/bandwidth
            delta = 2.0 * (signs[:, None] * M)
            new_sums = sums[None, :] - delta
            max_abs_new = np.max(np.abs(new_sums), axis=1)
            min_idx = np.argmin(max_abs_new)
            min_val = max_abs_new[min_idx]
            
            if min_val < score:
                signs[min_idx] = -signs[min_idx]
                sums = new_sums[min_idx] # Reuse calculated sums
                score = min_val
                return signs, sums, score, True
        except MemoryError:
            pass # Fallback to loop if memory fails (unlikely)
            
        return signs, sums, score, False

    # Phase 1: Greedy Single Flip (Vectorized)
    # Run until no improvement or time runs out.
    
    improved = True
    while improved:
        if time.time() > deadline - 30.0:
            break
        signs, sums, score, improved = greedy_single_flip_pass(signs, sums, score)
        if score < best_score:
            best_score = score
            best_signs = signs.copy()

    # Phase 2: Block Flips (Randomized)
    # Single flips get stuck in local minima. Block flips can escape.
    # We'll use SA-like acceptance for blocks to allow temporary worsening.
    
    T = 1.0
    signs = best_signs.copy()
    sums = M_T @ signs
    score = get_max_abs_sum(sums)
    
    # Precompute block views? No, slicing is fast.
    
    n_block_iters = 5000
    for it in range(n_block_iters):
        if time.time() > deadline - 5.0:
            break
            
        # Pick a random block size and start
        bs = rng.choice(block_sizes)
        if bs == 1:
            # Already handled by Phase 1, but let's allow it in SA
            start = rng.randint(n)
            end = start + 1
        else:
            start = rng.randint(0, n - bs + 1)
            end = start + bs
            
        block_signs = signs[start:end]
        delta_sums = -2.0 * (M[start:end, :] @ block_signs)
        new_sums = sums + delta_sums
        new_score = np.max(np.abs(new_sums))
        
        delta_score = new_score - score
        
        # Accept if improvement or with probability exp(-delta/T)
        if delta_score <= 0 or rng.rand() < np.exp(-delta_score / T):
            signs[start:end] = -block_signs
            sums = new_sums
            score = new_score
            
            if score < best_score:
                best_score = score
                best_signs = signs.copy()
                
        # Cool down
        T *= 0.999

    # Phase 3: Final Polish with Greedy Single Flips
    signs = best_signs.copy()
    sums = M_T @ signs
    score = get_max_abs_sum(sums)
    
    improved = True
    while improved:
        if time.time() > deadline - 1.0:
            break
        signs, sums, score, improved = greedy_single_flip_pass(signs, sums, score)
        if score < best_score:
            best_score = score
            best_signs = signs.copy()

    return [int(x) for x in best_signs]
