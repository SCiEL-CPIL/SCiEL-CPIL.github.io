"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _construct_kronecker(N, alpha):
    """Generate a Kronecker sequence based on fractional parts of alpha*n."""
    n = np.arange(N)
    # Use a quadratic phase or similar for better distribution
    # Simple Kronecker: sign(cos(2*pi*alpha*n))
    return np.sign(np.cos(2 * np.pi * alpha * n))

def _construct_welch_like(N):
    """
    Generate a Welch-like sequence.
    Welch sequences have low auto-correlation sidelobes.
    For N not a prime power, we can use a truncated or mapped version.
    Here we use a simple number-theoretic construction:
    h_n = 1 if n is a quadratic residue mod p, -1 otherwise, mapped to N.
    """
    # Find a prime p >= N
    p = N
    while True:
        is_prime = True
        for i in range(2, int(np.sqrt(p)) + 1):
            if p % i == 0:
                is_prime = False
                break
        if is_prime:
            break
        p += 1
        
    h = np.zeros(N)
    for n in range(N):
        # Legendre symbol
        if n == 0:
            h[n] = 1.0
        else:
            # pow(n, (p-1)//2, p) is 1 if residue, p-1 if non-residue
            ls = pow(n, (p-1)//2, p)
            h[n] = 1.0 if ls == 1 else -1.0
    return h

def _local_search_fast(h, time_limit, deadline):
    """
    Perform local search with incremental autoconvolution updates.
    Returns the best h found and its score.
    """
    h = h.copy()
    # Initial autoconvolution
    # Using FFT for the initial calculation is fine, or direct O(N^2) since N=400 is small
    # Direct O(N^2) is 160k ops, very fast.
    ac = np.zeros(2*N-1)
    for k in range(N):
        ac[k] = np.dot(h[:N-k], h[k:])
        ac[2*N-2-k] = ac[k] # Symmetry
        
    max_abs = np.max(np.abs(ac))
    s = np.sum(h)
    score = max_abs / (s*s)
    
    best_h = h.copy()
    best_score = score
    
    # Precompute indices for faster access? Maybe not needed with numpy.
    # We will iterate through indices and try flipping.
    # To be fast, we update ac in O(N) and check max in O(N).
    
    start_time = time.time()
    steps = 0
    no_improve_count = 0
    
    # Randomize the order of indices to avoid bias
    idx_order = np.random.permutation(N)
    current_idx_pos = 0
    
    while time.time() < deadline:
        if time.time() > time_limit + start_time:
            break
            
        i = idx_order[current_idx_pos % N]
        current_idx_pos += 1
        
        # Try flipping h[i]
        old_val = h[i]
        h[i] = -old_val
        
        # Update autoconvolution incrementally
        # Delta ac[k] = -2 * old_val * (h[i+k] + h[i-k])
        # Note: h[i] is already flipped, so the "old" contribution was old_val * neighbor.
        # New contribution is -old_val * neighbor.
        # Change = -2 * old_val * neighbor.
        
        # k ranges from -(N-1) to N-1.
        # ac index mapping: ac[k + N - 1] corresponds to lag k.
        
        # We need to update ac for all lags k where i+k or i-k is in [0, N-1].
        # Lag k:
        # Term 1: i is first index -> h[i] * h[i+k]. Valid if 0 <= i+k < N.
        # Term 2: i is second index -> h[i-k] * h[i]. Valid if 0 <= i-k < N.
        
        # Let's iterate over valid lags.
        # Lag k = i - j for j in [0, N-1] (j is the other index)
        # Actually, it's easier to think about the change in ac[m] where m is the array index.
        # ac[m] = sum_{i} h[i] h[i+m-N+1] ... this is confusing.
        
        # Let's stick to lag k.
        # ac_lag[k] = sum_{i=0}^{N-1-k} h[i] h[i+k] for k >= 0.
        # ac_lag[-k] = ac_lag[k].
        
        # Change in ac_lag[k] when flipping h[i]:
        # If i is in the sum for lag k:
        # 1. i <= N-1-k and i+k < N. Term h[i]h[i+k] changes by -2*old_val*h[i+k].
        # 2. i >= k and i-k >= 0. Term h[i-k]h[i] changes by -2*old_val*h[i-k].
        
        # So for a given i, the lags k affected are those where:
        # (i <= N-1-k) OR (i >= k)
        # k <= N-1-i OR k <= i
        
        # We can just update all k from -(N-1) to N-1.
        # For each k, check if it's affected.
        # But O(N^2) per step is too slow.
        # We need O(N).
        
        # The affected lags are k such that 0 <= i+k < N OR 0 <= i-k < N.
        # This is k in [-i, N-1-i] U [-(N-1-i), i] ... basically a range.
        # Actually, for any i, the lags k that involve h[i] are:
        # k = j - i for j in [0, N-1].
        # So k ranges from -i to N-1-i.
        
        # Let's compute the delta vector for these k values.
        # delta_ac[k] = -2 * old_val * (h[i+k] if i+k < N else 0 + h[i-k] if i-k >= 0 else 0)
        
        # We can vectorize this update.
        # Create an array of lags k
        k_min = -i
        k_max = N - 1 - i
        
        # Indices in ac array corresponding to these lags
        # ac array index = lag + N - 1
        idx_min = k_min + N - 1
        idx_max = k_max + N - 1
        
        # We need h[i+k] and h[i-k] for k in [k_min, k_max]
        # k = t - i where t is the other index?
        # Let's just loop over the affected range? No, vectorize.
        
        # For k in [k_min, k_max]:
        # term1: h[i+k] is valid if i+k in [0, N-1]
        # term2: h[i-k] is valid if i-k in [0, N-1]
        
        # Let's construct the neighbors.
        # k is the lag.
        # We want to add delta to ac[idx_min:idx_max+1]
        
        # Vectorized update:
        # k_vals = np.arange(k_min, k_max + 1)
        # neighbor1 = h[i + k_vals] where valid
        # neighbor2 = h[i - k_vals] where valid
        
        # This is tricky with numpy slicing if boundaries are crossed.
        # Given N=400, a simple loop over the ~N affected lags might be fast enough in Python if optimized?
        # Or we can do:
        
        # Let's try a hybrid: update ac in a loop over the affected lags.
        # Affected lags are approx N.
        # 400 iterations per step.
        # 120s / (400 ops) = 300 steps/s? No, Python loop is slow.
        # 10^6 Python ops/s approx. 400 ops per step -> 2500 steps/s.
        # 120s * 2500 = 300,000 steps. This is decent.
        
        # Let's implement the loop.
        # To speed up, we can precompute the "neighbor sum" for each i?
        # No, h changes.
        
        # Optimization:
        # We only need to check if the NEW max_abs is smaller.
        # We can update ac and then find max.
        
        # Let's do the update.
        # We'll use a temporary array for the delta to avoid modifying ac directly if we revert.
        # But we need to revert if it's worse.
        # So we must store old ac values or recompute?
        # Recomputing is O(N^2). Too slow.
        # Storing old values for ~N elements is O(N).
        
        # Let's just copy the affected slice of ac.
        ac_old_slice = ac[idx_min:idx_max+1].copy()
        
        # Calculate delta
        # k_vals = np.arange(k_min, k_max + 1)
        # We can compute the neighbor sums directly.
        
        # h[i+k] for k in [k_min, k_max]
        # i+k ranges from i+k_min to i+k_max
        # i+k_min = i - i = 0
        # i+k_max = i + N - 1 - i = N - 1
        # So i+k always in [0, N-1] for the range of k?
        # Wait. k_min = -i. i + (-i) = 0.
        # k_max = N-1-i. i + (N-1-i) = N-1.
        # So i+k is ALWAYS in [0, N-1] for k in [k_min, k_max].
        
        # What about i-k?
        # i - k_min = i - (-i) = 2i.
        # i - k_max = i - (N-1-i) = 2i - N + 1.
        
        # So i-k is valid if 0 <= i-k < N.
        
        # Let's construct the delta array.
        # delta[k] = -2 * old_val * (h[i+k] + (h[i-k] if valid else 0))
        
        # We can use numpy slicing for h[i+k] easily:
        # h[i+k] where k is [k_min, k_max] is h[0 : N] ?
        # i+k_min = 0. i+k_max = N-1.
        # So h[i+k] is exactly h[0:N].
        
        # h[i-k]:
        # k is [k_min, k_max].
        # i-k is [2i, 2i-N+1].
        # We need to mask valid indices.
        
        # Let's just do the loop for clarity and speed in PyPy/NumPy mix?
        # Actually, pure Python loop for 400 items is ~0.1ms.
        # 10k steps/sec. 120s -> 1.2M steps. This is good.
        
        # Let's implement the loop update.
        # It's safer and easier to debug.
        
        # Pre-compute valid k range?
        # k from -i to N-1-i.
        
        # We will update ac in place, but keep a backup of the slice.
        
        # Loop over k in range(k_min, k_max + 1):
        #   idx = k + N - 1
        #   neighbor_sum = h[i+k]
        #   if i-k >= 0 and i-k < N:
        #       neighbor_sum += h[i-k]
        #   ac[idx] -= 2 * old_val * neighbor_sum
        
        # This is ~400 iterations.
        
        # Let's try to vectorize the update to be faster.
        # h_ik = h[0:N] # Since i+k covers 0..N-1
        # h_ik_idx = np.arange(N)
        
        # For h[i-k]:
        # k_vals = np.arange(k_min, k_max + 1)
        # ik_vals = i - k_vals
        # mask = (ik_vals >= 0) & (ik_vals < N)
        # h_ik_valid = np.zeros(N)
        # h_ik_valid[mask] = h[ik_vals[mask]]
        
        # delta = -2 * old_val * (h_ik + h_ik_valid)
        # ac[idx_min:idx_max+1] += delta
        
        # This is vectorized! Much faster.
        
        k_vals = np.arange(k_min, k_max + 1)
        # i+k is always 0..N-1 in this range?
        # min i+k = i + (-i) = 0
        # max i+k = i + (N-1-i) = N-1
        # Yes.
        h_ik = h[0:N] # This is just h. Wait.
        # h[i+k] where k=0 is h[i].
        # h[i+k] where k=k_min=-i is h[0].
        # The order in k_vals is increasing.
        # i+k_vals is 0, 1, ..., N-1.
        # So h[i+k] is h[0], h[1], ..., h[N-1].
        # So h_ik is just h.
        
        # Now h[i-k]
        ik_vals = i - k_vals
        mask = (ik_vals >= 0) & (ik_vals < N)
        h_ik_sub = np.zeros(N)
        h_ik_sub[mask] = h[ik_vals[mask]]
        
        delta = -2.0 * old_val * (h + h_ik_sub)
        
        ac[idx_min:idx_max+1] += delta
        
        # Calculate new score
        # We only need to check if the max decreased.
        # We can check the max of the updated slice and the rest?
        # The rest didn't change.
        # So new_max = max(max(abs(ac[idx_min:idx_max+1])), max(abs(ac outside slice)))
        
        # To avoid full max calculation (O(N)), we can keep a global max tracker?
        # But the max might have been in the slice and decreased, so we need to recompute global max if it was the max.
        # Simplest: just compute max(abs(ac)). O(N).
        
        new_max_abs = np.max(np.abs(ac))
        new_score = new_max_abs / (s*s) # s didn't change?
        # Wait! Flipping a sign changes the sum s.
        # s_new = s - 2*old_val
        # s_new_sq = (s - 2*old_val)**2
        
        s_new = s - 2.0 * old_val
        if abs(s_new) < 1e-6:
            new_score = 1e10
        else:
            new_score = new_max_abs / (s_new * s_new)
            
        if new_score < best_score:
            best_score = new_score
            best_h = h.copy()
            s = s_new
            no_improve_count = 0
            # Optionally, re-randomize order if we've been going well
            if steps % 1000 == 0:
                idx_order = np.random.permutation(N)
                current_idx_pos = 0
        else:
            # Revert
            h[i] = old_val
            ac[idx_min:idx_max+1] = ac_old_slice
            no_improve_count += 1
            
            # If stuck, try a random flip (simulated annealing style)
            if no_improve_count > 500:
                # Flip a random element
                j = np.random.randint(N)
                old_val_j = h[j]
                h[j] = -old_val_j
                s = s - 2.0 * old_val_j
                
                # Update ac for j
                k_min_j = -j
                k_max_j = N - 1 - j
                idx_min_j = k_min_j + N - 1
                idx_max_j = k_max_j + N - 1
                ac_old_slice_j = ac[idx_min_j:idx_max_j+1].copy()
                
                k_vals_j = np.arange(k_min_j, k_max_j + 1)
                ik_vals_j = j - k_vals_j
                mask_j = (ik_vals_j >= 0) & (ik_vals_j < N)
                h_ik_sub_j = np.zeros(N)
                h_ik_sub_j[mask_j] = h[ik_vals_j[mask_j]]
                
                delta_j = -2.0 * old_val_j * (h + h_ik_sub_j)
                ac[idx_min_j:idx_max_j+1] += delta_j
                
                new_max_abs_j = np.max(np.abs(ac))
                if abs(s) < 1e-6:
                    new_score_j = 1e10
                else:
                    new_score_j = new_max_abs_j / (s * s)
                    
                if new_score_j < best_score:
                    best_score = new_score_j
                    best_h = h.copy()
                else:
                    # Revert random flip
                    h[j] = old_val_j
                    s = s + 2.0 * old_val_j
                    ac[idx_min_j:idx_max_j+1] = ac_old_slice_j
                    
                no_improve_count = 0
                idx_order = np.random.permutation(N)
                current_idx_pos = 0
                
        steps += 1
        
    return best_h, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_h = None
    best_val = 1e10
    
    np.random.seed(42)
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. All ones
    candidates.append(np.ones(N))
    
    # 2. Random
    candidates.append(np.random.choice([-1.0, 1.0], size=N))
    
    # 3. Kronecker
    candidates.append(_construct_kronecker(N, 0.5))
    candidates.append(_construct_kronecker(N, np.sqrt(2) % 1.0))
    
    # 4. Welch-like
    candidates.append(_construct_welch_like(N))
    
    # 5. Chirp
    x = np.arange(N)
    candidates.append(np.sign(np.cos(2 * np.pi * x * (x+1) / (2*N))))
    
    # 6. Block patterns
    h_block = np.ones(N)
    pos = 0
    while pos < N:
        block_len = np.random.randint(5, 20)
        sign = np.random.choice([-1.0, 1.0])
        h_block[pos:pos+block_len] = sign
        pos += block_len
    candidates.append(h_block)
    
    # Evaluate each candidate with local search
    for h_init in candidates:
        if time.time() > deadline:
            break
            
        # Ensure sum is not zero
        s = np.sum(h_init)
        if abs(s) < 10:
            h_init = -h_init if s < 0 else h_init
            # If still small, add bias? No, must be signed step.
            # If sum is 0, score is inf.
            # We can try to flip a few bits to make sum non-zero.
            # Or just skip if sum is 0?
            # Let's flip the first bit.
            if abs(s) < 1e-6:
                h_init[0] = -h_init[0]
        
        # Run local search
        # Allocate time per candidate
        # If we have 6 candidates, and 115s total, ~19s each.
        # But we want to stop early if we find a very good one.
        # Let's run for a fixed short time first, then extend the best.
        
        # Simple strategy: run local search for each candidate for 10s.
        # Then take the best and run it for the remaining time.
        
        t_start = time.time()
        h_opt, val_opt = _local_search_fast(h_init, time_limit=10.0, deadline=deadline)
        
        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt
            
    # If we have time left, polish the best one
    if best_h is not None and time.time() < deadline - 5:
        h_final, val_final = _local_search_fast(best_h, time_limit=100.0, deadline=deadline)
        if val_final < best_val:
            best_val = val_final
            best_h = h_final
            
    if best_h is None:
        return np.ones(N)
        
    return best_h
# EVOLVE-BLOCK-END
