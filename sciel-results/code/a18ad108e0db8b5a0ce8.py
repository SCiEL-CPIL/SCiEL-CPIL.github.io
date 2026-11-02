import numpy as np
import time

N = 400
DEADLINE = 148.0  # safety margin before 150s kill

def _compute_c3_bound(signs):
    """
    Compute the L1 norm of the autoconvolution (upper bound on C3).
    For a signed step function with N intervals of width dx on [-1/4, 1/4],
    the autoconvolution at lag k is R[k] = sum_i f[i]*f[i+k] * dx.
    The L1 norm is sum_k |R[k]| * dx.
    """
    dx = 0.5 / N
    n_fft = 2 * N
    F = np.fft.rfft(signs.astype(np.float64), n=n_fft)
    conv = np.fft.irfft(F * F, n=n_fft) * dx
    return float(np.sum(np.abs(conv)) * dx)

def _compute_R(signs):
    """Compute the full linear autocorrelation R[k] for k=0..N-1 (unscaled by dx)."""
    n_fft = 2 * N
    F = np.fft.rfft(signs.astype(np.float64), n=n_fft)
    R = np.fft.irfft(F * F, n=n_fft)
    return R[:N]  # R[0..N-1], where R[k] = sum_{i=0}^{N-1-k} f[i]*f[i+k]

def _objective_from_R(R):
    """Compute the L1 objective from autocorrelation array R (unscaled).
    Objective = dx * (|R[0]| + 2*sum_{k=1}^{N-1}|R[k]|)
    Since R[0] = N (constant for ±1 signs), we can drop it for comparison.
    """
    dx = 0.5 / N
    return dx * (R[0] + 2.0 * np.sum(np.abs(R[1:])))

def _block_flip_delta(signs, s, bs, R):
    """
    Compute the change in objective when flipping block [s, s+bs).
    Flipping signs in [s, s+bs) changes f[i] -> -f[i] for i in that range.
    
    The autocorrelation R[k] = sum_{i=0}^{N-1-k} f[i]*f[i+k].
    After flipping block [s, s+bs):
    - For a product f[i]*f[i+k], it changes sign if exactly one of i, i+k is in the block.
    - Both in block: product unchanged (both flipped).
    - Neither in block: product unchanged.
    
    So the change in R[k] = -2 * sum_{i: exactly one of i, i+k in [s, s+bs)} f[i]*f[i+k]
    
    We can compute this efficiently using prefix sums of f.
    """
    # For lag k, the pairs (i, i+k) where exactly one is in [s, s+bs):
    # i in [s, s+bs) and i+k NOT in [s, s+bs)
    #   i+k NOT in [s, s+bs) means i+k < s or i+k >= s+bs
    #   i in [s, s+bs) and i+k < s: impossible since k>=1, i+k > i >= s
    #   i in [s, s+bs) and i+k >= s+bs: i in [s+bs-k, s+bs) intersect [s, s+bs)
    #     = [max(s, s+bs-k), s+bs)
    # i NOT in [s, s+bs) and i+k in [s, s+bs):
    #   i+k in [s, s+bs) means i in [s-k, s+bs-k)
    #   i NOT in [s, s+bs): i < s or i >= s+bs
    #   i in [s-k, s) : i in [s-k, s) intersect valid range [0, N-1-k]
    #   i in [s+bs-k, s+bs) : but i >= s+bs and i < s+bs is empty
    #   Actually: i in [s-k, s+bs-k) and i NOT in [s, s+bs)
    #     = [s-k, s) union [s+bs-k, s+bs) intersect [s+bs, ...) -> [s+bs-k, s+bs) is within [s, s+bs) if s+bs-k >= s i.e. bs >= k
    #     Let me be more careful.
    
    # Simpler: compute new R[k] for affected lags directly.
    # A lag k is affected if the block [s, s+bs) overlaps with [0, N-1-k] in a way that
    # creates "exactly one in block" pairs.
    # This happens when s < N-k and s+bs > 0, i.e., the block intersects [0, N-1-k].
    # More precisely, lag k is affected if there exists i in [0, N-1-k] such that
    # exactly one of i, i+k is in [s, s+bs).
    # This is equivalent to: the block [s, s+bs) is not entirely before [0, N-1-k] 
    # and not entirely after, AND the block has positive length.
    # Actually, lag k is affected iff max(s, 0) < min(s+bs, N-k) OR 
    # there's an i with i in block but i+k not, or i not in block but i+k in block.
    # 
    # Let's just compute the delta for all k where it could be nonzero.
    # k ranges from 1 to N-1. For a given k, the affected i values are those where
    # (i in [s,s+bs) XOR i+k in [s,s+bs)).
    
    # For efficiency, we note that for a block flip, the change in R[k] is:
    # delta_R[k] = -2 * (sum_{i in A_k} f[i]*f[i+k])
    # where A_k = {i : i in [s,s+bs) XOR i+k in [s,s+bs)}
    
    # A_k consists of:
    # Part 1: i in [s, s+bs) AND i+k not in [s, s+bs)
    #   i+k not in [s, s+bs) => i+k < s OR i+k >= s+bs
    #   i in [s, s+bs) and i+k < s: impossible (k>=1, i>=s => i+k > s)
    #   i in [s, s+bs) and i+k >= s+bs: i >= s+bs-k, so i in [max(s, s+bs-k), min(s+bs-1, N-1-k)]
    # Part 2: i not in [s, s+bs) AND i+k in [s, s+bs)
    #   i+k in [s, s+bs) => i in [s-k, s+bs-k-1]
    #   i not in [s, s+bs): i < s OR i >= s+bs
    #   i in [s-k, s-1] (i < s part): i in [max(0, s-k), min(s-1, N-1-k)]
    #   i in [s+bs-k, s+bs-1] intersect i >= s+bs: empty (s+bs-k < s+bs for k>=1)
    
    # So:
    # Part 1 i-range: [max(s, s+bs-k), min(s+bs-1, N-1-k)]
    # Part 2 i-range: [max(0, s-k), min(s-1, N-1-k)]
    
    # This is getting complex. Let's just do a direct O(bs + k) computation per affected lag,
    # but only for lags that are actually affected.
    
    # A lag k is affected if the block [s, s+bs) intersects the "interaction window" [0, N-1-k].
    # Specifically, if s < N-k and s+bs > 0.
    
    # For simplicity and correctness, let's compute the new R values for all k in [1, N-1]
    # that could be affected, using the fact that only O(bs + N) lags are affected.
    # Actually, for a block of size bs, the affected lags are k in [1, N-1] where
    # the block [s, s+bs) is not entirely outside [0, N-1-k].
    # This is always true for k < N - s + bs or similar. Let's just compute all k=1..N-1
    # but use the efficient formula.
    
    # Alternative: just recompute R for the affected lags directly.
    # The number of affected lags is O(N) in worst case, but for small bs, it's O(bs + N).
    # Given N=400, even O(N^2) per move is 160000 ops, which is fast enough.
    # Let's just recompute R for all lags using the flip, in O(N*bs) time.
    
    # Actually, the simplest correct approach: create the flipped signs, compute R via FFT.
    # But that's O(N log N) per move, which is what we're trying to avoid.
    
    # Let's use the direct formula. For each k in 1..N-1:
    # delta_R[k] = -2 * (sum over Part1 + sum over Part2) of f[i]*f[i+k]
    
    # We can precompute prefix sums of f[i]*f[i+k] for each k, but that's O(N^2) storage.
    # Instead, for each k, compute the sum over the two ranges in O(range_length).
    
    # Total work: sum over k of (length of Part1 range + length of Part2 range)
    # Part1 range length: min(s+bs-1, N-1-k) - max(s, s+bs-k) + 1 (if valid)
    # Part2 range length: min(s-1, N-1-k) - max(0, s-k) + 1 (if valid)
    
    # For k < bs: Part1 range is [s, s+bs-k-1] (length bs-k), Part2 is [s-k, s-1] (length k)
    # For k >= bs: Part1 range is empty (s+bs-k <= s), Part2 is [s-k, s-1] (length k, capped)
    # Wait, for k >= bs: s+bs-k <= s, so Part1 is empty.
    # Part2: [max(0, s-k), min(s-1, N-1-k)]
    
    # So total work is O(N * min(bs, N)) = O(N*bs) for k < bs, and O(N*k) for k >= bs... 
    # No, for k >= bs, Part2 has length min(k, s) at most.
    # Total: sum_{k=1}^{min(bs,N-1)} (bs-k + k) + sum_{k=bs}^{N-1} k
    # = sum_{k=1}^{bs} bs + sum_{k=bs}^{N-1} k
    # = bs^2 + (N-1)(N)/2 - bs(bs-1)/2
    # = O(N^2) in worst case.
    
    # This is too slow. Let's just use FFT-based recomputation but with a trick:
    # We can compute the change in the objective without full FFT by noting that
    # the objective is a function of R, and R changes in a structured way.
    
    # Given the complexity, let's just use the FFT method but optimize it:
    # Create flipped array, compute R via FFT. This is O(N log N) ≈ 400*9 = 3600 ops.
    # With 5000 iterations, that's 18M ops, which is fast in numpy.
    
    new_signs = signs.copy()
    new_signs[s:s+bs] = -new_signs[s:s+bs]
    new_R = _compute_R(new_signs)
    new_obj = _objective_from_R(new_R)
    return new_obj, new_signs, new_R

def _simulated_annealing(signs, start_temp=2.0, end_temp=0.001, n_iter=8000, time_limit=20.0):
    """
    Simulated annealing with adaptive block-flip moves for the spin-glass Ising model.
    The energy is the L1 norm of the autocorrelation.
    """
    rng = np.random.RandomState(0)
    current = signs.astype(np.float64).copy()
    current_R = _compute_R(current)
    current_obj = _objective_from_R(current_R)
    best = current.copy()
    best_R = current_R.copy()
    best_obj = current_obj
    
    # Adaptive block size: start small, grow when stuck
    block_size = 2
    stuck_count = 0
    max_block = 64
    
    start_time = time.time()
    
    for i in range(n_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Cooling schedule: exponential
        temp = start_temp * (end_temp / start_temp) ** (i / max(n_iter - 1, 1))
        
        # Choose block size: adaptive
        # Use a mixture of small and large blocks
        r = rng.random()
        if r < 0.4:
            bs = rng.randint(1, 4)  # small
        elif r < 0.7:
            bs = rng.randint(4, 16)  # medium
        elif r < 0.9:
            bs = rng.randint(16, 32)  # large
        else:
            bs = rng.randint(32, max_block + 1)  # very large
        
        if bs > N:
            bs = N
        
        s = rng.randint(0, N - bs + 1)
        
        new_obj, new_signs, new_R = _block_flip_delta(current, s, bs, current_R)
        
        delta = new_obj - current_obj
        if delta < 0 or (temp > 1e-12 and rng.random() < np.exp(-delta / temp)):
            current = new_signs
            current_R = new_R
            current_obj = new_obj
            stuck_count = 0
            if current_obj < best_obj:
                best_obj = current_obj
                best = current.copy()
                best_R = current_R.copy()
        else:
            stuck_count += 1
    
    return best, best_obj

def _random_starts(num_starts, rng):
    """Generate diverse random initial sign patterns."""
    starts = []
    for i in range(num_starts):
        if i == 0:
            # All ones
            s = np.ones(N, dtype=np.float64)
        elif i == 1:
            # Alternating
            s = np.ones(N, dtype=np.float64)
            s[1::2] = -1.0
        elif i == 2:
            # Random
            s = rng.choice([1.0, -1.0], size=N)
        elif i == 3:
            # Low-frequency pattern (block structure)
            s = np.ones(N, dtype=np.float64)
            block_len = rng.randint(10, 50)
            for j in range(0, N, block_len):
                if rng.random() < 0.5:
                    s[j:j+block_len] = -1.0
        else:
            # Random with varying block structure
            s = np.ones(N, dtype=np.float64)
            pos = 0
            while pos < N:
                blen = rng.randint(5, 40)
                if rng.random() < 0.5:
                    s[pos:pos+blen] = -1.0
                pos += blen
        starts.append(s)
    return starts

def _local_refinement(signs, time_limit=10.0):
    """
    Local refinement: try single-bit flips and small block flips greedily.
    """
    rng = np.random.RandomState(777)
    current = signs.astype(np.float64).copy()
    current_R = _compute_R(current)
    current_obj = _objective_from_R(current_R)
    best = current.copy()
    best_obj = current_obj
    
    start_time = time.time()
    
    # Greedy single-bit flips
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        # Try all single-bit flips
        for i in range(N):
            if time.time() - start_time > time_limit:
                break
            current[i] = -current[i]
            new_R = _compute_R(current)
            new_obj = _objective_from_R(new_R)
            if new_obj < current_obj:
                current_obj = new_obj
                current_R = new_R
                improved = True
                if current_obj < best_obj:
                    best_obj = current_obj
                    best = current.copy()
            else:
                current[i] = -current[i]
    
    # Small block flips (size 2-4)
    for bs in [2, 3, 4]:
        if time.time() - start_time > time_limit:
            break
        for _ in range(200):
            if time.time() - start_time > time_limit:
                break
            s = rng.randint(0, N - bs + 1)
            current[s:s+bs] = -current[s:s+bs]
            new_R = _compute_R(current)
            new_obj = _objective_from_R(new_R)
            if new_obj < current_obj:
                current_obj = new_obj
                current_R = new_R
                if current_obj < best_obj:
                    best_obj = current_obj
                    best = current.copy()
            else:
                current[s:s+bs] = -current[s:s+bs]
    
    return best, best_obj

def construct():
    start_time = time.time()
    rng = np.random.RandomState(42)
    
    # Generate diverse initial patterns
    starts = _random_starts(6, rng)
    
    best_pattern = None
    best_score = float('inf')
    
    # Phase 1: Simulated annealing from multiple starts
    # Allocate time: 60s for SA, 20s for local refinement, 10s safety
    sa_time_budget = 60.0
    num_sa_runs = 4
    time_per_sa = sa_time_budget / num_sa
    
    for i in range(num_sa_runs):
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed - 30.0  # reserve 30s for refinement + safety
        if remaining < 5.0:
            break
        
        t_limit = min(time_per_sa, remaining)
        if t_limit < 2.0:
            break
        
        optimized, opt_score = _simulated_annealing(
            starts[i], 
            start_temp=2.0, 
            end_temp=0.001, 
            n_iter=10000, 
            time_limit=t_limit
        )
        
        if opt_score < best_score:
            best_score = opt_score
            best_pattern = optimized
    
    # Also try a couple more random starts if time permits
    for i in range(2):
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed - 30.0
        if remaining < 10.0:
            break
        
        rand_start = rng.choice([1.0, -1.0], size=N)
        t_limit = min(15.0, remaining)
        if t_limit < 3.0:
            break
        
        optimized, opt_score = _simulated_annealing(
            rand_start,
            start_temp=3.0,
            end_temp=0.001,
            n_iter=10000,
            time_limit=t_limit
        )
        
        if opt_score < best_score:
            best_score = opt_score
            best_pattern = optimized
    
    # Phase 2: Local refinement
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 15.0:
        refined, refined_score = _local_refinement(best_pattern, time_limit=min(20.0, DEADLINE - elapsed - 10.0))
        if refined_score < best_score:
            best_score = refined_score
            best_pattern = refined
    
    # Final safety check
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 5.0:
        # Return immediately
        result = np.sign(best_pattern).astype(int)
        if np.sum(result) == 0:
            result[0] = -result[0]
        return result
    
    result = np.sign(best_pattern).astype(int)
    
    # Constraint: heights must not sum to zero
    if np.sum(result) == 0:
        result[0] = -result[0]
    
    return result
