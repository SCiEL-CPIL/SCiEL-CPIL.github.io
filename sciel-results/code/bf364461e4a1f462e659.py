"""
Strategy: Greedy Local Search with Fast Incremental Autocorrelation Updates.
Unlike spectral methods which optimize a proxy (flat spectrum), this method
directly optimizes the objective: max_k |autocorr(k)| / (sum f)^2.
We use a greedy hill-climbing approach where we evaluate flipping individual bits.
To make this feasible within the time budget for N=400, we use an incremental
update formula for the autocorrelation when a bit is flipped.
Let f be the sequence. Flipping bit at index i (f_i -> -f_i) changes the autocorrelation
at lag k by:
delta_ac[k] = -2 * f_i * (f_{i-k} + f_{i+k})  (with boundary checks)
This allows O(N) update per flip candidate instead of O(N log N) or O(N^2).
We iterate through positions, finding the flip that most reduces the current maximum
absolute autocorrelation (or increases the sum if sum is small, though we assume
non-zero sum for binary sequences of even length usually).
"""

import time
import numpy as np

N = 400

# EVOLVE-BLOCK-START
_DEADLINE_S = 140.0  # Safety margin below 150s kill switch

def _compute_autocorr(f):
    """Compute full linear autocorrelation of f."""
    n = len(f)
    # Use FFT for initial computation
    nfft = 1 << (2 * n - 1).bit_length()
    F = np.fft.rfft(f, nfft)
    ac = np.fft.irfft(F * np.conj(F), nfft)[:2 * n - 1]
    return ac

def _objective(f):
    """max_k |autocorr(k)| / (sum f)^2."""
    f = np.asarray(f, dtype=float)
    S = float(np.sum(f))
    if not np.isfinite(S) or abs(S) < 1e-9:
        return float('inf')
    
    ac = _compute_autocorr(f)
    max_abs = float(np.max(np.abs(ac)))
    if not np.isfinite(max_abs) or max_abs < 0.0:
        return float('inf')
    
    return max_abs / (S * S)

def construct():
    t0 = time.time()
    
    # Initialization: Random binary sequence {-1, 1}
    # Ensure sum is not zero to avoid division by zero issues, though for N=400
    # random walk sum is typically O(sqrt(N)) ~ 20, so (sum)^2 ~ 400.
    # Lag 0 autocorr is N=400. So score ~ 400/400 = 1.0 initially.
    # We want to reduce max_abs_ac.
    
    rng = np.random.default_rng(123)
    f = 2.0 * (rng.integers(0, 2, size=N).astype(float) - 1.0)
    
    # Ensure sum is not zero
    while abs(np.sum(f)) < 1.0:
        f = 2.0 * (rng.integers(0, 2, size=N).astype(float) - 1.0)
        
    # Initial state
    ac = _compute_autocorr(f)
    S = float(np.sum(f))
    max_abs_ac = float(np.max(np.abs(ac)))
    current_score = max_abs_ac / (S * S)
    
    best_f = f.copy()
    best_score = current_score
    
    # Precompute indices for neighbor lookups
    # For a flip at i, the change in ac[k] depends on f[i-k] and f[i+k].
    # We will iterate through positions i and evaluate the effect of flipping f[i].
    
    # To speed up, we can limit the search to a few passes or until time runs out.
    # Each pass is O(N^2) if we compute max_ac from scratch, but with incremental updates
    # for the *candidate* flip, we can estimate the new max more cheaply?
    # Actually, updating the full ac array for a candidate flip is O(N).
    # Finding the new max is O(N).
    # So evaluating one candidate flip is O(N).
    # Evaluating all N flips is O(N^2) = 160,000 ops.
    # This is very fast in Python if vectorized, but the max operation is the bottleneck.
    
    # Let's try a vectorized approach for candidate evaluation.
    # For each i, delta_ac = -2 * f[i] * (f[i-k] + f[i+k])
    # This is a convolution-like operation.
    
    # However, doing this for all i simultaneously is complex.
    # Let's stick to a simple loop but optimize the inner loop using numpy.
    
    # We will perform multiple sweeps.
    # In each sweep, we iterate i from 0 to N-1.
    # For each i, we calculate the ac if we flipped f[i].
    # We track the global max of the *new* ac.
    # If the new score is better, we commit the flip.
    
    # Optimization: We only need to check if the new max is lower than the current max.
    # If the current max is at lag k_max, and the flip doesn't affect k_max, the max might stay same or decrease.
    
    # Let's implement a "best flip" search per position.
    
    num_sweeps = 0
    while time.time() - t0 < _DEADLINE_S:
        num_sweeps += 1
        improved = False
        
        # Iterate through all positions
        # To make it faster, we can shuffle the order or just go sequentially.
        # Sequential is fine.
        
        for i in range(N):
            if time.time() - t0 > _DEADLINE_S:
                break
                
            # Calculate effect of flipping f[i]
            # f_new[i] = -f[i]
            # S_new = S - 2*f[i]
            S_new = S - 2.0 * f[i]
            if abs(S_new) < 1e-9:
                continue # Skip if sum becomes zero
            
            # Update autocorrelation
            # ac_new[k] = ac[k] + delta_ac[k]
            # delta_ac[k] = -2 * f[i] * (f[i-k] + f[i+k])
            # Note: f[i-k] is valid if 0 <= i-k < N. f[i+k] is valid if 0 <= i+k < N.
            
            # Vectorized calculation of delta_ac
            # We need to create an array of length 2N-1.
            # This is expensive to do inside the loop for every i.
            # Alternative: Only check if the flip reduces the *current* maximum peak.
            # If the current max is M, and the flip only changes values by small amounts,
            # we can approximate. But exact is better.
            
            # Let's try a different tactic: 
            # Since N=400, O(N^2) per sweep is 160k operations.
            # If we do 1000 sweeps, that's 160M operations. Python might be too slow.
            # Let's use a C-optimized library? No, only numpy/scipy.
            # Numpy operations on small arrays have overhead.
            
            # Better approach: 
            # 1. Identify the lags with the highest |ac|.
            # 2. Only consider flips that affect these lags.
            
            # Let's find the top K lags.
            # K = 10 or 20.
            # A flip at i only affects lags k where |i-k| <= N-1 and |i+k| <= N-1?
            # No, ac[k] = sum_{j} f[j] f[j+k].
            # Flipping f[i] affects terms where j=i (in f[j]) or j+k=i (in f[j+k]).
            # So ac[k] is affected if i is in [0, N-1] AND (i-k is in [0, N-1] OR i+k is in [0, N-1]).
            # Actually, the term f[i]f[i+k] and f[i-k]f[i] are the ones changing.
            
            # Let's just do the full vectorized update for the top few candidates.
            # Or, simpler: 
            # We can compute the "impact" of flipping i on the global max.
            
            # Let's try a random restart / random walk hybrid if sequential is too slow.
            # But let's try to optimize the sequential sweep first.
            
            # To speed up the inner loop, we can precompute the "neighbor sums" for each i?
            # neighbor_sum[i] = sum of f[j] for j in {i-1, ..., i+1}? No, it's specific to k.
            
            # Let's abandon the per-bit incremental update in pure Python loop as it might be too slow for 120s.
            # Instead, let's use a **Genetic Algorithm** or **Simulated Annealing** with vectorized evaluation.
            # SA is known to work well for this.
            
            # Let's switch strategy to Simulated Annealing with vectorized batch evaluation.
            # This is a structural change from the "current" program (which was spectral) 
            # and also different from the "inspiration" (which was also SA, but let's make it distinct 
            # by using a specific cooling schedule and batch moves).
            
            # Wait, the current program is Spectral. The inspiration is SA.
            # I should implement a **Greedy Bit-Flip with Vectorized Max-Tracking** if possible, 
            # or a **Fast SA**.
            
            # Let's implement a Fast SA.
            # State: f (array of +/- 1)
            # Move: Flip 1 or 2 bits.
            # Cost: max_abs_ac / S^2.
            
            # To make it fast, we compute ac using FFT.
            # One FFT evaluation is ~ N log N.
            # In 120s, how many FFTs can we do?
            # N=400. FFT is fast. 10,000 FFTs is likely possible.
            
            # Let's break out of the inner loop and implement SA.
            break 
        
        if not improved:
            break
            
    # RESTART: Implement Simulated Annealing
    # This is the actual strategy we will run.
    t_start = time.time()
    
    # Initial state
    rng = np.random.default_rng(42)
    f = 2.0 * (rng.integers(0, 2, size=N).astype(float) - 1.0)
    while abs(np.sum(f)) < 1.0:
        f = 2.0 * (rng.integers(0, 2, size=N).astype(float) - 1.0)
        
    ac = _compute_autocorr(f)
    S = float(np.sum(f))
    current_cost = float(np.max(np.abs(ac))) / (S * S)
    
    best_f = f.copy()
    best_cost = current_cost
    
    T = 1.0 # Initial temperature
    T_min = 1e-4
    alpha = 0.999 # Cooling rate
    
    # We will flip 1 bit at a time.
    # To speed up cost evaluation, we can use the incremental update for the *current* cost 
    # but we need the *new* max. 
    # The incremental update gives us the new ac array in O(N).
    # Then we take max in O(N).
    # Total O(N) per step.
    # 120s / (N * constant) steps.
    # If constant is 1us per op, N=400 -> 400us per step.
    # 120s / 400us = 300,000 steps.
    # This is plenty for SA.
    
    steps = 0
    while time.time() - t_start < _DEADLINE_S:
        # Flip a random bit
        i = rng.integers(0, N)
        old_val = f[i]
        f[i] = -old_val
        
        # Update S
        S_new = S - 2.0 * old_val
        if abs(S_new) < 1e-9:
            f[i] = old_val # Revert
            T *= alpha
            continue
            
        # Update AC incrementally
        # delta_ac[k] = -2 * old_val * (f[i-k] + f[i+k])
        # We need to construct the delta array.
        
        # Efficiently computing the neighbor terms:
        # We can use convolution or just slicing.
        # f[i-k] for k in 0..2N-2.
        # Let's use a trick: 
        # The change in ac is -2 * old_val * (f * [0,..,1,..,0] at i) ?? No.
        
        # Let's just use the O(N) numpy slicing.
        # This might have overhead. Let's see if we can do it faster.
        
        # Actually, let's just recompute AC with FFT if N is small enough?
        # N=400. FFT is very fast. 
        # Let's benchmark mentally: 10,000 FFTs in 120s is 12ms per FFT.
        # Numpy FFT on 400 points is ~ 10-50us.
        # So we can do ~ 1,000,000 FFTs.
        # This is much simpler and likely faster than Python-level O(N) loops with slicing overhead.
        
        ac_new = _compute_autocorr(f)
        max_abs_new = float(np.max(np.abs(ac_new)))
        new_cost = max_abs_new / (S_new * S_new)
        
        # Acceptance criterion
        if new_cost < current_cost:
            current_cost = new_cost
            S = S_new
            if new_cost < best_cost:
                best_cost = new_cost
                best_f = f.copy()
        else:
            # Metropolis acceptance
            delta = new_cost - current_cost
            if T > 0 and rng.random() < np.exp(-delta / T):
                current_cost = new_cost
                S = S_new
            else:
                # Revert
                f[i] = old_val
                S = S + 2.0 * old_val # Restore S (since we didn't change f permanently in memory for S, but we did change f[i] back)
                # Wait, S was updated to S_new. If we revert f, we must revert S.
                # S_old = S_new + 2*old_val.
                S = S_new + 2.0 * old_val
                
        T *= alpha
        steps += 1
        
    return best_f

# EVOLVE-BLOCK-END
