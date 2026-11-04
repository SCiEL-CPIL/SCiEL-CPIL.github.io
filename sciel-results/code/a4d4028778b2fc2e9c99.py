import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin before 120s limit
    
    def compute_autocorr(f):
        """Compute aperiodic autocorrelation via FFT for speed."""
        # Zero-pad to at least 2N-1, use next power of 2 for FFT efficiency
        n_fft = 1
        while n_fft < 2 * N - 1:
            n_fft *= 2
        
        F = np.fft.rfft(f, n=n_fft)
        # Autocorrelation is inverse FFT of |F|^2
        # conv(f, f[::-1]) corresponds to circular autocorrelation if padded properly
        # Actually, for aperiodic autocorrelation R[k] = sum f[i]f[i+k]:
        # We can use: ifft(fft(f) * conj(fft(f))) gives circular autocorrelation
        # But we want linear. Let's just use the convolution approach with FFT.
        # conv(f, f_rev) where f_rev = f[::-1]
        # This gives R[-k] ... R[0] ... R[k]
        
        f_rev = f[::-1]
        F_rev = np.fft.rfft(f_rev, n=n_fft)
        
        # Linear convolution via FFT
        conv = np.fft.irfft(F * F_rev, n=n_fft)
        # The linear convolution is in the first 2N-1 elements
        R_full = conv[:2*N-1]
        
        # R_full[i] corresponds to lag i - (N-1)
        # We want to exclude lag 0 (index N-1)
        # Return the array with lag 0 set to 0 for easy max computation
        R = R_full.copy()
        R[N-1] = 0.0
        return R

    def max_abs_autocorr(f):
        R = compute_autocorr(f)
        return np.max(np.abs(R))

    # Initialize with random sequence
    rng = np.random.RandomState(42)
    
    # Try a few initial seeds to get a good starting point
    best_f = None
    best_val = float('inf')
    
    for seed in [42, 123, 456, 789, 1000]:
        if time.time() > deadline - 10:
            break
        f = rng.choice([-1.0, 1.0], size=N)
        val = max_abs_autocorr(f)
        if val < best_val:
            best_val = val
            best_f = f.copy()
    
    # If we didn't find any (shouldn't happen), just use random
    if best_f is None:
        best_f = rng.choice([-1.0, 1.0], size=N)
        best_val = max_abs_autocorr(best_f)
    
    # Simulated Annealing / Local Search
    # We use a greedy flip approach first, then simulated annealing to escape local minima
    
    current_f = best_f.copy()
    current_val = best_val
    current_R = compute_autocorr(current_f)
    
    # Phase 1: Greedy local search
    # Find the lag with max absolute autocorrelation
    # Try flipping bits to reduce the peak
    
    # Precompute which pairs contribute to which lags? 
    # Actually, for N=400, computing full autocorr is O(N log N) ~ 400*9 = 3600 ops.
    # Flipping a bit changes O(N) autocorr values.
    # We can update R incrementally.
    
    # Let's implement a fast local search using incremental updates.
    # R[lag] = sum_{i} f[i] f[i+lag]
    # If we flip f[k], then:
    # For all lags L such that i = k-L or i+L = k, the term changes.
    # Specifically, R[L] changes by:
    # -2 * f[k] * f[k+L] (if k+L < N)
    # -2 * f[k] * f[k-L] (if k-L >= 0)
    
    # This is efficient.
    
    # Let's use a priority queue or just iterate.
    # Since N=400, iterating over all k is fast.
    
    # We'll use a simple loop.
    
    # To speed up, we can track the current max and only check flips that might improve it.
    # But for simplicity and robustness, let's just do a few passes of greedy improvement.
    
    # Actually, a better approach for this specific problem (minimizing max autocorr)
    # is to use the "flip the bit that most reduces the peak" heuristic.
    
    # Let's implement a proper local search with incremental updates.
    
    # We'll maintain R array.
    # When flipping index k:
    # For each lag L in range(1, N):
    #   if k+L < N: R[L] -= 2 * f[k] * f[k+L]
    #   if k-L >= 0: R[L] -= 2 * f[k] * f[k-L]
    #   (and R[-L] changes similarly, but R is symmetric in abs, so we can just update R for L>0 and mirror)
    
    # Wait, R is symmetric: R[L] = R[-L]. So we only need to store L=1..N-1.
    # Let's redefine R to be length N, where R[0] is unused (lag 0), R[L] is autocorr at lag L.
    
    def compute_R_positive(f):
        """Compute R[L] for L=1..N-1."""
        R = np.zeros(N)
        for L in range(1, N):
            R[L] = np.dot(f[:N-L], f[L:])
        return R
    
    # This is O(N^2) = 160,000 ops. Fast enough for initialization.
    # For updates, we can do it incrementally.
    
    # Let's switch to this representation for speed in the search loop.
    
    current_R = compute_R_positive(current_f)
    current_max = np.max(np.abs(current_R[1:]))
    
    # Greedy Search
    # We want to find a flip that reduces the max absolute value.
    
    # Optimization: Only consider flips that affect the current peak lags.
    # But it's safer to check all flips. N=400 is small.
    
    # Let's run a loop of local search steps.
    
    max_iterations = 5000
    iteration = 0
    
    while time.time() < deadline - 5 and iteration < max_iterations:
        iteration += 1
        
        # Find the lag with the maximum absolute autocorrelation
        L_max = np.argmax(np.abs(current_R[1:])) + 1
        R_max = current_R[L_max]
        
        if np.abs(R_max) < 1e-9:
            break
            
        # Try flipping each bit to see if it reduces the global max
        best_improvement = 0
        best_flip = -1
        
        # We can prune: if a flip doesn't reduce the current peak, it can't improve the global max
        # unless it reduces another peak that is equal to the current max.
        # But for simplicity, let's just check all flips.
        
        # To make it faster, we can estimate the new max without full update.
        # But full update is O(N) per flip, so 400*400 = 160k ops per iteration.
        # 5000 iterations = 800M ops. Might be too slow in Python.
        
        # Let's optimize: only check flips that are "likely" to help.
        # Or, use a smaller step size.
        
        # Alternative: Use the fact that we only need to check if the new max is lower.
        # We can compute the change in R for each flip in O(N).
        
        # Let's just do it efficiently in numpy.
        
        # For each k, compute the change in R.
        # This is hard to vectorize easily.
        
        # Let's just do a simple loop but break early if we find a good flip.
        
        # Actually, let's just do a random walk with acceptance criteria (Simulated Annealing).
        # It's simpler and often works well.
        
        # Pick a random bit to flip
        k = rng.randint(N)
        
        # Compute change in R
        # R[L] changes by -2 * f[k] * f[k+L] (if k+L < N)
        # and -2 * f[k] * f[k-L] (if k-L >= 0)
        
        # Let's compute the new R for this flip
        old_f_k = current_f[k]
        current_f[k] *= -1
        
        # Update R
        # This is O(N)
        for L in range(1, N):
            delta = 0
            if k + L < N:
                delta -= 2 * old_f_k * current_f[k+L]  # Note: current_f[k] is now flipped
                # Wait, the term is f[k] f[k+L]. Old was old_f_k * f[k+L]. New is -old_f_k * f[k+L].
                # Change is -2 * old_f_k * f[k+L].
                # But we already flipped current_f[k], so we can't use it directly if we want to be careful.
                # Let's just recompute the contribution.
                pass
        
        # This is getting complicated. Let's just recompute R from scratch for simplicity,
        # but only if we find a better solution.
        
        # Actually, recomputing R is O(N^2) = 160k ops.
        # If we do this for every flip, it's too slow.
        
        # Let's go back to the greedy approach but with a limit on checks per iteration.
        
        # Check only a subset of flips?
        # Or, use the FFT-based autocorr which is O(N log N).
        
        # Let's use the FFT-based method for the full autocorr and just do a simple local search.
        
        # Flip a random bit
        k = rng.randint(N)
        current_f[k] *= -1
        
        # Compute new max
        new_val = max_abs_autocorr(current_f)
        
        # Accept if better
        if new_val < current_val:
            current_val = new_val
            current_R = compute_autocorr(current_f)
            if current_val < best_val:
                best_val = current_val
                best_f = current_f.copy()
        else:
            # Revert
            current_f[k] *= -1
        
    # Final check
    if best_val > current_val:
        best_f = current_f.copy()
        best_val = current_val
    
    return best_f
