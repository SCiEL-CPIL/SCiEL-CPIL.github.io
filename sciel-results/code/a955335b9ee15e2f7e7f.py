import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10 second margin
    
    # Initialize with a random sign sequence (zero mean)
    # Using a fixed seed for reproducibility in case the evaluator runs multiple times,
    # but random is generally fine. Let's use a deterministic but non-trivial start:
    # A simple alternating pattern is a good start, but random is better for breaking symmetry.
    # We'll use a pseudo-random sequence based on a simple LCG or just np.random with a seed.
    # Since we want to beat the seed (1.0), any zero-mean sequence should do.
    # We will optimize it.
    
    # Start with a random +/- 1 vector
    np.random.seed(42)
    h = np.random.choice([-1, 1], size=N)
    
    # Calculate initial autocorrelation
    # Linear autocorrelation: h * h
    # We can use FFT for speed, but N=400 is small. 
    # However, for the update step, we need the full vector A.
    # A[k] = sum(h[i] * h[i+k])
    
    def calc_autocorr(h):
        # Linear autocorrelation
        # Using np.correlate
        return np.correlate(h, h, mode='full')
    
    A = calc_autocorr(h)
    
    # The metric is likely related to the magnitude of the autocorrelation.
    # Common metrics: 
    # 1. Peak sidelobe: max(|A[k]| for k != 0)
    # 2. Integrated Sidelobe: sum(|A[k]| for k != 0)
    # 3. Total Energy: sum(|A[k]|)
    # 
    # The "upper bound on C3" suggests a global bound. 
    # Let's minimize the sum of absolute values of the autocorrelation (L1 norm).
    # This is robust.
    
    def calc_score(A):
        # Exclude the main lobe? The main lobe is A[N-1] (center).
        # For a zero-mean signal, the main lobe is still large (variance).
        # If the bound is on the *tail* or the *whole* thing, we minimize the whole thing.
        # Since h is +/- 1, sum(h^2) = N. The main lobe is N.
        # The sidelobes are smaller.
        # Minimizing the total L1 norm is a good proxy.
        return np.sum(np.abs(A))
    
    current_score = calc_score(A)
    best_h = h.copy()
    best_score = current_score
    
    # Hill Climbing / Simulated Annealing
    # We flip one bit at a time.
    # Change in A when flipping h[i]:
    # A_new[k] = A_old[k] - 2 * h[i] * h[i+k] (for valid i+k)
    # Note: np.correlate(h, h, 'full') has length 2N-1.
    # Index in full: j. h[j] * h[j+N-1-i] ... this is tricky to map directly to "shift k".
    
    # Let's use a simpler, though slightly slower, but safer approach for N=400:
    # Recalculate A after each flip? 
    # np.correlate for N=400 is very fast in C.
    # Let's benchmark mentally: 100,000 iterations * (correlate cost).
    # Correlate cost ~ N^2 = 160,000 ops. 100k * 160k = 16e9 ops. Too slow in Python loop.
    # We MUST use the incremental update or FFT.
    
    # Let's use the incremental update on the linear autocorrelation vector A.
    # A has length 2N-1.
    # A[m] = sum_{j} h[j] h[j + (m - (N-1))]
    # Let k = m - (N-1) be the lag. k ranges from -(N-1) to N-1.
    # A[m] = sum_{j} h[j] h[j+k]
    
    # If we flip h[i] to -h[i]:
    # Terms involving h[i] are:
    # 1. j = i: h[i] h[i+k] becomes -h[i] h[i+k]. Change: -2 h[i] h[i+k].
    # 2. j+k = i => j = i-k: h[i-k] h[i] becomes -h[i-k] h[i]. Change: -2 h[i-k] h[i].
    
    # So for each lag k, A[k] changes by:
    # delta_A[k] = -2 * ( h[i] * h[i+k] + h[i-k] * h[i] )
    # Note: h[i] * h[i+k] is the term where the first index is i.
    # h[i-k] * h[i] is the term where the second index is i.
    
    # We need to update A for all valid k.
    # k ranges from -(N-1) to N-1.
    # h[i+k] is valid if 0 <= i+k < N.
    # h[i-k] is valid if 0 <= i-k < N.
    
    def update_A(A, h, i):
        """
        Updates A in-place assuming h[i] has been flipped.
        Actually, let's compute the delta and apply it.
        """
        old_h_i = h[i]
        # The change in A for lag k is -2 * old_h_i * (h[i+k] + h[i-k])
        # Wait, h[i] is flipped, so new_h_i = -old_h_i.
        # New term 1: (-old_h_i) * h[i+k] = -old_h_i * h[i+k]
        # Old term 1: old_h_i * h[i+k]
        # Diff 1: -2 * old_h_i * h[i+k]
        
        # New term 2: h[i-k] * (-old_h_i) = -h[i-k] * old_h_i
        # Old term 2: h[i-k] * old_h_i
        # Diff 2: -2 * h[i-k] * old_h_i
        
        # Total diff for lag k: -2 * old_h_i * (h[i+k] + h[i-k])
        
        # Valid k range:
        # i+k in [0, N-1] => k in [-i, N-1-i]
        # i-k in [0, N-1] => k in [i-(N-1), i]
        
        # We iterate over all k and check bounds.
        # Since N=400, this loop is 400 iterations.
        # We do this for each flip.
        
        # To make it faster, we can precompute the valid range.
        # But 400 iterations is tiny.
        
        # Let's just loop k from -(N-1) to N-1
        # A index m = k + N - 1
        
        for k in range(-(N-1), N):
            term1 = 0.0
            term2 = 0.0
            
            idx1 = i + k
            if 0 <= idx1 < N:
                term1 = h[idx1]
                
            idx2 = i - k
            if 0 <= idx2 < N:
                term2 = h[idx2]
                
            delta = -2.0 * old_h_i * (term1 + term2)
            A[k + N - 1] += delta

    # Optimization: 
    # We can vectorize the update if we use FFT, but the logic is complex.
    # For N=400, the Python loop for update is ~400 steps.
    # 10,000 flips * 400 steps = 4,000,000 steps. Very fast.
    # We can do ~1,000,000 flips in 100 seconds?
    # 1M flips * (400 inner loop + overhead) ~ 400M ops. 
    # Python is ~10M ops/sec. So 400M ops is 40 seconds.
    # This is feasible.
    
    # Let's implement the loop.
    
    steps = 0
    while time.time() < deadline:
        # Pick a random index to flip
        i = np.random.randint(0, N)
        
        # Calculate new score efficiently?
        # We need to update A, then calc score.
        # Score is sum(abs(A)).
        # We can track the score incrementally too!
        
        # Old A values at affected indices?
        # The delta affects all k. So all A[m] change.
        # So we must recalculate sum(abs(A)) from scratch?
        # Or track the change in sum(abs(A)).
        
        # Let A_old be the current A.
        # We compute delta_A for all k.
        # New A = A_old + delta_A.
        # New Score = sum(|A_old + delta_A|).
        # Old Score = sum(|A_old|).
        
        # This requires computing the new A or at least the new absolute values.
        # Since delta_A is non-zero for all k (mostly), we have to touch all elements.
        
        # Let's just update A and recalculate score.
        
        # Save old A to rollback if rejected?
        # Or compute delta, apply, check, rollback.
        
        # Optimization:
        # Compute delta_A vector.
        # delta_A[k+N-1] = -2 * h[i] * (h[i+k] + h[i-k])
        
        # We can construct delta_A using vectorized numpy operations.
        
        # h_i = h[i]
        # valid_k range for i+k: k in [-i, N-1-i]
        # valid_k range for i-k: k in [i-N+1, i]
        
        # Let's create a zero array for delta_A
        # delta_A = np.zeros(2*N-1)
        
        # This allocation is expensive.
        # Let's stick to the loop for delta calculation, it's simple.
        
        # Actually, let's use a different approach:
        # Just recalculate A from scratch using FFT?
        # FFT of 400 is fast.
        # np.fft.fft(h) -> O(N log N).
        # Power spectrum -> O(N).
        # Inverse FFT -> O(N log N).
        # Total ~ 400 * 9 = 3600 ops.
        # Python overhead for numpy calls is ~10us.
        # 100,000 iterations * 10us = 1 second.
        # 1,000,000 iterations * 10us = 10 seconds.
        # This is MUCH faster than the O(N^2) correlate or O(N) python loop update.
        
        # Let's use FFT to compute autocorrelation.
        
        def autocorr_fft(h):
            # Zero pad to next power of 2 or 2N to avoid circular wrap
            n_fft = 1 << (2*N - 1).bit_length()
            f = np.fft.fft(h, n_fft)
            power = f * np.conj(f)
            a = np.fft.ifft(power).real
            # The autocorrelation is in a[0:N-1] and a[n_fft-N+1:n_fft]
            # Standard linear autocorrelation:
            # a[0] is lag 0
            # a[1] is lag 1
            # ...
            # a[N-1] is lag N-1
            # a[N] is lag -N+1 (wrap around)
            # We want the linear autocorrelation vector of length 2N-1.
            # It is a[0:N-1] concatenated with a[n_fft-N+1:n_fft]
            # Wait, ifft gives circular correlation.
            # For zero-padded input, the circular correlation is the linear correlation.
            # The order is: lag 0, 1, ..., N-1, -(N-1), ..., -1
            # So we take a[0:N-1] and a[n_fft-N+1:n_fft]
            # Let's verify.
            # If h = [1, 0], n_fft=4.
            # f = [1, 1, 1, 1]. power = [1,1,1,1]. ifft = [1,0,0,0].
            # a[0:1] = [1]. a[4-2+1:4] = a[3:4] = [0].
            # Result [1, 0]. Correct (lag 0: 1, lag -1: 0).
            
            return np.concatenate([a[0:N-1], a[n_fft-N+1:n_fft]])
        
        # This function allocates memory. 
        # 1M calls might be slow due to allocation.
        # But N=400 is small.
        
        # Let's try this FFT approach.
        
        # Reset A to FFT based
        A = autocorr_fft(h)
        current_score = np.sum(np.abs(A))
        
        while time.time() < deadline:
            i = np.random.randint(0, N)
            h[i] = -h[i]
            
            A_new = autocorr_fft(h)
            new_score = np.sum(np.abs(A_new))
            
            if new_score < current_score:
                current_score = new_score
                A = A_new
                if current_score < best_score:
                    best_score = current_score
                    best_h = h.copy()
            else:
                # Revert
                h[i] = -h[i]
                # A remains the old one (since we reverted h, A_new is invalid, but we kept A)
                # Actually, we didn't update A if rejected.
                # But we computed A_new. We discard it.
            
            steps += 1
            
            # Check time more frequently?
            # 1000 steps check?
            if steps % 1000 == 0:
                if time.time() > deadline:
                    break

    return best_h

# EVOLVE-BLOCK-END
