"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Initialize with random signs
    # Using a fixed seed for reproducibility during development, 
    # but random is fine for the evaluator.
    rng = np.random.default_rng(42)
    s = rng.choice([-1, 1], size=N)
    
    best_psl = np.inf
    best_s = s.copy()
    
    # Helper to compute PSL (Peak Sidelobe Level)
    # We want to minimize max_{k != 0} |autocorr[k]|
    # Autocorr can be computed via FFT for speed.
    
    def get_psl(seq):
        # Compute autocorrelation using FFT
        # Zero pad to avoid circular convolution issues
        n_fft = 1 << (2 * N - 1).bit_length()
        f = np.fft.rfft(seq, n=n_fft)
        acf = np.fft.irfft(f * np.conjugate(f), n=n_fft)[:N]
        # acf[0] is the energy, acf[1:] are the sidelobes
        # We look at max absolute value of sidelobes
        return np.max(np.abs(acf[1:]))
    
    current_psl = get_psl(s)
    if current_psl < best_psl:
        best_psl = current_psl
        best_s = s.copy()
        
    # Greedy optimization
    # We try flipping individual bits to reduce the PSL.
    # To speed up, we can look at the worst lag and try to fix it.
    
    while time.time() < deadline:
        # Compute autocorrelation for current s
        n_fft = 1 << (2 * N - 1).bit_length()
        f = np.fft.rfft(s, n=n_fft)
        acf = np.fft.irfft(f * np.conjugate(f), n=n_fft)[:N]
        
        # Find the worst sidelobe lag
        # We ignore lag 0
        abs_acf = np.abs(acf)
        abs_acf[0] = 0
        worst_lag = np.argmax(abs_acf)
        
        if worst_lag == 0:
            break
            
        # Try to flip a bit to reduce the contribution to this lag
        # The autocorrelation at lag k is sum s[i] s[i+k]
        # Flipping s[j] changes the autocorrelation at lag k by:
        # -2 * s[j] * (s[j-k] if j>=k else 0 + s[j+k] if j+k<N else 0) ... roughly
        # Actually, let's just try flipping a few candidate bits.
        
        # A simple strategy: flip the bit that is most "out of phase" with the neighbors contributing to the worst lag.
        # Or simply try flipping bits at indices related to the worst lag.
        
        # Let's try flipping a random subset of bits or the bit that minimizes the new PSL.
        # Checking all N bits is O(N * FFT) which is too slow if done naively for many iterations.
        # N=400. FFT is ~400 log 400 ~ 3000 ops. 400 flips * 3000 = 1.2M ops per iteration.
        # 110s / 1.2M is ~100 iterations? Might be slow in Python.
        
        # Optimized Greedy:
        # 1. Identify worst lag k.
        # 2. The term contributing to acf[k] is sum_{i} s[i] s[i+k].
        # 3. Flipping s[j] affects acf[k] by -2 * s[j] * (s[j-k] + s[j+k]) (with bounds).
        # 4. We want to reduce |acf[k]|.
        # 5. So we should flip s[j] if s[j] * (s[j-k] + s[j+k]) has the same sign as acf[k].
        
        # Let's compute the "influence" of each bit j on the worst lag k.
        # Influence[j] = s[j] * (s[j-k] if j>=k else 0 + s[j+k] if j+k<N else 0)
        # If acf[k] > 0, we want Influence[j] > 0 to flip (to make it negative, reducing sum).
        # If acf[k] < 0, we want Influence[j] < 0 to flip.
        
        k = worst_lag
        sign_acf = np.sign(acf[k])
        
        # Calculate influence vector
        # s is 0-indexed.
        # For a given j, the neighbors are j-k and j+k.
        # We can compute this vectorized.
        
        # Create arrays for indices
        # left neighbors: j-k >= 0 => j >= k
        # right neighbors: j+k < N => j < N-k
        
        influence = np.zeros(N)
        
        # Contribution from s[j] * s[j-k]
        # Valid for j in [k, N-1]
        if k > 0:
            # s[j] * s[j-k]
            # We want to know if flipping s[j] reduces the sum.
            # Sum decreases by 2 * s[j] * s[j-k] if we flip s[j]?
            # Original term: s[j] s[j-k]. New term: -s[j] s[j-k]. Change: -2 s[j] s[j-k].
            # We want Change to have opposite sign to acf[k].
            # So we want -2 s[j] s[j-k] * sign(acf[k]) < 0 => s[j] s[j-k] * sign(acf[k]) > 0.
            
            # Let's just calculate the change in acf[k] if we flip j.
            # delta_acf_k = -2 * s[j] * ( (s[j-k] if j>=k else 0) + (s[j+k] if j+k<N else 0) )
            
            # Vectorized calculation of delta
            # Term 1: j >= k
            # s[j] * s[j-k]
            t1 = s[k:] * s[:N-k] # s[j] for j=k..N-1, s[j-k] for j=k..N-1 is s[0..N-1-k]
            # t1[i] corresponds to j = i+k. So t1[i] = s[i+k] s[i].
            # This affects acf[k].
            # We map this back to index j = i+k.
            
            # Term 2: j + k < N
            # s[j] * s[j+k]
            t2 = s[:N-k] * s[k:] # s[j] for j=0..N-1-k, s[j+k] for j=0..N-1-k is s[k..N-1]
            # t2[i] corresponds to j = i.
            
            # Combine
            # delta[j] = -2 * ( (s[j-k] if j>=k else 0) + (s[j+k] if j+k<N else 0) ) * s[j]
            
            delta = np.zeros(N)
            # For j >= k: delta[j] += -2 * s[j-k] * s[j]
            delta[k:] += -2 * (s[:N-k] * s[k:])
            # For j < N-k: delta[j] += -2 * s[j+k] * s[j]
            delta[:N-k] += -2 * (s[k:] * s[:N-k])
            
            # We want to pick j that minimizes |acf[k] + delta[j]|
            # Or more simply, if acf[k] > 0, pick j with most negative delta.
            # If acf[k] < 0, pick j with most positive delta.
            
            if acf[k] > 0:
                best_j = np.argmin(delta)
            else:
                best_j = np.argmax(delta)
                
            # Check if flipping this bit actually improves the PSL (not just this lag)
            # It might improve this lag but worsen others.
            # We'll flip it if it reduces the specific lag, and re-check global PSL.
            
            s[best_j] *= -1
            new_psl = get_psl(s)
            
            if new_psl < best_psl:
                best_psl = new_psl
                best_s = s.copy()
            else:
                # Revert if it didn't improve global best
                s[best_j] *= -1
                # If we can't improve with single bit flips on worst lag, 
                # we might be stuck. Let's try random restarts or random flips.
                
                # To avoid getting stuck in local minima, occasionally do a random flip
                # and accept if it improves global PSL.
                if np.random.rand() < 0.1:
                    r_j = np.random.randint(N)
                    s[r_j] *= -1
                    r_psl = get_psl(s)
                    if r_psl < best_psl:
                        best_psl = r_psl
                        best_s = s.copy()
                    else:
                        s[r_j] *= -1
                        
    return best_s
# EVOLVE-BLOCK-END
