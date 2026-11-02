import numpy as np
import time

N = 400

def _calc_acf(s):
    """Compute full autocorrelation array A[0..N-1]"""
    n_fft = 1
    while n_fft < 2 * N:
        n_fft *= 2
    s_f = s.astype(np.float64)
    F = np.fft.rfft(s_f, n=n_fft)
    P = F * np.conj(F)
    acf = np.fft.irfft(P, n=n_fft)
    return np.real(acf[:N])

def _max_abs_acf_from_array(A):
    return float(np.max(np.abs(A[1:])))

def _init_sine(freq, phase):
    """Initialize sign vector from a low-frequency sine wave."""
    x = np.linspace(-0.25, 0.25, N)
    wave = np.sin(2 * np.pi * freq * x + phase)
    s = np.where(wave >= 0, 1.0, -1.0)
    return s

def _apply_block_flip(s, A, start, length):
    """
    Flip a block of signs in s and update A accordingly.
    Returns the new A.
    """
    if length <= 0 or start < 0 or start + length > N:
        return A
    
    # Extract the block
    block = s[start:start+length]
    
    # The change in s is delta_s = -2 * s_old for indices in the block
    # A_new[k] = sum_{j} s_new[j] s_new[j+k]
    # A_new[k] = sum_{j not in B} s[j]s[j+k] + sum_{j in B} (-s[j])s_new[j+k] ... 
    # Actually, it's easier to think:
    # A_new = A + delta_A
    # delta_A[k] = -2 * ( sum_{j in B} s[j] s[j+k] + sum_{j in B} s[j] s[j-k] ) 
    # Wait, if we flip s[j] to -s[j] for j in B:
    # Term j in B, j+k not in B: s_new[j]s_new[j+k] = -s[j]s[j+k]. Change: -2 s[j]s[j+k].
    # Term j not in B, j+k in B: s_new[j]s_new[j+k] = s[j](-s[j+k]). Change: -2 s[j]s[j+k].
    # Term j in B, j+k in B: s_new[j]s_new[j+k] = (-s[j])(-s[j+k]) = s[j]s[j+k]. Change: 0.
    
    # So delta_A[k] = -2 * ( sum_{j in B, j+k not in B} s[j]s[j+k] + sum_{j not in B, j+k in B} s[j]s[j+k] )
    
    # Let's compute this efficiently.
    # Let B be the indices [start, start+length).
    # For a given k, the pairs (j, j+k) are affected if exactly one of them is in B.
    
    # We can compute the contribution of the block to the autocorrelation more directly.
    # Let's just recompute the affected parts of A.
    # The max index affected in A is start+length-1 + (N-1-start) ... no.
    # The autocorrelation A[k] depends on s[j] and s[j+k].
    # If we change s in [start, start+L), then A[k] changes for k such that
    # the window [j, j+k] intersects [start, start+L) in a "non-trivial" way?
    # Actually, any k where the pair (j, j+k) has one index in B and one outside B is changed by -2*s[j]*s[j+k].
    # If both are in B, the product is unchanged.
    # If both are outside B, the product is unchanged.
    
    # So we only need to update A[k] for k where the overlap of B and B-k is not B itself (i.e., k is not a multiple of L? No, just not 0 relative to block).
    # Actually, if k is such that B and B+k are disjoint, then all pairs are (in, out) or (out, in).
    # If k is small, B and B+k overlap. The overlapping part has both indices in B, so no change.
    # The non-overlapping part has change.
    
    # A simpler approach:
    # delta_A = -2 * ( cross_corr(s_B, s) - auto_corr(s_B) ) ? No.
    
    # Let's stick to the explicit sum but vectorized.
    # For each k, we need sum_{j in B, j+k not in B} s[j]s[j+k] + sum_{j not in B, j+k in B} s[j]s[j+k].
    # This is equivalent to:
    # Let mask_B be 1 in B, 0 elsewhere.
    # Let mask_Bk be mask_B shifted by k.
    # We want sum where mask_B != mask_Bk.
    # Sum = sum_{j: mask_B[j]=1, mask_B[j+k]=0} s[j]s[j+k] + sum_{j: mask_B[j]=0, mask_B[j+k]=1} s[j]s[j+k]
    
    # This is getting complicated to vectorize per k.
    # Since N=400, and we do this many times, we can just update A directly using the property:
    # A_new = A - 2 * (s_B * s) convolved with ...?
    
    # Let's try a different perspective.
    # A = s * s (circular? No, linear).
    # If we flip a block, we are adding a perturbation d where d[j] = -2s[j] for j in B, 0 otherwise.
    # s_new = s + d.
    # A_new = (s+d) * (s+d) = s*s + s*d + d*s + d*d.
    # A_new = A + cross_corr(s, d) + cross_corr(d, s) + auto_corr(d).
    
    # d is non-zero only in B.
    # auto_corr(d) is non-zero only for lags -L+1 to L-1.
    # cross_corr(s, d) is non-zero for lags ... well, d is sparse, so we can compute this fast.
    
    # Let's compute d.
    d = np.zeros(N)
    d[start:start+length] = -2.0 * s[start:start+length]
    
    # Now compute A_new = A + cc(s,d) + cc(d,s) + ac(d).
    # cc(s,d)[k] = sum_j s[j] d[j+k]
    # Since d is zero outside B, this is sum_{j+k in B} s[j] d[j+k] = sum_{m in B} s[m-k] d[m].
    # So cc(s,d)[k] = sum_{m in B} s[m-k] d[m].
    # We can compute this for all k efficiently?
    # Or just for the k's that matter?
    # The change in A is non-zero for k in [-N+1, N-1].
    
    # Let's just compute the full delta using FFT? No, that's O(N log N).
    # Since N=400, O(N^2) is 160,000 ops. If we do 10,000 flips, that's 1.6e9 ops. Might be too slow in Python loops.
    # But we can vectorize the delta calculation.
    
    # Let's construct the delta array directly.
    # delta_A[k] = sum_j (s_new[j]s_new[j+k] - s[j]s[j+k])
    
    # We can compute the new A for the affected range?
    # Actually, the simplest robust way for N=400 is to just recompute A if the block is large, or use the d-method if small.
    # Given N=400, recomputing A via FFT takes ~0.1ms.
    # If we do 100,000 flips in 140s, that's 1.4ms per flip. FFT is fine.
    # But local search with single bits was slow because of the Python loop overhead.
    # Block flips reduce the number of iterations needed.
    
    # Let's optimize the d-method.
    # d is sparse.
    # cc(s,d)[k] = sum_{m in B} s[m-k] d[m]
    # Let's compute this for all k.
    # We can use convolution.
    # cc(s,d) is the cross-correlation.
    # np.correlate(s, d, mode='full')?
    # Yes! np.correlate is implemented in C and fast.
    
    # A_new = A + np.correlate(s, d, 'full')[N-1:N-1+N] + np.correlate(d, s, 'full')[N-1:N-1+N] + np.correlate(d, d, 'full')[N-1:N-1+N]
    # Wait, correlate(s,d) gives sum s[n] d[n+k]? Or d[n] s[n+k]?
    # numpy.correlate(a, v) is sum a[n+k] v[n].
    # So correlate(s, d) is sum s[n+k] d[n]. This is cc(s,d)[k] if we define cc[k] = sum s[j] d[j-k]?
    # Let's check indices.
    # A[k] = sum s[j] s[j+k].
    # A_new[k] = sum (s[j]+d[j]) (s[j+k]+d[j+k])
    # = A[k] + sum s[j]d[j+k] + sum d[j]s[j+k] + sum d[j]d[j+k]
    
    # Term 1: sum s[j] d[j+k]. Let m = j+k => j = m-k. Sum s[m-k] d[m].
    # This is cross-correlation of s and d with lag k.
    # np.correlate(s, d, 'full') gives c[k] = sum_n s[n] d[n+k]? No.
    # Documentation: c[k] = sum_n a[n+k] v[n].
    # So if a=s, v=d, c[k] = sum s[n+k] d[n].
    # This matches Term 1 if we swap indices?
    # Term 1 is sum s[j] d[j+k]. Let n=j. sum s[n] d[n+k].
    # This is c[-k] of correlate(s,d)?
    # c[-k] = sum s[n-k] d[n]. Let m=n-k => n=m+k. sum s[m] d[m+k]. Yes.
    
    # So Term 1[k] = correlate(s, d, 'full')[N-1-k] ?
    # Let's just use the fact that A is symmetric (for real s).
    # Actually, let's just recompute A using FFT if the block size is large, and use a simpler update if small?
    # No, let's just use the FFT method for simplicity and reliability, as N=400 is small.
    # The overhead of FFT is small compared to the search logic.
    
    s[start:start+length] = -s[start:start+length]
    A_new = _calc_acf(s)
    return A_new

def _local_search_sa(s, A, current_val, deadline):
    """
    Simulated Annealing with block flips.
    """
    T = 1.0
    cooling = 0.99
    min_T = 0.01
    
    best_val = current_val
    best_s = s.copy()
    
    # Random block sizes
    # Start with small blocks, occasionally larger
    block_sizes = [1, 2, 4, 8, 16, 32]
    
    while time.time() < deadline and T > min_T:
        # Choose a block size
        # Probabilistic choice: smaller blocks more frequent
        r = np.random.random()
        if r < 0.4:
            length = 1
        elif r < 0.7:
            length = 2
        elif r < 0.85:
            length = 4
        elif r < 0.95:
            length = 8
        else:
            length = 16
            
        # Choose start position
        if length > N:
            continue
        start = np.random.randint(0, N - length + 1)
        
        # Apply flip
        old_s = s[start:start+length].copy()
        s[start:start+length] = -s[start:start+length]
        A_new = _calc_acf(s)
        val_new = _max_abs_acf_from_array(A_new)
        
        # Accept or reject
        if val_new < current_val or np.random.random() < np.exp(-(val_new - current_val) / T):
            A = A_new
            current_val = val_new
            if current_val < best_val:
                best_val = current_val
                best_s = s.copy()
        else:
            # Revert
            s[start:start+length] = old_s
            # A is unchanged
            
        T *= cooling
        
        # Occasionally restart temperature if stuck?
        # Or just keep cooling.
        
    return best_s, best_val

def construct():
    start_time = time.time()
    deadline = 140.0
    
    best_val = float('inf')
    best_seq = np.ones(N)
    
    # Try different initializations
    freqs = [1, 2, 3, 4, 5, 6, 8, 10, 12]
    
    # Number of restarts
    num_restarts = 20
    
    for _ in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        # Choose initialization
        if np.random.random() < 0.5:
            freq = np.random.choice(freqs)
            phase = np.random.uniform(0, 2 * np.pi)
            s = _init_sine(freq, phase)
        else:
            s = np.random.randint(0, 2, size=N) * 2 - 1
            
        A = _calc_acf(s)
        current_val = _max_abs_acf_from_array(A)
        
        # Run SA
        s_new, val_new = _local_search_sa(s, A, current_val, deadline)
        
        if val_new < best_val:
            best_val = val_new
            best_seq = s_new.copy()
            
        # If we found a very good solution, we can spend more time polishing
        if best_val < 0.01:
             # Do a final polish with single-bit flips?
             pass

    return best_seq
