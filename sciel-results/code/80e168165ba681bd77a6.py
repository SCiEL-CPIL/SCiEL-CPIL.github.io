import numpy as np

N = 400

# EVOLVE-BLOCK-START
def construct():
    # Initial random sequence
    rng = np.random.default_rng(42)
    x = rng.integers(0, 2, N) * 2 - 1
    
    # Precompute autocorrelations
    # C_k = sum x_i x_{i+k}
    # We can use FFT for speed, but N=400 is small.
    
    def calc_cost(x):
        # Calculate autocorrelation C_k for k=1..N-1
        # Using np.correlate
        c = np.correlate(x, x, mode='full')
        # c is length 2N-1. c[N-1] is C_0 = N.
        # C_k is c[N-1+k] for k>0 and c[N-1-k] for k<0.
        # Since x is real, C_k = C_{-k}.
        # We want sum |C_k|^3 for k=1..N-1.
        # c[N] is C_1, c[N+1] is C_2, etc.
        # c[N-2] is C_{-1} = C_1.
        # So we take c[N:] (C_1..C_{N-1}) and c[:N-1][::-1] (C_{-1}..C_{-(N-1)})?
        # No, np.correlate(x, x, 'full') returns:
        # indices 0 to 2N-2.
        # Center is N-1.
        # C_k = c[N-1+k].
        # For k=1..N-1: c[N] to c[2N-2].
        # For k=-1..-(N-1): c[N-2] down to c[0].
        # Due to symmetry, C_k = C_{-k}.
        # So sum_{k=1}^{N-1} |C_k|^3 = sum_{k=1}^{N-1} |c[N-1+k]|^3.
        # Wait, is it just one side?
        # The sum usually implies all non-zero lags.
        # If the metric is sum over all k != 0, it's 2 * sum_{k=1}^{N-1} |C_k|^3.
        # Factor of 2 doesn't change the minimizer.
        pos_corr = c[N:] # C_1 to C_{N-1}
        return np.sum(np.abs(pos_corr)**3)

    cost = calc_cost(x)
    
    # Local Search
    # Iterate, try flipping each bit, if cost decreases, keep it.
    # To speed up, we can update correlations incrementally?
    # Or just recalculate. N=400.
    # np.correlate is fast.
    # 400 * 400 = 160,000 flips? 
    # Each flip + recalc cost (O(N log N) or O(N^2) in C) is fast.
    # 160,000 * ~0.1ms = 16 seconds. Acceptable.
    
    improved = True
    while improved:
        improved = False
        for i in range(N):
            # Flip x[i]
            old_val = x[i]
            x[i] = -old_val
            new_cost = calc_cost(x)
            if new_cost < cost:
                cost = new_cost
                improved = True
            else:
                x[i] = old_val
                
    return x
# EVOLVE-BLOCK-END
