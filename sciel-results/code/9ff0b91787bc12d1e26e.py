import numpy as np
import time

N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    time_limit = 110.0
    
    def get_min_correlation(h):
        # Calculate C_k = sum h_i h_{i+k} for k=1..N-1
        # Using FFT for speed? N=100 is small, direct calculation is fine.
        # C_k = sum_{i=0}^{N-1} h_i h_{(i+k)%N}
        # Note: The problem statement says "shifted 1-h".
        # Overlap of h and shifted (1-h) is sum h_i (1-h_{i+k}) = sum h_i - sum h_i h_{i+k}.
        # To minimize max overlap, we minimize max (N/2 - C_k).
        # This is equivalent to maximizing min C_k.
        min_c = np.inf
        for k in range(1, N):
            # Circular shift
            c_k = np.sum(h * np.roll(h, k))
            if c_k < min_c:
                min_c = c_k
        return min_c

    def ternary_search_optimize_i(h, i):
        # Optimize h[i] to maximize min_correlation
        # h is the current array
        # We fix all h[j] for j != i
        
        def objective(x):
            # Temporarily set h[i] = x
            # Calculate min correlation
            # We can optimize this by pre-computing parts that don't depend on x
            # But N=100 is small, just recompute.
            old_h_i = h[i]
            h[i] = x
            val = get_min_correlation(h)
            h[i] = old_h_i
            return val

        lo = 0.0
        hi = 1.0
        # Ternary search for maximum of a concave function
        # The function min_k (A_k + B_k x) is concave.
        for _ in range(50): # 50 iterations is plenty for double precision
            m1 = lo + (hi - lo) / 3.0
            m2 = hi - (hi - lo) / 3.0
            f1 = objective(m1)
            f2 = objective(m2)
            if f1 < f2:
                lo = m1
            else:
                hi = m2
        return (lo + hi) / 2.0

    # Initialization: Sinusoidal
    # h_i = 0.5 + A * sin(2*pi*i/N)
    # Choose A such that values are in [0,1]. A=0.5 gives [0,1].
    # Sum will be approx N/2.
    h = 0.5 + 0.5 * np.sin(2 * np.pi * np.arange(N) / N)
    
    # Ensure sum is exactly N/2 and bounds [0,1]
    # The sinusoid has mean 0.5, so sum is N/2.
    # Values are in [0,1].
    # Check sum
    s = np.sum(h)
    if abs(s - N/2) > 1e-9:
        h += (N/2 - s) / N
        h = np.clip(h, 0, 1)
        # Re-fix sum after clip
        s = np.sum(h)
        if abs(s - N/2) > 1e-12:
            # Distribute difference
            h += (N/2 - s) / N
            h = np.clip(h, 0, 1)

    best_h = h.copy()
    best_val = get_min_correlation(best_h)
    
    # Coordinate Ascent
    # Multiple passes
    max_passes = 500
    tol = 1e-10
    
    for p in range(max_passes):
        if time.time() - start_time > time_limit:
            break
        
        changed = False
        for i in range(N):
            old_h_i = h[i]
            new_h_i = ternary_search_optimize_i(h, i)
            h[i] = new_h_i
            
            if abs(new_h_i - old_h_i) > tol:
                changed = True
                
            # Check time inside loop? Maybe too slow. Check every few i?
            # N=100, ternary search 50 steps, get_min_correlation 100 shifts * 100 mults = 10k ops.
            # 50 * 10k = 500k ops per i. 100 i's = 50M ops per pass.
            # 500 passes = 25B ops. This might be too slow in Python.
            # Optimization: Precompute dependencies for ternary search.
            
        if not changed:
            break
            
        # Evaluate current best
        val = get_min_correlation(h)
        if val > best_val:
            best_val = val
            best_h = h.copy()
            
    # Final projection
    diff = N/2 - np.sum(best_h)
    best_h += diff / N
    best_h = np.clip(best_h, 0, 1)
    
    # One last check/fix if clip broke sum
    diff = N/2 - np.sum(best_h)
    if abs(diff) > 1e-12:
        # Add to elements not at bound
        mask = (best_h > 0) & (best_h < 1)
        if np.sum(mask) > 0:
            best_h[mask] += diff / np.sum(mask)
        else:
            best_h += diff / N
            best_h = np.clip(best_h, 0, 1)

    return best_h
# EVOLVE-BLOCK-END
