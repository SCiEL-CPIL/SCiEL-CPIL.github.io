"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

TIME_LIMIT = 115  # safety margin before 120s kill

def _compute_c3_bound(heights):
    """
    Compute the upper bound on C3 from the autoconvolution of the signed step function.
    For a step function f(x) with N equal intervals on [-1/4, 1/4],
    the autoconvolution (f*f)(t) is computed via FFT-based convolution.
    The bound is dx * max_k |conv[k]| where conv is the discrete autoconvolution.
    """
    h = np.asarray(heights, dtype=np.float64)
    n = len(h)
    dx = 0.5 / n  # total length 0.5, n intervals
    
    # Pad h for linear autoconvolution
    h_pad = np.zeros(2 * n - 1)
    h_pad[:n] = h
    
    # Linear autoconvolution via FFT
    f_fft = np.fft.fft(h_pad, n=2 * (2 * n - 1))
    conv = np.fft.ifft(f_fft * np.conj(f_fft)).real
    
    # The continuous autoconvolution at grid points is dx * conv[k]
    max_abs = np.max(np.abs(conv))
    c3_bound = dx * max_abs
    
    return c3_bound

def _thue_morse(n):
    """Generate Thue-Morse sequence of length n as +/-1 values."""
    seq = [1]
    while len(seq) < n:
        seq += [-x for x in seq]
    return np.array(seq[:n], dtype=np.float64)

def _local_search_fast(h, time_budget=10.0):
    """
    Local search using O(N) autoconvolution update.
    """
    h = np.array(h, dtype=np.float64)
    n = len(h)
    dx = 0.5 / n
    
    # Initial autoconvolution
    h_pad = np.zeros(2 * n - 1)
    h_pad[:n] = h
    f_fft = np.fft.fft(h_pad, n=2 * (2 * n - 1))
    conv = np.fft.ifft(f_fft * np.conj(f_fft)).real
    current_bound = dx * np.max(np.abs(conv))
    
    start_time = time.time()
    improved = True
    iters = 0
    
    while improved and time.time() - start_time < time_budget:
        improved = False
        iters += 1
        best_gain = 0
        best_idx = -1
        
        for i in range(n):
            # Calculate change in autoconvolution if we flip h[i]
            # h_new[i] = -h[i]
            # Delta conv at lag k:
            # conv_new[k] = conv[k] - 4*h[i]*(h[i-k] + h[i+k]) + 4*delta[k,0]
            # Note: indices must be wrapped or bounded.
            # The autoconvolution is defined for lags -(n-1) to n-1.
            # In our padded array, index 0 corresponds to lag 0?
            # No, np.fft.ifft of |F|^2 gives circular correlation.
            # We zero-padded to 2n-1, so the result is linear correlation.
            # The result array `conv` has length 2n-1.
            # Index 0 is lag 0? No, usually the output of ifft is centered?
            # Let's check:
            # a = [1, 0, 0]
            # fft -> [1, 1, 1]
            # |fft|^2 -> [1, 1, 1]
            # ifft -> [1, 0, 0]
            # So index 0 is lag 0.
            # Wait, linear convolution of [1, 0] with [1, 0] is [1, 0, 0].
            # If we pad to 3: [1, 0, 0].
            # Convolution:
            # lag -1: 0
            # lag 0: 1*1 = 1
            # lag 1: 0
            # So index 0 is lag 0.
            
            # The formula for change in autoconvolution at lag k (relative to center?)
            # Let's use the property:
            # C_new = C - 2*h_i * (h_i * delta_i + delta_i * h_i) + ...
            # Actually, simpler:
            # C_new[k] = sum_j h_new[j] h_new[j-k]
            # h_new[j] = h[j] - 2 h_i delta_{j,i}
            # C_new[k] = sum_j (h[j] - 2 h_i delta_{j,i}) (h[j-k] - 2 h_i delta_{j-k,i})
            # = sum_j h[j]h[j-k] - 2 h_i sum_j h[j] delta_{j-k,i} - 2 h_i sum_j h[j-k] delta_{j,i} + 4 h_i^2 delta_{j,i} delta_{j-k,i}
            # = C[k] - 2 h_i h[k+i] - 2 h_i h[i-k] + 4 h_i^2 delta_{k,0}
            
            # Note: h[j] is 0 for j outside [0, n-1].
            # So we only need to update lags k where the terms are non-zero.
            # The terms h[k+i] and h[i-k] are non-zero when indices are in [0, n-1].
            
            # Instead of updating the whole conv array, we can just compute the new max.
            # But updating the whole array is O(N).
            # Let's try to compute the new max efficiently.
            # The change is sparse? No, it affects all lags where h[k+i] or h[i-k] is non-zero.
            # That's effectively all lags.
            
            # Let's just recompute the max using the delta formula on the fly?
            # No, we need the new conv values.
            
            # Optimization:
            # The delta is -2*h_i*(h[k+i] + h[i-k]) + 4*delta[k,0].
            # We can construct the delta vector.
            
            # However, constructing the delta vector for each i is O(N).
            # So each flip evaluation is O(N).
            # Total per iteration: O(N^2).
            # With N=400, N^2 = 160,000 ops.
            # If we do 1000 iterations, 160,000,000 ops.
            # Python is slow. 160M ops might take ~10-20 seconds.
            # This is feasible within 115s if we optimize the inner loop.
            
            # Let's use NumPy to vectorize the delta calculation.
            
            # h[i] is the value at i.
            # We need h[k+i] for all k.
            # Let's create a vector of h shifted.
            
            # This is getting complicated. Let's stick to the FFT method but optimize the search.
            # The FFT method was O(N log N) per eval.
            # N=400, N log N ~ 3600.
            # 1000 evals ~ 3.6M ops. Very fast.
            # The bottleneck in the previous code was likely the Python loop overhead or too many iterations.
            
            # Let's use the FFT method but with a tighter loop and more restarts.
            
            h_new = h.copy()
            h_new[i] = -h_new[i]
            new_bound = _compute_c3_bound(h_new)
            
            gain = current_bound - new_bound
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        
        if best_idx >= 0:
            h[best_idx] = -h[best_idx]
            current_bound -= best_gain
            improved = True
    
    return h, current_bound

def construct():
    start_time = time.time()
    
    best_h = np.ones(N)
    best_bound = _compute_c3_bound(best_h)
    
    # Thue-Morse is a very strong candidate
    tm = _thue_morse(N)
    tm_bound = _compute_c3_bound(tm)
    if tm_bound < best_bound:
        best_bound = tm_bound
        best_h = tm.copy()
    
    # Random restarts
    np.random.seed(42)
    
    # Time budget for each local search phase
    # Total time 115s.
    # Let's do 5 phases of local search with different starts.
    
    starts = []
    starts.append(tm)
    for _ in range(4):
        starts.append(np.random.choice([-1.0, 1.0], size=N))
    
    for start_idx, init_h in enumerate(starts):
        elapsed = time.time() - start_time
        remaining = TIME_LIMIT - elapsed - 5 # 5s safety
        
        if remaining < 10:
            break
            
        # Allocate time to this phase
        # Later phases get less time? Or equal?
        # Let's give equal time to each phase.
        phase_time = min(remaining / (len(starts) - start_idx), 20.0)
        
        h_copy = init_h.copy()
        h_opt, bound_opt = _local_search_fast(h_copy, time_budget=phase_time)
        
        if bound_opt < best_bound:
            best_bound = bound_opt
            best_h = h_opt.copy()
            
    return best_h

# EVOLVE-BLOCK-END
