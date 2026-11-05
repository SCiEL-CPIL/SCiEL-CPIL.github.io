"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

TIME_LIMIT = 145  # safety margin before 150s kill


def _compute_c3_bound(heights):
    """
    Compute the upper bound on C3 from the autoconvolution of the signed step function.
    For a step function f(x) with N equal intervals on [-1/4, 1/4],
    the autoconvolution (f*f)(t) is computed via FFT-based convolution.
    The bound is dx * max_k |conv[k]| for k in [1, N-1] (excluding lag 0).
    """
    h = np.asarray(heights, dtype=np.float64)
    n = len(h)
    dx = 0.5 / n  # total length 0.5, n intervals
    
    # Pad h for linear autoconvolution
    h_pad = np.zeros(2 * n - 1)
    h_pad[:n] = h
    
    # Linear autoconvolution via FFT
    fft_size = 1
    while fft_size < 2 * (2 * n - 1) - 1:
        fft_size *= 2
        
    f_fft = np.fft.fft(h_pad, n=fft_size)
    conv_full = np.fft.ifft(f_fft * np.conj(f_fft)).real
    # Extract the linear convolution part (length 2n-1)
    conv = conv_full[:2 * n - 1]
    
    # The continuous autoconvolution at grid points is dx * conv[k]
    # Exclude lag 0 (the DC/self term) - use lags 1 through n-1
    # conv has length 2n-1, lag k corresponds to conv[k] for k=0..2n-2
    # We want lags 1 to n-1 (positive lags up to n-1)
    max_abs = np.max(np.abs(conv[1:n]))
    c3_bound = dx * max_abs
    
    return c3_bound


def _compute_c3_bound_and_conv(heights):
    """Compute C3 bound and also return the convolution array for reuse."""
    h = np.asarray(heights, dtype=np.float64)
    n = len(h)
    dx = 0.5 / n
    
    h_pad = np.zeros(2 * n - 1)
    h_pad[:n] = h
    
    fft_size = 1
    while fft_size < 2 * (2 * n - 1) - 1:
        fft_size *= 2
        
    f_fft = np.fft.fft(h_pad, n=fft_size)
    conv_full = np.fft.ifft(f_fft * np.conj(f_fft)).real
    conv = conv_full[:2 * n - 1]
    
    max_abs = np.max(np.abs(conv[1:n]))
    c3_bound = dx * max_abs
    
    return c3_bound, conv


def _thue_morse(n):
    """Generate Thue-Morse sequence of length n as +/-1 values."""
    seq = [1]
    while len(seq) < n:
        seq += [-x for x in seq]
    return np.array(seq[:n], dtype=np.float64)


def _zadoff_chu_product(n, p1=7, p2=11):
    """
    Generate a sequence based on the sign of the imaginary part of the product
    of two Zadoff-Chu sequences with distinct prime moduli.
    """
    k = np.arange(n, dtype=np.float64)
    phase1 = -np.pi * k * (k + 1) / p1
    phase2 = -np.pi * k * (k + 1) / p2
    total_phase = phase1 + phase2
    seq = np.sign(np.sin(total_phase))
    seq[seq == 0] = 1.0
    return seq.astype(np.float64)


def _chaotic_sequence(n, r=3.99, x0=0.12345, seed=42):
    """
    Generate a signed sequence from the logistic map near the edge of chaos.
    """
    x = x0
    seq = np.empty(n, dtype=np.float64)
    for i in range(n):
        seq[i] = 1.0 if x > 0.5 else -1.0
        x = r * x * (1.0 - x)
        if x < 0:
            x = 0.0
        elif x > 1:
            x = 1.0
    return seq


def _local_search(h, max_iter=5000, time_budget=10, start_time=0):
    """
    Local search: try flipping individual bits (changing sign) to reduce the C3 bound.
    Uses best-improvement strategy with time-based stopping.
    """
    h = np.array(h, dtype=np.float64)
    current_bound, conv = _compute_c3_bound_and_conv(h)
    dx = 0.5 / N
    n = len(h)
    
    if start_time == 0:
        start_time = time.time()
    
    improved = True
    iters = 0
    while improved and iters < max_iter:
        if time.time() - start_time > time_budget:
            break
        
        improved = False
        iters += 1
        best_gain = 0.0
        best_idx = -1
        
        # Try flipping each position using analytic perturbation
        h_pad = np.zeros(2 * n - 1)
        h_pad[:n] = h
        
        for i in range(n):
            # Perturbation from flipping h[i]: conv_new = conv - 2*h[i]*shift(h_pad, i)
            if i == 0:
                pert = -2.0 * h[i] * h_pad
            else:
                pert = np.zeros(2 * n - 1)
                pert[i:] = -2.0 * h[i] * h_pad[:2*n-1-i]
            
            new_conv = conv + pert
            new_max = np.max(np.abs(new_conv[1:n]))
            new_bound = dx * new_max
            gain = current_bound - new_bound
            
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        
        if best_idx >= 0:
            h[best_idx] = -h[best_idx]
            current_bound -= best_gain
            # Recompute conv for correctness
            _, conv = _compute_c3_bound_and_conv(h)
            improved = True
    
    return h, current_bound


def _pairwise_flip_search(h, max_iter=2000, time_budget=10, start_time=0):
    """
    Enhanced local search: try flipping pairs of bits to escape local minima.
    Uses analytic perturbation for speed.
    """
    h = np.array(h, dtype=np.float64)
    current_bound, conv = _compute_c3_bound_and_conv(h)
    dx = 0.5 / N
    n = len(h)
    
    if start_time == 0:
        start_time = time.time()
    
    improved = True
    iters = 0
    while improved and iters < max_iter:
        if time.time() - start_time > time_budget:
            break
        
        improved = False
        iters += 1
        best_gain = 0.0
        best_pair = None
        
        # Try flipping pairs (i, j) where i < j
        # To keep this tractable, sample a subset of pairs
        np.random.seed(42 + iters)
        indices = np.random.choice(n, size=min(60, n), replace=False)
        
        h_pad = np.zeros(2 * n - 1)
        h_pad[:n] = h
        
        for ii in range(len(indices)):
            if time.time() - start_time > time_budget:
                break
            i = indices[ii]
            for jj in range(ii + 1, len(indices)):
                j = indices[jj]
                
                # Perturbation from flipping both i and j
                if i == 0:
                    pert_i = -2.0 * h[i] * h_pad
                else:
                    pert_i = np.zeros(2 * n - 1)
                    pert_i[i:] = -2.0 * h[i] * h_pad[:2*n-1-i]
                
                if j == 0:
                    pert_j = -2.0 * h[j] * h_pad
                else:
                    pert_j = np.zeros(2 * n - 1)
                    pert_j[j:] = -2.0 * h[j] * h_pad[:2*n-1-j]
                
                new_conv = conv + pert_i + pert_j
                new_max = np.max(np.abs(new_conv[1:n]))
                new_bound = dx * new_max
                gain = current_bound - new_bound
                
                if gain > best_gain:
                    best_gain = gain
                    best_pair = (i, j)
        
        if best_pair is not None:
            i, j = best_pair
            h[i] = -h[i]
            h[j] = -h[j]
            current_bound -= best_gain
            _, conv = _compute_c3_bound_and_conv(h)
            improved = True
    
    return h, current_bound


def construct():
    start_time = time.time()
    
    # Initialize with multiple starting points
    best_h = np.ones(N)
    best_bound = _compute_c3_bound(best_h)
    
    # Try Thue-Morse sequence
    tm = _thue_morse(N)
    tm_bound = _compute_c3_bound(tm)
    if tm_bound < best_bound:
        best_bound = tm_bound
        best_h = tm.copy()
    
    # Try chaotic sequences with various parameters
    chaotic_params = [
        (3.99, 0.12345),
        (3.999, 0.12345),
        (3.99, 0.45678),
        (3.999, 0.45678),
        (3.99, 0.78901),
        (3.999, 0.78901),
        (3.95, 0.12345),
        (3.95, 0.45678),
        (3.99, 0.23456),
        (3.999, 0.34567),
    ]
    for r, x0 in chaotic_params:
        if time.time() - start_time > TIME_LIMIT - 20:
            break
        h_chaos = _chaotic_sequence(N, r=r, x0=x0)
        bound = _compute_c3_bound(h_chaos)
        if bound < best_bound:
            best_bound = bound
            best_h = h_chaos.copy()
    
    # Try Zadoff-Chu product sequences
    if time.time() - start_time < TIME_LIMIT - 20:
        zc_params = [(7, 11), (5, 13), (7, 13), (11, 17), (5, 7)]
        for p1, p2 in zc_params:
            h_zc = _zadoff_chu_product(N, p1=p1, p2=p2)
            bound = _compute_c3_bound(h_zc)
            if bound < best_bound:
                best_bound = bound
                best_h = h_zc.copy()
    
    # Try random initializations
    np.random.seed(42)
    for trial in range(20):
        if time.time() - start_time > TIME_LIMIT - 20:
            break
        
        h = np.random.choice([-1.0, 1.0], size=N)
        bound = _compute_c3_bound(h)
        if bound < best_bound:
            best_bound = bound
            best_h = h.copy()
    
    # Local search from best initialization
    remaining = TIME_LIMIT - (time.time() - start_time)
    if remaining > 15:
        local_time = min(remaining - 10, 50)
        best_h, best_bound = _local_search(best_h, max_iter=5000, time_budget=local_time, start_time=start_time)
    
    # Pairwise flip search to escape local minima
    remaining = TIME_LIMIT - (time.time() - start_time)
    if remaining > 20:
        pair_time = min(remaining - 15, 30)
        best_h, best_bound = _pairwise_flip_search(best_h, max_iter=2000, time_budget=pair_time, start_time=start_time)
    
    # Additional restarts if time permits
    remaining = TIME_LIMIT - (time.time() - start_time)
    if remaining > 25:
        for trial in range(5):
            if time.time() - start_time > TIME_LIMIT - 10:
                break
            
            # Mix of random and chaotic initializations
            if trial % 2 == 0:
                h = np.random.choice([-1.0, 1.0], size=N)
            else:
                h = _chaotic_sequence(N, r=3.99 + 0.001 * trial, x0=0.1 + 0.01 * trial)
            
            bound = _compute_c3_bound(h)
            
            if bound < best_bound:
                local_time = min(TIME_LIMIT - (time.time() - start_time) - 5, 15)
                h, bound = _local_search(h, max_iter=2000, time_budget=local_time, start_time=start_time)
                if bound < best_bound:
                    best_bound = bound
                    best_h = h.copy()
    
    # Final local polish
    remaining = TIME_LIMIT - (time.time() - start_time)
    if remaining > 10:
        polish_time = min(remaining - 5, 15)
        best_h, best_bound = _local_search(best_h, max_iter=3000, time_budget=polish_time, start_time=start_time)
    
    # Final verification
    final_bound = _compute_c3_bound(best_h)
    
    return best_h


# Test
if __name__ == "__main__":
    h = construct()
    print(f"Length: {len(h)}")
    print(f"First 20: {h[:20]}")
    print(f"C3 bound: {_compute_c3_bound(h)}")
    print(f"Score: {1.0 / _compute_c3_bound(h)}")
# EVOLVE-BLOCK-END
