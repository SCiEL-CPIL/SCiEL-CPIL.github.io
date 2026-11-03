import numpy as np
import time
from scipy.optimize import minimize

N = 400
_FSIZE = 1
while _FSIZE < 2 * N - 1:
    _FSIZE *= 2

# Precompute basis vectors for efficient FFT updates
# Shape: (FSIZE//2 + 1, N)
_freqs = np.arange(_FSIZE // 2 + 1)
_basis_vectors = np.exp(-2j * np.pi * np.outer(_freqs, np.arange(N)) / _FSIZE)

def compute_raw_value(x):
    """
    Computes the raw value: max(|autoconvolution|) / (sum(x))^2.
    """
    X = np.fft.rfft(x, _FSIZE)
    conv = np.fft.irfft(X * X, _FSIZE)
    conv = conv[:2*N-1]
    max_abs_conv = np.max(np.abs(conv))
    integral = np.sum(x)
    if abs(integral) < 1e-12:
        return float('inf')
    return max_abs_conv / (integral * integral)

def _soft_max_abs_and_grad(h, beta=50.0):
    """
    Computes the soft-max absolute autoconvolution value and its analytical gradient.
    Returns: value, gradient
    """
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10, np.zeros_like(h)
    
    H = np.fft.rfft(h, _FSIZE)
    ac = np.fft.irfft(H * H, _FSIZE)[:2*N-1]
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    
    if m < 1e-15:
        S = 0.0
        grad_S = np.zeros_like(h)
    else:
        scaled = beta * (abs_ac - m)
        scaled = np.clip(scaled, -50, 50)
        exp_scaled = np.exp(scaled)
        sum_exp = np.sum(exp_scaled)
        log_sum = np.log(sum_exp)
        S = m + log_sum / beta
        
        # Weights for the soft-max
        w = exp_scaled / sum_exp
        
        # v_k = w_k * sign(ac_k)
        v = w * np.sign(ac)
        
        # Gradient of S w.r.t h
        # S = soft_max(|ac|)
        # dS/dh_i = 2 * sum_k v_k h_{k-i} 
        # This is a cross-correlation between v and h.
        
        fsize_corr = 1
        while fsize_corr < 3 * N:
            fsize_corr *= 2
            
        V = np.fft.rfft(v, fsize_corr)
        Hc = np.fft.rfft(h, fsize_corr)
        # Cross-correlation: V * conj(Hc)
        C = np.fft.irfft(V * np.conj(Hc), fsize_corr)
        
        # C[i] = sum_n v[n+i] conj(h[n])
        # For real h, C[i] = sum_n v[n+i] h[n]
        # We need the first N elements of this correlation to form the gradient vector.
        grad_S = 2 * C[:N]
    
    # Objective f = S / s^2
    # df/dh_i = (dS/dh_i * s^2 - S * 2 * s) / s^4
    # = (dS/dh_i * s - 2 * S) / s^3
    
    grad = (grad_S * s - 2 * S) / (s ** 3)
    
    return S / (s * s), grad

def _local_search_flip_vectorized(h, max_passes=5, deadline=None):
    """
    Vectorized local search: try flipping each element's sign.
    Uses FFT updates but processes flips in a way that allows for early exit or batch processing.
    To make it distinct and potentially faster, we evaluate all single-flip improvements in a single pass.
    """
    h = h.copy()
    h = np.sign(h)
    h[h == 0] = 1
    best_val = compute_raw_value(h)
    H = np.fft.rfft(h, _FSIZE)
    integral = np.sum(h)
    
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        # Iterate through indices. 
        # Optimization: We can pre-calculate the effect of flipping index i.
        # Flipping h[i] -> -h[i] changes H by -2*h[i]*basis_i.
        # New H = H + delta_i.
        # New Conv = irfft((H + delta_i)^2).
        
        # For speed, we just do a simple loop but with strict time checking.
        # Vectorization of the *evaluation* of all N flips at once is memory intensive (N FFTs).
        # Instead, we stick to efficient scalar updates but ensure we don't miss improvements.
        
        for i in range(N):
            if deadline is not None and time.time() > deadline:
                break
            
            old_val = h[i]
            h[i] = -old_val
            
            delta = -2 * old_val * _basis_vectors[:, i]
            H_new = H + delta
            conv = np.fft.irfft(H_new * H_new, _FSIZE)[:2*N-1]
            new_integral = integral - 2 * old_val
            
            if abs(new_integral) < 1e-12:
                new_val = float('inf')
            else:
                new_val = np.max(np.abs(conv)) / (new_integral * new_integral)
            
            if new_val < best_val:
                best_val = new_val
                H = H_new
                integral = new_integral
                improved = True
            else:
                h[i] = old_val
        
        if not improved:
            break
            
    return h, best_val

def _peak_targeted_search(h, deadline=None):
    """
    Targets the peak of the autoconvolution.
    Identifies the lag with max abs conv, and tries to flip small blocks of signs 
    to reduce that specific peak.
    """
    h = h.copy()
    h = np.sign(h)
    h[h == 0] = 1
    best_val = compute_raw_value(h)
    H = np.fft.rfft(h, _FSIZE)
    integral = np.sum(h)
    
    # Find peak lag
    conv = np.fft.irfft(H * H, _FSIZE)[:2*N-1]
    peak_lag = np.argmax(np.abs(conv))
    
    # Heuristic: The peak at lag L is influenced by pairs (i, j) where |i-j| is related to L.
    # We try flipping small windows of size 2-4.
    window_sizes = [2, 3, 4]
    
    improved = True
    while improved and (deadline is None or time.time() < deadline):
        improved = False
        
        for ws in window_sizes:
            for start in range(0, N - ws + 1, 2): # Step 2 to save time
                if deadline is not None and time.time() > deadline:
                    break
                
                # Flip block
                block_indices = np.arange(start, start + ws)
                old_block = h[block_indices].copy()
                h[block_indices] = -h[block_indices]
                
                # Update H
                delta = np.zeros(_FSIZE//2+1, dtype=complex)
                for idx in block_indices:
                    delta += -2 * old_block[idx - start] * _basis_vectors[:, idx]
                    
                H_new = H + delta
                conv_new = np.fft.irfft(H_new * H_new, _FSIZE)[:2*N-1]
                new_integral = integral - 2 * np.sum(old_block)
                
                if abs(new_integral) < 1e-12:
                    new_val = float('inf')
                else:
                    new_val = np.max(np.abs(conv_new)) / (new_integral * new_integral)
                
                if new_val < best_val:
                    best_val = new_val
                    H = H_new
                    integral = new_integral
                    improved = True
                else:
                    h[block_indices] = old_block
                    
    return h, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin
    
    # 1. Baseline
    best_x = np.ones(N, dtype=float)
    best_val = compute_raw_value(best_x)
    
    # 2. Generate Diverse Candidates
    candidates = [best_x.copy()]
    t = np.arange(N)
    
    # Chirps
    for freq in [0.5, 1.0, 1.5, 2.0, 3.0]:
        phase = 2 * np.pi * freq * (t**2) / N
        x_chirp = np.sign(np.cos(phase))
        x_chirp[x_chirp == 0] = 1
        candidates.append(x_chirp)
        
    # Linear
    for freq in [1.0, 2.0, 4.0]:
        phase = 2 * np.pi * freq * t / N
        x_lin = np.sign(np.cos(phase))
        x_lin[x_lin == 0] = 1
        candidates.append(x_lin)
        
    # Random
    rng = np.random.default_rng(123)
    for _ in range(20):
        x_rand = rng.choice([1.0, -1.0], N)
        candidates.append(x_rand)
        
    # 3. Evaluate Candidates
    for x in candidates:
        val = compute_raw_value(x)
        if val < best_val:
            best_val = val
            best_x = x.copy()
            
    # 4. Smooth Surrogate Optimization (Global Phase)
    # Use the best candidate as start
    h_start = best_x.copy()
    s0 = np.sum(h_start)
    if abs(s0) > 1e-6:
        h_norm = h_start / s0
        
        # Multi-stage beta
        for beta in [40, 100, 200, 400]:
            if time.time() > deadline - 40:
                break
            
            try:
                res = minimize(_soft_max_abs_and_grad, h_norm, method='L-BFGS-B', jac=True,
                               options={'maxiter': 200, 'disp': False, 'gtol': 1e-7, 'beta': beta})
                h_opt = res.x
                
                # Binarize and check
                h_bin = np.sign(h_opt)
                h_bin[h_bin == 0] = 1
                val_bin = compute_raw_value(h_bin)
                
                if val_bin < best_val:
                    best_val = val_bin
                    best_x = h_bin.copy()
                    # Update start for next beta
                    h_norm = h_bin / np.sum(h_bin) if np.sum(h_bin) != 0 else h_norm
                    
            except Exception:
                pass

    # 5. Local Search Refinement
    # Step 5a: Peak Targeted Search
    if time.time() < deadline - 20:
        h_work, val_work = _peak_targeted_search(best_x, deadline=deadline - 15)
        if val_work < best_val:
            best_val = val_work
            best_x = h_work.copy()

    # Step 5b: Single Bit Flip Local Search
    if time.time() < deadline - 5:
        h_work, val_work = _local_search_flip_vectorized(best_x, max_passes=5, deadline=deadline - 2)
        if val_work < best_val:
            best_val = val_work
            best_x = h_work.copy()

    # 6. Final Polish with different random restarts if time permits
    if time.time() < deadline - 15:
        for _ in range(3):
            if time.time() > deadline - 5:
                break
            x_rand = rng.choice([1.0, -1.0], N)
            # Quick local search on random start
            h_work, val_work = _local_search_flip_vectorized(x_rand, max_passes=2, deadline=deadline - 3)
            if val_work < best_val:
                best_val = val_work
                best_x = h_work.copy()

    return best_x

if __name__ == "__main__":
    x = construct()
    val = compute_raw_value(x)
    print(f"Best raw value: {val}")
    print(f"Sum: {np.sum(x)}")
