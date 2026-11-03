import numpy as np
import time

N = 400

def _autoconv_lin(x):
    """
    Compute linear autoconvolution using FFT.
    c[k] = sum_i x[i] x[i-k]
    """
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    fx = np.fft.rfft(x, size)
    return np.fft.irfft(fx * fx, size)[:2 * n - 1]

def _peak_sidelobe(x):
    """
    Compute max |c[k]| for k != 0 / S^2
    """
    s = np.sum(x)
    if abs(s) < 1e-12:
        return float('inf')
    c = _autoconv_lin(x)
    if len(c) > 1:
        return float(np.max(np.abs(c[1:]))) / (s * s)
    return 0.0

def _local_search_fast(x, max_iters=50000, time_budget=20.0):
    """
    Fast local search using incremental autoconvolution updates.
    x is a ±1 array.
    """
    start_time = time.time()
    n = len(x)
    
    # Precompute initial autoconvolution
    c = _autoconv_lin(x).copy()
    s = np.sum(x)
    current_peak = float(np.max(np.abs(c[1:]))) / (s * s) if abs(s) > 1e-12 else float('inf')
    best_x = x.copy()
    best_peak = current_peak
    
    no_improve = 0
    max_no_improve = 300
    
    for iteration in range(max_iters):
        if time.time() - start_time > time_budget:
            break
        
        improved = False
        # Try several random flips
        for _ in range(20):
            idx = np.random.randint(n)
            v = x[idx]
            
            # Compute delta in c for flipping x[idx]
            # c[k] = sum_i x[i] x[i-k]
            # Flipping x[idx]: delta = -2*v
            # Affected c[k]: 
            #   When i = idx: term x[idx]*x[idx-k] changes by -2v*x[idx-k]
            #   When i-k = idx (i = idx+k): term x[idx+k]*x[idx] changes by -2v*x[idx+k]
            # So dc[k] = -2v * (x[idx-k] + x[idx+k]) for valid indices
            
            # c is indexed 0 to 2n-2, where c[j] corresponds to lag j-(n-1)
            # lag = k - (n-1), so k = lag + (n-1)
            # For a given idx, the lags affected are:
            #   lag1 = idx - idx = 0? No...
            #   Let me think in terms of the convolution index.
            #   c[j] = sum_i x[i] x[i - (j - (n-1))] = sum_i x[i] x[i - j + n - 1]
            #   So c[j] involves x[i] and x[i - j + n - 1]
            #   When i = idx: x[idx] and x[idx - j + n - 1]
            #   When i - j + n - 1 = idx => i = idx + j - n + 1: x[idx + j - n + 1] and x[idx]
            
            # So for each j, dc[j] = -2v * (x[idx - j + n - 1] + x[idx + j - n + 1])
            # where we only include terms where indices are valid (0 to n-1)
            
            # Build the delta vector
            dc = np.zeros(2 * n - 1)
            for j in range(2 * n - 1):
                i1 = idx - j + n - 1
                i2 = idx + j - n + 1
                term = 0.0
                if 0 <= i1 < n:
                    term += x[i1]
                if 0 <= i2 < n:
                    term += x[i2]
                if term != 0:
                    dc[j] = -2.0 * v * term
            
            new_c = c + dc
            new_s = s - 2.0 * v
            if abs(new_s) < 1e-12:
                continue
            new_peak = float(np.max(np.abs(new_c[1:]))) / (new_s * new_s)
            
            if new_peak < current_peak:
                # Accept the flip
                x[idx] = -x[idx]
                c = new_c
                s = new_s
                current_peak = new_peak
                improved = True
                no_improve = 0
                if current_peak < best_peak:
                    best_peak = current_peak
                    best_x = x.copy()
                break
        
        if not improved:
            no_improve += 1
            if no_improve > max_no_improve:
                break
    
    return best_x, best_peak

def construct():
    deadline = time.time() + 140.0
    rng = np.random.default_rng(42)
    
    best_x = None
    best_val = float('inf')
    
    # Multiple random restarts with local search
    n_restarts = 0
    while time.time() < deadline - 15.0:
        n_restarts += 1
        
        # Random ±1 initialization
        x_init = rng.choice([-1.0, 1.0], size=N)
        
        # Local search
        x_ls, val_ls = _local_search_fast(x_init, max_iters=30000, time_budget=18.0)
        
        if val_ls < best_val:
            best_val = val_ls
            best_x = x_ls.copy()
        
        # If we've found something good, do a refinement pass
        if best_val < 0.005 and time.time() < deadline - 20.0:
            x_ref, val_ref = _local_search_fast(best_x, max_iters=20000, time_budget=10.0)
            if val_ref < best_val:
                best_val = val_ref
                best_x = x_ref.copy()
        
        if time.time() > deadline - 20.0:
            break
    
    # Fallback
    if best_x is None:
        best_x = np.ones(N)
    
    return best_x
