"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START

def _compute_R_positive(f):
    """Compute R[L] for L=1..N-1 where R[L] = sum_{i=0}^{N-L-1} f[i]*f[i+L]."""
    R = np.zeros(N)
    for L in range(1, N):
        R[L] = np.dot(f[:N-L], f[L:])
    return R

def _peak_from_R(R):
    """Get peak autocorrelation from R array (R[0] unused)."""
    return np.max(np.abs(R[1:]))

def _local_search_incremental(f, time_limit, rng):
    """Local search with incremental autocorrelation updates."""
    n = len(f)
    f = f.copy()
    R = _compute_R_positive(f)
    current_peak = _peak_from_R(R)
    best_peak = current_peak
    best_f = f.copy()
    
    start_time = time.time()
    no_improve = 0
    max_no_improve = 200
    
    while time.time() - start_time < time_limit:
        if no_improve >= max_no_improve:
            break
        
        # Find the lag with maximum absolute autocorrelation
        L_max = np.argmax(np.abs(R[1:])) + 1
        R_max = R[L_max]
        
        # Try flipping each bit and see if it reduces the peak
        improved = False
        # Shuffle indices to avoid bias
        indices = np.arange(n)
        rng.shuffle(indices)
        
        for k in indices:
            if time.time() - start_time > time_limit:
                break
            
            old_val = f[k]
            f[k] = -old_val
            
            # Update R incrementally
            # When we flip f[k], for each lag L:
            # R[L] changes by -2 * old_val * f[k+L] if k+L < n
            # R[L] changes by -2 * old_val * f[k-L] if k-L >= 0
            # But we need to be careful: after flipping, f[k] = -old_val
            
            # The change in R[L] for lag L:
            # Old contribution at position k for lag L: 
            #   if k >= L: old_val * f[k-L] (from f[k-L]*f[k])
            #   if k+L < n: old_val * f[k+L] (from f[k]*f[k+L])
            # New contribution:
            #   if k >= L: -old_val * f[k-L]
            #   if k+L < n: -old_val * f[k+L]
            # Change: -2 * old_val * (f[k-L] if k>=L else 0) - 2 * old_val * (f[k+L] if k+L<n else 0)
            
            # But we can vectorize this update
            # For lag L, the change is:
            # delta_R[L] = -2 * old_val * (f[k-L] if k>=L else 0 + f[k+L] if k+L<n else 0)
            
            # Let's compute this efficiently
            delta = np.zeros(n)
            # For lags L where k-L >= 0, i.e., L <= k
            if k > 0:
                delta[1:k+1] = -2 * old_val * f[k-1::-1]  # f[k-L] for L=1..k
            # For lags L where k+L < n, i.e., L < n-k
            if k + 1 < n:
                delta[1:n-k] += -2 * old_val * f[k+1:]  # f[k+L] for L=1..n-k-1
            
            R += delta
            
            new_peak = _peak_from_R(R)
            
            if new_peak < current_peak - 1e-12:
                current_peak = new_peak
                improved = True
                if current_peak < best_peak:
                    best_peak = current_peak
                    best_f = f.copy()
                no_improve = 0
                break
            else:
                # Revert
                f[k] = old_val
                R -= delta
        
        if not improved:
            no_improve += 1
    
    return best_f, best_peak

def _thue_morse_init(n):
    """Generate Thue-Morse sequence of length n."""
    h = np.zeros(n)
    for i in range(n):
        count = bin(i).count('1')
        h[i] = 1.0 if count % 2 == 0 else -1.0
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    rng = np.random.RandomState(42)
    
    best_h = None
    best_peak = np.inf
    
    # Try multiple initializations
    inits = []
    
    # 1. Thue-Morse
    h_tm = _thue_morse_init(N)
    inits.append(h_tm)
    
    # 2. Random sequences
    for _ in range(5):
        inits.append(rng.choice([-1.0, 1.0], size=N))
    
    # 3. Alternating pattern
    h_alt = np.ones(N)
    h_alt[1::2] = -1.0
    inits.append(h_alt)
    
    # 4. Block pattern: +++++-----+++++-----...
    h_block = np.ones(N)
    block_size = 8
    for i in range(N):
        if (i // block_size) % 2 == 1:
            h_block[i] = -1.0
    inits.append(h_block)
    
    # Evaluate all initializations and run local search on the best few
    init_peaks = []
    for h in inits:
        R = _compute_R_positive(h)
        peak = _peak_from_R(R)
        init_peaks.append((peak, h))
    
    # Sort by peak and take the top 3 for local search
    init_peaks.sort(key=lambda x: x[0])
    
    for peak_init, h_init in init_peaks[:3]:
        if time.time() >= deadline - 5.0:
            break
        
        remaining = deadline - time.time() - 2.0
        if remaining <= 0:
            break
        
        h_opt, peak_opt = _local_search_incremental(h_init, remaining, rng)
        
        if peak_opt < best_peak:
            best_peak = peak_opt
            best_h = h_opt
    
    # Final polish with remaining time
    if best_h is not None and time.time() < deadline - 1.0:
        h_final, peak_final = _local_search_incremental(best_h, deadline - time.time() - 0.5, rng)
        if peak_final < best_peak:
            best_peak = peak_final
            best_h = h_final
    
    if best_h is None:
        best_h = np.ones(N)
        
    return best_h

# EVOLVE-BLOCK-END
