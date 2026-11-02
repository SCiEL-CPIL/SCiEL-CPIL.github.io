"""Combined approach: discrete hypercube local search with FFT-accelerated peak tracking."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

START_TIME = time.time()
TIME_LIMIT = 115
SAFETY_MARGIN = 5

def _check_deadline():
    return (time.time() - START_TIME) > (TIME_LIMIT - SAFETY_MARGIN)

def _autoconv_fft(x):
    """Compute linear autoconvolution using FFT."""
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    X = np.fft.rfft(x, n=L_fft)
    C = np.fft.irfft(X * X, n=L_fft)[:L]
    return C

def _objective(x):
    """Compute max(|autoconv|) / (sum x)^2."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    ac = _autoconv_fft(x)
    return np.max(np.abs(ac)) / (s * s)

def construct():
    best_x = np.ones(N)
    best_val = _objective(best_x)
    
    if _check_deadline():
        return best_x
    
    # Phase 1: Generate diverse initializations on the hypercube {+1, -1}^N
    rng = np.random.default_rng(42)
    initializations = []
    
    # All ones
    initializations.append(np.ones(N))
    
    # Alternating patterns
    for period in [2, 3, 4, 5, 7, 10, 13, 17, 20, 25, 30, 40, 50]:
        pattern = np.ones(N)
        for i in range(N):
            if (i % period) >= period // 2:
                pattern[i] = -1
        initializations.append(pattern)
    
    # Random hypercube points
    for _ in range(30):
        initializations.append(rng.choice([-1.0, 1.0], size=N))
    
    # Structured patterns: blocks of +1 and -1
    for block_size in [4, 8, 16, 32, 64]:
        pattern = np.ones(N)
        for start in range(0, N, block_size * 2):
            pattern[start:start+block_size] = -1
        initializations.append(pattern)
    
    # Evaluate all initializations
    for x0 in initializations:
        if _check_deadline():
            break
        val = _objective(x0)
        if val < best_val:
            best_val = val
            best_x = x0.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 2: Discrete local search - flip single bits to reduce peak autoconv
    # Key insight: flipping x[j] changes autoconv[k] by -2*x[j]*x[j+k-(N-1)] 
    # We track the current peak and try to reduce it
    
    x = best_x.copy()
    current_val = best_val
    
    # Precompute FFT length for fast autoconv
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    
    max_iter = 20000
    stall_counter = 0
    max_stall = 500
    
    for iteration in range(max_iter):
        if _check_deadline():
            break
        
        # Compute current autoconv and find peak
        ac = _autoconv_fft(x)
        abs_ac = np.abs(ac)
        max_idx = np.argmax(abs_ac)
        max_val_ac = abs_ac[max_idx]
        
        # The shift corresponding to max_idx in full convolution
        shift = max_idx - (N - 1)
        
        # Try flipping each bit and see which most reduces the peak
        # For efficiency, we'll sample a subset of bits if N is large
        # But N=400 is manageable for O(N) per iteration
        
        best_flip_idx = -1
        best_flip_val = current_val
        
        # Sample up to 50 random bits to flip (for speed)
        if N <= 100:
            candidates = np.arange(N)
        else:
            candidates = rng.choice(N, size=min(50, N), replace=False)
        
        for j in candidates:
            # Flipping x[j] means x[j] -> -x[j]
            # Change in autoconv at lag k: delta[k] = -2 * x[j] * x[j+k-(N-1)] (with bounds)
            # More precisely, for full conv index k, autoconv[k] = sum_i x[i]*x[k-i]
            # Flipping x[j]: new_autoconv[k] = old_autoconv[k] - 2*x[j]*x[k-j] (if k-j in [0,N-1])
            # But we need to be careful with the convolution definition
            
            # Simpler: just compute the new objective for this flip
            x_trial = x.copy()
            x_trial[j] *= -1
            trial_val = _objective(x_trial)
            
            if trial_val < best_flip_val:
                best_flip_val = trial_val
                best_flip_idx = j
        
        if best_flip_idx >= 0 and best_flip_val < current_val:
            x[best_flip_idx] *= -1
            current_val = best_flip_val
            stall_counter = 0
            
            if current_val < best_val:
                best_val = current_val
                best_x = x.copy()
        else:
            stall_counter += 1
            if stall_counter > max_stall:
                # Restart from a random point or best_x with perturbation
                if rng.random() < 0.3:
                    # Perturb best_x
                    x = best_x.copy()
                    n_flips = rng.integers(1, 5)
                    flip_indices = rng.choice(N, size=n_flips, replace=False)
                    x[flip_indices] *= -1
                else:
                    # Random restart
                    x = rng.choice([-1.0, 1.0], size=N)
                    current_val = _objective(x)
                    if current_val < best_val:
                        best_val = current_val
                        best_x = x.copy()
                stall_counter = 0
    
    # Phase 3: Final polish with single-bit flips
    if not _check_deadline():
        x = best_x.copy()
        current_val = best_val
        for _ in range(5000):
            if _check_deadline():
                break
            
            j = rng.integers(0, N)
            x_trial = x.copy()
            x_trial[j] *= -1
            trial_val = _objective(x_trial)
            
            if trial_val < current_val:
                x = x_trial
                current_val = trial_val
                if current_val < best_val:
                    best_val = current_val
                    best_x = x.copy()
    
    return best_x
# EVOLVE-BLOCK-END
