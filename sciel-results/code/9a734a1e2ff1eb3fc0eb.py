"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_acf(s):
    """Compute autocorrelation using FFT."""
    n = len(s)
    f = np.fft.fft(s, n=2*n)
    acf = np.fft.ifft(f * np.conj(f)).real[:n]
    return acf

def _peak_sidelobe(acf):
    """Peak absolute sidelobe (excluding main lobe)."""
    return np.max(np.abs(acf[1:]))

def _eval_flip(s, acf, i):
    """Compute peak sidelobe after flipping position i without full FFT."""
    si = s[i]
    new_acf = acf.copy()
    # For k where i+k < N: new_acf[k] -= 2*si*s[i+k]
    k_max = N - i
    if k_max > 0:
        new_acf[:k_max] -= 2 * si * s[i:i+k_max]
    # For k where i-k >= 0, i.e., k <= i: new_acf[k] -= 2*s[k]*si
    if i > 0:
        new_acf[:i+1] -= 2 * s[:i+1] * si
    return _peak_sidelobe(new_acf)

def _local_search(s, acf, current_val, deadline):
    """Iteratively flip the best single position until no improvement or deadline."""
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        best_flip_val = current_val
        best_flip_idx = -1
        
        for i in range(N):
            new_val = _eval_flip(s, acf, i)
            if new_val < best_flip_val:
                best_flip_val = new_val
                best_flip_idx = i
        
        if best_flip_idx >= 0:
            s[best_flip_idx] = -s[best_flip_idx]
            acf = _compute_acf(s)
            current_val = _peak_sidelobe(acf)
            improved = True
    
    return current_val

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    best_seq = None
    best_val = np.inf
    
    np.random.seed(42)
    
    # Try multiple random restarts with local search
    for restart in range(30):
        if time.time() > deadline - 5:
            break
        
        # Random initialization
        s = np.random.choice([-1.0, 1.0], size=N)
        acf = _compute_acf(s)
        current_val = _peak_sidelobe(acf)
        
        if current_val < best_val:
            best_val = current_val
            best_seq = s.copy()
        
        # Local search to improve
        current_val = _local_search(s, acf, current_val, deadline)
        
        if current_val < best_val:
            best_val = current_val
            best_seq = s.copy()
        
        # Early stop if we've found a very good solution
        if best_val <= 3.0:
            break
    
    if best_seq is None:
        return np.ones(N)
    
    return best_seq
# EVOLVE-BLOCK-END
