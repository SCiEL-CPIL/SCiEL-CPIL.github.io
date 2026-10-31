import numpy as np
from scipy.signal import windows
import time

N = 400

def _get_max_sidelobe(f):
    """Compute the maximum absolute value of the autoconvolution sidelobes."""
    # Use FFT for efficient autoconvolution
    f_fft = np.fft.fft(f, n=2*N)
    ac = np.fft.ifft(f_fft * f_fft).real
    # The peak is at index N-1 (for zero-based indexing of full convolution of length N)
    center = N - 1
    sidelobes = np.delete(ac, center)
    return np.max(np.abs(sidelobes))

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_f = None
    best_val = float('inf')
    
    # Strategy 1: Search Kaiser windows with varying beta
    for beta in np.arange(0, 30, 0.5):
        if time.time() > deadline:
            break
        f = windows.kaiser(N, beta)
        norm = np.linalg.norm(f)
        if norm == 0:
            continue
        f = f / norm
        
        val = _get_max_sidelobe(f)
        if val < best_val:
            best_val = val
            best_f = f.copy()
    
    # Strategy 2: Try other window types
    window_types = ['hamming', 'hann', 'blackman', 'blackmanharris', 'nuttall']
    for wtype in window_types:
        if time.time() > deadline:
            break
        try:
            f = windows.get_window(wtype, N)
            norm = np.linalg.norm(f)
            if norm == 0:
                continue
            f = f / norm
            
            val = _get_max_sidelobe(f)
            if val < best_val:
                best_val = val
                best_f = f.copy()
        except:
            pass
    
    # Strategy 3: Generate smooth band-limited signals via sum of sines
    # This creates smooth, oscillating height profiles
    np.random.seed(42)
    for trial in range(50):
        if time.time() > deadline:
            break
        
        # Generate a random sum of low-frequency sine waves
        n_harmonics = np.random.randint(3, 15)
        freqs = np.random.uniform(0.5, 5.0, size=n_harmonics)
        phases = np.random.uniform(0, 2*np.pi, size=n_harmonics)
        amps = np.random.uniform(0.1, 1.0, size=n_harmonics)
        
        # Sample at N points
        x = np.linspace(0, 2*np.pi, N)
        f = np.zeros(N)
        for k in range(n_harmonics):
            f += amps[k] * np.sin(freqs[k] * x + phases[k])
        
        # Ensure it's a "signed" step function - we can keep it continuous
        # but normalize it
        norm = np.linalg.norm(f)
        if norm == 0:
            continue
        f = f / norm
        
        val = _get_max_sidelobe(f)
        if val < best_val:
            best_val = val
            best_f = f.copy()
    
    # Strategy 4: Local optimization around best Kaiser window
    if best_f is not None:
        # Try small perturbations of the best window found
        for _ in range(20):
            if time.time() > deadline:
                break
            
            # Create a perturbed version
            f_pert = best_f.copy()
            # Add small random noise
            noise = np.random.normal(0, 0.01, size=N)
            f_pert += noise
            
            # Re-normalize
            norm = np.linalg.norm(f_pert)
            if norm == 0:
                continue
            f_pert = f_pert / norm
            
            val = _get_max_sidelobe(f_pert)
            if val < best_val:
                best_val = val
                best_f = f_pert.copy()
    
    # Fallback
    if best_f is None:
        return np.ones(N)
    
    return best_f
