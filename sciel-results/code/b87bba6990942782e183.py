import numpy as np
import time

N = 400
DEADLINE = 140.0  # Safety margin before 150s kill

def _compute_c3(signed_step):
    """
    Compute the C3 upper bound: max |autoconv(t)| for the signed step function.
    """
    M = 2 * N  # fine grid for accuracy
    dx = 0.5 / M
    
    # Build fine grid
    m_indices = np.arange(M)
    i_indices = (m_indices * N // M)
    i_indices = np.clip(i_indices, 0, N - 1)
    f_fine = signed_step.astype(np.float64)[i_indices]
    
    # Autoconvolution via FFT
    fft_f = np.fft.fft(f_fine, n=2 * M)
    autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real * dx
    
    c3 = np.max(np.abs(autoconv))
    return c3

def _structured_initialization():
    """Generate multiple structured candidates and return the best."""
    candidates = []
    
    # All ones (baseline)
    candidates.append(np.ones(N, dtype=np.int8))
    
    # Alternating
    candidates.append(np.tile(np.array([1, -1], dtype=np.int8), N // 2))
    
    # Random
    np.random.seed(42)
    for _ in range(10):
        candidates.append(np.random.choice([-1, 1], size=N).astype(np.int8))
    
    best = None
    best_c3 = np.inf
    for c in candidates:
        c3 = _compute_c3(c)
        if c3 < best_c3:
            best_c3 = c3
            best = c.copy()
    
    return best, best_c3

def construct():
    start_time = time.time()
    
    # Get a good structured start
    best_step, best_c3 = _structured_initialization()
    
    # Simulated annealing parameters
    T_start = 0.05
    T_end = 0.001
    max_iter = 50000
    
    # Precompute for speed
    M = 2 * N
    dx = 0.5 / M
    m_indices = np.arange(M)
    i_indices = (m_indices * N // M)
    i_indices = np.clip(i_indices, 0, N - 1)
    
    def _fast_c3(signed_step):
        f_fine = signed_step.astype(np.float64)[i_indices]
        fft_f = np.fft.fft(f_fine, n=2 * M)
        autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real * dx
        return np.max(np.abs(autoconv))
    
    current_step = best_step.copy()
    current_c3 = best_c3
    T = T_start
    cooling_rate = (T_end / T_start) ** (1.0 / max_iter)
    
    for iteration in range(max_iter):
        if time.time() - start_time > DEADLINE - 5.0:
            break
        
        # Propose a move: flip 1-3 random positions
        r = np.random.random()
        if r < 0.7:
            num_flips = 1
        elif r < 0.9:
            num_flips = 2
        else:
            num_flips = 3
        
        idxs = np.random.choice(N, size=num_flips, replace=False)
        old_vals = current_step[idxs].copy()
        current_step[idxs] = -current_step[idxs]
        
        new_c3 = _fast_c3(current_step)
        delta = new_c3 - current_c3
        
        # Accept if better, or with probability exp(-delta/T) if worse
        if delta < 0 or (T > 1e-6 and np.random.random() < np.exp(-delta / T)):
            current_c3 = new_c3
            if new_c3 < best_c3:
                best_c3 = new_c3
                best_step = current_step.copy()
        else:
            # Reject: revert
            current_step[idxs] = old_vals
        
        T *= cooling_rate
    
    return best_step
