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
    
    # Precompute fine grid mapping
    M = 2 * N
    dx = 0.5 / M
    m_indices = np.arange(M)
    i_indices = (m_indices * N // M)
    i_indices = np.clip(i_indices, 0, N - 1)
    
    # Build fine grid signal
    current_step = best_step.copy()
    f_fine = current_step.astype(np.float64)[i_indices]
    
    # Compute initial autoconvolution
    fft_f = np.fft.fft(f_fine, n=2 * M)
    autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real * dx
    current_c3 = np.max(np.abs(autoconv))
    
    # Precompute which fine-grid positions correspond to each coarse index
    # For each coarse index i, find all fine indices m where i_indices[m] == i
    coarse_to_fine = [[] for _ in range(N)]
    for m in range(M):
        coarse_to_fine[i_indices[m]].append(m)
    
    # Convert to arrays for fast access
    coarse_to_fine_arrays = [np.array(coarse_to_fine[i], dtype=np.int64) for i in range(N)]
    
    # Simulated annealing parameters
    T_start = 0.05
    T_end = 0.001
    max_iter = 200000  # More iterations possible with faster evaluation
    
    def _update_autoconv_single_flip(autoconv, f_fine, flip_coarse_idx, old_val, new_val):
        """
        Update autoconvolution after flipping one coarse position.
        Returns new autoconv (copy) and new c3.
        """
        # The change in f_fine is at all fine positions corresponding to flip_coarse_idx
        fine_positions = coarse_to_fine_arrays[flip_coarse_idx]
        delta_f = np.zeros(M, dtype=np.float64)
        delta_f[fine_positions] = new_val - old_val  # = -2*old_val
        
        # The autoconvolution is (f * f)(t) = sum_i f(i) f(i+t)
        # Change: (f + delta_f) * (f + delta_f) = f*f + 2*f*delta_f + delta_f*delta_f
        # f * delta_f is a convolution: (f * delta_f)(t) = sum_i f(i) delta_f(i+t)
        # Since delta_f is nonzero only at fine_positions, this is:
        # sum_{j in fine_positions} delta_f[j] * f[j-t] (with appropriate wrapping)
        
        # Use FFT for the cross term: fft(f) * fft(delta_f)
        fft_delta = np.fft.fft(delta_f, n=2 * M)
        cross_term = np.fft.ifft(fft_f * fft_delta).real * dx
        
        # delta_f * delta_f: since delta_f has constant value d at fine_positions,
        # (delta_f * delta_f)(t) = d^2 * (number of j in fine_positions such that j+t in fine_positions)
        # For simplicity, compute via FFT
        delta_conv = np.fft.ifft(fft_delta * np.conj(fft_delta)).real * dx
        
        new_autoconv = autoconv + 2 * cross_term + delta_conv
        new_c3 = np.max(np.abs(new_autoconv))
        
        return new_autoconv, new_c3
    
    T = T_start
    cooling_rate = (T_end / T_start) ** (1.0 / max_iter)
    
    for iteration in range(max_iter):
        if time.time() - start_time > DEADLINE - 5.0:
            break
        
        # Propose a move: flip 1 random position (faster)
        flip_idx = np.random.randint(N)
        old_val = current_step[flip_idx]
        new_val = -old_val
        
        # Update autoconvolution incrementally
        new_autoconv, new_c3 = _update_autoconv_single_flip(
            autoconv, f_fine, flip_idx, float(old_val), float(new_val)
        )
        
        delta = new_c3 - current_c3
        
        # Accept if better, or with probability exp(-delta/T) if worse
        if delta < 0 or (T > 1e-6 and np.random.random() < np.exp(-delta / T)):
            # Accept the move
            current_step[flip_idx] = new_val
            f_fine[coarse_to_fine_arrays[flip_idx]] = new_val
            autoconv = new_autoconv
            current_c3 = new_c3
            
            if new_c3 < best_c3:
                best_c3 = new_c3
                best_step = current_step.copy()
        else:
            pass
        
        T *= cooling_rate
    
    return best_step
