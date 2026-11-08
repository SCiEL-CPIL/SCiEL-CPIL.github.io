import numpy as np
import time
import math

N = 400
DEADLINE = 145.0
M = 2 * N  # Fine grid resolution (2x N is sufficient for step function)
dx = 0.5 / M

# Precompute the index mapping from fine grid to coarse intervals
_FINE_TO_COARSE = np.clip(np.arange(M) * N // M, 0, N - 1)

# Precompute FFT of each basis vector for fast incremental updates
_basis_fft = np.zeros((N, 2 * M), dtype=np.complex128)
for k in range(N):
    vec = np.zeros(M, dtype=np.float64)
    mask = (_FINE_TO_COARSE == k)
    vec[mask] = 1.0
    _basis_fft[k] = np.fft.fft(vec, n=2 * M)

# Precompute negated basis for fast updates
_neg_basis_fft = -_basis_fft

def _compute_c3(signed_step):
    """
    Compute the C3 upper bound for a signed step function.
    C3 = max_t |autoconv(t)| where autoconv is the convolution of f with itself.
    """
    f_fine = signed_step[_FINE_TO_COARSE].astype(np.float64)
    fft_f = np.fft.fft(f_fine, n=2 * M)
    autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real * dx
    c3 = np.max(np.abs(autoconv))
    return c3

def _compute_c3_from_fft(fft_f):
    """Compute C3 directly from an FFT array."""
    power = fft_f * np.conj(fft_f)
    autoconv = np.fft.ifft(power).real * dx
    return np.max(np.abs(autoconv))

def construct():
    start_time = time.time()
    np.random.seed(42)
    
    # Initialize with all ones (the seed)
    best_step = np.ones(N, dtype=np.int8)
    best_c3 = _compute_c3(best_step)
    
    # Try diverse random initializations
    init_candidates = [best_step]
    for _ in range(10):
        if time.time() - start_time > DEADLINE - 35.0:
            break
        random_step = np.random.choice([-1, 1], size=N).astype(np.int8)
        init_candidates.append(random_step)
    
    for step in init_candidates:
        c3_val = _compute_c3(step)
        if c3_val < best_c3:
            best_step = step.copy()
            best_c3 = c3_val
    
    # Simulated annealing with fast incremental FFT updates
    current_step = best_step.copy()
    f_fine = current_step[_FINE_TO_COARSE].astype(np.float64)
    current_fft = np.fft.fft(f_fine, n=2 * M)
    current_c3 = _compute_c3_from_fft(current_fft)
    
    # SA parameters
    T0 = 0.05
    T_min = 1e-8
    cooling_rate = 0.99995
    T = T0
    
    sa_start = time.time()
    
    for iteration in range(1000000):
        # Time check
        if iteration % 500 == 0:
            if time.time() - start_time > DEADLINE - 8.0:
                break
        
        # Choose move: single flip
        idx = np.random.randint(N)
        
        # Apply move using incremental FFT update
        old_val = current_step[idx]
        current_step[idx] = -old_val
        # Flipping sign: new_fft = old_fft - 2*old_val*basis_fft[idx]
        current_fft -= 2.0 * old_val * _basis_fft[idx]
        
        new_c3 = _compute_c3_from_fft(current_fft)
        delta = new_c3 - current_c3
        
        # Acceptance criterion
        if delta < 0:
            current_c3 = new_c3
            if new_c3 < best_c3:
                best_c3 = new_c3
                best_step = current_step.copy()
        else:
            # Metropolis acceptance
            if T > T_min:
                accept_prob = math.exp(-delta / T)
                if np.random.random() < accept_prob:
                    current_c3 = new_c3
                else:
                    # Reject: revert
                    current_step[idx] = old_val
                    current_fft += 2.0 * old_val * _basis_fft[idx]
            else:
                # Reject: revert
                current_step[idx] = old_val
                current_fft += 2.0 * old_val * _basis_fft[idx]
        
        # Cool down
        T *= cooling_rate
    
    # Final greedy polish: multiple passes
    if time.time() - start_time < DEADLINE - 2.0:
        polish_step = best_step.copy()
        f_fine = polish_step[_FINE_TO_COARSE].astype(np.float64)
        polish_fft = np.fft.fft(f_fine, n=2 * M)
        polish_c3 = _compute_c3_from_fft(polish_fft)
        
        for polish_pass in range(3):
            if time.time() - start_time > DEADLINE - 1.0:
                break
            improved = False
            for idx in range(N):
                if time.time() - start_time > DEADLINE - 0.5:
                    break
                old_val = polish_step[idx]
                polish_step[idx] = -old_val
                polish_fft -= 2.0 * old_val * _basis_fft[idx]
                new_c3 = _compute_c3_from_fft(polish_fft)
                if new_c3 < polish_c3:
                    polish_c3 = new_c3
                    improved = True
                else:
                    polish_step[idx] = old_val
                    polish_fft += 2.0 * old_val * _basis_fft[idx]
            if not improved:
                break
        
        best_step = polish_step
        best_c3 = polish_c3
    
    return best_step
