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
# basis_fft[k] = FFT of the fine-grid vector that is 1 on interval k and 0 elsewhere
_basis_fft = np.zeros((N, 2 * M), dtype=np.complex128)
for k in range(N):
    vec = np.zeros(M, dtype=np.float64)
    mask = (_FINE_TO_COARSE == k)
    vec[mask] = 1.0
    _basis_fft[k] = np.fft.fft(vec, n=2 * M)

# Precompute the full FFT for the all-ones vector
_all_ones_fft = np.sum(_basis_fft, axis=0)

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
    autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real * dx
    return np.max(np.abs(autoconv))

def construct():
    start_time = time.time()
    np.random.seed(42)
    
    # Initialize with all ones (the seed)
    best_step = np.ones(N, dtype=np.int8)
    best_c3 = _compute_c3(best_step)
    
    # Try more diverse random initializations
    for trial in range(20):
        if time.time() - start_time > DEADLINE - 30.0:
            break
        random_step = np.random.choice([-1, 1], size=N).astype(np.int8)
        c3_val = _compute_c3(random_step)
        if c3_val < best_c3:
            best_step = random_step.copy()
            best_c3 = c3_val
    
    # Simulated annealing with fast incremental FFT updates
    current_step = best_step.copy()
    # Compute initial FFT for current state
    f_fine = current_step[_FINE_TO_COARSE].astype(np.float64)
    current_fft = np.fft.fft(f_fine, n=2 * M)
    current_c3 = _compute_c3_from_fft(current_fft)
    
    # SA parameters
    T0 = 0.02  # Lower initial temperature for more exploitation
    T_min = 1e-7
    cooling_rate = 0.9998  # Slower cooling for longer exploration
    T = T0
    
    elapsed_limit = DEADLINE - 5.0  # Leave 5s for final polish
    sa_start = time.time()
    
    # Precompute negated basis for fast updates
    _neg_basis_fft = -_basis_fft  # precompute once
    
    for iteration in range(500000):
        # Time check every 1000 iterations to reduce overhead
        if iteration % 1000 == 0:
            if time.time() - sa_start > elapsed_limit - sa_start:
                break
            if time.time() - start_time > DEADLINE - 5.0:
                break
        
        # Choose move: single flip (primary) or occasional double flip
        if np.random.random() < 0.9:
            idx = np.random.randint(N)
            idxs = [idx]
        else:
            idxs = list(np.random.choice(N, size=2, replace=False))
        
        # Apply move using incremental FFT update
        old_vals = current_step[idxs].copy()
        for idx in idxs:
            current_step[idx] = -current_step[idx]
            # Update FFT: flipping sign of interval k means adding 2*(-basis_fft[k])
            # since new = old - 2*old_val*basis, and old_val was ±1
            current_fft += 2.0 * old_vals[idxs.index(idx)] * _neg_basis_fft[idx]
        
        new_c3 = _compute_c3_from_fft(current_fft)
        delta = new_c3 - current_c3
        
        # Acceptance criterion
        if delta < 0:
            # Always accept improvements
            current_c3 = new_c3
            if new_c3 < best_c3:
                best_c3 = new_c3
                best_step = current_step.copy()
        else:
            # Metropolis acceptance: accept with probability exp(-delta/T)
            if T > 1e-10:
                accept_prob = math.exp(-delta / T)
                if np.random.random() < accept_prob:
                    current_c3 = new_c3
                else:
                    # Reject: revert
                    for i, idx in enumerate(idxs):
                        current_step[idx] = old_vals[i]
                        current_fft += 2.0 * old_vals[i] * _basis_fft[idx]
            else:
                # Reject: revert
                for i, idx in enumerate(idxs):
                    current_step[idx] = old_vals[i]
                    current_fft += 2.0 * old_vals[i] * _basis_fft[idx]
        
        # Cool down
        T *= cooling_rate
    
    # Final greedy polish: try each single flip in sequence
    if time.time() - start_time < DEADLINE - 1.0:
        polish_step = best_step.copy()
        f_fine = polish_step[_FINE_TO_COARSE].astype(np.float64)
        polish_fft = np.fft.fft(f_fine, n=2 * M)
        polish_c3 = _compute_c3_from_fft(polish_fft)
        for idx in range(N):
            if time.time() - start_time > DEADLINE - 0.5:
                break
            old_val = polish_step[idx]
            polish_step[idx] = -old_val
            polish_fft += 2.0 * old_val * _neg_basis_fft[idx]
            new_c3 = _compute_c3_from_fft(polish_fft)
            if new_c3 < polish_c3:
                polish_c3 = new_c3
            else:
                polish_step[idx] = old_val
                polish_fft += 2.0 * old_val * _basis_fft[idx]
        best_step = polish_step
        best_c3 = polish_c3
    
    return best_step
