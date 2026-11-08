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
    
    # Try diverse random initializations to find a good starting point for SA
    init_candidates = []
    init_candidates.append(np.ones(N, dtype=np.int8))
    
    # Random blocks
    for _ in range(5):
        signs = np.random.choice([-1, 1], size=N//10).astype(np.int8)
        block_step = np.repeat(signs, 10).astype(np.int8)
        init_candidates.append(block_step)
        
    # Pure random
    for _ in range(5):
        init_candidates.append(np.random.choice([-1, 1], size=N).astype(np.int8))
        
    best_init_step = init_candidates[0]
    best_init_c3 = _compute_c3(best_init_step)
    for step in init_candidates[1:]:
        c3_val = _compute_c3(step)
        if c3_val < best_init_c3:
            best_init_c3 = c3_val
            best_init_step = step.copy()
            
    if best_init_c3 < best_c3:
        best_c3 = best_init_c3
        best_step = best_init_step.copy()

    # Precompute negated basis for fast updates
    _neg_basis_fft = -_basis_fft
    
    def run_sa_chain(init_step, time_limit):
        current_step = init_step.copy()
        f_fine = current_step[_FINE_TO_COARSE].astype(np.float64)
        current_fft = np.fft.fft(f_fine, n=2 * M)
        current_c3 = _compute_c3_from_fft(current_fft)
        
        # SA parameters
        T0 = 0.05 
        T_min = 1e-8
        cooling_rate = 0.9999
        T = T0
        
        best_step_local = current_step.copy()
        best_c3_local = current_c3
        
        for iteration in range(1000000): 
            if iteration % 1000 == 0:
                if time.time() > DEADLINE - 5.0:
                    break
                if time.time() > time_limit:
                    break
            
            idx = np.random.randint(N)
            old_val = current_step[idx]
            
            current_step[idx] = -old_val
            current_fft += 2.0 * old_val * _neg_basis_fft[idx]
            
            new_c3 = _compute_c3_from_fft(current_fft)
            delta = new_c3 - current_c3
            
            if delta < 0:
                current_c3 = new_c3
                if new_c3 < best_c3_local:
                    best_c3_local = new_c3
                    best_step_local = current_step.copy()
            else:
                if T > 1e-10:
                    if delta / T < 20.0:
                        if np.random.random() < math.exp(-delta / T):
                            current_c3 = new_c3
                        else:
                            current_step[idx] = old_val
                            current_fft += 2.0 * old_val * _basis_fft[idx]
                    else:
                        current_step[idx] = old_val
                        current_fft += 2.0 * old_val * _basis_fft[idx]
                else:
                    current_step[idx] = old_val
                    current_fft += 2.0 * old_val * _basis_fft[idx]
            
            T *= cooling_rate
            
        return best_step_local, best_c3_local

    sa_deadline = DEADLINE - 10.0
    
    # Run 1: From best init
    step1, c3_1 = run_sa_chain(best_init_step, start_time + 20.0)
    if c3_1 < best_c3:
        best_c3 = c3_1
        best_step = step1.copy()
        
    # Run 2-5: From random inits
    while time.time() < sa_deadline:
        random_step = np.random.choice([-1, 1], size=N).astype(np.int8)
        step_r, c3_r = run_sa_chain(random_step, time.time() + 5.0)
        if c3_r < best_c3:
            best_c3 = c3_r
            best_step = step_r.copy()

    # Final Polish: Local search on the best step found
    if time.time() < DEADLINE - 1.0:
        polish_step = best_step.copy()
        f_fine = polish_step[_FINE_TO_COARSE].astype(np.float64)
        polish_fft = np.fft.fft(f_fine, n=2 * M)
        polish_c3 = _compute_c3_from_fft(polish_fft)
        
        for _ in range(5):
            if time.time() > DEADLINE - 0.5:
                break
            improved = False
            for idx in range(N):
                if time.time() > DEADLINE - 0.5:
                    break
                old_val = polish_step[idx]
                polish_step[idx] = -old_val
                polish_fft += 2.0 * old_val * _neg_basis_fft[idx]
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
