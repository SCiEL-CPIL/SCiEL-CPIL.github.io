import numpy as np
import time

N = 400
DEADLINE = 145.0

def _compute_c3(signed_step):
    """
    Compute the C3 upper bound for a signed step function.
    C3 = max_t |autoconv(t)| where autoconv is the convolution of f with itself.
    """
    M = 2 * N  # fine grid points
    dx = 0.5 / M
    
    # Vectorized construction of fine grid
    m_indices = np.arange(M)
    i_indices = (m_indices * N // M)
    i_indices = np.clip(i_indices, 0, N-1)
    f_fine = signed_step[i_indices]
    
    # Autoconvolution via FFT
    fft_f = np.fft.fft(f_fine, n=2*M)
    autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real * dx
    
    c3 = np.max(np.abs(autoconv))
    
    return c3

def _compute_c3_soft(signed_step, beta=50.0):
    """
    Compute a smooth surrogate of C3 using log-sum-exp.
    This is differentiable and helps gradient-based optimization.
    """
    M = 2 * N
    dx = 0.5 / M
    
    m_indices = np.arange(M)
    i_indices = (m_indices * N // M)
    i_indices = np.clip(i_indices, 0, N-1)
    f_fine = signed_step[i_indices]
    
    fft_f = np.fft.fft(f_fine, n=2*M)
    autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real * dx
    
    # Soft max of |autoconv|
    abs_autoconv = np.abs(autoconv)
    scaled = beta * abs_autoconv
    max_val = np.max(scaled)
    log_sum_exp = max_val + np.log(np.sum(np.exp(scaled - max_val)))
    soft_c3 = log_sum_exp / beta
    
    return soft_c3

def _soft_c3_from_autoconv(autoconv, beta=50.0):
    """Compute soft C3 from precomputed autoconvolution."""
    abs_autoconv = np.abs(autoconv)
    scaled = beta * abs_autoconv
    max_val = np.max(scaled)
    log_sum_exp = max_val + np.log(np.sum(np.exp(scaled - max_val)))
    return log_sum_exp / beta

def _compute_autoconv(signed_step):
    """Compute autoconvolution for a signed step function."""
    M = 2 * N
    dx = 0.5 / M
    
    m_indices = np.arange(M)
    i_indices = (m_indices * N // M)
    i_indices = np.clip(i_indices, 0, N-1)
    f_fine = signed_step[i_indices]
    
    fft_f = np.fft.fft(f_fine, n=2*M)
    autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real * dx
    
    return autoconv

def _gradient_soft_c3(signed_step, beta=50.0):
    """
    Compute gradient of soft C3 with respect to signed_step values.
    Uses finite differences or analytic gradient.
    """
    # For simplicity, use finite difference with small step
    eps = 1e-5
    grad = np.zeros(N)
    
    base_autoconv = _compute_autoconv(signed_step)
    base_soft = _soft_c3_from_autoconv(base_autoconv, beta)
    
    for i in range(N):
        # Forward difference
        perturbed = signed_step.copy()
        perturbed[i] += eps
        perturbed_autoconv = _compute_autoconv(perturbed)
        perturbed_soft = _soft_c3_from_autoconv(perturbed_autoconv, beta)
        grad[i] = (perturbed_soft - base_soft) / eps
    
    return grad

def construct():
    start_time = time.time()
    
    # Initialize with all ones (the seed)
    best_step = np.ones(N, dtype=np.int8)
    best_c3 = _compute_c3(best_step)
    
    np.random.seed(42)
    
    # Try a few random initializations
    for trial in range(10):
        if time.time() - start_time > DEADLINE - 10.0:
            break
        random_step = np.random.choice([-1, 1], size=N).astype(np.int8)
        c3_val = _compute_c3(random_step)
        if c3_val < best_c3:
            best_step = random_step.copy()
            best_c3 = c3_val
    
    # Local search with single and multi-flip moves
    max_iterations = 100000
    no_improve_count = 0
    patience = 2000
    
    for iteration in range(max_iterations):
        if time.time() - start_time > DEADLINE - 5.0:
            break
        
        r = np.random.random()
        if r < 0.8:
            num_flips = 1
        elif r < 0.95:
            num_flips = 2
        else:
            num_flips = 3
        
        if num_flips == 1:
            idxs = [np.random.randint(N)]
        elif num_flips == 2:
            idxs = list(np.random.choice(N, size=2, replace=False))
        else:
            idxs = list(np.random.choice(N, size=3, replace=False))
        
        old_vals = best_step[idxs].copy()
        
        for idx in idxs:
            best_step[idx] = -best_step[idx]
        
        new_c3 = _compute_c3(best_step)
        
        if new_c3 < best_c3:
            best_c3 = new_c3
            no_improve_count = 0
        else:
            for i, idx in enumerate(idxs):
                best_step[idx] = old_vals[i]
            no_improve_count += 1
        
        if no_improve_count >= patience:
            if time.time() - start_time > DEADLINE - 10.0:
                break
            
            # Use gradient-based local search with soft surrogate
            # Convert to continuous variables in [-1, 1]
            x = best_step.astype(np.float64)
            
            # Perform several gradient steps
            for grad_iter in range(50):
                if time.time() - start_time > DEADLINE - 5.0:
                    break
                
                grad = _gradient_soft_c3(x, beta=50.0)
                step_size = 0.1 / (1 + 0.01 * grad_iter)
                x -= step_size * grad
                
                # Project back to [-1, 1]
                x = np.clip(x, -1, 1)
                
                # Check if we should update best
                # Convert to signed step for evaluation
                signed_candidate = np.sign(x)
                signed_candidate[signed_candidate == 0] = 1
                c3_candidate = _compute_c3(signed_candidate)
                
                if c3_candidate < best_c3:
                    best_step = signed_candidate.astype(np.int8)
                    best_c3 = c3_candidate
                    x = best_step.astype(np.float64)
                    no_improve_count = 0
            
            no_improve_count = 0
    
    return best_step
