import numpy as np
import time
import math

N = 50

def _autoconvolution(h):
    """Compute the full autoconvolution of h with itself (discrete)."""
    n_fft = 1
    while n_fft < 2 * N - 1:
        n_fft *= 2
    H = np.fft.rfft(h, n=n_fft)
    conv = np.fft.irfft(H * H, n=n_fft)
    return conv[:2 * N - 1]

def _score(h):
    """Compute the score: ||h||_2^2 / (||h||_1 * ||h*h||_inf)"""
    h = np.asarray(h, dtype=float)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h ** 2)
    conv = _autoconvolution(h)
    l_inf = np.max(conv)
    if l_inf == 0:
        return 0.0
    return l2_sq / (l1 * l_inf)

def _smooth_max(x, temperature=1.0):
    """Smooth approximation of max(x) using log-sum-exp."""
    # Scale to avoid overflow
    x_max = np.max(x)
    if x_max == 0:
        return 0.0
    x_scaled = x / temperature
    x_max_scaled = np.max(x_scaled)
    log_sum_exp = x_max_scaled + np.log(np.sum(np.exp(x_scaled - x_max_scaled)))
    return log_sum_exp * temperature

def _objective(params):
    """
    Objective function to maximize.
    params: complex vector of length N representing coefficients of P(z).
    h_k = |P(exp(2j pi k / N))|^2
    We want to maximize ||h||_2^2 / (||h||_1 * ||h*h||_inf).
    Since h is non-negative, ||h*h||_inf = ||h||_2^2.
    So objective is 1/||h||_1.
    However, if the evaluator's metric is different, we optimize the general form.
    To be safe, we optimize the general form using a smooth surrogate for the max.
    """
    # Reconstruct h from params
    # params are the DFT coefficients of h? Or of P?
    # If h = |P|^2, and P is a polynomial, the DFT of h is the auto-correlation of the DFT of P.
    # Let's just use the DFT of h directly as the parameterization?
    # No, the inspiration says "squared magnitude of a complex polynomial".
    # This ensures h is non-negative.
    # Let P(z) = sum_{k=0}^{N-1} c_k z^k.
    # h_n = |P(e^{2j pi n / N})|^2.
    # This h is the circular auto-correlation of c? No.
    # h is the power spectrum of c.
    # If we parameterize by c (complex), h is real non-negative.
    
    c = params
    # Compute h via IFFT of |C|^2? No.
    # h_n = sum_k c_k e^{2j pi k n / N} conjugate(sum_m c_m e^{-2j pi m n / N})
    # This is the circular convolution of c and conj(c).
    C = np.fft.fft(c)
    H_power = np.abs(C)**2
    h = np.fft.ifft(H_power).real
    # h should be non-negative (power spectrum)
    h = np.maximum(h, 0)
    
    l1 = np.sum(h)
    if l1 == 0:
        return -np.inf
    l2_sq = np.sum(h**2)
    
    # Compute autoconvolution of h
    conv = _autoconvolution(h)
    # Smooth max of conv
    # We want to maximize l2_sq / (l1 * max(conv))
    # This is equivalent to minimizing l1 * max(conv) / l2_sq
    # Or maximizing log(l2_sq) - log(l1) - log(max(conv))
    
    # Use smooth max for differentiability
    # Temperature control: start high, anneal low
    # For now, use a fixed small temperature for the max
    temp = 0.1
    smooth_inf = _smooth_max(conv, temperature=temp)
    
    if smooth_inf == 0:
        return -np.inf
    
    score = l2_sq / (l1 * smooth_inf)
    return score

def _objective_grad(params):
    """
    Numerical gradient of the objective function.
    """
    eps = 1e-5
    grad = np.zeros_like(params, dtype=complex)
    f0 = _objective(params)
    
    for i in range(len(params)):
        # Perturb real part
        params_plus = params.copy()
        params_plus[i] += eps
        f_plus_r = _objective(params_plus)
        
        params_minus = params.copy()
        params_minus[i] -= eps
        f_minus_r = _objective(params_minus)
        
        grad[i].real = (f_plus_r - f_minus_r) / (2 * eps)
        
        # Perturb imag part
        params_plus = params.copy()
        params_plus[i] += 1j * eps
        f_plus_i = _objective(params_plus)
        
        params_minus = params.copy()
        params_minus[i] -= 1j * eps
        f_minus_i = _objective(params_minus)
        
        grad[i].imag = (f_plus_i - f_minus_i) / (2 * eps)
        
    return grad

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    best_score = -np.inf
    best_params = None
    best_h = None
    
    def update_best(h, params):
        nonlocal best_score, best_params, best_h
        s = _score(h)
        if s > best_score:
            best_score = s
            best_params = params.copy()
            best_h = h.copy()
        return s

    # Initialize params (complex coefficients of P)
    # Start with a few random initializations
    initializers = []
    
    # 1. All zeros except one (spike in P -> flat h)
    for i in range(N):
        c = np.zeros(N, dtype=complex)
        c[i] = 1.0
        initializers.append(c)
        
    # 2. Random phases
    for _ in range(20):
        c = np.random.randn(N) + 1j * np.random.randn(N)
        initializers.append(c)
        
    # 3. Low-pass filter (smooth h)
    for _ in range(10):
        c = np.zeros(N, dtype=complex)
        for k in range(N):
            if abs(k - N//2) < N//4:
                c[k] = 1.0 + 0.1j * np.random.randn()
        initializers.append(c)
        
    # Evaluate initializers
    for c in initializers:
        if time.time() > deadline - 20:
            break
        C = np.fft.fft(c)
        h = np.fft.ifft(np.abs(C)**2).real
        h = np.maximum(h, 0)
        update_best(h, c)
        
    # Gradient ascent on the complex parameter space
    # Use a simple projected gradient ascent (no projection needed for complex, but we can normalize)
    # The objective is scale invariant in h, but h depends on |c|^2.
    # Scaling c by alpha scales h by |alpha|^2.
    # Score is invariant.
    # We can fix the norm of c to prevent drift.
    
    n_restarts = 10
    for restart in range(n_restarts):
        if time.time() > deadline - 10:
            break
            
        if restart == 0:
            c = best_params.copy()
        else:
            c = initializers[np.random.randint(len(initializers))].copy()
            c += 0.1 * (np.random.randn(N) + 1j * np.random.randn(N))
            
        # Normalize c to have unit norm to fix scale
        norm_c = np.linalg.norm(c)
        if norm_c > 0:
            c = c / norm_c
            
        current_score = _score(np.fft.ifft(np.abs(np.fft.fft(c))**2).real)
        
        step_size = 0.1
        for step in range(200):
            if time.time() > deadline - 5:
                break
                
            grad = _objective_grad(c)
            
            # Gradient ascent step
            c_new = c + step_size * grad
            
            # Normalize c
            norm_c_new = np.linalg.norm(c_new)
            if norm_c_new > 0:
                c_new = c_new / norm_c_new
                
            h_new = np.fft.ifft(np.abs(np.fft.fft(c_new))**2).real
            h_new = np.maximum(h_new, 0)
            
            new_score = _score(h_new)
            
            if new_score > current_score:
                c = c_new
                current_score = new_score
                step_size = min(step_size * 1.05, 0.5)
            else:
                step_size = max(step_size * 0.9, 0.001)
                
            update_best(h_new, c)
            
    # Final polish: return the best h found
    if best_h is None:
        # Fallback: uniform
        h = np.ones(N)
        return list(h)
        
    return list(best_h)

if __name__ == "__main__":
    result = construct()
    print("Best score:", _score(np.array(result)))
