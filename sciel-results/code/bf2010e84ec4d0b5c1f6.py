"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy: Discrete spectral search with gradient-based optimization in the
frequency domain. We parameterize the step function by its DFT coefficients,
enforce non-negativity via soft-plus projection, and directly minimize the
maximum value of the autoconvolution using projected gradient descent.
The key insight is that the autoconvolution peak is related to the inverse DFT
of the squared magnitude spectrum. By working in the frequency domain, we can
directly control the energy distribution of the autoconvolution.
"""

import numpy as np
import time

N = 600
TIME_LIMIT = 115  # safety margin before 120s kill

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    conv = np.convolve(f, f, mode='full')
    return np.max(conv) / (s * s)

def soft_plus(x, beta=10.0):
    """Numerically stable softplus: log(1 + exp(beta*x))/beta."""
    x = np.asarray(x, dtype=np.float64)
    return np.logaddexp(0, beta * x) / beta

def score_and_grad(F, beta=10.0):
    """
    Compute score and gradient with respect to F (complex DFT coefficients).
    Returns (score, grad_real, grad_imag) where grad is with respect to
    the real and imaginary parts of F.
    
    We use the chain rule:
    f = Re(ifft(F))
    f_proj = soft_plus(f, beta)
    score = max(conv(f_proj)) / (sum(f_proj))^2
    
    The gradient is computed via automatic differentiation through the
    soft-plus projection and the autoconvolution.
    """
    # Inverse DFT to time domain
    f_raw = np.real(np.fft.ifft(F))
    
    # Soft-plus projection
    f_proj = soft_plus(f_raw, beta)
    
    # Compute score
    s = np.sum(f_proj)
    if s < 1e-15:
        return np.inf, np.zeros(N), np.zeros(N)
    
    conv = np.convolve(f_proj, f_proj, mode='full')
    max_idx = np.argmax(conv)
    score = conv[max_idx] / (s * s)
    
    # Compute gradient using finite differences for simplicity
    # This is expensive but robust. We use a small step size.
    eps = 1e-6
    grad_real = np.zeros(N)
    grad_imag = np.zeros(N)
    
    # To save time, we compute gradient only for a subset of coefficients
    # and use random sampling of coordinates
    half = N // 2
    coords = list(range(half + 1))
    
    for k in coords:
        # Perturb real part
        F_pert = F.copy()
        F_pert[k] += eps
        if k != 0 and k != half:
            F_pert[N - k] += eps
        
        f_pert = np.real(np.fft.ifft(F_pert))
        f_pert_proj = soft_plus(f_pert, beta)
        s_pert = np.sum(f_pert_proj)
        if s_pert < 1e-15:
            grad_real[k] = 0
            if k != 0 and k != half:
                grad_real[N - k] = 0
        else:
            conv_pert = np.convolve(f_pert_proj, f_pert_proj, mode='full')
            score_pert = conv_pert[np.argmax(conv_pert)] / (s_pert * s_pert)
            grad_real[k] = (score_pert - score) / eps
            if k != 0 and k != half:
                grad_real[N - k] = grad_real[k]
        
        # Perturb imaginary part
        F_pert = F.copy()
        F_pert[k] += 1j * eps
        if k != 0 and k != half:
            F_pert[N - k] -= 1j * eps
        
        f_pert = np.real(np.fft.ifft(F_pert))
        f_pert_proj = soft_plus(f_pert, beta)
        s_pert = np.sum(f_pert_proj)
        if s_pert < 1e-15:
            grad_imag[k] = 0
            if k != 0 and k != half:
                grad_imag[N - k] = 0
        else:
            conv_pert = np.convolve(f_pert_proj, f_pert_proj, mode='full')
            score_pert = conv_pert[np.argmax(conv_pert)] / (s_pert * s_pert)
            grad_imag[k] = (score_pert - score) / eps
            if k != 0 and k != half:
                grad_imag[N - k] = -grad_imag[k]
    
    return score, grad_real, grad_imag

def optimize_from_F(F_init, start_time, max_time, n_iter=200, lr=0.01, beta=10.0):
    """
    Optimize DFT coefficients using projected gradient descent.
    """
    F = F_init.copy()
    best_F = F.copy()
    best_score = compute_score(np.maximum(np.real(np.fft.ifft(F)), 0.0))
    
    for it in range(n_iter):
        if time.time() - start_time > max_time:
            break
        
        score, grad_real, grad_imag = score_and_grad(F, beta)
        
        if np.isinf(score):
            break
        
        # Update F
        F -= lr * (grad_real + 1j * grad_imag)
        
        # Enforce conjugate symmetry
        half = N // 2
        for k in range(1, half + 1):
            F[N - k] = np.conj(F[k])
        
        # Compute current score
        f_curr = np.maximum(np.real(np.fft.ifft(F)), 0.0)
        curr_score = compute_score(f_curr)
        
        if curr_score < best_score:
            best_score = curr_score
            best_F = F.copy()
        
        # Adaptive learning rate
        if curr_score > best_score * 1.1:
            lr *= 0.5
        else:
            lr *= 1.05
    
    return best_F, best_score

def construct():
    start_time = time.time()
    
    # Initial best: constant function
    best_f = np.ones(N)
    best_score = compute_score(best_f)
    
    rng = np.random.default_rng(42)
    
    # Run multiple restarts with different random phase initializations
    n_restarts = 30
    half = N // 2
    
    for r in range(n_restarts):
        if time.time() - start_time > TIME_LIMIT - 5:
            break
        
        # Create random phase initialization
        F_init = np.zeros(N, dtype=np.complex128)
        F_init[0] = 1.0  # DC term real and positive
        
        # Random phases for other frequencies
        for k in range(1, half + 1):
            phase = rng.uniform(0, 2 * np.pi)
            F_init[k] = np.exp(1j * phase)
            F_init[N - k] = np.conj(F_init[k])
        
        # Optimize
        F_opt, score_opt = optimize_from_F(F_init, start_time, TIME_LIMIT - 5, 
                                            n_iter=100, lr=0.005, beta=10.0)
        
        # Convert back to time domain
        f_opt = np.maximum(np.real(np.fft.ifft(F_opt)), 0.0)
        
        # Compute actual score
        actual_score = compute_score(f_opt)
        
        if actual_score < best_score:
            best_score = actual_score
            best_f = f_opt
    
    # Ensure nonnegative
    best_f = np.maximum(best_f, 0.0)
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
