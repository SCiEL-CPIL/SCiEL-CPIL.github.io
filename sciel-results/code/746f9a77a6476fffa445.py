import numpy as np
from scipy import optimize
from scipy.fft import fft, ifft
import time

N = 600

def _fft_convolve(h):
    """FFT-based autoconvolution for speed."""
    n = len(h)
    size = 2 * n
    H = fft(h, size)
    conv = np.real(ifft(H * H, size)[:2*n-1])
    return conv

def compute_autoconv_max(h):
    """Compute the normalized maximum of the autoconvolution of step function h."""
    h = np.asarray(h, dtype=float)
    conv = _fft_convolve(h)
    total = sum(h)
    if total < 1e-10:
        return 0.0
    return max(conv) / (total * total)

def _soft_max_conv(h, T):
    """Smooth approximation of max(autoconv) using log-sum-exp."""
    conv = _fft_convolve(h)
    conv_max = np.max(conv)
    shifted = (conv - conv_max) / T
    shifted = np.clip(shifted, -100, 100)
    return conv_max + T * np.log(np.sum(np.exp(shifted)))

def _fourier_optimize(h_init, max_iter=200, deadline=None):
    """Optimize h via alternating Fourier-space gradient descent and projection.
    
    Key idea: The autoconvolution in frequency domain is |H|^2. We want to
    minimize max of ifft(|H|^2) / (sum(h))^2. We use a smooth surrogate and
    gradient descent in the complex Fourier domain, then project back to
    non-negative real space.
    """
    h = h_init.copy()
    n = len(h)
    
    # Precompute FFT sizes
    size = 2 * n
    
    # Start with a reasonable temperature
    T = 0.05
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        h = np.maximum(h, 0)
        total = np.sum(h)
        if total < 1e-10:
            break
        
        # Compute current score
        score = compute_autoconv_max(h)
        
        # Compute gradient of soft-max objective w.r.t. h
        # f(h) = soft_max_conv(h, T) / total^2
        # We need df/dh
        
        conv = _fft_convolve(h)
        conv_max = np.max(conv)
        shifted = (conv - conv_max) / T
        shifted = np.clip(shifted, -100, 100)
        weights = np.exp(shifted) / np.sum(np.exp(shifted))  # softmax weights
        soft_max_val = conv_max + T * np.log(np.sum(np.exp(shifted)))
        
        # d(soft_max)/d(conv_k) = weights[k]
        # d(conv)/d(h) is the autoconvolution operator (linear)
        # conv = ifft(fft(h) * fft(h))[:2n-1]
        # d(conv_k)/d(h_j) = h_{k-j} + h_{j-k} (with appropriate indexing)
        
        # Use FFT to compute gradient efficiently
        # grad_conv_k = sum_j d(conv_k)/d(h_j) * d(soft_max)/d(conv_k)
        # = sum_k weights[k] * (h_{k-j} + h_{j-k})
        
        # This is a convolution of weights with h
        # Let w = weights, then grad[j] = sum_k w[k] * (h[k-j] + h[j-k])
        # = 2 * (w * h)[j] (circular, but we need linear)
        
        # Compute via FFT
        W = fft(weights, size)
        H = fft(h, size)
        grad_conv = np.real(ifft(W * H, size))[:n]
        grad_conv *= 2  # because of the symmetric structure
        
        # Now d(soft_max)/d(h_j) = grad_conv[j]
        # d(score)/d(h_j) = (grad_conv[j] * total^2 - soft_max_val * 2 * total) / total^4
        # = grad_conv[j] / total^2 - 2 * soft_max_val / (total^3)
        
        grad_score = grad_conv / (total * total) - 2.0 * soft_max_val / (total ** 3)
        
        # Gradient descent with adaptive step size
        # Normalize gradient
        grad_norm = np.linalg.norm(grad_score)
        if grad_norm < 1e-15:
            break
        
        # Adaptive step size
        step = 0.1 / (grad_norm + 1e-10)
        
        # Line search: try a few step sizes
        best_h_trial = h.copy()
        best_score_trial = score
        
        for alpha in [1.0, 0.5, 0.25, 0.1, 0.05]:
            h_trial = h - alpha * step * grad_score
            h_trial = np.maximum(h_trial, 0)
            total_trial = np.sum(h_trial)
            if total_trial < 1e-10:
                continue
            score_trial = compute_autoconv_max(h_trial)
            if score_trial < best_score_trial:
                best_score_trial = score_trial
                best_h_trial = h_trial.copy()
        
        h = best_h_trial
        
        # Decay temperature
        T *= 0.995
        if T < 0.005:
            T = 0.005
    
    return h

def _simulated_annealing(h_init, max_iter=500, deadline=None):
    """Simulated annealing with Gaussian perturbations in Fourier space."""
    h = h_init.copy()
    n = len(h)
    size = 2 * n
    
    best_h = h.copy()
    best_score = compute_autoconv_max(h)
    
    T_start = 0.5
    T_end = 0.01
    T_current = T_start
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Decay temperature
        progress = it / max_iter
        T_current = T_start * (T_end / T_start) ** progress
        
        # Perturb in Fourier space
        H = fft(h, size)
        # Add Gaussian noise to non-DC components
        noise = np.random.normal(0, T_current, size)
        noise[0] = 0  # Keep DC (total sum) roughly fixed
        H_new = H + noise
        h_new = np.real(ifft(H_new, size))[:n]
        h_new = np.maximum(h_new, 0)
        
        total_new = np.sum(h_new)
        if total_new < 1e-10:
            continue
        
        score_new = compute_autoconv_max(h_new)
        
        # Accept or reject
        if score_new < best_score:
            best_score = score_new
            best_h = h_new.copy()
            h = h_new.copy()
        else:
            # Accept with probability exp(-(score_new - best_score) / T_current)
            delta = score_new - best_score
            if delta < 0:
                h = h_new.copy()
            else:
                accept_prob = np.exp(-delta / (T_current + 1e-10))
                if np.random.random() < accept_prob:
                    h = h_new.copy()
    
    return best_h

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for 150s limit
    
    # Multiple initializations and optimization strategies
    
    best_score = np.inf
    best_h = None
    
    # Strategy 1: Fourier gradient descent from different initializations
    np.random.seed(42)
    
    initializations = []
    
    # All ones
    initializations.append(np.ones(N))
    
    # Gaussian-like
    x = np.linspace(-0.25, 0.25, N)
    for sigma in [0.05, 0.1, 0.15]:
        h_gauss = np.exp(-x**2 / (2 * sigma**2))
        initializations.append(h_gauss)
    
    # Cosine window
    h_cos = 0.5 * (1 + np.cos(2 * np.pi * x))
    initializations.append(h_cos)
    
    # Random smooth functions (low-pass filtered noise)
    for seed in [1, 2, 3]:
        np.random.seed(seed)
        h_rand = np.random.rand(N)
        # Low-pass filter via FFT
        H_rand = fft(h_rand, 2*N)
        # Keep only low frequencies
        cutoff = 20
        H_rand[cutoff:-cutoff] = 0
        h_rand = np.real(ifft(H_rand, 2*N))[:N]
        h_rand = np.maximum(h_rand, 0)
        if np.sum(h_rand) > 1e-10:
            initializations.append(h_rand)
    
    # Run Fourier gradient descent on each initialization
    for i, h_init in enumerate(initializations):
        if time.time() > deadline - 30:
            break
        
        h_opt = _fourier_optimize(h_init, max_iter=100, deadline=deadline - 20)
        score_opt = compute_autoconv_max(h_opt)
        
        if score_opt < best_score:
            best_score = score_opt
            best_h = h_opt.copy()
    
    # Strategy 2: Simulated annealing from best found
    if time.time() < deadline - 15 and best_h is not None:
        h_sa = _simulated_annealing(best_h, max_iter=300, deadline=deadline - 10)
        score_sa = compute_autoconv_max(h_sa)
        if score_sa < best_score:
            best_score = score_sa
            best_h = h_sa.copy()
    
    # Final local refinement with L-BFGS-B
    if time.time() < deadline - 10 and best_h is not None:
        def soft_objective(x, T=0.01):
            h = np.maximum(x, 0)
            total = np.sum(h)
            if total < 1e-10:
                return 1e10
            conv = _fft_convolve(h)
            conv_max = np.max(conv)
            shifted = (conv - conv_max) / T
            shifted = np.clip(shifted, -100, 100)
            soft_max = conv_max + T * np.log(np.sum(np.exp(shifted)))
            return soft_max / (total * total)
        
        try:
            res = optimize.minimize(
                lambda x: soft_objective(x), best_h,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            h_refined = np.maximum(res.x, 0)
            score_refined = compute_autoconv_max(h_refined)
            if score_refined < best_score:
                best_score = score_refined
                best_h = h_refined.copy()
        except Exception:
            pass
    
    if best_h is None:
        best_h = np.ones(N)
    
    return list(best_h)
