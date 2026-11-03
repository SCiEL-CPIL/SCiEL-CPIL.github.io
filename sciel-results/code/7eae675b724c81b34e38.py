import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import logsumexp

def _autoconv(f):
    """Compute autoconvolution (convolution f*f) using FFT."""
    n = len(f)
    F_full = np.fft.fft(f, n=2*n)
    conv_full = np.fft.ifft(F_full * F_full)[:2*n-1].real
    return conv_full

def _compute_score(f):
    """Compute the normalized max of autoconvolution."""
    n = len(f)
    f = np.asarray(f, dtype=np.float64)
    if np.all(f == 0):
        return 0.0
    conv = _autoconv(f)
    max_conv = np.max(conv)
    sum_f = np.sum(f)
    norm = sum_f * sum_f
    if norm == 0:
        return 0.0
    return max_conv / norm

def construct():
    n = 600
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    def time_remaining():
        return deadline - time.time()

    # Softmax parameterization: f_i = exp(u_i) / sum(exp(u_j))
    # This ensures f_i >= 0 and sum(f) = 1.
    # We minimize max(f*f).
    
    # Temperature for soft max approximation of the max of convolution
    # Lower temperature -> sharper max approximation, but less smooth gradient
    T = 0.01 
    
    # Initial guess: Gaussian bump in time domain
    x = np.linspace(-0.25, 0.25, n, endpoint=False)
    f_init = np.exp(-x**2 / (2 * 0.08**2))
    f_init = f_init / np.sum(f_init)
    
    # Convert f to u: u_i = log(f_i) + C. 
    # Since softmax is shift-invariant, we can set sum(u)=0 or just use log(f) directly.
    # To avoid log(0), add small epsilon.
    u_init = np.log(f_init + 1e-10)
    # Center u to prevent overflow in exp
    u_init = u_init - np.mean(u_init)

    def softmax(u):
        # Numerically stable softmax
        u_max = np.max(u)
        exp_u = np.exp(u - u_max)
        return exp_u / np.sum(exp_u)

    def objective_and_grad(u):
        f = softmax(u)
        conv = _autoconv(f)
        
        # Soft max of conv: log(sum(exp(conv/T)))
        # We want to minimize this.
        # Note: conv can be negative? No, f>=0 => conv >= 0.
        # But for stability, we subtract max.
        conv_max = np.max(conv)
        log_sum_exp_val = logsumexp(conv / T)
        
        # Value of the soft max: T * log_sum_exp_val
        # Actually, we want max(conv). 
        # soft_max = T * log(sum(exp(conv/T)))
        val = T * log_sum_exp_val
        
        # Gradient of soft_max w.r.t conv: (1/T) * exp(conv/T) / sum(exp(conv/T))
        # = exp((conv - conv_max)/T) / sum(exp((conv - conv_max)/T))
        # Let's compute weights
        conv_shifted = (conv - conv_max) / T
        exp_conv = np.exp(conv_shifted)
        weights = exp_conv / np.sum(exp_conv)
        
        # Gradient of val w.r.t conv is just weights (since d/dx (T log sum exp(x/T)) = exp(x/T)/sum...)
        # Wait, d/dx [ T * log(sum(exp(y/T))) ] where y=conv.
        # d/dy log(sum(exp(y/T))) = (1/T) exp(y/T) / sum(exp(y/T)).
        # So d val / d conv = weights.
        
        # Now backprop through convolution.
        # conv[k] = sum_i f[i] f[k-i]
        # d conv[k] / d f[j] = sum_i (delta_{j,i} f[k-i] + f[j] delta_{j,k-i}) 
        # This is effectively the convolution of f with itself, but shifted?
        # Actually, d (f*f) / d f = f * f (convolution) ? No.
        # Let's use FFT for gradient.
        # Conv = ifft(fft(f)^2)
        # d Conv / d f in frequency domain:
        # F = fft(f)
        # C = ifft(F^2)
        # d C / d f: 
        # In frequency domain, C_freq = F^2.
        # d C_freq[k] / d F[m] = 2 F[m] if k=m, else 0? No, F^2 is pointwise.
        # d (F^2) / d F = 2F.
        # So grad_C_freq = 2 * F * grad_C (in freq domain? No, gradient is in time domain).
        
        # Let's do it in time domain via FFT.
        # grad_f = ifft( 2 * fft(f) * fft(grad_conv) ) ?
        # Let's verify.
        # J = sum_k grad_conv[k] * conv[k]
        # conv[k] = sum_i f[i] f[k-i]
        # dJ/df[j] = sum_k grad_conv[k] * (f[k-j] + f[j-k]) 
        # This is 2 * (grad_conv * f) [j] ?
        # Yes, the gradient of the scalar product <grad_conv, f*f> w.r.t f is 2 * (grad_conv * f) (circular convolution).
        # We are using linear convolution via FFT padding, but since f is zero-padded to 2n, and we take the first 2n-1, 
        # the circular convolution of length 2n of the padded signals is equivalent to linear convolution.
        
        # grad_conv is defined on indices 0..2n-2.
        # We need to pad grad_conv to length 2n with zeros.
        grad_conv_padded = np.zeros(2*n)
        grad_conv_padded[:2*n-1] = weights
        
        # Compute gradient in frequency domain
        F = np.fft.fft(f, n=2*n)
        G = np.fft.fft(grad_conv_padded, n=2*n)
        grad_f_freq = 2 * F * G
        grad_f = np.fft.ifft(grad_f_freq)[:n].real
        
        # Now we need gradient w.r.t u.
        # f_i = softmax(u)_i
        # d f_i / d u_j = f_i (delta_ij - f_j)
        # grad_u_j = sum_i grad_f_i * d f_i / d u_j
        # grad_u_j = sum_i grad_f_i * f_i (delta_ij - f_j)
        # grad_u_j = grad_f_j f_j - f_j (sum_i grad_f_i f_i)
        
        term1 = grad_f * f
        term2 = np.sum(grad_f * f)
        grad_u = term1 - f * term2
        
        return val, grad_u

    # Optimize u using L-BFGS-B
    # We can run multiple iterations or just one long one.
    # Given time constraints, L-BFGS-B is efficient.
    
    best_u = u_init.copy()
    best_val, best_grad = objective_and_grad(best_u)
    
    # Check time
    if time_remaining() > 10:
        res = minimize(
            objective_and_grad,
            best_u,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 1000, 'maxfun': 10000, 'maxls': 50}
        )
        
        if res.success or res.fun < best_val:
            best_u = res.x
            best_val, _ = objective_and_grad(best_u)
            
    # Convert back to f
    f = softmax(best_u)
    
    # Refine with a few steps of gradient descent in time domain if time permits?
    # Or just return the softmax result.
    # The softmax result is smooth and non-negative.
    
    # Ensure exact non-negativity and sum
    f = np.maximum(f, 0)
    f = f / np.sum(f)
    
    return list(f)
