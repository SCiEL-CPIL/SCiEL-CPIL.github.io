import numpy as np
from scipy.optimize import minimize
import time

N = 600

def _compute_score(f):
    """
    Computes the normalized maximum of the autoconvolution.
    f is assumed to be nonnegative and normalized such that sum(f) = 1.
    The score is max(autoconv(f)).
    """
    if np.any(f < -1e-12):
        return 1e10
    f = np.clip(f, 0, None)
    s = np.sum(f)
    if s < 1e-9:
        return 1e10
    f_norm = f / s
    # Compute autoconvolution via FFT
    fft_f = np.fft.fft(f_norm, n=2*N)
    fft_conv = fft_f * fft_f
    conv = np.fft.ifft(fft_conv).real
    return np.max(conv)

def construct():
    start_time = time.time()
    deadline = 120.0
    safety_margin = 10.0
    
    # Initial guess: Uniform distribution in time domain
    f_best = np.ones(N) / N
    best_score = _compute_score(f_best)
    
    # Strategy: Optimize in the frequency domain.
    # Let F_k = |F_k| * exp(i * theta_k).
    # We fix phases theta_k to be random or zero? 
    # Actually, for a real signal, F_{N-k} = conj(F_k).
    # The time domain signal is f_n = IDFT(F).
    # We want f_n >= 0.
    # Objective: min max_n (autoconv(f))_n.
    # autoconv(f) = IDFT(|F|^2).
    # So we want to minimize the max of IDFT(|F|^2).
    # Let G_k = |F_k|^2. G_k is real and non-negative.
    # Also, G_k must correspond to a valid PSD of some signal f with f_n >= 0.
    # This is a complex constraint.
    
    # Alternative approach:
    # Optimize the log-amplitudes in the frequency domain?
    # Or optimize the time-domain signal using a better surrogate?
    
    # Let's try optimizing the time-domain signal but using a smooth max surrogate
    # to make the objective smooth and differentiable, and using L-BFGS-B with
    # analytic gradients computed via FFT.
    
    def objective_and_grad(f_raw):
        # f_raw are the variables. We map them to non-negative f.
        # Use softplus or exp. exp is good for positivity.
        # But exp can overflow. Let's use sigmoid or softplus scaled.
        # Let f = exp(f_raw). Then sum(f) = 1 constraint is hard.
        # Instead, let f = exp(f_raw) / sum(exp(f_raw)).
        # This ensures non-negativity and normalization.
        
        # To avoid overflow in exp, shift f_raw by its max.
        f_shifted = f_raw - np.max(f_raw)
        f_unnorm = np.exp(f_shifted)
        f = f_unnorm / np.sum(f_unnorm)
        
        # Compute autoconvolution
        fft_f = np.fft.fft(f, n=2*N)
        fft_conv = fft_f * fft_f
        conv = np.fft.ifft(fft_conv).real
        
        # Smooth max surrogate: log-sum-exp with temperature
        # To make it differentiable and closer to max, use a high temperature?
        # No, log-sum-exp is always smooth. We can use a "soft max" with a parameter.
        # Or just use the actual max? L-BFGS-B might struggle with non-differentiability.
        # Let's use a smooth approximation: max(x) ~ (1/beta) * log(sum(exp(beta*x)))
        # As beta -> inf, it approaches max.
        # Let's try beta = 10.0 initially.
        beta = 10.0
        # Avoid overflow in exp(beta * conv)
        conv_max = np.max(conv)
        conv_shifted = conv - conv_max
        soft_max = conv_max + (1.0/beta) * np.log(np.sum(np.exp(beta * conv_shifted)))
        
        # Gradient of soft_max w.r.t. f
        # d(soft_max)/d(conv) = exp(beta * (conv - conv_max)) / sum(exp(beta * (conv - conv_max)))
        # Let weights = exp(beta * (conv - conv_max))
        weights = np.exp(beta * conv_shifted)
        weights /= np.sum(weights)
        
        # d(conv)/d(f)
        # conv = IDFT(FFT(f) * FFT(f))
        # Let F = FFT(f). conv_n = sum_k f_k f_{k-n}
        # This is a linear operator in f? No, it's quadratic.
        # Gradient of conv w.r.t. f is related to the cross-correlation of f with itself shifted?
        # Actually, conv = f * f.
        # d(conv)/d(f) is a matrix.
        # Let's compute the gradient numerically? No, too slow.
        # Let's use the fact that conv = IDFT(|F|^2).
        # d(conv)/d(f) = 2 * IDFT( F * conj(FFT(df)) )?
        # Let's derive:
        # conv_m = sum_n f_n f_{n-m}
        # d(conv_m)/d(f_k) = f_{k-m} + f_{k+m}
        # So the gradient vector g_k = sum_m weights_m * (f_{k-m} + f_{k+m})
        # This is a convolution of weights with (f shifted + f reversed)?
        # Let's compute it via FFT.
        # G_k (gradient) = IDFT( FFT(weights) * (FFT(f) + FFT(f_reversed)) )?
        # Actually, d(conv)/d(f) is a circulant matrix?
        # Let's just use the FFT property.
        # conv = IDFT( F * F )
        # d(conv)/d(f) = ?
        # Let's use the chain rule.
        # d(soft_max)/d(f_k) = sum_m weights_m * d(conv_m)/d(f_k)
        # d(conv_m)/d(f_k) = f_{k-m} + f_{k+m} (indices mod N)
        # So grad_k = sum_m weights_m * f_{k-m} + sum_m weights_m * f_{k+m}
        # This is 2 * (weights * f)_k (circular convolution)
        # Because sum_m weights_m f_{k-m} is (weights * f)_k
        # And sum_m weights_m f_{k+m} is (weights * f_rev)_k which is also (weights * f)_k if symmetric?
        # Wait, f is not symmetric.
        # But weights (from autoconv) is symmetric?
        # conv is symmetric? No, autoconvolution of a general function is symmetric around 0?
        # f * f (n) = sum_k f(k) f(n-k).
        # (f*f)(-n) = sum_k f(k) f(-n-k) = sum_j f(j) f(n-j) = (f*f)(n).
        # Yes, autoconvolution is even. So weights are even.
        # So grad = 2 * IDFT( FFT(weights) * FFT(f) )
        
        fft_weights = np.fft.fft(weights, n=N)
        fft_f = np.fft.fft(f, n=N)
        grad_conv = 2.0 * np.fft.ifft(fft_weights * fft_f).real
        
        # Now, we need d(f)/d(f_raw)
        # f = exp(f_raw) / S, where S = sum(exp(f_raw))
        # Let u_k = exp(f_raw_k)
        # f_k = u_k / S
        # d(f_k)/d(f_raw_j) = (delta_{kj} u_k S - u_k u_j) / S^2
        # = delta_{kj} f_k - f_k f_j
        # So grad_obj_j = sum_k grad_conv_k * (delta_{kj} f_k - f_k f_j)
        # = grad_conv_j * f_j - f_j * sum_k (grad_conv_k * f_k)
        # Let dot = sum_k (grad_conv_k * f_k)
        dot = np.dot(grad_conv, f)
        grad_obj = grad_conv * f - f * dot
        
        return soft_max, grad_obj

    # Initial guess in log-space
    # f = 1/N => log(f) = -log(N)
    f_raw_best = -np.log(N) * np.ones(N)
    
    # Run optimization
    for i in range(5):
        if time.time() > deadline - safety_margin:
            break
            
        if i == 0:
            f_raw_current = f_raw_best.copy()
        else:
            # Random perturbation
            f_raw_current = f_raw_best + np.random.normal(0, 0.5, N)
            
        # L-BFGS-B
        try:
            res = minimize(objective_and_grad, f_raw_current, method='L-BFGS-B', 
                           jac=True, options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50})
            
            if res.success or res.fun < best_score:
                f_candidate = np.exp(res.x - np.max(res.x))
                f_candidate /= np.sum(f_candidate)
                score_candidate = _compute_score(f_candidate)
                if score_candidate < best_score:
                    best_score = score_candidate
                    f_raw_best = res.x.copy()
                    f_best = f_candidate
        except Exception:
            pass

    return f_best.tolist()

if __name__ == "__main__":
    construct()
