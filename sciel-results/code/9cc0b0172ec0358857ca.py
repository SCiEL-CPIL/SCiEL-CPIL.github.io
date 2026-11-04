import numpy as np
from scipy.optimize import minimize
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # Target for flat convolution
    num_coeffs = 2 * N - 1
    target_val = 1.0 / num_coeffs
    
    def objective_and_grad(h):
        # Normalize h to sum to 1 for evaluation of C?
        # No, if we use penalty, h is not normalized during opt.
        # But C scales with h^2.
        # If we don't normalize, the target depends on scale.
        # Better to normalize h inside the objective function.
        
        s = np.sum(h)
        if abs(s) < 1e-10:
            return 1e10, np.zeros_like(h)
            
        h_norm = h / s
        
        c = np.convolve(h_norm, h_norm)
        
        # Objective: sum of squared deviations from target
        diff = c - target_val
        obj = np.sum(diff ** 2)
        
        # Gradient
        # d obj / d h_norm = 2 * sum_k (c_k - t) * d c_k / d h_norm
        # d c_k / d h_norm is the convolution kernel?
        # c_k = sum_i h_norm[i] h_norm[k-i]
        # d c_k / d h_norm[j] = h_norm[k-j] + h_norm[j-k] (with bounds)
        # This is equivalent to convolving the gradient of c w.r.t h_norm with h_norm?
        # Actually, let g = c - t.
        # grad_c = 2 * g
        # d c / d h = convolve(grad_c, h)? No.
        # c = conv(h, h).
        # d c_k / d h_j = h_{k-j} + h_{j-k} (symmetric)
        # This is equivalent to: 
        # grad_h = convolve(2*g, h) + convolve(2*g, h)[::-1]?
        # Actually, for real h, d/dh_j sum_k (sum_i h_i h_{k-i} - t)^2
        # = 2 sum_k (c_k - t) (h_{k-j} + h_{j-k})
        # This is 2 * (g * h)_j + 2 * (g * h_flipped)_j?
        # Note that conv(g, h) gives sum_i g_i h_{k-i}.
        # Let's use FFT for speed and simplicity of gradient.
        
        # FFT Gradient
        # C(f) = H(f)^2
        # d C(f) / d H(f) = 2 H(f)
        # d obj / d H(f) = 2 * IFFT( 2 * (C(f) - T(f)) * 2 * H(f) ) ?
        # T(f) is FFT of target vector.
        
        H = np.fft.fft(h_norm)
        C = np.fft.fft(c)
        T = np.fft.fft(np.full(num_coeffs, target_val))
        
        # Gradient in freq domain
        # d obj / d C = 2 * (c - t)
        # d C / d H = 2 * H
        # d H / d h = IFFT
        # Chain rule: d obj / d h = IFFT( d obj / d C * d C / d H )
        # Note: C = H^2 is in freq domain of the convolution.
        # Actually, c = IFFT( H * H )? No, c = IFFT( H^2 )?
        # Convolution theorem: FFT(c) = H * H (element-wise).
        # So C_f = H^2.
        # d obj / d h = IFFT( 2 * (c - t) * 2 * H )?
        # Wait, obj = sum_k (c_k - t)^2.
        # This is not simply related to C_f unless we use Parseval.
        # sum (c-t)^2 = sum (C_f - T_f)^2 / N?
        # Yes, if we normalize FFT properly.
        
        # Let's stick to time domain gradient for correctness, using convolution.
        # grad_c = 2 * (c - target_val)
        # d c / d h = convolve(grad_c, h) + convolve(grad_c, h[::-1])?
        # Let's verify for N=2.
        # h=[a,b]. c=[a^2, 2ab, b^2].
        # c0=a^2. dc0/da=2a.
        # conv(grad_c, h):
        # grad_c = [2(a^2-t), 4ab, 2(b^2-t)]
        # conv(g, h) = [2a^3-t2a, 2a^2b+2ab^2-t2b, ...]
        # This is getting complicated.
        
        # Simpler: Use numerical gradient? No, too slow.
        # Use the property: grad_h = 2 * convolve(c - t, h) + 2 * convolve(c - t, h)[::-1]?
        # Actually, if we use the FFT method:
        # obj = || c - t ||^2
        # grad_h = 2 * IFFT( FFT(c-t) * 2 * H )
        
        grad_f = 2 * np.fft.fft(c - target_val) * 2 * H
        grad_h_norm = np.real(np.fft.ifft(grad_f))
        
        # Now map back to h
        # h_norm = h / s
        # d h_norm / d h = (1/s) * I - (h/s^2) * 1^T
        # grad_h = (1/s) * (grad_h_norm - h * sum(grad_h_norm)/s)
        
        grad_h = (1.0 / s) * (grad_h_norm - h * (np.sum(grad_h_norm) / s))
        
        return obj, grad_h

    # Initial guesses
    best_h = None
    best_val = np.inf
    
    np.random.seed(42)
    starts = [
        np.random.randn(N),
        np.sin(np.linspace(0, 2*np.pi, N)),
        np.cos(np.linspace(0, 2*np.pi, N)),
        np.ones(N)
    ]
    
    for h0 in starts:
        if time.time() > deadline:
            break
            
        # Normalize
        s = np.sum(h0)
        if s == 0: h0 += 1e-6
        h0 = h0 / s
        
        res = minimize(objective_and_grad, h0, method='L-BFGS-B', jac=True, 
                       options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
        
        if res.success or res.fun < best_val:
            h_opt = res.x
            val = abs(2 * N * np.max(np.convolve(h_opt, h_opt)) / np.sum(h_opt) ** 2)
            if val < best_val:
                best_val = val
                best_h = h_opt
                
    if best_h is None:
        return np.ones(N)
        
    return best_h
