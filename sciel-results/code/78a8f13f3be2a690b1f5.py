import numpy as np
import time
from scipy.optimize import minimize
import sys

N = 600
START_TIME = time.time()
TIME_LIMIT = 145.0

K = 32
M_STEP = N * K
M_FFT = 1 << 15

def compute_score(heights):
    """Compute the normalized maximum of autoconvolution."""
    heights = np.asarray(heights, dtype=np.float64)
    if np.any(heights < -1e-15):
        return np.inf
    heights = np.maximum(heights, 0.0)
    n = len(heights)
    w = 0.5 / n
    f_fine = np.zeros(M_FFT, dtype=np.float64)
    f_fine[:M_STEP] = np.repeat(heights, K)
    F = np.fft.rfft(f_fine)
    conv = np.fft.irfft(F * F, n=M_FFT)
    max_conv = np.max(conv)
    integral_f = np.sum(heights) * w
    if integral_f <= 1e-10:
        return np.inf
    return max_conv / (integral_f ** 2)

def compute_soft_max_score_and_grad(heights, beta):
    """
    Compute the soft-max normalized autoconvolution score and its gradient.
    Score = SoftMax_beta(conv) / (integral_f)^2
    """
    heights = np.asarray(heights, dtype=np.float64)
    heights = np.maximum(heights, 0.0)
    n = len(heights)
    w = 0.5 / n
    
    f_fine = np.zeros(M_FFT, dtype=np.float64)
    f_fine[:M_STEP] = np.repeat(heights, K)
    F = np.fft.rfft(f_fine)
    conv = np.fft.irfft(F * F, n=M_FFT)
    
    # Soft max of conv
    # To avoid overflow, subtract max
    max_c = np.max(conv)
    if max_c == 0:
        return 0.0, np.zeros_like(heights)
    
    exp_conv = np.exp(beta * (conv - max_c))
    sum_exp = np.sum(exp_conv)
    soft_max_val = max_c + np.log(sum_exp) / beta
    
    integral_f = np.sum(heights) * w
    if integral_f <= 1e-10:
        return 1e10, np.zeros_like(heights)
    
    score = soft_max_val / (integral_f ** 2)
    
    # Gradient calculation
    # d(soft_max)/dh = sum( p_i * d(conv_i)/dh )
    # where p_i = exp(beta*(conv_i - max_c)) / sum_exp
    
    # d(conv_i)/dh is the convolution of f with a shifted delta, effectively 
    # related to the cross-correlation.
    # conv = f * f. d(conv_k)/dh_j = sum_m f_{k-m-j} (with appropriate indexing)
    # In frequency domain: d(conv)/dh = 2 * IFFT(F * IFFT(H)) ... no.
    # conv = IFFT(F^2). d(conv)/dh = IFFT(2 * F * dF/dh).
    # dF/dh is the FFT of the impulse at position j.
    # So d(conv)/dh is the cross-correlation of f with itself, shifted.
    
    # Efficient calculation of gradient:
    # We need G = d(soft_max)/dh = sum_i p_i * (d conv_i / dh)
    # Note that d conv_i / dh is the vector of partial derivatives of conv at index i w.r.t h.
    # Let's compute the gradient of the soft max w.r.t conv first:
    # grad_conv = p (vector of length M_FFT)
    # Then grad_h = d(conv)/dh . grad_conv
    # d(conv)/dh is a linear operator. 
    # conv = IFFT(F^2). 
    # Let g = grad_conv.
    # We want h_grad such that h_grad = d(soft_max)/dh.
    # d(soft_max) = g . d(conv) = g . IFFT(d(F^2)) = g . IFFT(2 F dF)
    # Using Parseval/inner product properties:
    # <g, IFFT(A)> = <FFT(g), A> (real parts handled carefully)
    # Let G_fft = rfft(g).
    # Then d(soft_max) = <G_fft, 2 F dF> = <2 F G_fft, dF>
    # So the gradient w.r.t F is 2 F G_fft.
    # And dF = FFT(dh).
    # So grad_h = IFFT(2 F G_fft) / M_FFT (scaling depends on FFT convention)
    # NumPy FFT: IFFT(FFT(x)) = x.
    # So grad_h = IFFT(2 * F * G_fft)
    
    G_fft = np.fft.rfft(g)
    # dF/dh_j is the FFT of a vector that is 1 at j and 0 elsewhere? 
    # Actually, we are computing the gradient w.r.t the vector h.
    # The chain rule gives us the gradient w.r.t F, then w.r.t h.
    # grad_F = 2 * F * G_fft
    # grad_h = IFFT(grad_F)
    # Note: The FFT of h is F. The relationship is linear.
    
    grad_h_fine = np.fft.irfft(2 * F * G_fft, n=M_FFT)
    
    # grad_h_fine is the gradient w.r.t the fine grid f_fine.
    # We need to map it back to h.
    # f_fine is constructed by repeating h.
    # d(f_fine)/dh is a repeat matrix.
    # grad_h = sum over blocks of grad_h_fine
    grad_h = grad_h_fine[:M_STEP].reshape(N, K).sum(axis=1)
    
    # Now account for the denominator (integral_f)^2
    # score = S / I^2
    # d(score)/dh = (dS/dh * I^2 - S * 2 * I * dI/dh) / I^4
    #             = (dS/dh * I - 2 * S * dI/dh) / I^2
    # dI/dh = w (constant vector)
    
    dS_dh = grad_h
    dI_dh = w
    grad_score = (dS_dh * integral_f - 2 * score * dI_dh) / (integral_f ** 2)
    
    return score, grad_score

def construct():
    global START_TIME
    if time.time() - START_TIME > TIME_LIMIT:
        return [1.0] * N

    best_h = np.ones(N, dtype=np.float64)
    best_score = compute_score(best_h)
    
    # Initializations
    # 1. Uniform
    # 2. Two rectangles (gap in middle)
    # 3. Gaussian-like
    # 4. Random
    
    rng = np.random.default_rng(42)
    candidates = []
    
    # Uniform
    candidates.append(np.ones(N))
    
    # Two rectangles
    for gap_frac in [0.1, 0.2, 0.3]:
        h = np.zeros(N)
        gap = int(gap_frac * N)
        left_end = (N - gap) // 2
        right_start = left_end + gap
        if left_end > 0:
            h[:left_end] = 1.0
        if right_start < N:
            h[right_start:] = 1.0
        candidates.append(h)
        
    # Gaussian
    x = np.linspace(-1, 1, N)
    for sigma in [0.2, 0.4, 0.6]:
        h = np.exp(-x**2 / (2*sigma**2))
        candidates.append(h)
        
    # Random
    for _ in range(3):
        h = np.abs(rng.normal(1.0, 0.5, N))
        candidates.append(h)
    
    # Optimization Loop
    # We will use L-BFGS-B on the soft-max objective.
    # We start with a low beta (smooth) and increase it (sharp).
    
    betas = [10.0, 50.0, 100.0, 500.0]
    
    bounds = [(0.0, None)] * N
    
    for h_init in candidates:
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
            
        h_current = h_init.copy()
        
        # Normalize initial guess to have integral 1? 
        # The score is scale invariant, so normalization doesn't change score,
        # but it might help the optimizer stay in a good region.
        # However, L-BFGS-B handles the scale invariant landscape fine if we use the ratio.
        
        for beta in betas:
            if time.time() - START_TIME > TIME_LIMIT - 5:
                break
                
            def objective(h):
                val, _ = compute_soft_max_score_and_grad(h, beta)
                return val
            
            def objective_grad(h):
                val, grad = compute_soft_max_score_and_grad(h, beta)
                return grad
                
            try:
                result = minimize(objective, h_current, method='L-BFGS-B',
                                  jac=objective_grad, bounds=bounds,
                                  options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-5})
                
                if result.fun < best_score:
                    best_score = result.fun
                    best_h = np.maximum(result.x, 0.0)
                
                # Update current for next beta
                h_current = np.maximum(result.x, 0.0)
                
            except Exception as e:
                # If optimization fails, break
                break
    
    # Final check with exact max (not soft max) to get true score
    final_score = compute_score(best_h)
    if final_score < best_score:
        best_score = final_score
        # best_h is already set
        
    # One last local refinement with exact max gradient (subgradient)
    # This helps "snap" to the true peak minimization if soft max was too smooth.
    if time.time() - START_TIME < TIME_LIMIT - 2:
        def exact_obj(h):
            return compute_score(h)
        
        def exact_grad(h):
            h = np.maximum(h, 0.0)
            n = len(h)
            w = 0.5 / n
            f_fine = np.zeros(M_FFT, dtype=np.float64)
            f_fine[:M_STEP] = np.repeat(h, K)
            F = np.fft.rfft(f_fine)
            conv = np.fft.irfft(F * F, n=M_FFT)
            max_conv = np.max(conv)
            integral_f = np.sum(h) * w
            if integral_f <= 1e-10:
                return 1e10, np.zeros_like(h)
            score = max_conv / (integral_f ** 2)
            
            peak_idx = np.argmax(conv)
            # Gradient of max is the gradient of conv at peak_idx
            # d(conv_peak)/dh = IFFT(2 * F * FFT(delta_peak))
            # FFT(delta_peak) is a complex exponential
            # But we can use the same logic as soft max with p being a delta.
            g = np.zeros(M_FFT)
            g[peak_idx] = 1.0
            G_fft = np.fft.rfft(g)
            grad_h_fine = np.fft.irfft(2 * F * G_fft, n=M_FFT)
            grad_h = grad_h_fine[:M_STEP].reshape(N, K).sum(axis=1)
            
            dI_dh = w
            grad_score = (grad_h * integral_f - 2 * score * dI_dh) / (integral_f ** 2)
            return grad_score
            
        try:
            result = minimize(exact_obj, best_h, method='L-BFGS-B',
                              jac=exact_grad, bounds=bounds,
                              options={'maxiter': 50, 'ftol': 1e-10})
            if result.fun < best_score:
                best_score = result.fun
                best_h = np.maximum(result.x, 0.0)
        except Exception:
            pass

    best_h = np.maximum(best_h, 0.0)
    return [max(0.0, float(v)) for v in best_h]

if __name__ == "__main__":
    res = construct()
    assert len(res) == N
    assert all(h >= 0 for h in res)
    print("Constructed optimized step function.")
    print(f"Score: {compute_score(np.array(res))}")
