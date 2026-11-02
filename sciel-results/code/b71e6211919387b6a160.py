import numpy as np
from scipy.optimize import minimize
import time

N = 600
TOTAL_WIDTH = 0.5
W = TOTAL_WIDTH / N
N_FFT = 2 * N

def compute_score(heights):
    """Compute normalized max of autoconvolution for step function heights."""
    heights = np.asarray(heights, dtype=float)
    heights = np.maximum(heights, 0.0)
    
    f_fft = np.fft.rfft(heights, n=N_FFT)
    conv = np.fft.irfft(f_fft * np.conj(f_fft), n=N_FFT)[:N] * W
    
    l2_sq = np.sum(heights**2) * W
    if l2_sq < 1e-15:
        return float('inf')
    
    return np.max(conv) / l2_sq

def objective_and_grad(heights, alpha=100.0):
    """
    Smooth-max surrogate objective and its analytical gradient.
    obj = smoothmax(conv) / l2_sq
    """
    h = np.maximum(heights, 0.0)
    
    f_fft = np.fft.rfft(h, n=N_FFT)
    conv = np.fft.irfft(f_fft * np.conj(f_fft), n=N_FFT)[:N] * W
    
    l2_sq = np.sum(h**2) * W
    if l2_sq < 1e-15:
        return 1e10, np.zeros(N)
    
    # Smooth max via log-sum-exp
    max_c = np.max(conv)
    shifted = conv - max_c
    exp_vals = np.exp(alpha * shifted)
    exp_sum = exp_vals.sum()
    smooth_max = max_c + np.log(exp_sum) / alpha
    
    obj = smooth_max / l2_sq
    
    # Gradient of smooth_max w.r.t. conv_k
    weights = exp_vals / exp_sum
    
    # Gradient of conv w.r.t. h_i:
    # d(conv_k)/d(h_i) = W * (h_{k-i} if 0<=k-i<N else 0) + W * (h_{i-k} if 0<=i-k<N else 0)
    # Sum over k: grad_conv[i] = W * (sum_j weights[j+i]*h[j] for j<N-i) + W * (sum_j weights[i-j]*h[j] for j<=i)
    
    grad_conv = np.zeros(N)
    for i in range(N):
        if N - i > 0:
            grad_conv[i] += np.dot(weights[i:N], h[:N-i])
        if i + 1 > 0:
            grad_conv[i] += np.dot(weights[:i+1][::-1], h[:i+1])
    
    grad_conv *= W
    
    # Full gradient of obj = smooth_max / l2_sq
    grad = grad_conv / l2_sq - 2 * smooth_max * W * h / (l2_sq ** 2)
    
    return obj, grad

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    x0 = np.ones(N)
    best_heights = x0.copy()
    best_score = compute_score(best_heights)
    
    # Primary optimization from uniform start
    try:
        result = minimize(
            objective_and_grad,
            x0,
            jac=True,
            method='L-BFGS-B',
            bounds=[(0, None)] * N,
            options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 100, 'maxfun': 500}
        )
        if time.time() < deadline - 15:
            candidate = np.maximum(result.x, 0.0)
            score = compute_score(candidate)
            if score < best_score:
                best_score = score
                best_heights = candidate
    except Exception:
        pass
    
    # Second optimization from linear ramp start
    if time.time() < deadline - 50:
        x1 = np.linspace(1.0, 0.3, N)
        try:
            result2 = minimize(
                objective_and_grad,
                x1,
                jac=True,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 100, 'maxfun': 400}
            )
            if time.time() < deadline - 15:
                candidate2 = np.maximum(result2.x, 0.0)
                score2 = compute_score(candidate2)
                if score2 < best_score:
                    best_score = score2
                    best_heights = candidate2
        except Exception:
            pass
    
    # Third optimization from two-block start
    if time.time() < deadline - 50:
        x2 = np.zeros(N)
        x2[:N//3] = 1.0
        x2[2*N//3:] = 1.0
        try:
            result3 = minimize(
                objective_and_grad,
                x2,
                jac=True,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 150, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 100, 'maxfun': 300}
            )
            if time.time() < deadline - 15:
                candidate3 = np.maximum(result3.x, 0.0)
                score3 = compute_score(candidate3)
                if score3 < best_score:
                    best_score = score3
                    best_heights = candidate3
        except Exception:
            pass
    
    return best_heights.tolist()
