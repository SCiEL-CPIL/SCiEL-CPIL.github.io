"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _compute_overlap(h, N):
    """
    Compute the overlap of h with shifted 1-h for all shifts.
    Overlap at shift k = sum_i h[i] * (1 - h[(i+k) % N])
                       = sum_i h[i] - sum_i h[i]*h[(i+k)%N]
                       = N/2 - autocorr[k]
    where autocorr[k] = sum_i h[i]*h[(i+k)%N]
    
    So minimizing max overlap = minimizing max(N/2 - autocorr[k])
    = N/2 - min_k autocorr[k]
    
    To minimize the maximum overlap, we want to maximize the minimum autocorrelation.
    """
    # autocorr[k] = sum_i h[i] * h[(i+k) % N]
    # Using FFT: autocorr = ifft(fft(h) * conj(fft(h)))
    H = np.fft.fft(h)
    power = H * np.conj(H)
    autocorr = np.fft.ifft(power).real
    # autocorr[k] corresponds to circular correlation at lag k
    # overlap[k] = N/2 - autocorr[k]
    overlap = N / 2.0 - autocorr
    return overlap


def _max_overlap(h, N):
    """Return the maximum overlap value."""
    overlap = _compute_overlap(h, N)
    return np.max(overlap)


def _objective(params, N):
    """
    Objective: minimize the maximum overlap.
    We use a smooth surrogate: log-sum-exp of overlaps for smoothness.
    Also add small penalty for going out of [0,1] range.
    """
    h = params
    # Soft penalty for bounds
    bound_pen = 1e-6 * (np.sum(np.maximum(0, -h)**2) + np.sum(np.maximum(0, h - 1)**2))
    
    overlap = _compute_overlap(h, N)
    # Use log-sum-exp as smooth max
    beta = 50.0
    m = np.max(overlap)
    if m > 1e10:
        return 1e10 + bound_pen
    lse = m + np.log(np.sum(np.exp(beta * (overlap - m)))) / beta
    return lse + bound_pen


def _objective_and_grad(params, N):
    """Objective with gradient for L-BFGS-B."""
    h = params
    bound_pen = 1e-6 * (np.sum(np.maximum(0, -h)**2) + np.sum(np.maximum(0, h - 1)**2))
    
    H = np.fft.fft(h)
    power = H * np.conj(H)
    autocorr = np.fft.ifft(power).real
    overlap = N / 2.0 - autocorr
    
    beta = 50.0
    m = np.max(overlap)
    if m > 1e10:
        return 1e10 + bound_pen, np.zeros_like(h)
    
    exp_terms = np.exp(beta * (overlap - m))
    sum_exp = np.sum(exp_terms)
    lse = m + np.log(sum_exp) / beta
    
    # Gradient of log-sum-exp w.r.t. overlap: exp_terms / sum_exp
    grad_overlap = exp_terms / sum_exp  # shape (N,)
    
    # Gradient of overlap w.r.t. h:
    # overlap[k] = N/2 - sum_i h[i]*h[(i+k)%N]
    # d(overlap[k])/d(h[j]) = -sum_i [delta_{i,j} h[(i+k)%N] + h[i] delta_{(i+k)%N, j}]
    #                        = -h[(j+k)%N] - h[(j-k)%N]
    # So d(overlap)/d(h) = -sum_k grad_overlap[k] * (h[(j+k)%N] + h[(j-k)%N])
    # This is a convolution-like operation.
    # sum_k w[k] * h[(j+k)%N] = ifft(fft(w) * fft(h))[j] where w is circular
    # Actually: sum_k w[k] h[(j+k)%N] is circular cross-correlation.
    # Let's use FFT: 
    # A[j] = sum_k w[k] h[(j+k)%N]
    # This is the circular convolution of w_rev with h, where w_rev[k] = w[-k mod N]
    # Or: A = ifft(fft(w) * fft(h)) gives sum_k w[k] h[j-k]... let me be careful.
    
    # Circular convolution: (w * h)[j] = sum_k w[k] h[(j-k)%N]
    # We want: sum_k w[k] h[(j+k)%N] = sum_k w[k] h[(j-(-k))%N]
    # Let v[k] = w[-k % N], then sum_k w[k] h[(j+k)%N] = sum_k v[-k] h[(j+k)%N]
    # = sum_m v[m] h[(j-m)%N] where m = -k
    # = (v * h)[j] circular convolution
    
    w = grad_overlap
    v = np.zeros(N)
    v[0] = w[0]
    v[1:] = w[-1:0:-1]  # v[k] = w[-k mod N]
    
    W = np.fft.fft(v)
    Hf = np.fft.fft(h)
    A = np.fft.ifft(W * Hf).real  # A[j] = sum_k v[k] h[(j-k)%N] = sum_k w[k] h[(j+k)%N]
    
    # Similarly for h[(j-k)%N]:
    # sum_k w[k] h[(j-k)%N] = (w * h)[j] circular convolution
    B = np.fft.ifft(np.fft.fft(w) * Hf).real
    
    # d(overlap[k])/d(h[j]) = -h[(j+k)%N] - h[(j-k)%N]
    # So grad_h[j] = -sum_k grad_overlap[k] * (h[(j+k)%N] + h[(j-k)%N])
    # = -(A[j] + B[j])
    grad_h = -(A + B)
    
    # Scale by beta for log-sum-exp gradient
    grad_lse = beta * grad_h / beta  # Wait, let me redo.
    
    # lse = m + log(sum(exp(beta*(overlap-m))))/beta
    # dlse/d(overlap[k]) = exp(beta*(overlap[k]-m)) / sum_exp = grad_overlap[k]
    # dlse/d(h[j]) = sum_k grad_overlap[k] * d(overlap[k])/d(h[j])
    # = sum_k grad_overlap[k] * (-h[(j+k)%N] - h[(j-k)%N])
    # = -(A[j] + B[j])
    
    grad_obj = -(A + B)
    
    # Bound penalty gradient
    bp = 2e-6 * (np.where(h < 0, -1, 0) + np.where(h > 1, 1, 0))
    
    return lse + bound_pen, grad_obj + bp


def construct():
    start_time = time.time()
    deadline = 120.0  # safety margin
    
    # Start from constant solution
    h0 = np.full(N, 0.5)
    
    best_h = h0.copy()
    best_score = _max_overlap(h0, N)
    
    # Try multiple initializations and local optimizations
    n_trials = 10
    rng = np.random.default_rng(42)
    
    for trial in range(n_trials):
        if time.time() - start_time > deadline - 10:
            break
        
        # Generate perturbed initial guess
        if trial == 0:
            h_init = h0.copy()
        else:
            # Add random perturbation
            noise_scale = 0.1 * (1.0 / (trial + 1))
            h_init = h0 + rng.normal(0, noise_scale, N)
            h_init = np.clip(h_init, 0, 1)
            # Normalize to sum = N/2
            s = np.sum(h_init)
            if s > 0:
                h_init = h_init * (N / 2.0) / s
                h_init = np.clip(h_init, 0, 1)
                # Re-normalize after clipping
                s2 = np.sum(h_init)
                if s2 > 0:
                    h_init = h_init * (N / 2.0) / s2
        
        # Use L-BFGS-B with bounds [0,1] and equality constraint sum=N/2
        # For simplicity, use bounds and rely on the optimizer; we'll project afterwards
        bounds = [(0, 1)] * N
        
        try:
            result = minimize(
                _objective_and_grad,
                h_init,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
            )
        except Exception:
            continue
        
        if time.time() - start_time > deadline - 10:
            break
        
        h_opt = result.x
        # Project to satisfy sum = N/2 and bounds
        h_opt = np.clip(h_opt, 0, 1)
        s = np.sum(h_opt)
        if s > 0:
            h_opt = h_opt * (N / 2.0) / s
            h_opt = np.clip(h_opt, 0, 1)
            # Iterate projection a few times
            for _ in range(10):
                s = np.sum(h_opt)
                if abs(s - N/2.0) < 1e-15:
                    break
                h_opt = h_opt * (N / 2.0) / s
                h_opt = np.clip(h_opt, 0, 1)
        
        # Ensure exact sum
        diff = N / 2.0 - np.sum(h_opt)
        if abs(diff) > 1e-15:
            # Distribute difference
            h_opt += diff / N
            h_opt = np.clip(h_opt, 0, 1)
            # Final adjustment on the largest element
            s_final = np.sum(h_opt)
            if abs(s_final - N/2.0) > 1e-15:
                # Adjust one element
                idx = np.argmax(h_opt) if s_final < N/2.0 else np.argmin(h_opt)
                h_opt[idx] += N/2.0 - s_final
                h_opt = np.clip(h_opt, 0, 1)
        
        score = _max_overlap(h_opt, N)
        
        if score < best_score:
            best_score = score
            best_h = h_opt.copy()
    
    # Final polish: a few more iterations from the best
    if time.time() - start_time < deadline - 5:
        try:
            result = minimize(
                _objective_and_grad,
                best_h,
                method='L-BFGS-B',
                bounds=[(0, 1)] * N,
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 100}
            )
            h_polish = result.x
            h_polish = np.clip(h_polish, 0, 1)
            for _ in range(20):
                s = np.sum(h_polish)
                if abs(s - N/2.0) < 1e-15:
                    break
                h_polish = h_polish * (N / 2.0) / s
                h_polish = np.clip(h_polish, 0, 1)
            
            diff = N / 2.0 - np.sum(h_polish)
            if abs(diff) > 1e-15:
                h_polish += diff / N
                h_polish = np.clip(h_polish, 0, 1)
                s_final = np.sum(h_polish)
                if abs(s_final - N/2.0) > 1e-15:
                    idx = np.argmax(h_polish) if s_final < N/2.0 else np.argmin(h_polish)
                    h_polish[idx] += N/2.0 - s_final
                    h_polish = np.clip(h_polish, 0, 1)
            
            score_polish = _max_overlap(h_polish, N)
            if score_polish < best_score:
                best_h = h_polish
        except Exception:
            pass
    
    # Ensure the returned array has exact sum N/2
    s = np.sum(best_h)
    if abs(s - N/2.0) > 1e-12:
        best_h = best_h * (N / 2.0) / s
        best_h = np.clip(best_h, 0, 1)
    
    return best_h
# EVOLVE-BLOCK-END
