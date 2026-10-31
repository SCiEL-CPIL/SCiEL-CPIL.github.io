"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _overlap_vector(h):
    """Compute overlap for all shifts: overlap[k] = sum_i h[i]*(1-h[(i-k)%N])"""
    # h has shape (N,)
    # For shift k, we need h[i] * (1 - h[(i-k) % N])
    # This is a circular cross-correlation-like quantity
    # overlap[k] = sum_i h[i] - sum_i h[i]*h[(i-k)%N]
    # = sum(h) - circular_conv(h, h)[k]  ... wait, let me be careful
    # sum_i h[i]*h[(i-k)%N] is the circular autocorrelation at lag k
    # Actually: circular_corr[k] = sum_i h[i] * h[(i-k) mod N]
    # So overlap[k] = sum(h) - circular_corr[k]
    
    # Compute circular autocorrelation using FFT
    # autocorr[k] = sum_i h[i] * h[(i-k) % N]
    # Using FFT: ifft(fft(h) * conj(fft(h))) gives sum_i h[i] * conj(h[i-k]) for real = autocorr
    # Actually np.fft.ifft(np.fft.fft(h) * np.conj(np.fft.fft(h))) gives 
    # circular correlation: result[k] = sum_j h[j] * h[j-k] (with appropriate indexing)
    
    H = np.fft.fft(h)
    autocorr = np.real(np.fft.ifft(H * np.conj(H)))
    # autocorr[k] = sum_j h[j] * h[(j-k) % N]
    # We want overlap[k] = sum_i h[i] * (1 - h[(i-k)%N]) = sum(h) - sum_i h[i]*h[(i-k)%N]
    # sum_i h[i]*h[(i-k)%N] = autocorr[k] (by definition above)
    total = np.sum(h)
    overlaps = total - autocorr
    return overlaps

def _max_overlap(h):
    """Return the maximum overlap value."""
    overlaps = _overlap_vector(h)
    return np.max(overlaps)

def _soft_max_overlap(h, beta=50.0):
    """Smooth surrogate for max overlap using log-sum-exp."""
    overlaps = _overlap_vector(h)
    # Soft max: (1/beta) * log(sum(exp(beta * x)))
    m = np.max(overlaps)
    return m + np.log(np.sum(np.exp(beta * (overlaps - m)))) / beta

def _soft_max_grad(h, beta=50.0):
    """Gradient of soft max overlap w.r.t. h."""
    overlaps = _overlap_vector(h)
    m = np.max(overlaps)
    exps = np.exp(beta * (overlaps - m))
    weights = exps / np.sum(exps)
    # d(soft_max)/d(h[i]) = sum_k weights[k] * d(overlap[k])/d(h[i])
    # overlap[k] = sum_j h[j] * (1 - h[(j-k)%N])
    # d(overlap[k])/d(h[i]) = (1 - h[(i-k)%N]) - h[i] * delta... 
    # Actually: overlap[k] = sum_j h[j] - sum_j h[j]*h[(j-k)%N]
    # d/dh[i] of overlap[k] = 1 - [h[(i-k)%N] + h[i] * (d/dh[i]) h[(j-k)%N] summed...]
    # Let me be more careful.
    # overlap[k] = sum_j h[j] * (1 - h[(j-k) mod N])
    # d(overlap[k])/d(h[i]) = (1 - h[(i-k) mod N]) + sum_j h[j] * (-1) * d(h[(j-k) mod N])/d(h[i])
    # d(h[(j-k) mod N])/d(h[i]) = 1 if (j-k) mod N == i, i.e., j = (i+k) mod N
    # So = (1 - h[(i-k)%N]) - h[(i+k)%N]
    
    # For each i, d(soft_max)/d(h[i]) = sum_k weights[k] * ((1 - h[(i-k)%N]) - h[(i+k)%N])
    # Let's compute this efficiently.
    # term1[i] = sum_k weights[k] * (1 - h[(i-k)%N]) = 1 - sum_k weights[k] * h[(i-k)%N]
    # term2[i] = sum_k weights[k] * h[(i+k)%N]
    
    # circular convolution of weights with h
    W = np.fft.fft(weights)
    H = np.fft.fft(h)
    # sum_k weights[k] * h[(i-k)%N] = ifft(fft(weights) * fft(h))[i]
    conv1 = np.real(np.fft.ifft(W * H))
    # sum_k weights[k] * h[(i+k)%N] = ifft(fft(weights) * conj(fft(h)))[i] ... 
    # Actually sum_k weights[k] * h[(i+k)%N]: let m = i+k, k = m-i
    # = sum_m weights[m-i] * h[m] = circular convolution of weights reversed with h
    # = ifft(conj(fft(weights)) * fft(h))[i] ... let me verify
    # (f*g)[i] = sum_k f[k] * g[i-k]
    # We want sum_k weights[k] * h[i+k] = sum_k weights[k] * h[(i+k)%N]
    # Let l = -k, then = sum_l weights[-l] * h[i-l] = (weights_rev * h)[i] where weights_rev[l] = weights[-l]
    # weights_rev = weights flipped: weights_rev[k] = weights[N-k] = weights[-k mod N]
    W_rev = np.fft.fft(np.roll(weights, 1))  # roll by 1 to reverse circularly... 
    # Actually simpler: weights_rev[k] = weights[(-k) % N]
    weights_rev = np.array([weights[(-k) % N] for k in range(N)])
    W_rev = np.fft.fft(weights_rev)
    conv2 = np.real(np.fft.ifft(W_rev * H))
    
    grad = (1 - conv1) - conv2
    return grad

def construct():
    deadline = time.time() + 140.0  # 10s safety margin
    
    # Initial guess: try a periodic pattern
    # Start with a simple structure and optimize
    h0 = np.full(N, 0.5)
    
    best_h = h0.copy()
    best_score = _max_overlap(h0)
    
    beta = 30.0
    
    try:
        res = minimize(
            _soft_max_overlap,
            h0,
            args=(beta,),
            method='L-BFGS-B',
            jac=_soft_max_grad,
            bounds=[(0, 1)] * N,
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        
        if res.success or res.fun < best_score:
            h_opt = np.clip(res.x, 0, 1)
            # Normalize to ensure sum = N/2
            s = np.sum(h_opt)
            if s > 0:
                h_opt = h_opt * (N / 2.0) / s
                h_opt = np.clip(h_opt, 0, 1)
                # Re-normalize after clipping
                s2 = np.sum(h_opt)
                if abs(s2 - N/2.0) > 1e-6 and s2 > 0:
                    h_opt = h_opt * (N / 2.0) / s2
                    h_opt = np.clip(h_opt, 0, 1)
            
            score = _max_overlap(h_opt)
            if score < best_score:
                best_h = h_opt
                best_score = score
        
        # Try with higher beta for sharper max approximation
        if time.time() < deadline - 20:
            beta2 = 80.0
            res2 = minimize(
                _soft_max_overlap,
                best_h,
                args=(beta2,),
                method='L-BFGS-B',
                jac=_soft_max_grad,
                bounds=[(0, 1)] * N,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if res2.fun < best_score:
                h_opt = np.clip(res2.x, 0, 1)
                s = np.sum(h_opt)
                if s > 0:
                    h_opt = h_opt * (N / 2.0) / s
                    h_opt = np.clip(h_opt, 0, 1)
                    s2 = np.sum(h_opt)
                    if abs(s2 - N/2.0) > 1e-6 and s2 > 0:
                        h_opt = h_opt * (N / 2.0) / s2
                        h_opt = np.clip(h_opt, 0, 1)
                
                score = _max_overlap(h_opt)
                if score < best_score:
                    best_h = h_opt
                    best_score = score
                    
    except Exception:
        pass
    
    # Final normalization to ensure exact sum
    s = np.sum(best_h)
    if s > 0 and abs(s - N/2.0) > 1e-10:
        best_h = best_h * (N / 2.0) / s
        best_h = np.clip(best_h, 0, 1)
        # One more pass
        s2 = np.sum(best_h)
        if s2 > 0 and abs(s2 - N/2.0) > 1e-10:
            # Adjust a few elements
            diff = N/2.0 - s2
            idx = np.argsort(best_h)
            if diff > 0:
                for i in idx[::-1]:
                    if best_h[i] < 1.0:
                        add = min(diff, 1.0 - best_h[i])
                        best_h[i] += add
                        diff -= add
                        if diff <= 1e-12:
                            break
            else:
                for i in idx:
                    if best_h[i] > 0.0:
                        sub = min(-diff, best_h[i])
                        best_h[i] -= sub
                        diff += sub
                        if diff >= -1e-12:
                            break
    
    return best_h
# EVOLVE-BLOCK-END
