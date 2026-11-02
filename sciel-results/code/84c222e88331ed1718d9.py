import numpy as np
import time

N = 100

def _overlap_max(h):
    """Compute max cyclic overlap of h with 1-h, scaled by step width."""
    w = 2.0 / N
    H = np.fft.fft(h)
    autocorr = np.fft.ifft(H * np.conj(H)).real
    overlaps = N / 2.0 - autocorr
    return w * np.max(overlaps)

def _smooth_overlap_grad(h, beta=200.0):
    """
    Compute the smooth surrogate objective (log-sum-exp of overlaps) and its gradient.
    Objective: L = (1/beta) * log(sum_k exp(beta * overlap_k))
    overlap_k = w * (N/2 - autocorr_k)
    autocorr_k = sum_i h_i * h_{i+k}
    d(autocorr_k)/dh_j = h_{j-k} + h_{j+k} (for k != 0, N/2)
    d(overlap_k)/dh_j = -w * (h_{j-k} + h_{j+k})
    """
    w = 2.0 / N
    H = np.fft.fft(h)
    autocorr = np.fft.ifft(H * np.conj(H)).real
    overlaps = w * (N / 2.0 - autocorr)
    
    # Normalize overlaps for numerical stability in log-sum-exp
    max_ov = np.max(overlaps)
    shifted = beta * (overlaps - max_ov)
    log_sum_exp = max_ov + np.log(np.sum(np.exp(shifted))) / beta
    
    # Compute gradient
    # grad_j = sum_k (overlap_k / sum(exp(beta*overlap_k))) * d(overlap_k)/dh_j
    # probs_k = exp(beta * overlap_k) / sum(exp(beta * overlap_k))
    exp_terms = np.exp(shifted)
    probs = exp_terms / np.sum(exp_terms)
    
    # d(overlap_k)/dh_j = -w * (h_{j-k} + h_{j+k})
    # grad = sum_k probs_k * (-w) * (h_{j-k} + h_{j+k})
    # This is a convolution-like operation.
    # Let's compute it via FFT or direct loop.
    # grad_j = -w * sum_k probs_k * (h_{j-k} + h_{j+k})
    # = -w * (sum_k probs_k * h_{j-k} + sum_k probs_k * h_{j+k})
    # Note: sum_k probs_k * h_{j-k} is a circular convolution of probs and h.
    # sum_k probs_k * h_{j+k} is also related.
    
    # Let P = probs.
    # term1_j = sum_k P_k h_{j-k} = (P * h)_j (circular conv)
    # term2_j = sum_k P_k h_{j+k} = sum_k P_k h_{j+k}
    # Note that sum_k P_k h_{j+k} = sum_m P_{m-j} h_m (let m = j+k)
    # This is a correlation.
    # Actually, term2_j = sum_k P_k h_{j+k} = (P_rev * h)_j where P_rev[k] = P[-k]
    
    P_fft = np.fft.fft(probs)
    h_fft = np.fft.fft(h)
    
    # term1 = ifft(fft(P) * fft(h))
    term1 = np.fft.ifft(P_fft * h_fft).real
    
    # term2: sum_k P_k h_{j+k}
    # Let P_flip[k] = P[(-k) % N]
    P_flip = np.flip(probs)
    P_flip[0] = probs[0] # flip doesn't change index 0
    P_flip_fft = np.fft.fft(P_flip)
    term2 = np.fft.ifft(P_flip_fft * h_fft).real
    
    grad = -w * (term1 + term2)
    
    return log_sum_exp, grad

def construct():
    deadline = time.time() + 110
    
    # Initial guess: all 0.5
    h0 = np.full(N, 0.5)
    best_h = h0.copy()
    best_val = _overlap_max(h0)
    
    # Try structured initializations
    inits = [
        np.full(N, 0.5),
        np.tile([0.0, 1.0], N // 2).astype(float),
        np.sin(np.linspace(0, 2 * np.pi, N, endpoint=False)) * 0.3 + 0.5,
        np.cos(np.linspace(0, 2 * np.pi, N, endpoint=False)) * 0.3 + 0.5,
    ]
    
    # Project to sum = N/2
    def project(h):
        h = np.clip(h, 0, 1)
        diff = N / 2.0 - np.sum(h)
        h = h + diff / N
        h = np.clip(h, 0, 1)
        # Re-adjust sum
        for _ in range(10):
            diff = N / 2.0 - np.sum(h)
            if abs(diff) < 1e-10:
                break
            h = h + diff / N
            h = np.clip(h, 0, 1)
        return h
    
    for h_init in inits:
        if time.time() > deadline - 10:
            break
        h = project(h_init)
        val = _overlap_max(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
        
        # Gradient descent on smooth surrogate
        # Start with larger beta, decrease over time? Or keep fixed?
        # Let's try a few betas.
        betas = [50, 100, 200, 500]
        
        for beta in betas:
            if time.time() > deadline - 5:
                break
            curr_h = h.copy()
            # Adaptive step size
            step = 0.01
            no_improve = 0
            max_iter = 500
            
            for it in range(max_iter):
                if time.time() > deadline - 2:
                    break
                obj, grad = _smooth_overlap_grad(curr_h, beta)
                
                # Check if gradient is small
                if np.max(np.abs(grad)) < 1e-5:
                    break
                
                # Gradient step
                new_h = curr_h - step * grad
                new_h = project(new_h)
                
                # Evaluate true objective
                new_val = _overlap_max(new_h)
                curr_val = _overlap_max(curr_h)
                
                if new_val < curr_val:
                    curr_h = new_h
                    no_improve = 0
                    if new_val < best_val:
                        best_val = new_val
                        best_h = curr_h.copy()
                    # Increase step if improving significantly?
                    # step *= 1.05
                else:
                    no_improve += 1
                    step *= 0.5
                    if step < 1e-6:
                        break
                    if no_improve > 10:
                        break
                        
    return best_h
