import numpy as np
from scipy.optimize import minimize
import time

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    def max_overlap(h):
        """Compute max overlap of h with shifted 1-h."""
        H = np.fft.fft(h)
        C = np.real(np.fft.ifft(H * np.conj(H)))
        # overlap(s) = N/2 - C(s), max overlap = N/2 - min(C)
        min_C = np.min(C)
        return N/2 - min_C
    
    def smooth_max_overlap(h, beta):
        """Smooth approximation of max overlap using softmax."""
        H = np.fft.fft(h)
        C = np.real(np.fft.ifft(H * np.conj(H)))
        # We want to minimize max_s overlap(s) = N/2 - min_s C(s)
        # = maximize min_s C(s)
        # Use log-sum-exp for smooth max of (N/2 - C)
        overlaps = N/2 - C
        # Softmax approximation of max
        shifted = beta * overlaps
        shifted -= np.max(shifted)
        logsum = np.log(np.sum(np.exp(shifted)))
        return np.max(shifted)/beta + logsum/beta
    
    def smooth_max_overlap_grad(h, beta):
        """Gradient of smooth max overlap."""
        H = np.fft.fft(h)
        C = np.real(np.fft.ifft(H * np.conj(H)))
        overlaps = N/2 - C
        shifted = beta * overlaps
        shifted -= np.max(shifted)
        weights = np.exp(shifted) / np.sum(np.exp(shifted))
        # d(overlaps)/dh: overlap(s) = N/2 - sum_i h[i]*h[(i+s)%N]
        # d overlap(s)/dh[j] = -sum_i (delta_{i,j} h[(i+s)%N] + delta_{(i+s)%N, j} h[i])
        # = -h[(j+s)%N] - h[(j-s)%N]
        # So gradient of weighted sum = -sum_s weights[s] * (h[(j+s)%N] + h[(j-s)%N])
        # This is -2 * (weights * h) circular convolution, but careful with s=0
        # Actually: sum_s w[s]*(h[(j+s)%N] + h[(j-s)%N]) = 2*sum_s w[s]*h[(j+s)%N] (since s runs over all)
        # But for s=0, h[(j+0)%N] appears twice, so it's 2*conv(w,h)[j] where conv is circular
        grad_h = -2.0 * np.real(np.fft.ifft(np.fft.fft(weights) * H))
        return beta * grad_h
    
    def objective_and_grad(x, beta):
        """Objective and gradient for L-BFGS-B. x = [h (N vars), t (1 var)].
        Minimize t subject to t >= smooth_max_overlap(h, beta) - small, sum(h)=N/2.
        Actually, let's just minimize smooth_max_overlap directly with constraints.
        """
        h = x[:N]
        val = smooth_max_overlap(h, beta)
        grad = smooth_max_overlap_grad(h, beta)
        return val, grad
    
    x0 = np.full(N, 0.5)
    
    constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - N/2}]
    bounds = [(0, 1)] * N
    
    best_h = x0.copy()
    best_val = max_overlap(x0)
    
    # Multiple beta values from coarse to fine
    betas = [2.0, 5.0, 10.0, 20.0, 50.0]
    
    for beta in betas:
        if time.time() > deadline - 15:
            break
        
        # Try multiple starting points
        n_starts = 3
        for start_idx in range(n_starts):
            if time.time() > deadline - 15:
                break
            
            if start_idx == 0:
                h_init = best_h.copy()
            elif start_idx == 1:
                # Structured pattern: alternating blocks
                h_init = np.zeros(N)
                for i in range(N):
                    if (i // 7) % 2 == 0:
                        h_init[i] = 0.7
                    else:
                        h_init[i] = 0.3
                h_init = h_init / np.sum(h_init) * N/2
                h_init = np.clip(h_init, 0, 1)
            else:
                # Random
                rng = np.random.RandomState(start_idx * 1234 + 7)
                h_init = rng.uniform(0, 1, N)
                h_init = h_init / np.sum(h_init) * N/2
                h_init = np.clip(h_init, 0, 1)
            
            try:
                def obj(h):
                    return smooth_max_overlap(h, beta)
                
                def obj_grad(h):
                    return smooth_max_overlap_grad(h, beta)
                
                result = minimize(obj, h_init, jac=obj_grad, method='L-BFGS-B',
                                  bounds=bounds, constraints=constraints,
                                  options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8})
                
                h = np.clip(result.x, 0, 1)
                # Fix sum precisely
                diff = N/2 - np.sum(h)
                h += diff / N
                h = np.clip(h, 0, 1)
                # Re-fix sum after clipping
                diff = N/2 - np.sum(h)
                if abs(diff) > 1e-12:
                    # Distribute to elements not at bounds
                    mask = (h > 1e-8) & (h < 1 - 1e-8)
                    if np.sum(mask) > 0:
                        h[mask] += diff / np.sum(mask)
                    else:
                        h += diff / N
                    h = np.clip(h, 0, 1)
                
                val = max_overlap(h)
                if val < best_val:
                    best_h = h.copy()
                    best_val = val
            except Exception:
                pass
    
    # Final polish: use the actual max_overlap with SLSQP from best point
    if time.time() < deadline - 10:
        try:
            result = minimize(max_overlap, best_h, method='SLSQP', bounds=bounds,
                              constraints=constraints, options={'maxiter': 200, 'ftol': 1e-10})
            h = np.clip(result.x, 0, 1)
            diff = N/2 - np.sum(h)
            h += diff / N
            h = np.clip(h, 0, 1)
            diff = N/2 - np.sum(h)
            if abs(diff) > 1e-12:
                mask = (h > 1e-8) & (h < 1 - 1e-8)
                if np.sum(mask) > 0:
                    h[mask] += diff / np.sum(mask)
                else:
                    h += diff / N
                h = np.clip(h, 0, 1)
            val = max_overlap(h)
            if val < best_val:
                best_h = h.copy()
                best_val = val
        except Exception:
            pass
    
    # Ensure sum is exactly N/2
    diff = N/2 - np.sum(best_h)
    if abs(diff) > 1e-14:
        mask = (best_h > 1e-8) & (best_h < 1 - 1e-8)
        if np.sum(mask) > 0:
            best_h[mask] += diff / np.sum(mask)
        else:
            best_h += diff / N
        best_h = np.clip(best_h, 0, 1)
    
    return best_h
