import numpy as np
import time
from scipy.optimize import minimize

_START_TIME = time.time()
_TIME_LIMIT = 140.0

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _compute_corrs_fft(h):
    """Compute circular autocorrelations C_k for k=0..N-1."""
    n = len(h)
    H = np.fft.fft(h)
    C = np.fft.ifft(H * np.conj(H)).real
    return C

def _project_to_simplex(v, target_sum):
    """Project v onto {x : 0 <= x <= 1, sum(x) = target_sum}."""
    n = len(v)
    # First project onto {x >= 0, sum = target_sum}
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho_candidates = np.nonzero(u * np.arange(1, n + 1) > (cssv - target_sum))[0]
    if len(rho_candidates) == 0:
        rho = n - 1
    else:
        rho = rho_candidates[-1]
    theta = (cssv[rho] - target_sum) / (rho + 1.0)
    w = np.maximum(v - theta, 0)
    
    # Clip to [0, 1]
    w = np.clip(w, 0, 1)
    
    # Adjust to ensure sum = target_sum
    for _ in range(20):
        diff = target_sum - np.sum(w)
        if abs(diff) < 1e-12:
            break
        if diff > 0:
            mask = w < 1.0
            cnt = np.sum(mask)
            if cnt == 0:
                break
            w[mask] += diff / cnt
        else:
            mask = w > 0.0
            cnt = np.sum(mask)
            if cnt == 0:
                break
            w[mask] += diff / cnt
        w = np.clip(w, 0, 1)
    
    return w

def _compute_min_corr(h):
    """Compute min_{k=1..N-1} C_k."""
    C = _compute_corrs_fft(h)
    # C[0] is C_0. We want min of C[1:] and C[0] is excluded?
    # The task says "shifted 1-h". Shifts are non-zero.
    # So we look at k=1..N-1.
    return np.min(C[1:])

def construct():
    N = 100
    target_sum = N / 2.0
    best_h = np.full(N, 0.5)
    best_min_corr = _compute_min_corr(best_h)
    
    # Initializations
    inits = []
    inits.append(np.full(N, 0.5))
    
    # Structured inits
    for freq in [1, 2, 3, 4, 5]:
        x = np.linspace(0, 2*np.pi, N, endpoint=False)
        inits.append(0.5 + 0.4 * np.sin(freq * x))
        inits.append(0.5 + 0.4 * np.cos(freq * x))
        
    # Block inits
    for frac in [0.3, 0.4, 0.5, 0.6, 0.7]:
        k = int(N * frac)
        inits.append(np.concatenate([np.full(k, 0.0), np.full(N-k, 2.0*target_sum/(N-k))]))
        inits.append(np.concatenate([np.full(k, 2.0*target_sum/k), np.full(N-k, 0.0)]))
        
    # Random
    np.random.seed(42)
    for _ in range(10):
        inits.append(np.random.uniform(0, 1, N))

    betas = [10, 50, 100, 500, 1000, 2000]
    
    for h0 in inits:
        if _time_remaining() < 10.0:
            break
        h = _project_to_simplex(h0, target_sum)
        
        for beta in betas:
            if _time_remaining() < 5.0:
                break
            
            # Projected Gradient Ascent on J(h)
            step_size = 1.0 / (beta * N) # Heuristic
            for it in range(200):
                if _time_remaining() < 5.0:
                    break
                
                C = _compute_corrs_fft(h)
                # Weights for k=1..N-1
                C_nonzero = np.delete(C, 0)
                # Soft min weights
                min_C = np.min(C_nonzero)
                shifted = C_nonzero - min_C
                exp_terms = np.exp(-beta * shifted)
                Z = np.sum(exp_terms)
                W_nonzero = exp_terms / Z
                
                # Construct full W vector with W[0]=0
                W = np.zeros(N)
                W[1:] = W_nonzero
                # Note: C is symmetric, so W is symmetric.
                # Gradient of J is (2/beta) * (h * W)
                # h * W is circular convolution
                grad = (2.0 / beta) * np.fft.ifft(np.fft.fft(h) * np.fft.fft(W)).real
                
                # Line search or fixed step
                h_new = h + step_size * grad
                h_new = _project_to_simplex(h_new, target_sum)
                
                # Check improvement
                new_min_corr = _compute_min_corr(h_new)
                if new_min_corr > best_min_corr:
                    best_min_corr = new_min_corr
                    best_h = h_new
                    h = h_new
                else:
                    # Reduce step size
                    step_size *= 0.5
                    if step_size < 1e-6:
                        break

    # Final Polish with Subgradient on Hard Min
    if _time_remaining() > 10.0:
        h = best_h.copy()
        step_size = 0.01
        for it in range(500):
            if _time_remaining() < 5.0:
                break
            
            C = _compute_corrs_fft(h)
            min_corr = np.min(C[1:])
            # Find indices where min is achieved
            mask = C[1:] < min_corr + 1e-6
            k_indices = np.nonzero(mask)[0] + 1
            
            # Subgradient of min_corr is average of gradients of active C_k
            # Grad of C_k is h_{j+k} + h_{j-k}
            grad = np.zeros(N)
            for k in k_indices:
                grad += np.roll(h, -k) + np.roll(h, k)
            grad /= len(k_indices)
            
            h_new = h + step_size * grad
            h_new = _project_to_simplex(h_new, target_sum)
            
            new_min_corr = _compute_min_corr(h_new)
            if new_min_corr > best_min_corr:
                best_min_corr = new_min_corr
                best_h = h_new
                h = h_new
                step_size = min(step_size * 1.05, 0.1)
            else:
                step_size *= 0.5
                if step_size < 1e-6:
                    break

    # Ensure exact sum and bounds
    best_h = np.clip(best_h, 0, 1)
    diff = target_sum - np.sum(best_h)
    if abs(diff) > 1e-12:
        mask = (best_h > 0) & (best_h < 1)
        if np.sum(mask) > 0:
            best_h[mask] += diff / np.sum(mask)
            best_h = np.clip(best_h, 0, 1)
            
    return best_h
