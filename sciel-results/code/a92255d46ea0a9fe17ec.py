import numpy as np
import time
from scipy.optimize import minimize

N = 600

def compute_score(h):
    """Compute the normalized maximum of autoconvolution."""
    h = np.maximum(h, 0)
    area = np.sum(h)
    max_val = np.max(h)
    if area == 0 or max_val == 0:
        return float('inf')
    conv = np.convolve(h, h, mode='full')
    return np.max(conv) / (area * max_val)

def soft_max_conv(h, lam=10.0):
    """Compute soft-max of autoconvolution for smooth optimization."""
    h = np.maximum(h, 0)
    area = np.sum(h)
    max_val = np.max(h)
    if area == 0 or max_val == 0:
        return float('inf')
    conv = np.convolve(h, h, mode='full')
    # Scale conv to avoid overflow in exp
    # max conv is roughly area * max_val
    # We want to minimize max(conv) / (area * max_val)
    # So we can just minimize soft_max(conv) / (area * max_val)
    # But area and max_val are also variables.
    # It's better to minimize log(sum(exp(lam * conv))) - lam * (log(area) + log(max_val)) ?
    # No, just minimize the ratio using the soft-max in the numerator.
    max_conv_soft = np.log(np.sum(np.exp(lam * (conv - np.max(conv))))) / lam + np.max(conv)
    return max_conv_soft / (area * max_val)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Candidate shapes
    candidates = []
    
    # 1. Uniform
    h_uni = np.ones(N)
    candidates.append(h_uni)
    
    # 2. Delta
    h_delta = np.zeros(N)
    h_delta[N//2] = 1
    candidates.append(h_delta)
    
    # 3. Box (width N/2)
    h_box = np.zeros(N)
    h_box[N//4 : 3*N//4] = 1
    candidates.append(h_box)
    
    # 4. Gaussian
    x = np.arange(N)
    center = N//2
    sigma = N/10
    h_gauss = np.exp(-(x-center)**2 / (2*sigma**2))
    candidates.append(h_gauss)
    
    # 5. Two spikes
    h_two = np.zeros(N)
    h_two[N//4] = 1
    h_two[3*N//4] = 1
    candidates.append(h_two)
    
    best_h = None
    best_score = float('inf')
    
    for h in candidates:
        s = compute_score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
            
    # If we have time, run L-BFGS-B from the best candidate
    if time.time() < deadline - 30:
        # Use soft-max objective for smoothness
        lam = 10.0
        
        def obj(x):
            h = np.maximum(x, 0)
            if np.sum(h) == 0 or np.max(h) == 0:
                return 1e10
            return soft_max_conv(h, lam)
        
        def grad(x):
            # Gradient of soft_max_conv is complex.
            # L-BFGS-B can use numerical gradient if we don't provide one.
            # But providing one is faster.
            # Let's skip analytical gradient and let SciPy handle it?
            # No, SciPy L-BFGS-B requires a gradient or uses finite differences.
            # Finite differences on N=600 variables might be slow.
            # Let's provide a numerical gradient?
            # Actually, let's just use the gradient of the exact score? No, it's non-smooth.
            # Let's stick to Adam with restarts if L-BFGS-B is too slow.
            pass

        # Given the time limit and complexity of gradient, let's stick to a robust Adam implementation
        # but with better initialization and restarts.
        
        np.random.seed(42)
        x = np.log(best_h + 1e-6) # Inverse softplus approx
        
        lr = 0.05
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        m = np.zeros(N)
        v = np.zeros(N)
        t = 0
        
        best_h_opt = best_h.copy()
        best_score_opt = best_score
        
        while time.time() < deadline:
            t += 1
            h = np.exp(x) # Use exp instead of softplus for simplicity? 
            # Softplus is better for non-negativity. Let's use softplus.
            h = np.where(x > 30, x, np.log1p(np.exp(x)))
            
            area = np.sum(h)
            max_val = np.max(h)
            if area == 0 or max_val == 0:
                break
                
            conv = np.convolve(h, h, mode='full')
            max_conv = np.max(conv)
            score = max_conv / (area * max_val)
            
            if score < best_score_opt:
                best_score_opt = score
                best_h_opt = h.copy()
                
            # Gradient
            k_max = np.argmax(conv)
            j_idx = k_max - np.arange(N)
            valid = (j_idx >= 0) & (j_idx < N)
            grad_conv = np.zeros(N)
            grad_conv[valid] = 2.0 * h[j_idx[valid]]
            
            is_max = (h == max_val)
            n_max = np.sum(is_max)
            grad_max_val = is_max / n_max if n_max > 0 else np.zeros(N)
            
            denom = (area * max_val) ** 2
            grad_h = (grad_conv * area * max_val - max_conv * (max_val + area * grad_max_val)) / denom
            
            # Gradient of h w.r.t x
            grad_x = grad_h * (1.0 / (1.0 + np.exp(-np.clip(x, -50, 50))))
            
            # Adam
            m = beta1 * m + (1 - beta1) * grad_x
            v = beta2 * v + (1 - beta2) * grad_x ** 2
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            x -= lr * m_hat / (np.sqrt(v_hat) + eps)
            
            # Restart if stuck
            if t % 100 == 0:
                if time.time() > start_time + 30:
                    # Try a random perturbation
                    x_pert = x + np.random.normal(0, 0.1, N)
                    h_pert = np.where(x_pert > 30, x_pert, np.log1p(np.exp(x_pert)))
                    score_pert = compute_score(h_pert)
                    if score_pert < best_score_opt:
                        x = x_pert
                        m = np.zeros(N)
                        v = np.zeros(N)
                        t = 0

    if best_score_opt < best_score:
        best_h = best_h_opt
        best_score = best_score_opt
        
    return list(best_h)
