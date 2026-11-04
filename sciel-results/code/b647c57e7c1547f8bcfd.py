"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s
    
    target_sum = N / 2.0
    
    def compute_overlaps(h):
        """Compute overlap C(k) = sum_i h_i * (1 - h_{(i+k)%N}) for all k."""
        H = np.fft.fft(h)
        power = H * np.conj(H)
        ac = np.fft.ifft(power).real
        overlaps = target_sum - ac
        return overlaps
    
    def project_h(h):
        """Project h to [0,1]^N with sum = N/2 using iterative Bregman projection."""
        h = np.clip(h, 0.0, 1.0)
        for _ in range(100):
            s = np.sum(h)
            if abs(s - target_sum) < 1e-14:
                break
            diff = (target_sum - s) / N
            mask = (h > 1e-6) & (h < 1.0 - 1e-6)
            n_mask = np.sum(mask)
            if n_mask == 0:
                break
            h[mask] += diff
            h = np.clip(h, 0.0, 1.0)
        # Final precision fix
        s = np.sum(h)
        if abs(s - target_sum) > 1e-12:
            mask = (h > 1e-6) & (h < 1.0 - 1e-6)
            if np.sum(mask) > 0:
                h[mask] += (target_sum - s) / np.sum(mask)
                h = np.clip(h, 0.0, 1.0)
        return h
    
    def objective_and_grad(h, beta=50.0):
        """Compute soft-max objective and gradient using FFT-based convolution."""
        overlaps = compute_overlaps(h)
        # Log-sum-exp for max approximation
        scaled = beta * overlaps
        max_s = np.max(scaled)
        lse = max_s + np.log(np.sum(np.exp(scaled - max_s)))
        obj = lse / beta
        
        # Softmax weights
        w = np.exp(scaled - max_s)
        w = w / np.sum(w)
        
        # Gradient: dObj/dh[i] = sum_k w[k] * dC[k]/dh[i]
        # C[k] = sum_j h[j](1 - h[j+k])
        # dC[k]/dh[i] = (1 - h[i+k]) - h[i-k]
        # sum_k w[k] * (1 - h[i+k]) = 1 - (w * h)[i]
        # sum_k w[k] * h[i-k] = (w * h)[i]
        # So grad[i] = 1 - 2*(w*h)[i]
        W = np.fft.fft(w)
        H = np.fft.fft(h)
        conv = np.fft.ifft(W * H).real
        grad = 1.0 - 2.0 * conv
        return obj, grad
    
    def objective_for_scipy(x):
        """Objective function for scipy.optimize.minimize."""
        h = project_h(x)
        overlaps = compute_overlaps(h)
        return np.max(overlaps)
    
    def grad_for_scipy(x):
        """Gradient for scipy.optimize.minimize."""
        h = project_h(x)
        _, grad = objective_and_grad(h, beta=50.0)
        return grad
    
    # Start from uniform with small perturbation
    np.random.seed(42)
    h0 = np.full(N, 0.5) + 0.01 * np.random.randn(N)
    h0 = project_h(h0)
    
    # Phase 1: Gradient descent with adaptive learning rate
    lr = 0.05
    best_h = h0.copy()
    best_obj = np.max(compute_overlaps(h0))
    h = h0.copy()
    
    for it in range(300):
        if time.time() > deadline - 20:
            break
        
        obj, grad = objective_and_grad(h, beta=50.0)
        
        if best_obj <= 49.75 + 1e-6:
            break
        
        h_new = project_h(h - lr * grad)
        obj_new = np.max(compute_overlaps(h_new))
        
        if obj_new < best_obj:
            best_obj = obj_new
            best_h = h_new.copy()
            lr = min(lr * 1.03, 0.15)
        else:
            lr *= 0.7
            if lr < 1e-8:
                break
        
        h = h_new
    
    # Phase 2: L-BFGS-B refinement from best point
    if time.time() < deadline - 15:
        try:
            res = minimize(
                objective_for_scipy,
                best_h,
                method='L-BFGS-B',
                jac=grad_for_scipy,
                bounds=[(0.0, 1.0)] * N,
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
            )
            h_lbfgs = project_h(res.x)
            obj_lbfgs = np.max(compute_overlaps(h_lbfgs))
            if obj_lbfgs < best_obj:
                best_obj = obj_lbfgs
                best_h = h_lbfgs.copy()
        except Exception:
            pass
    
    # Phase 3: Multiple restarts with different perturbations
    for r in range(8):
        if time.time() > deadline - 10:
            break
        
        np.random.seed(42 + r * 7919)
        scale = 0.02 + 0.03 * r
        h_r = np.full(N, 0.5) + scale * np.random.randn(N)
        h_r = project_h(h_r)
        
        obj_r = np.max(compute_overlaps(h_r))
        if obj_r < best_obj:
            best_obj = obj_r
            best_h = h_r.copy()
            
            # Short gradient descent from this point
            h = h_r.copy()
            lr = 0.04
            for it2 in range(150):
                if time.time() > deadline - 5:
                    break
                obj, grad = objective_and_grad(h, beta=50.0)
                if best_obj <= 49.75 + 1e-6:
                    break
                h_new = project_h(h - lr * grad)
                obj_new = np.max(compute_overlaps(h_new))
                if obj_new < best_obj:
                    best_obj = obj_new
                    best_h = h_new.copy()
                    lr = min(lr * 1.02, 0.12)
                else:
                    lr *= 0.75
                    if lr < 1e-8:
                        break
                h = h_new
            
            # L-BFGS-B polish
            if time.time() < deadline - 8:
                try:
                    res = minimize(
                        objective_for_scipy,
                        best_h,
                        method='L-BFGS-B',
                        jac=grad_for_scipy,
                        bounds=[(0.0, 1.0)] * N,
                        options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 30}
                    )
                    h_lbfgs = project_h(res.x)
                    obj_lbfgs = np.max(compute_overlaps(h_lbfgs))
                    if obj_lbfgs < best_obj:
                        best_obj = obj_lbfgs
                        best_h = h_lbfgs.copy()
                except Exception:
                    pass
    
    # Final precision adjustment
    final_h = project_h(best_h)
    s = np.sum(final_h)
    if abs(s - target_sum) > 1e-10:
        mask = (final_h > 1e-6) & (final_h < 1.0 - 1e-6)
        if np.sum(mask) > 0:
            final_h[mask] += (target_sum - s) / np.sum(mask)
            final_h = np.clip(final_h, 0.0, 1.0)
    
    # Verify constraints
    final_h = np.clip(final_h, 0.0, 1.0)
    s = np.sum(final_h)
    if abs(s - target_sum) > 1e-8:
        # Last resort: adjust a few elements
        mask = (final_h > 0.01) & (final_h < 0.99)
        if np.sum(mask) > 0:
            final_h[mask] += (target_sum - s) / np.sum(mask)
            final_h = np.clip(final_h, 0.0, 1.0)
    
    return final_h

# EVOLVE-BLOCK-END
