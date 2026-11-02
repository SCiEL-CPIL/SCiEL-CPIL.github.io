"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy import optimize

N = 600

def compute_score(heights):
    """Compute the normalized maximum of the autoconvolution."""
    heights = np.asarray(heights, dtype=np.float64)
    n = len(heights)
    # Compute autoconvolution via FFT
    f = np.fft.fft(heights)
    ac = np.fft.ifft(f * np.conj(f)).real
    s = np.sum(heights)
    if s < 1e-12:
        return 0.0
    return np.max(ac) * N / (s * s)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    # Initialize h
    # Start with uniform, but perturb to escape symmetry
    h = np.ones(N)
    h += 0.01 * np.random.randn(N)
    h = np.maximum(h, 0)
    h /= np.linalg.norm(h)
    
    best_h = h.copy()
    best_score = compute_score(best_h)
    
    # Softmax temperature
    T = 1.0 
    # Learning rate
    lr = 0.1
    
    for i in range(5000):
        if time.time() > deadline:
            break
        
        # Compute ac
        f = np.fft.fft(h)
        ac = np.fft.ifft(f * np.conj(f)).real
        
        # Compute weights for soft-max
        # ac values are positive. Scale them for softmax stability.
        # Max ac is around 1/N if ||h||_2=1? No, for uniform ||h||_2=1, ac[0]=1.
        # So ac[0] is the largest.
        ac_scaled = ac / T
        
        # Softmax weights
        ac_max = np.max(ac_scaled)
        exp_ac = np.exp(ac_scaled - ac_max)
        w = exp_ac / np.sum(exp_ac)
        
        # Gradient of sum(w_k * ac_k) wrt h
        # grad = 2 * ifft(conj(fft(w)) * fft(h))
        fw = np.fft.fft(w)
        fh = np.fft.fft(h)
        grad = 2.0 * np.fft.ifft(np.conj(fw) * fh).real
        
        # Update h
        h_new = h - lr * grad
        
        # Enforce non-negativity
        h_new = np.maximum(h_new, 0)
        
        # Normalize to unit L2 norm
        norm = np.linalg.norm(h_new)
        if norm < 1e-12:
            # If it died, restart with uniform
            h_new = np.ones(N)
            h_new += 0.01 * np.random.randn(N)
            h_new = np.maximum(h_new, 0)
            norm = np.linalg.norm(h_new)
        h_new /= norm
        
        # Check score
        score = compute_score(h_new)
        if score < best_score:
            best_score = score
            best_h = h_new.copy()
            h = h_new.copy()
            # Maybe decrease T to sharpen the penalty on the max?
            T *= 0.95
            if T < 0.01:
                T = 0.01
        else:
            # If no improvement, adjust learning rate
            lr *= 0.9
            if lr < 1e-5:
                break
                
    # Refinement: Use scipy minimize on the best candidate found
    # We optimize the log-sum-exp of the autoconvolution, which is a smooth surrogate for the max
    # Objective: minimize log(sum(exp(ac / T))) subject to h >= 0, sum(h) = 1 (or fixed norm)
    # To handle constraints, we can use a change of variables h = exp(z) / sum(exp(z)) or just minimize on h with projection.
    # However, scipy's L-BFGS-B can handle bounds.
    
    # Let's try a few restarts from the best_h found, or just polish best_h.
    # We will use a higher temperature T for the log-sum-exp to make it smooth.
    
    def objective(z):
        """
        Objective function for scipy.
        z: parameter vector. We use z = log(h) to enforce h > 0.
        But h can be 0, so log is problematic. 
        Instead, let's minimize directly on h with bounds [0, inf].
        But gradient computation for max-ac is hard.
        Let's stick to the log-sum-exp surrogate.
        """
        h_val = np.exp(z)
        # Normalize h_val to sum 1? Or L2 norm 1?
        # The score is scale invariant. Let's fix sum(h) = 1 for optimization stability.
        h_norm = h_val / np.sum(h_val)
        
        # Compute autoconvolution
        f = np.fft.fft(h_norm)
        ac = np.fft.ifft(f * np.conj(f)).real
        
        # Log-sum-exp of ac / T
        # T should be small enough to approximate max, but large enough for smoothness.
        # The scale of ac is roughly 1/N if sum=1.
        # Let's use a fixed T for this phase, e.g., 0.01 * max(ac)
        ac_max = np.max(ac)
        if ac_max < 1e-12:
            return 0.0
        T_opt = max(0.01 * ac_max, 1e-6)
        
        # log-sum-exp
        # log(sum(exp(ac/T))) = max(ac/T) + log(sum(exp((ac - max(ac))/T)))
        shifted = (ac - ac_max) / T_opt
        lse = ac_max / T_opt + np.log(np.sum(np.exp(shifted)))
        
        return lse

    def gradient(z):
        h_val = np.exp(z)
        h_norm = h_val / np.sum(h_val)
        
        f = np.fft.fft(h_norm)
        ac = np.fft.ifft(f * np.conj(f)).real
        
        ac_max = np.max(ac)
        if ac_max < 1e-12:
            return np.zeros_like(z)
        
        T_opt = max(0.01 * ac_max, 1e-6)
        shifted = (ac - ac_max) / T_opt
        exp_shifted = np.exp(shifted)
        sum_exp = np.sum(exp_shifted)
        w = exp_shifted / sum_exp
        
        # Gradient of LSE wrt h
        # d/dh sum(w_k * ac_k) = 2 * ifft(conj(fft(w)) * fft(h))
        fw = np.fft.fft(w)
        fh = np.fft.fft(h_norm)
        grad_h = 2.0 * np.fft.ifft(np.conj(fw) * fh).real
        
        # Gradient wrt z: h = exp(z), so dh/dz = diag(h)
        # Also h_norm = h / sum(h).
        # This derivative is complex. Let's use numerical gradient if it's too slow or tricky?
        # No, let's derive it.
        # Let S = sum(h). h_n = h/S.
        # ac = h_n * h_n (circular).
        # d ac / d h_n = 2 * (h_n * I) (convolution operator).
        # d h_n / d h = (1/S) * (I - h h^T / S).
        # So d ac / d h = (2/S) * (Conv(h_n) - (h^T Conv(h_n)) h / S).
        # Then d LSE / d h = sum_k w_k d ac_k / d h.
        # Let G = 2 * Conv(w) * h_n (this is the grad_h above, but scaled by 1/S? No, grad_h was for h_n directly).
        # Actually, grad_h computed above is d/dh_n sum(w*ac).
        # So d LSE / d h = (1/S) * (grad_h - (h^T grad_h) h / S).
        
        h_dot_grad = np.dot(h, grad_h)
        grad_h_unnorm = (1.0 / np.sum(h)) * (grad_h - (h_dot_grad / np.sum(h)) * h)
        
        # d h / d z = h
        grad_z = grad_h_unnorm * h
        
        return grad_z

    # Try polishing best_h
    # Convert best_h to z = log(best_h + epsilon)
    eps = 1e-12
    z0 = np.log(best_h + eps)
    
    try:
        result = optimize.minimize(
            objective, 
            z0, 
            method='L-BFGS-B', 
            jac=gradient,
            options={'maxiter': 100, 'gtol': 1e-8, 'ftol': 1e-12}
        )
        
        if result.success or result.fun < objective(z0):
            z_opt = result.x
            h_opt = np.exp(z_opt)
            h_opt = h_opt / np.sum(h_opt) # Normalize to sum 1
            # Scale back to original norm if needed, but score is invariant.
            # Let's check score
            score_opt = compute_score(h_opt)
            if score_opt < best_score:
                best_score = score_opt
                best_h = h_opt.copy()
                
    except Exception:
        pass

    return best_h.tolist()

# EVOLVE-BLOCK-END
