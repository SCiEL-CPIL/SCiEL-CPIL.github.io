"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import eigh

# Precompute the circulant autoconvolution matrix properties.
# The autoconvolution c_k = sum_{i=0}^{N-1} h_i h_{i+k} (circular) can be expressed via FFT.
# c = IFFT(|FFT(h)|^2)
# The objective is max_k |c_k| / (sum(h)^2).
#
# Strategy:
# 1. Use spectral decomposition of the circulant matrix corresponding to autoconvolution.
#    The circulant matrix C has eigenvalues lambda_j = |H_j|^2 where H_j = FFT(h)_j.
#    Actually, the autoconvolution vector is c = IFFT(|FFT(h)|^2).
#    The max absolute value of c is the max over k of |IFFT(|FFT(h)|^2)_k|.
#
# 2. Since the problem is homogeneous (scaling h doesn't change the ratio), we can fix sum(h) = 1.
#    Then the objective is max_k |c_k|.
#
# 3. We use a smooth surrogate: soft-max of |c_k| using log-sum-exp.
#    Let z_k = |c_k|. The soft-max is log(sum(exp(z_k / tau))) * tau for small tau.
#    But we need the max, not log-sum-exp of exp(z). Actually, max(z) ≈ log(sum(exp(z/tau))) * tau as tau -> 0.
#    However, z_k can be negative? No, c_k is real, but |c_k| is non-negative.
#    Wait, c_k = sum h_i h_{i+k}. This can be negative. So |c_k| is non-negative.
#
# 4. We minimize soft_max(|c|) subject to sum(h) = 1.
#    We use gradient descent with projection or constrained optimization.
#
# Let's use scipy.optimize.minimize with SLSQP or trust-constr.
# But N=400, so the vector is 400-dim. The objective involves FFTs, which are fast.

def compute_objective_and_grad(h, tau=0.1):
    """
    Compute the soft-max of |autoconvolution| and its gradient.
    h: array of shape (N,)
    tau: temperature for soft-max
    """
    # Compute autoconvolution via FFT
    H = np.fft.fft(h)
    power = np.abs(H)**2
    c = np.real(np.fft.ifft(power))
    
    # Clip tiny negative values due to numerical error
    c = np.maximum(c, 0)  # Actually c can be negative in theory, but |c| is what we want
    # Wait, the autoconvolution c_k = sum_i h_i * h_{(i+k) mod N}.
    # This is not necessarily non-negative. So we need |c_k|.
    
    abs_c = np.abs(c)
    
    # Soft-max: log(sum(exp(abs_c / tau))) * tau
    # To avoid overflow, subtract max
    m = np.max(abs_c)
    if m < 1e-12:
        return 0.0, np.zeros_like(h)
    
    exp_terms = np.exp((abs_c - m) / tau)
    sum_exp = np.sum(exp_terms)
    obj = (m + tau * np.log(sum_exp))
    
    # Gradient: d/dh [tau * log(sum exp(|c_k|/tau))]
    # = sum_k (exp(|c_k|/tau) / sum_j exp(|c_j|/tau)) * sign(c_k) * d|c_k|/dh
    # d|c_k|/dh = sign(c_k) * dc_k/dh
    # dc_k/dh_i = h_{(i+k) mod N} + h_{(i-k) mod N} (for k != 0)
    # Actually, c_k = sum_i h_i h_{i+k}
    # dc_k/dh_j = h_{j+k} + h_{j-k} (with modular indexing)
    # But this is for the circular convolution.
    
    # Let's compute the gradient using the chain rule.
    # Let f_k = |c_k|
    # df_k/dh = sign(c_k) * (shift(h, k) + shift(h, -k))
    # where shift(h, k)_j = h_{(j+k) mod N}
    
    # The soft-max gradient weights
    weights = exp_terms / sum_exp  # shape (N,)
    
    grad = np.zeros_like(h)
    for k in range(N):
        if abs_c[k] < 1e-12:
            continue
        s = np.sign(c[k])
        if s == 0:
            continue
        # dc_k/dh = roll(h, k) + roll(h, -k)
        dc_dh = np.roll(h, k) + np.roll(h, -k)
        # But for k=0, dc_0/dh = 2*h
        # For k=N/2 (if N even), dc_k/dh = 2*roll(h, k)
        # Let's handle generally
        grad += weights[k] * s * dc_dh
    
    # The objective is obj, gradient is grad * tau? No.
    # d/dh [tau * log(sum exp(f_k/tau))] = sum_k (exp(f_k/tau)/sum_j exp(f_j/tau)) * df_k/dh
    # So grad is already the correct gradient of the soft-max objective.
    
    return obj, grad


def construct():
    deadline = time.time() + 140.0  # Safety margin
    
    # Initial guess: random or ones
    h0 = np.ones(N)
    
    # We want to minimize max|c_k| subject to sum(h) = 1
    # Let's use a smooth optimization
    
    best_h = h0.copy()
    best_obj = np.inf
    
    # Try a few different initializations
    np.random.seed(42)
    
    for init_idx in range(10):
        if time.time() > deadline:
            break
        
        if init_idx == 0:
            h_init = np.ones(N)
        else:
            h_init = np.random.randn(N)
            h_init += 2.0  # Make mostly positive
            h_init /= np.sum(h_init)
        
        # Use L-BFGS-B with equality constraint sum(h)=1
        # Or use penalty method
        
        # Let's use a simple projected gradient descent
        # Or use scipy minimize with constraint
        
        # For speed, let's use a few iterations of projected gradient
        h = h_init.copy()
        h /= np.sum(h)  # Normalize
        
        tau = 0.5
        lr = 0.01
        n_iters = 500
        
        for it in range(n_iters):
            if time.time() > deadline:
                break
            
            obj, grad = compute_objective_and_grad(h, tau)
            
            # Project gradient onto the constraint sum(h)=1
            # The constraint is g(h) = sum(h) - 1 = 0
            # Projected gradient: grad - (sum(grad)/N) * ones
            grad_proj = grad - np.mean(grad)
            
            # Step
            h_new = h - lr * grad_proj
            
            # Project back to sum(h)=1
            h_new = h_new - np.mean(h_new) + 1.0/N
            # Actually, to satisfy sum(h)=1 exactly:
            h_new = h_new / np.sum(h_new) if np.sum(h_new) != 0 else h
            
            # Check convergence
            if np.max(np.abs(h_new - h)) < 1e-8:
                h = h_new
                break
            
            h = h_new
        
        # Evaluate the true objective (max abs)
        H = np.fft.fft(h)
        power = np.abs(H)**2
        c = np.real(np.fft.ifft(power))
        true_obj = np.max(np.abs(c)) / (np.sum(h)**2)
        
        if true_obj < best_obj:
            best_obj = true_obj
            best_h = h.copy()
        
        # Try reducing tau for a more precise search
        if init_idx < 3:
            # Do a second phase with smaller tau
            tau = 0.1
            lr = 0.005
            n_iters2 = 300
            
            for it in range(n_iters2):
                if time.time() > deadline:
                    break
                
                obj, grad = compute_objective_and_grad(h, tau)
                grad_proj = grad - np.mean(grad)
                h_new = h - lr * grad_proj
                h_new = h_new / np.sum(h_new) if np.sum(h_new) != 0 else h
                h = h_new
            
            H = np.fft.fft(h)
            power = np.abs(H)**2
            c = np.real(np.fft.ifft(power))
            true_obj = np.max(np.abs(c)) / (np.sum(h)**2)
            
            if true_obj < best_obj:
                best_obj = true_obj
                best_h = h.copy()
    
    # Normalize to make the ratio well-defined
    if np.sum(best_h) != 0:
        best_h = best_h / np.sum(best_h)
    
    return best_h
# EVOLVE-BLOCK-END
