"""Program for minimizing the normalized maximum autoconvolution of a step function."""
import numpy as np
from scipy.optimize import minimize

N = 600

# EVOLVE-BLOCK-START
def construct():
    """
    Represent the step function as h = |z|^2 where z is a complex vector.
    We minimize the maximum of the autoconvolution using gradient descent in the 
    complex domain with Wirtinger calculus, projecting back to the constraint 
    sum(|z|^2) = N after each step.
    """
    
    def get_h(z):
        return np.abs(z)**2

    def compute_obj_grad(z):
        h = get_h(z)
        # h is already normalized if we maintain sum(|z|^2)=N
        # conv = h * h
        conv = np.convolve(h, h, mode='full')
        
        # Objective is max(conv) / N
        max_val = np.max(conv)
        obj = max_val / N
        
        # Find index of max
        k_star = np.argmax(conv)
        
        # Gradient of max(conv) w.r.t h
        # d(conv[k])/d(h[j]) = h[k-j] + h[j]
        # We need to handle boundaries: h[i] is 0 if i < 0 or i >= N
        grad_h = np.zeros(N)
        
        # For each j in 0..N-1:
        # term1: h[k_star - j] if 0 <= k_star - j < N else 0
        # term2: h[j]
        
        # Vectorized calculation
        # indices for h[j] is just 0..N-1
        idx_j = np.arange(N)
        
        # indices for h[k_star - j]
        idx_k_minus_j = k_star - idx_j
        
        # mask for valid indices
        mask_valid = (idx_k_minus_j >= 0) & (idx_k_minus_j < N)
        
        # h[k_star - j] where valid, else 0
        h_shifted = np.zeros(N)
        h_shifted[mask_valid] = h[idx_k_minus_j[mask_valid]]
        
        grad_h = (h_shifted + h)
        
        # The objective is max(conv)/N. 
        # The gradient of max is the gradient of the specific term that is max.
        # So grad_obj_h = grad_h / N
        
        # Chain rule: z -> h -> obj
        # h_j = z_j * conj(z_j)
        # d(obj)/dz_j = d(obj)/dh_j * dh_j/dz_j
        # dh_j/dz_j = conj(z_j)
        grad_z = grad_h * np.conj(z) / N
        
        return obj, grad_z

    # Initialize z
    np.random.seed(42)
    # Start with a uniform distribution to avoid local minima near zero?
    # Or random?
    z = np.random.randn(N) + 1j * np.random.randn(N)
    
    # Normalize z so that sum(|z|^2) = N
    norm_z = np.linalg.norm(z)
    z = z * (np.sqrt(N) / norm_z)
    
    # Adam optimizer parameters
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    lr = 0.01
    
    m = np.zeros_like(z)
    v = np.zeros_like(z)
    
    best_obj = 1e10
    best_z = z.copy()
    
    # Number of iterations
    # N=600 is small. 2000 iterations should be fast.
    for t in range(2000):
        obj, grad_z = compute_obj_grad(z)
        
        if obj < best_obj:
            best_obj = obj
            best_z = z.copy()
        
        # Adam update
        m = beta1 * m + (1 - beta1) * grad_z
        v = beta2 * v + (1 - beta2) * (grad_z ** 2)
        
        m_hat = m / (1 - beta1 ** (t + 1))
        v_hat = v / (1 - beta2 ** (t + 1))
        
        z = z - lr * m_hat / (np.sqrt(v_hat) + eps)
        
        # Project back to constraint sum(|z|^2) = N
        norm_z = np.linalg.norm(z)
        if norm_z > 0:
            z = z * (np.sqrt(N) / norm_z)
            
        # Adaptive learning rate decay
        if t % 500 == 0 and t > 0:
            lr *= 0.8

    h = get_h(best_z)
    # Ensure sum is exactly N
    h = h * (N / np.sum(h))
    
    return list(h)
# EVOLVE-BLOCK-END
