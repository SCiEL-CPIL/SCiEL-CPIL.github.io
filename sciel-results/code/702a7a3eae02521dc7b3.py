"""
Program for minimizing the normalized maximum of autoconvolution of a
nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Approach:
- The uniform (box) function gives a triangular autoconvolution.
- We use an iterative proportional fitting / projected gradient method to
  find a shape that minimizes the peak of its autoconvolution while keeping
  the L2 norm fixed and all values nonnegative.
- This is a convex optimization problem in the space of nonnegative vectors
  with fixed L2 norm. We use a simple projected gradient descent with
  line search on the objective max(conv(h, h)) / ||h||_2^2.
- We start from a Gaussian-like profile which may have a flatter convolution
  peak than the box in some regimes, then optimize.

Actually, for the discrete case with fixed L2 norm, the minimum of the max
autoconvolution is achieved by the uniform distribution (by symmetry and
convexity arguments in the discrete Fourier domain). However, we implement
a robust optimization that verifies this and can find any improvement.
"""

import numpy as np

N = 600

# EVOLVE-BLOCK-START
def construct():
    # Start with a slightly tapered (Gaussian) profile, then optimize
    x = np.linspace(-0.5, 0.5, N)
    sigma = 0.35
    h = np.exp(-0.5 * (x / sigma) ** 2)
    h = np.maximum(h, 1e-12)
    
    # Normalize to unit L2 norm
    h = h / np.sqrt(np.sum(h ** 2))
    
    # Objective: max(autoconv) / ||h||^2 = max(autoconv) since ||h||=1
    def objective(h):
        ac = np.convolve(h, h, mode='full')
        return np.max(ac)
    
    # Projected gradient descent with projection onto {h >= 0, ||h|| = 1}
    best_h = h.copy()
    best_obj = objective(h)
    
    lr = 0.01
    for it in range(5000):
        ac = np.convolve(h, h, mode='full')
        # Gradient of max(ac) w.r.t. h: if max is at index k,
        # d/dh_i ac_k = h_{k-i} (for valid indices)
        k_max = np.argmax(ac)
        # Gradient contribution
        grad = np.zeros(N)
        # ac_k = sum_i h_i * h_{k-i}
        # d ac_k / d h_j = h_{k-j} + h_{j-k} (for the full convolution indexing)
        # Actually for autoconv: ac_m = sum_i h_i h_{m-i}
        # d ac_m / d h_j = h_{m-j} (if m-j in [0,N-1]) + h_j (if j-m in [0,N-1])... 
        # Let's just use: grad_j = h_{k_max - j} + h_{j - k_max} where defined
        idx1 = k_max - np.arange(N)
        idx2 = np.arange(N) - k_max
        mask1 = (idx1 >= 0) & (idx1 < N)
        mask2 = (idx2 >= 0) & (idx2 < N)
        grad[mask1] += h[idx1[mask1]]
        grad[mask2] += h[idx2[mask2]]
        
        # Also account for the normalization constraint ||h||=1
        # Use Lagrangian: L = max(ac) - lambda*(||h||^2 - 1)
        # But simpler: just project after step
        
        # Gradient step
        h_new = h - lr * grad
        
        # Project: nonnegative and unit L2 norm
        h_new = np.maximum(h_new, 0)
        norm = np.sqrt(np.sum(h_new ** 2))
        if norm > 1e-12:
            h_new = h_new / norm
        
        obj_new = objective(h_new)
        
        if obj_new < best_obj:
            best_obj = obj_new
            best_h = h_new.copy()
        
        h = h_new
        
        # Adaptive learning rate
        if obj_new > best_obj:
            lr *= 0.95
        else:
            lr = min(lr * 1.01, 0.1)
    
    return list(best_h)
# EVOLVE-BLOCK-END
