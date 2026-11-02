"""
Program for minimizing the normalized maximum of autoconvolution of a
nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Approach:
- We use Iterative Proportional Fitting (IPF) / Multiplicative Weights Update
  (Mirror Descent) to minimize the peak of the autoconvolution.
- Unlike gradient descent, this method uses exponential updates based on the
  gradient of the peak constraint, which naturally enforces non-negativity
  and can converge to flatter peaks more efficiently.
- The update rule is h_j <- h_j * exp(-eta * grad_j), followed by L2 normalization.
"""

import numpy as np

N = 600

# EVOLVE-BLOCK-START
def construct():
    # Start with a uniform distribution (box function)
    # This is a strong baseline. The optimization will taper the edges if beneficial.
    h = np.ones(N)
    h = h / np.sqrt(np.sum(h ** 2))
    
    best_h = h.copy()
    best_obj = np.max(np.convolve(h, h, mode='full'))
    
    # Hyperparameters
    eta = 0.1  # Learning rate for multiplicative updates
    max_iters = 2000
    patience = 100
    no_improve_count = 0
    
    for it in range(max_iters):
        # Compute autoconvolution
        ac = np.convolve(h, h, mode='full')
        current_obj = np.max(ac)
        
        if current_obj < best_obj:
            best_obj = current_obj
            best_h = h.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > patience:
                break
        
        # Find the peak index
        k_max = np.argmax(ac)
        
        # Compute gradient of the peak w.r.t. h
        # d/dh_j (sum_i h_i h_{k-i}) = h_{k-j} + h_{j-k} (if indices valid)
        # Note: In full convolution, ac_m = sum_{i=0}^{N-1} h_i h_{m-i} where h is 0 outside [0, N-1]
        # Actually, standard np.convolve(h, h, 'full') has length 2N-1.
        # ac[m] = sum_{i} h[i] h[m-i]
        # d ac[m] / d h[j] = h[m-j] + h[j-m]
        
        j_idx = np.arange(N)
        idx1 = k_max - j_idx
        idx2 = j_idx - k_max
        
        grad = np.zeros(N)
        mask1 = (idx1 >= 0) & (idx1 < N)
        mask2 = (idx2 >= 0) & (idx2 < N)
        
        # The gradient contribution from the first term h_{k-j}
        grad[mask1] += h[idx1[mask1]]
        # The gradient contribution from the second term h_{j-k}
        grad[mask2] += h[idx2[mask2]]
        
        # Multiplicative update: h <- h * exp(-eta * grad)
        # To avoid underflow/overflow, we can normalize the exponent or use a small eta
        # Since h is normalized to L2=1, and grad is roughly O(1), eta=0.1 is reasonable.
        # We subtract the mean of the log-update to maintain stability? No, we renormalize.
        
        # Clip the exponent to prevent numerical issues
        exponent = -eta * grad
        # Center the exponent to avoid large magnitude shifts that might cause underflow
        # This step is optional but helps stability.
        exponent -= np.mean(exponent)
        
        h_new = h * np.exp(exponent)
        
        # Ensure non-negativity (exp is always positive, so this is safe)
        # Normalize to L2 = 1
        norm = np.sqrt(np.sum(h_new ** 2))
        if norm < 1e-12:
            # Fallback to uniform if it collapses
            h = np.ones(N)
            h = h / np.sqrt(np.sum(h ** 2))
            continue
            
        h = h_new / norm
        
        # Adaptive learning rate
        if current_obj > best_obj:
            eta *= 0.95
        else:
            eta = min(eta * 1.01, 0.5)

    return list(best_h)
# EVOLVE-BLOCK-END
