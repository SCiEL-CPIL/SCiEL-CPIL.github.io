"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Construct the step heights h to minimize the maximum overlap of h with shifted 1-h.
    
    Strategy:
    1. Start with a non-uniform alternating pattern (a, 1-a) which breaks the 
       symmetry of the uniform solution and provides a lower initial peak overlap 
       in certain lags, serving as a better starting point for refinement.
    2. Use a spectral/fourier-based intuition: by optimizing the continuous 
       heights via gradient descent on a Log-Sum-Exp surrogate of the max 
       overlap, we effectively adjust the 'phase' and 'amplitude' of the 
       underlying patterns to minimize the peak sidelobe level.
    3. Refine using gradient descent with projection to constraints.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # 1. Initial Guess: Alternating Pattern
    # From Inspiration 2: a = (N-2) / (2(N-1)) minimizes the local alternating overlap
    a = (N - 2) / (2 * (N - 1))
    h = np.zeros(N)
    for i in range(N):
        if i % 2 == 0:
            h[i] = a
        else:
            h[i] = 1 - a
            
    # Ensure exact sum N/2
    current_sum = np.sum(h)
    target_sum = N / 2.0
    if current_sum > 0:
        h = h * (target_sum / current_sum)
    h = np.clip(h, 0.0, 1.0)
    
    # 2. Optimization Setup
    beta = 10.0  # Sharpness of LSE approximation
    lr = 0.05    # Learning rate
    
    def compute_overlaps(h_vec):
        """
        Compute linear overlaps C[k] for all shifts k.
        C[k] = sum_i h[i] * (1 - h[i+k]) where indices are valid.
        We compute all non-zero shifts.
        """
        # Using correlate for efficiency if possible, but manual is clearer for gradients
        # Let's stick to the efficient vectorized gradient approach from the current program
        # but ensure we are computing the linear (not circular) overlaps if that's the metric.
        # The prompt says "shifted 1-h". Usually implies linear convolution/correlation in this context.
        # However, the current program used circular roll. Let's check the inspiration.
        # Inspiration 1 used np.correlate(h, g, mode='full') and took max excluding center.
        # This is linear overlap.
        # The current program used np.roll (circular).
        # Let's use linear overlap as it is more standard for "overlap with shifted" in finite sequences.
        
        g = 1.0 - h_vec
        # np.correlate(h_vec, g, mode='full')
        # The result length is 2N-1.
        # The center (shift 0) is at index N-1.
        # We want max of all shifts k != 0.
        
        c = np.correlate(h_vec, g, mode='full')
        # Exclude the center index N-1
        # The shifts are from -(N-1) to N-1.
        # We want max over all k != 0.
        
        # For gradient computation, we need to be careful.
        # Let's define the objective as max over k in 1..N-1 of C(k) and C(-k).
        # Due to symmetry? No, C(k) is not necessarily symmetric unless h is symmetric.
        # C(k) = sum h[i](1-h[i+k])
        # C(-k) = sum h[i](1-h[i-k]) = sum h[i+k](1-h[i]) = sum h[j](1-h[j-k]) ... 
        # Actually, let's just compute the gradient for the specific shifts used in the max.
        
        # To keep it simple and robust, let's compute the overlaps explicitly for the gradient step.
        # N=100 is small enough that O(N^2) per step is fine (10,000 ops).
        
        overlaps = np.zeros(N-1)
        # Shifts 1 to N-1
        for k in range(1, N):
            # h[i] * (1-h[i+k]) for i where i+k < N
            # i ranges from 0 to N-k-1
            h_part = h_vec[:N-k]
            g_part = 1.0 - h_vec[k:]
            overlaps[k-1] = np.sum(h_part * g_part)
            
            # Also consider the "other side" shift? 
            # "max overlap of h with shifted 1-h"
            # Usually this implies max over all non-zero shifts.
            # Shift k and shift -k are distinct in linear correlation.
            # Let's include both to be safe.
            # Shift -k: h[i] * (1-h[i-k]) where i-k >= 0 => i >= k
            # i ranges from k to N-1
            h_part2 = h_vec[k:]
            g_part2 = 1.0 - h_vec[:N-k]
            overlaps[k-1] = max(overlaps[k-1], np.sum(h_part2 * g_part2))
            
        return overlaps

    def objective_and_grad(h_vec):
        overlaps = compute_overlaps(h_vec)
        
        # Stable LSE
        scaled = beta * overlaps
        max_s = np.max(scaled)
        lse = max_s + np.log(np.sum(np.exp(scaled - max_s)))
        obj = lse / beta
        
        # Gradient of LSE wrt overlaps is softmax(beta * overlaps)
        probs = np.exp(scaled - max_s)
        probs /= np.sum(probs)
        
        # Gradient of obj wrt h
        grad_h = np.zeros(N)
        
        for idx, k in enumerate(range(1, N)):
            p = probs[idx]
            if p < 1e-9:
                continue
            
            # We need dC_k/dh and dC_-k/dh
            # C_k = sum_{i=0}^{N-k-1} h[i](1-h[i+k])
            # dC_k/dh[j] = (1-h[j+k]) if j < N-k, minus h[j-k] if j >= k
            # Wait, the term is h[i](1-h[i+k]).
            # If j = i, term is (1-h[j+k]).
            # If j = i+k, term is -h[j-k].
            
            # Contribution from C_k (positive shift k)
            # For j in 0..N-k-1: derivative wrt h[j] is (1-h[j+k])
            # For j in k..N-1: derivative wrt h[j] is -h[j-k]
            
            # Vectorized update for C_k
            # grad_h[0:N-k] += p * (1 - h_vec[k:N])
            # grad_h[k:N] += p * (-h_vec[0:N-k])
            
            grad_h[:N-k] += p * (1.0 - h_vec[k:])
            grad_h[k:] += p * (-h_vec[:N-k])
            
            # Contribution from C_-k (negative shift -k, which is the max with the other side)
            # We took max(C_k, C_-k). The gradient follows the one that is larger.
            # Let's compute C_k and C_-k separately to know which one contributes.
            
            c_k = np.sum(h_vec[:N-k] * (1.0 - h_vec[k:]))
            c_m_k = np.sum(h_vec[k:] * (1.0 - h_vec[:N-k]))
            
            # Actually, my compute_overlaps took the max. 
            # For the gradient, we should only backprop through the active max.
            # If c_k >= c_m_k, we used c_k.
            # If c_m_k > c_k, we used c_m_k.
            
            # I already added the gradient for c_k above.
            # I need to add the gradient for c_m_k only if it was the max.
            # But I added p * grad(c_k).
            # If c_m_k was the max, I should have added p * grad(c_m_k) instead.
            
            # Let's correct this:
            # Subtract the gradient of c_k if c_m_k > c_k, and add gradient of c_m_k.
            # Or better, compute both and select.
            
            # Let's restart the gradient loop logic to be precise.
            
        # Let's rewrite the gradient calculation to be precise about the max.
        grad_h = np.zeros(N)
        for idx, k in enumerate(range(1, N)):
            p = probs[idx]
            if p < 1e-9:
                continue
            
            c_k = np.sum(h_vec[:N-k] * (1.0 - h_vec[k:]))
            c_m_k = np.sum(h_vec[k:] * (1.0 - h_vec[:N-k]))
            
            if c_k >= c_m_k:
                # Gradient of C_k
                # d/dh[j] sum_{i=0}^{N-k-1} h[i](1-h[i+k])
                # j in [0, N-k): term is (1-h[j+k])
                # j in [k, N): term is -h[j-k]
                grad_h[:N-k] += p * (1.0 - h_vec[k:])
                grad_h[k:] += p * (-h_vec[:N-k])
            else:
                # Gradient of C_-k
                # C_-k = sum_{i=k}^{N-1} h[i](1-h[i-k])
                # Let j be index in h.
                # If j = i, term is (1-h[j-k]) for j in [k, N)
                # If j = i-k, term is -h[j+k] for j in [0, N-k)
                grad_h[k:] += p * (1.0 - h_vec[:N-k])
                grad_h[:N-k] += p * (-h_vec[k:])
                
        return obj, grad_h

    # Perform gradient descent
    for step in range(50):
        if time.time() > deadline:
            break
            
        obj, grad = objective_and_grad(h)
        
        # Gradient step
        h_new = h - lr * grad
        
        # Projection to constraints: 0 <= h_i <= 1, sum h_i = N/2
        h_new = np.clip(h_new, 0.0, 1.0)
        s = np.sum(h_new)
        if s > 1e-9:
            h_new = h_new * (N / 2.0) / s
        h_new = np.clip(h_new, 0.0, 1.0)
        
        h = h_new

    # Final precision polish: ensure sum is exactly N/2
    sum_h = np.sum(h)
    diff = (N / 2.0) - sum_h
    if abs(diff) > 1e-12:
        if h[0] + diff <= 1.0 and h[0] + diff >= 0.0:
            h[0] += diff
        else:
            h = h * (N / 2.0) / sum_h

    return h

# EVOLVE-BLOCK-END
