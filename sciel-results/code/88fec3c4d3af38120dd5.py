"""
Erdos minimum overlap problem:
Choose h[i] in [0,1] for i in 0..N-1 with sum(h) = N/2.
Minimize the maximum over shifts k of overlap_k = sum_i h[i]*(1-h[(i+k) mod N]).

We formulate this as minimizing t subject to:
  sum_i h[i] = N/2
  0 <= h[i] <= 1
  For all k: sum_i h[i]*(1-h[(i+k) mod N]) <= t

This is a non-convex QCQP (quadratic constraints with bilinear terms).
We use a penalty method + gradient descent with a smooth max surrogate.

Key insight: overlap_k = sum_i h[i] - sum_i h[i]*h[(i+k) mod N]
               = N/2 - sum_i h[i]*h[(i+k) mod N]   (using the sum constraint)

So minimizing max_k overlap_k is equivalent to maximizing min_k (sum_i h[i]*h[(i+k) mod N]).
Let Q_k = sum_i h[i]*h[(i+k) mod N]. We want to maximize the minimum Q_k.

This is a maximin problem. We can use a smooth surrogate:
  f(h) = -logsumexp(-Q_k / tau)  (for some temperature tau)
Minimize f(h) subject to constraints.

We'll use a projected gradient method with:
1. Log-sum-exp smoothing for the max
2. Penalty for the sum constraint
3. Projection onto [0,1]^N after each step

Additionally, we'll try spectral initialization:
The optimal h should have structure related to the eigenvectors of the circulant matrix.
For a circulant matrix C with C_{i,j} = 1 if (i-j) mod N is in some set,
the eigenvectors are Fourier modes. The problem has translational symmetry, so
we expect the solution to be a sum of a few Fourier modes.

Strategy:
1. Initialize with a few harmonic modes
2. Use projected gradient descent with log-sum-exp smoothing
3. Anneal the temperature tau
4. Final polish with exact max evaluation
"""

import numpy as np
import time

N = 100
TIME_LIMIT = 130.0  # Leave margin for the 150s kill

def construct():
    start_time = time.time()
    
    # Precompute the circulant structure
    # Q_k = sum_i h[i] * h[(i+k) % N]
    # We can compute all Q_k efficiently using FFT
    
    def compute_Q(h):
        """Compute Q_k = sum_i h[i] * h[(i+k) % N] for all k."""
        # This is the circular cross-correlation of h with itself
        H = np.fft.rfft(h, N)
        Q = np.fft.irfft(H * np.conj(H), N)[:N]
        # Q[k] = sum_i h[i] * h[(i+k) % N]
        return Q
    
    def compute_overlap(h):
        """Compute overlap_k = N/2 - Q_k for all k."""
        Q = compute_Q(h)
        return N / 2.0 - Q
    
    def project_box(h):
        """Project onto [0, 1]^N."""
        return np.clip(h, 0.0, 1.0)
    
    def enforce_sum(h, target_sum=N/2.0):
        """Adjust h to satisfy sum constraint, keeping in [0,1]."""
        # Simple approach: scale and clip, then iterate
        for _ in range(50):
            s = np.sum(h)
            if abs(s - target_sum) < 1e-12:
                break
            diff = target_sum - s
            # Distribute difference evenly, then clip
            h = h + diff / N
            h = np.clip(h, 0.0, 1.0)
            # If clipping changed the sum, redistribute among non-bound variables
            s2 = np.sum(h)
            if abs(s2 - target_sum) > 1e-12:
                # Find non-bound variables
                non_bound = (h > 1e-8) & (h < 1.0 - 1e-8)
                if np.any(non_bound):
                    h[non_bound] += (target_sum - s2) / np.sum(non_bound)
                    h = np.clip(h, 0.0, 1.0)
                else:
                    break
        return h
    
    # Gradient of the log-sum-exp surrogate
    # f(h) = -logsumexp(-Q_k / tau)
    # df/dh_i = (1/tau) * sum_k exp(-Q_k/tau) / sum_j exp(-Q_j/tau) * dQ_k/dh_i
    # Q_k = sum_j h[j] * h[(j+k)%N]
    # dQ_k/dh_i = h[(i-k)%N] + h[(i+k)%N]  (for k != 0; for k=0 it's 2*h[i])
    
    def surrogate_grad(h, tau):
        """Gradient of -logsumexp(-Q/tau) with respect to h."""
        Q = compute_Q(h)
        # Soft max weights
        x = -Q / tau
        x_max = np.max(x)
        exp_x = np.exp(x - x_max)
        weights = exp_x / np.sum(exp_x)  # weights[k] proportional to exp(-Q_k/tau)
        
        # dQ_k/dh_i for each i
        # Q_k = sum_j h[j] * h[(j+k)%N]
        # dQ_k/dh_i = h[(i-k)%N] + h[(i+k)%N]  (for k != 0)
        # For k=0: dQ_0/dh_i = 2*h[i]
        
        # grad_i = (1/tau) * sum_k weights[k] * dQ_k/dh_i
        #        = (1/tau) * [h[i] * 2*weights[0] + sum_{k=1}^{N-1} weights[k] * (h[(i-k)%N] + h[(i+k)%N])]
        
        # The sum over k of weights[k] * h[(i-k)%N] is a circular convolution of weights with h
        wh = np.fft.rfft(weights, N)
        H = np.fft.rfft(h, N)
        conv = np.fft.irfft(wh * H, N)[:N]
        # conv[i] = sum_k weights[k] * h[(i-k)%N]
        
        # Similarly sum_k weights[k] * h[(i+k)%N]
        wh_flip = np.fft.rfft(weights[::-1] if N % 2 == 0 else weights, N)
        # Actually, let's just compute it directly for the reverse
        # sum_k weights[k] * h[(i+k)%N] = sum_k weights[-k % N] * h[(i-k)%N]
        # So it's conv of weights[::-1] (with appropriate wrapping) with h
        weights_rev = np.zeros(N)
        weights_rev[0] = weights[0]
        for k in range(1, N):
            weights_rev[k] = weights[N-k]
        wh2 = np.fft.rfft(weights_rev, N)
        conv2 = np.fft.irfft(wh2 * H, N)[:N]
        
        # grad = (1/tau) * (2*weights[0]*h + conv + conv2)
        # But wait, for k=0, dQ_0/dh_i = 2*h[i], and weights[0] is included in both conv and conv2
        # Actually, let me reconsider:
        # dQ_k/dh_i = sum_j [d(h[j])/dh_i * h[(j+k)%N] + h[j] * d(h[(j+k)%N])/dh_i]
        #           = h[(i+k)%N] + h[(i-k)%N]  (for k != 0)
        # For k=0: dQ_0/dh_i = 2*h[i]
        
        # So: sum_k weights[k] * dQ_k/dh_i
        #     = weights[0] * 2*h[i] + sum_{k=1}^{N-1} weights[k] * (h[(i+k)%N] + h[(i-k)%N])
        #     = 2*weights[0]*h[i] + sum_k weights[k]*h[(i+k)%N] + sum_k weights[k]*h[(i-k)%N] - 2*weights[0]*h[i] + ... 
        #     Actually: sum_{k=1}^{N-1} weights[k]*h[(i+k)%N] = conv2[i] - weights[0]*h[i]
        #     and sum_{k=1}^{N-1} weights[k]*h[(i-k)%N] = conv[i] - weights[0]*h[i]
        # So total = 2*weights[0]*h[i] + (conv2[i] - weights[0]*h[i]) + (conv[i] - weights[0]*h[i])
        #          = conv[i] + conv2[i]
        
        grad = (1.0 / tau) * (conv + conv2)
        return grad
    
    # Try multiple initializations and pick the best
    best_h = None
    best_score = np.inf
    
    num_restarts = 5
    for restart in range(num_restarts):
        if time.time() - start_time > TIME_LIMIT - 10:
            break
        
        # Initialize with a mixture of Fourier modes
        if restart == 0:
            # Start with constant
            h = np.full(N, 0.5)
        elif restart == 1:
            # Add a cosine mode
            h = 0.5 + 0.3 * np.cos(2 * np.pi * 1 * np.arange(N) / N)
            h = np.clip(h, 0, 1)
        elif restart == 2:
            # Add a different cosine mode
            h = 0.5 + 0.4 * np.cos(2 * np.pi * 2 * np.arange(N) / N)
            h = np.clip(h, 0, 1)
        elif restart == 3:
            # Sine mode
            h = 0.5 + 0.3 * np.sin(2 * np.pi * 1 * np.arange(N) / N)
            h = np.clip(h, 0, 1)
        else:
            # Random
            h = np.random.rand(N) * 0.5 + 0.25
        
        h = project_box(h)
        h = enforce_sum(h)
        
        # Simulated annealing on tau
        tau_start = 2.0
        tau_end = 0.01
        n_anneal = 80
        
        for step in range(n_anneal):
            if time.time() - start_time > TIME_LIMIT - 5:
                break
            
            # Anneal tau
            progress = step / max(1, n_anneal - 1)
            tau = tau_start * (tau_end / tau_start) ** progress
            
            # Compute gradient
            grad = surrogate_grad(h, tau)
            
            # Also add a penalty for sum constraint violation
            s = np.sum(h)
            grad += 0.5 * (s - N / 2.0)
            
            # Gradient descent step
            lr = 0.1 / (1 + step * 0.01)
            h_new = h - lr * grad
            
            # Project
            h_new = project_box(h_new)
            h_new = enforce_sum(h_new)
            
            # Check improvement
            Q_old = compute_Q(h)
            Q_new = compute_Q(h_new)
            # We want to maximize min Q_k, so minimize -min Q_k
            # Use the smooth surrogate value
            def surrogate_value(h_curr, tau_curr):
                Q_curr = compute_Q(h_curr)
                x = -Q_curr / tau_curr
                x_max = np.max(x)
                return -np.log(np.sum(np.exp(x - x_max))) - x_max
            
            val_old = surrogate_value(h, tau)
            val_new = surrogate_value(h_new, tau)
            
            if val_new < val_old or np.random.rand() < np.exp((val_new - val_old) / tau):
                h = h_new
        
        # Final polish: use exact max with smaller steps
        for polish in range(30):
            if time.time() - start_time > TIME_LIMIT - 3:
                break
            
            Q = compute_Q(h)
            k_max = np.argmax(N / 2.0 - Q)  # k with max overlap
            
            # Gradient of the exact max (subgradient)
            # d(overlap_k)/dh_i for the maximizing k
            # overlap_k = N/2 - sum_j h[j]*h[(j+k)%N]
            # d/dh_i = -h[(i+k)%N] - h[(i-k)%N]
            
            grad_exact = -h[(np.arange(N) + k_max) % N] - h[(np.arange(N) - k_max) % N]
            
            s = np.sum(h)
            grad_exact += 0.5 * (s - N / 2.0)
            
            lr = 0.05
            h_new = h - lr * grad_exact
            h_new = project_box(h_new)
            h_new = enforce_sum(h_new)
            
            # Accept if it improves the max overlap
            ov_old = np.max(N / 2.0 - Q)
            Q_new = compute_Q(h_new)
            ov_new = np.max(N / 2.0 - Q_new)
            
            if ov_new <= ov_old + 1e-12:
                h = h_new
        
        # Evaluate
        Q = compute_Q(h)
        overlaps = N / 2.0 - Q
        score = np.max(overlaps)
        
        if score < best_score:
            best_score = score
            best_h = h.copy()
    
    # If we didn't get a good solution, return the constant
    if best_h is None:
        best_h = np.full(N, 0.5)
    
    # Final check: ensure sum is exactly N/2
    best_h = enforce_sum(best_h)
    best_h = project_box(best_h)
    
    # Verify sum
    if abs(np.sum(best_h) - N / 2.0) > 1e-6:
        # Fallback to constant
        best_h = np.full(N, 0.5)
    
    return best_h
