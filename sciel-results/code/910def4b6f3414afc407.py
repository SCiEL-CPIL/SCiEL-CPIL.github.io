"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _autoconv_max_abs(x):
    """Compute max absolute value of autoconvolution of x."""
    # Autoconvolution: (x * x)[k] = sum_i x[i] x[i+k]
    # Use FFT for efficiency
    n = len(x)
    size = 2 * n
    X = np.fft.rfft(x, size)
    conv = np.fft.irfft(X * X, size)[:n]
    return np.max(np.abs(conv))

def _integral_sq(x):
    """Squared integral (sum of squares for discrete step function)."""
    return np.sum(x * x)

def _objective(x):
    """Objective: max_abs_autoconv / integral_sq. Lower is better."""
    denom = _integral_sq(x)
    if denom < 1e-12:
        return 1e10
    return _autoconv_max_abs(x) / denom

def _gradient_num(x, eps=1e-6):
    """Numerical gradient of the objective."""
    grad = np.zeros_like(x)
    f0 = _objective(x)
    for i in range(len(x)):
        xp = x.copy()
        xp[i] += eps
        xm = x.copy()
        xm[i] -= eps
        grad[i] = (_objective(xp) - _objective(xm)) / (2 * eps)
    return grad

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline with 10s safety margin
    
    # Initialize with random signs
    np.random.seed(42)
    x = np.random.choice([-1.0, 1.0], size=N)
    
    best_x = x.copy()
    best_score = _objective(best_x)
    
    # Local search: flip individual bits to reduce objective
    # Use O(N) incremental update for autoconvolution when flipping one bit
    
    def _flip_effect(x, i):
        """Compute change in max_abs_autoconv when flipping x[i]."""
        # When we flip x[i], the autoconvolution changes.
        # conv[k] = sum_j x[j] x[j+k]
        # Flipping x[i] changes conv by: -2*x[i]*x[i+k] for each k where j=i or j+k=i
        # i.e., for k such that i is in [0, N-1] and i+k is in [0, N-1]
        # And for k such that i-k is in [0, N-1] (i.e., j = i-k)
        
        # Let's compute the delta to conv
        old_val = x[i]
        new_val = -old_val
        delta = new_val - old_val  # = -2*old_val
        
        # conv[k] = sum_j x[j]*x[j+k]
        # Terms involving x[i]: 
        #   j = i: x[i]*x[i+k] for k in [-i, N-1-i]
        #   j+k = i => j = i-k: x[i-k]*x[i] for k in [-(N-1-i), i]
        
        # The change in conv[k] is delta * (x[i+k] + x[i-k]) where both are valid
        # But we need to be careful about double counting when j=i and j+k=i (k=0)
        
        # Actually, let's just recompute the affected parts of conv
        # For efficiency, precompute the current conv
        
        return None  # Placeholder
    
    # Simpler approach: maintain running conv and update incrementally
    def _compute_conv(x):
        n = len(x)
        size = 2 * n
        X = np.fft.rfft(x, size)
        conv = np.fft.irfft(X * X, size)[:n]
        return conv
    
    def _flip_update_conv(x, conv, i):
        """Update conv when flipping x[i]. Returns new conv."""
        new_conv = conv.copy()
        old_val = x[i]
        delta = -2 * old_val  # change in x[i]
        
        # conv[k] = sum_j x[j] x[j+k]
        # Change comes from terms where j=i or j+k=i
        # j=i: term is x[i]*x[i+k], changes by delta*x[i+k]
        # j+k=i => j=i-k: term is x[i-k]*x[i], changes by x[i-k]*delta
        # When k=0, both apply: x[i]*x[i] appears once in sum, so total change is delta*x[i] + x[i]*delta = 2*delta*x[i]
        # Wait, let me reconsider.
        
        # conv[k] = sum_{j=0}^{N-1-k} x[j]*x[j+k]
        # For a fixed k, the terms involving index i are:
        #   j = i (if i <= N-1-k): x[i]*x[i+k]
        #   j = i-k (if i-k >= 0 and i-k <= N-1-k): x[i-k]*x[i]
        # If both conditions hold (i <= N-1-k and i-k >= 0), and if i == i-k (k=0), it's the same term.
        
        # Let's just iterate over all k and update
        for k in range(N):
            # j = i: valid if i + k < N
            if i + k < N:
                new_conv[k] += delta * x[i + k]
            # j = i - k: valid if i - k >= 0 and i - k + k < N (always true if i-k >= 0)
            # Actually j = i-k must satisfy 0 <= i-k <= N-1-k, i.e., k <= i and i-k <= N-1-k => i <= N-1
            if i - k >= 0:
                # j = i-k, and j+k = i, so term is x[i-k]*x[i]
                # But we already counted this if i+k < N and k=0? No, different k.
                # For k > 0, j=i-k is different from j=i.
                # But wait, when k=0, j=i and j=i-k=i are the same.
                if k > 0:
                    new_conv[k] += x[i - k] * delta
                else:
                    # k=0: the term x[i]*x[i] is counted once. 
                    # The change is: new term is (-x[i])*(-x[i]) = x[i]^2, old was x[i]^2. No change?
                    # Wait, no. If we flip x[i], x[i] becomes -x[i].
                    # Old term: x[i]*x[i] = x[i]^2
                    # New term: (-x[i])*(-x[i]) = x[i]^2
                    # So no change for k=0 from the j=i term!
                    # But from j=i-k=i-k with k=0, it's the same term.
                    # So actually for k=0, flipping x[i] doesn't change conv[0] at all?
                    # Let's verify: conv[0] = sum_j x[j]^2. Flipping x[i] doesn't change x[i]^2.
                    # So conv[0] is invariant under sign flips! That's a key insight.
                    pass
        
        return new_conv
    
    # Since conv[0] is invariant, and we're minimizing max|conv|/sum(x^2),
    # and sum(x^2) = N for ±1 values, the objective is just max|conv|/N.
    # So we want to minimize max|conv|.
    
    # Let's use a simpler local search with full recomputation but limit iterations
    def _local_search(x, max_iters=5000):
        """Flip bits to reduce max_abs_autoconv."""
        x = x.copy()
        conv = _compute_conv(x)
        current_max = np.max(np.abs(conv))
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            best_i = -1
            best_new_max = current_max + 1e-10
            
            # Try flipping each bit
            for i in range(N):
                # Compute new conv after flipping x[i]
                # delta = -2*x[i]
                delta = -2 * x[i]
                
                new_conv = conv.copy()
                for k in range(N):
                    if i + k < N:
                        new_conv[k] += delta * x[i + k]
                    if i - k >= 0 and k > 0:
                        new_conv[k] += x[i - k] * delta
                
                new_max = np.max(np.abs(new_conv))
                if new_max < best_new_max:
                    best_new_max = new_max
                    best_i = i
                    best_new_conv = new_conv
            
            if best_i >= 0:
                x[best_i] = -x[best_i]
                conv = best_new_conv
                current_max = best_new_max
            else:
                break
        
        return x, current_max
    
    # Run local search
    best_x, best_max = _local_search(best_x, max_iters=3000)
    best_score = best_max / N  # since sum(x^2) = N for ±1
    
    # Try multiple random restarts if time permits
    num_restarts = 5
    for r in range(num_restarts):
        if time.time() > deadline - 20:
            break
        
        x_init = np.random.choice([-1.0, 1.0], size=N)
        x_res, max_res = _local_search(x_init, max_iters=1000)
        score_res = max_res / N
        
        if score_res < best_score:
            best_score = score_res
            best_x = x_res
    
    # Return the signed step heights
    return best_x
# EVOLVE-BLOCK-END
