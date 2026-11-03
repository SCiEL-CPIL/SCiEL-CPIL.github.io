"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import sys

N = 100

def _overlap_matrix(h):
    """Compute overlap_k = sum_i h_i * (1 - h_{(i+k) mod N}) for all k."""
    N = len(h)
    overlaps = np.zeros(N)
    for k in range(N):
        shifted = np.roll(h, -k)
        overlaps[k] = np.sum(h * (1.0 - shifted))
    return overlaps

def _max_overlap(h):
    return np.max(_overlap_matrix(h))

def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill
    
    # Initial guess: alternating pattern or random
    # Try a few initializations
    best_h = None
    best_score = np.inf
    
    # Initialization 1: alternating 0, 1, 0, 1, ...
    h1 = np.zeros(N)
    h1[::2] = 1.0
    h1[1::2] = 0.0
    # Adjust to make sum exactly N/2
    # Already sum = N/2 if N is even
    
    # Initialization 2: random binary
    np.random.seed(42)
    h2 = np.zeros(N)
    idx = np.random.permutation(N)
    h2[idx[:N//2]] = 1.0
    
    # Initialization 3: smooth sinusoidal
    x = np.linspace(0, 2*np.pi, N, endpoint=False)
    h3 = 0.5 + 0.5*np.cos(x)
    h3 = np.clip(h3, 0, 1)
    # Normalize to sum N/2
    h3 = h3 / h3.sum() * (N/2)
    h3 = np.clip(h3, 0, 1)
    
    candidates = [h1, h2, h3]
    
    for h0 in candidates:
        if time.time() > deadline - 30:
            break
        h = h0.copy()
        
        # Use log-sum-exp smooth surrogate for max
        def loss(h, alpha=10.0):
            ov = _overlap_matrix(h)
            # Soft max via log-sum-exp
            m = np.max(ov)
            return m + np.log(np.sum(np.exp(alpha*(ov - m))))/alpha
        
        # Gradient descent with projection
        lr = 0.01
        alpha = 5.0
        for it in range(5000):
            if time.time() > deadline - 5:
                break
            
            ov = _overlap_matrix(h)
            m = np.max(ov)
            # Soft max gradient
            exps = np.exp(alpha*(ov - m))
            weights = exps / np.sum(exps)
            
            # Gradient of soft-max w.r.t. h
            # d(soft_max)/dh_i = sum_k weights_k * d(overlap_k)/dh_i
            # overlap_k = sum_j h_j * (1 - h_{(j+k) mod N})
            # d(overlap_k)/dh_i = (1 - h_{(i+k) mod N}) + h_{(i-k) mod N} * (-1) 
            # Wait, let me be more careful.
            # overlap_k = sum_j h_j (1 - h_{(j+k) mod N})
            # d/dh_i [overlap_k] = (1 - h_{(i+k) mod N}) + sum_j h_j * d/dh_i[1 - h_{(j+k) mod N}]
            # The second term: d/dh_i[1 - h_{(j+k) mod N}] = -1 if (j+k) mod N == i, i.e., j = (i-k) mod N
            # So = (1 - h_{(i+k) mod N}) - h_{(i-k) mod N}
            
            grad = np.zeros(N)
            for k in range(N):
                wk = weights[k]
                term = (1.0 - np.roll(h, -k)) - np.roll(h, k)
                grad += wk * term
            
            # Project gradient step
            h_new = h - lr * grad
            # Project to [0,1] and adjust sum to N/2
            h_new = np.clip(h_new, 0, 1)
            # Adjust to make sum exactly N/2
            diff = N/2 - h_new.sum()
            if abs(diff) > 1e-10:
                # Distribute the difference
                h_new = h_new + diff / N
                h_new = np.clip(h_new, 0, 1)
                # Final fix: if still off, adjust individual elements
                diff = N/2 - h_new.sum()
                if abs(diff) > 1e-10:
                    # Add to elements that have room
                    for i in range(N):
                        if diff > 0 and h_new[i] < 1.0:
                            add = min(diff, 1.0 - h_new[i])
                            h_new[i] += add
                            diff -= add
                        elif diff < 0 and h_new[i] > 0.0:
                            sub = min(-diff, h_new[i])
                            h_new[i] -= sub
                            diff += sub
                        if abs(diff) < 1e-10:
                            break
            
            h = h_new
            lr = max(lr * 0.999, 0.001)
        
        score = _max_overlap(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()
    
    if best_h is None:
        return np.full(N, 0.5)
    
    # Ensure sum is exactly N/2
    diff = N/2 - best_h.sum()
    if abs(diff) > 1e-10:
        for i in range(N):
            if diff > 0 and best_h[i] < 1.0:
                add = min(diff, 1.0 - best_h[i])
                best_h[i] += add
                diff -= add
            elif diff < 0 and best_h[i] > 0.0:
                sub = min(-diff, best_h[i])
                best_h[i] -= sub
                diff += sub
            if abs(diff) < 1e-10:
                break
    
    return best_h
