"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_value(h):
    """Compute the task value: abs(2*N*max(conv(h,h))/sum(h)^2)"""
    s = np.sum(h)
    if s == 0:
        return np.inf
    conv = np.convolve(h, h)
    return abs(2 * N * np.max(conv) / (s * s))

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # 120s wall-clock limit with safety margin
    
    best_h = np.ones(N)
    best_val = _compute_value(best_h)  # should be 2.0
    
    # Strategy: We want to minimize abs(2*N*max(conv)/sum^2)
    # Note: conv[0] = sum(h^2) which is always positive and is the max when h is all positive
    # If we make some h negative, sum(h) can be small while sum(h^2) stays moderate
    # But we need sum(h) != 0
    
    # Key insight: max(conv) is at least conv[0] = sum(h^2) by Cauchy-Schwarz related bounds
    # Actually conv[0] = sum(h_i^2), and this is the largest element when h is non-negative
    # If we allow negative values, other lags might be larger in magnitude
    
    # Let's think about this differently:
    # value = 2*N*|max_k conv[k]| / sum(h)^2
    # We want to minimize this.
    # 
    # If h is all ones: conv[0] = N, sum = N, value = 2*N*N/N^2 = 2
    # 
    # What if h has mean zero? Then sum(h) = 0, which is invalid.
    # What if h has small mean but large variance?
    # Let h = c + v where sum(v) = 0 and sum(v^2) is large relative to c^2*N
    # sum(h) = N*c
    # conv(h,h)[0] = sum(h_i^2) = N*c^2 + sum(v_i^2)
    # For other lags: conv[k] = sum(h_i * h_{i+k}) = c^2*(N-|k|) + c*sum(v_i + v_{i+k}) + sum(v_i*v_{i+k})
    # Since sum(v) = 0, the cross terms simplify
    # 
    # The max conv is likely at lag 0: sum(h^2)
    # value ≈ 2*N*sum(h^2) / (N*c)^2 = 2*sum(h^2)/(N*c^2)
    # = 2*(N*c^2 + sum(v^2))/(N*c^2) = 2 + 2*sum(v^2)/(N*c^2)
    # This is > 2! So making variance large increases the value.
    
    # What if we make the mean large relative to variance?
    # h = c + small perturbation
    # value ≈ 2*(N*c^2 + sum(v^2))/(N*c^2) ≈ 2 + 2*sum(v^2)/(N*c^2)
    # Still >= 2.
    
    # Wait, I need to reconsider. The max is over SIGNED entries.
    # If we can make all conv entries negative... but conv[0] = sum(h^2) >= 0 always.
    # So max(conv) >= conv[0] = sum(h^2) >= 0.
    # Therefore value = 2*N*max(conv)/sum(h)^2 >= 2*N*sum(h^2)/sum(h)^2 >= 2*N*N*mean(h^2)/(N*mean(h))^2 = 2*mean(h^2)/mean(h)^2
    # By Cauchy-Schwarz, mean(h^2) >= mean(h)^2, so value >= 2.
    # Equality when all h_i are equal (constant).
    
    # So the minimum possible value is 2, achieved by any constant vector!
    # The seed already achieves this.
    
    # But wait - let me re-examine. The problem says "lower is better" and seed gives 2.0.
    # If 2 is the theoretical minimum, then we can't do better than 2.
    
    # Let me verify with a simple example:
    # h = [1, -1, 1, -1] (N=4)
    # sum = 0, invalid
    
    # h = [1, 1, -1, -1] (N=4)  
    # sum = 0, invalid
    
    # h = [2, 1, -1, -1] (N=4)
    # sum = 1
    # conv[0] = 4+1+1+1 = 7
    # conv[1] = 2*1 + 1*(-1) + (-1)*(-1) = 2-1+1 = 2
    # conv[2] = 2*(-1) + 1*(-1) = -2-1 = -3
    # conv[3] = 2*(-1) = -2
    # max = 7
    # value = 2*4*7/1 = 56 >> 2
    
    # So indeed, the minimum is 2, achieved by constant vectors.
    # The seed is already optimal!
    
    # But the task says "score is seed/value" so if value=2, score=1.0
    # And we want score > 1.0, meaning value < 2.
    # But I just argued value >= 2 always.
    
    # Let me re-read the formula: abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h) ** 2)
    # The abs is on the whole thing. And np.max takes the maximum of signed entries.
    
    # Hmm, what if sum(h) is negative? Then sum(h)^2 is still positive.
    # What if max(conv) is negative? That can't happen since conv[0] = sum(h^2) >= 0.
    
    # I'm convinced the minimum is 2. Let me just return the seed.
    # But let me try a few random vectors to be sure.
    
    np.random.seed(42)
    for trial in range(100):
        if time.time() > deadline - 5:
            break
        h = np.random.randn(N)
        val = _compute_value(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    # Try structured vectors
    for trial in range(50):
        if time.time() > deadline - 5:
            break
        # Try vectors with one large value and rest small
        h = np.zeros(N)
        h[0] = 1.0
        h[1:] = np.random.randn(N-1) * 0.01
        val = _compute_value(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    return best_h
# EVOLVE-BLOCK-END
