import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a sequence of 50 nonnegative heights.
    Strategy: Search for optimal 2-point and 3-point clusters.
    Based on observation that a 2-point cluster improved the score.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    best_score = 0.0
    best_h = [1.0] * N
    
    def evaluate(h):
        h = np.array(h)
        l1 = np.sum(h)
        if l1 == 0:
            return 0.0
        l2_sq = np.sum(h**2)
        # Compute autoconvolution
        # Using np.convolve for full convolution
        conv = np.convolve(h, h, mode='full')
        l_inf = np.max(conv)
        if l_inf == 0:
            return 0.0
        return l2_sq / (l1 * l_inf)

    # 1. Try uniform (seed)
    h_uniform = [1.0] * N
    score_uniform = evaluate(h_uniform)
    best_score = score_uniform
    best_h = h_uniform
    
    # 2. Try 2-point clusters
    # h = [a, b, 0, ...]
    # We can vary position and ratio
    # Due to symmetry, position might not matter for the ratio, 
    # but let's try a few positions just in case.
    
    for pos in [0, N//2, N-1]:
        for ratio in np.linspace(0.1, 10.0, 100):
            if time.time() > deadline:
                return best_h
            
            # a = ratio, b = 1. Normalize later?
            # The ratio is scale invariant if we normalize L1=1?
            # Let's just use a=ratio, b=1.
            h = [0.0] * N
            h[pos] = ratio
            h[pos+1] = 1.0 if pos+1 < N else 0.0
            if h[pos+1] == 0: # Handle edge case
                h[pos] = 1.0
                h[pos-1] = ratio if pos-1 >= 0 else 0.0
                if h[pos-1] == 0:
                    continue # Single point
                
            score = evaluate(h)
            if score > best_score:
                best_score = score
                best_h = h
    
    # 3. Try 3-point clusters
    for pos in [0, N//2, N-2]:
        for r1 in np.linspace(0.5, 2.0, 20):
            for r2 in np.linspace(0.5, 2.0, 20):
                if time.time() > deadline:
                    return best_h
                
                h = [0.0] * N
                # Check bounds
                if pos < 0 or pos+2 >= N:
                    continue
                h[pos] = r1
                h[pos+1] = 1.0
                h[pos+2] = r2
                
                score = evaluate(h)
                if score > best_score:
                    best_score = score
                    best_h = h

    return best_h
# EVOLVE-BLOCK-END
