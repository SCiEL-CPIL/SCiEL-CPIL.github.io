import time
import numpy as np

N = 600

def compute_score(h):
    s = np.sum(h)
    if s == 0:
        return 0.0
    conv = np.convolve(h, h, mode='full')
    max_conv = np.max(conv)
    return max_conv / (s * s)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # The uniform distribution is the analytical global minimizer for the
    # normalized peak autoconvolution of non-negative functions on a continuous domain.
    # For discrete step functions, we test if any structured distribution improves the score.
    
    # Try several candidate distributions and pick the best
    candidates = []
    
    # 1. Uniform
    h_uniform = np.full(N, 1.0)
    candidates.append(h_uniform)
    
    # 2. Raised cosine (smooth taper at edges)
    x = np.linspace(-1, 1, N)
    h_cosine = 0.5 * (1 + np.cos(np.pi * x))
    h_cosine = np.maximum(h_cosine, 1e-10)  # Ensure nonnegative
    candidates.append(h_cosine)
    
    # 3. Slightly concave (higher in middle)
    h_concave = 1.0 + 0.1 * np.cos(np.pi * x)
    h_concave = np.maximum(h_concave, 1e-10)
    candidates.append(h_concave)
    
    # 4. Slightly convex (lower in middle)
    h_convex = 1.0 - 0.1 * np.cos(np.pi * x)
    h_convex = np.maximum(h_convex, 1e-10)
    candidates.append(h_convex)
    
    # Evaluate all candidates
    best_h = None
    best_score = float('inf')
    
    for h in candidates:
        s = np.sum(h)
        if s == 0:
            continue
        conv = np.convolve(h, h, mode='full')
        max_conv = np.max(conv)
        score = max_conv / (s * s)
        
        if score < best_score:
            best_score = score
            best_h = h
    
    if time.time() > deadline:
        return (best_h if best_h is not None else h_uniform).tolist()
        
    return (best_h if best_h is not None else h_uniform).tolist()
# EVOLVE-BLOCK-END
