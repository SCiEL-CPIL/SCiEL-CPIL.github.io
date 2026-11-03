import numpy as np
import time

N = 600
DELTA = 1.0 / 1200.0
TARGET_NORM_SQ = 1.0 / DELTA  # 1200

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # Leave margin for 150s limit
    
    best_h = None
    best_score = float('inf')
    
    # Initial seed: constant
    h_const = np.ones(N)
    h_const /= np.sqrt(np.sum(h_const**2) / TARGET_NORM_SQ)
    # Actually just normalize
    norm = np.sqrt(np.sum(h_const**2))
    h_const = h_const * (np.sqrt(TARGET_NORM_SQ) / norm)
    
    conv = np.convolve(h_const, h_const, mode='full')
    score = DELTA * np.max(conv)
    best_h = h_const
    best_score = score
    
    # Random search
    rng = np.random.default_rng(42)
    
    iterations = 0
    while time.time() < deadline:
        # Generate random non-negative vector
        h = rng.random(N)
        
        # Normalize
        norm = np.sqrt(np.sum(h**2))
        if norm == 0:
            continue
        h = h * (np.sqrt(TARGET_NORM_SQ) / norm)
        
        # Compute convolution
        conv = np.convolve(h, h, mode='full')
        curr_score = DELTA * np.max(conv)
        
        if curr_score < best_score:
            best_score = curr_score
            best_h = h
        
        iterations += 1
        
        # Optional: try some structured patterns
        # e.g. alternating
        if iterations % 10 == 0:
             h_alt = np.array([1.0 if i%2==0 else 0.0 for i in range(N)])
             norm = np.sqrt(np.sum(h_alt**2))
             h_alt = h_alt * (np.sqrt(TARGET_NORM_SQ) / norm)
             conv = np.convolve(h_alt, h_alt, mode='full')
             curr_score = DELTA * np.max(conv)
             if curr_score < best_score:
                 best_score = curr_score
                 best_h = h_alt

    return list(best_h)
