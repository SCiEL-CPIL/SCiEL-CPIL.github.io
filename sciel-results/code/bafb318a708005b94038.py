import numpy as np
import random

N = 50

def evaluate_mask(x):
    """
    Calculate the ratio for a binary mask x.
    x is an array of 0s and 1s.
    """
    s = np.sum(x)
    if s == 0:
        return 0.0
    # Auto-correlation counts
    # r_k = sum_i x_i x_{i+k}
    # We can use FFT or direct convolution
    # Since N is small, direct is fine, but FFT is cleaner for 'full' mode if we want to be safe.
    # np.convolve(x, x, mode='full') gives counts.
    conv = np.convolve(x, x, mode='full')
    max_peak = np.max(conv)
    if max_peak == 0:
        return 0.0
    return s / max_peak

def greedy_construct(seed):
    """
    Greedy construction with a specific random seed.
    """
    random.seed(seed)
    S = set()
    x = np.zeros(N, dtype=int)
    
    for _ in range(N):
        best_score = -1
        best_idx = -1
        candidates = [i for i in range(N) if x[i] == 0]
        
        # Shuffle candidates to break ties randomly
        random.shuffle(candidates)
        
        for j in candidates:
            # Try adding j
            x[j] = 1
            score = evaluate_mask(x)
            x[j] = 0
            
            if score > best_score:
                best_score = score
                best_idx = j
        
        if best_idx != -1:
            x[best_idx] = 1
            S.add(best_idx)
            
    # Convert to heights
    if len(S) == 0:
        h = np.zeros(N)
        h[0] = 1.0
    else:
        h = np.zeros(N)
        for i in S:
            h[i] = 1.0 / len(S)
            
    return list(h)

def construct():
    best_h = None
    best_score = -1
    
    # Run multiple restarts
    for i in range(200):
        h = greedy_construct(i)
        # Calculate actual score using the formula from the task
        h_arr = np.asarray(h)
        l1 = np.sum(h_arr)
        if l1 == 0: continue
        l2_sq = np.sum(h_arr**2)
        conv = np.convolve(h_arr, h_arr, mode='full')
        max_peak = np.max(conv)
        if max_peak == 0: continue
        score = l2_sq / (l1 * max_peak)
        
        if score > best_score:
            best_score = score
            best_h = h
            
    if best_h is None:
        # Fallback
        h = np.zeros(N)
        h[0] = 1.0
        return list(h)
        
    return best_h

def compute_ratio(h):
    h = np.asarray(h)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    conv = np.convolve(h, h, mode='full')
    max_peak = np.max(conv)
    if max_peak == 0:
        return 0.0
    return l2_sq / (l1 * max_peak)
