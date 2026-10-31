import numpy as np
import random

N = 50

def construct():
    # Find a large Sidon set in {0, ..., N-1}
    # A Sidon set has distinct pairwise sums i+j (i <= j)
    # This minimizes the peak of the autoconvolution for uniform weights,
    # maximizing the ratio R = ||h||_2^2 / (||h||_1 * max(h*h))
    
    best_S = []
    best_size = 0
    
    # Increase random restarts to ensure we find the maximum size Sidon set
    # For N=50, max size is around 7-8.
    for _ in range(500):
        S = []
        used_sums = set()
        candidates = list(range(N))
        random.shuffle(candidates)
        
        for i in candidates:
            new_sums = [i + i]
            for j in S:
                new_sums.append(i + j)
                
            if any(s in used_sums for s in new_sums):
                continue
                
            S.append(i)
            used_sums.update(new_sums)
            
        if len(S) > best_size:
            best_size = len(S)
            best_S = S
            
    if not best_S:
        best_S = [0]
        
    h = np.zeros(N)
    for i in best_S:
        h[i] = 1.0 / len(best_S)
        
    return list(h)

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
