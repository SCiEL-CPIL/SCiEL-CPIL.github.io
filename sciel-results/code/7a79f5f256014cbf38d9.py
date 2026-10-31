import numpy as np
import random

N = 50

def construct():
    best_set = []
    best_size = 0
    
    # Number of random restarts
    restarts = 200
    
    for _ in range(restarts):
        S = []
        sums = set()
        
        # Randomized greedy: shuffle indices
        indices = list(range(N))
        random.shuffle(indices)
        
        for x in indices:
            # Check if x can be added to S
            valid = True
            new_sums = []
            
            # Check against existing elements in S
            for y in S:
                s = x + y
                if s in sums:
                    valid = False
                    break
                new_sums.append(s)
            
            # Check self-sum
            s_self = 2 * x
            if s_self in sums:
                valid = False
            else:
                new_sums.append(s_self)
                
            if valid:
                S.append(x)
                for s in new_sums:
                    sums.add(s)
                    
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
            
    # If no set found (should not happen), fallback to single element
    if not best_set:
        best_set = [0]
        
    h = np.zeros(N)
    k = len(best_set)
    for i in best_set:
        h[i] = 1.0 / k
        
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
