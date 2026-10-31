import numpy as np
import random

N = 50

def construct():
    best_set = []
    # Random restarts to find a large Sidon set
    # We try to find a set of size as large as possible
    # Theoretical max for N=50 is likely 7 or 8
    
    # Helper to check if x can be added to S (with sums set)
    # S is a list, sums is a set of i+j for i<=j in S
    
    max_size = 0
    best_S = []
    
    # Try multiple random orderings
    for _ in range(2000):
        indices = list(range(N))
        random.shuffle(indices)
        S = []
        sums = set()
        
        for x in indices:
            # Check if x can be added
            can_add = True
            # Check 2x
            if 2*x in sums:
                can_add = False
                break
            # Check x+s for s in S
            new_sums = []
            for s in S:
                val = x + s
                if val in sums:
                    can_add = False
                    break
                new_sums.append(val)
            
            if can_add:
                S.append(x)
                sums.add(2*x)
                sums.update(new_sums)
        
        if len(S) > max_size:
            max_size = len(S)
            best_S = S[:]
            
            # If we reached a theoretically high bound, we can stop early?
            # Max size is approx sqrt(N) = 7.
            # If we find 8, it's very good.
            if max_size >= 8:
                break

    # Fallback if no set found (should not happen)
    if not best_S:
        best_S = [0]
        
    h = np.zeros(N)
    k = len(best_S)
    for i in best_S:
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
