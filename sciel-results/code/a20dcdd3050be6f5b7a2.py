"""Program to find a,b with sigma(a)=sigma(b) minimizing |a/b - 3/2|, where error must be > 0."""
import math
import time
import bisect

# EVOLVE-BLOCK-START
import time
import numpy as np

MAX_N = 10_000_000
TIME_LIMIT = 80.0  # safety margin (evaluator kills at 90s)

def construct():
    t_start = time.time()
    deadline = t_start + TIME_LIMIT

    # --- Build sigma (divisor sum) via vectorized sieve up to MAX_N ---
    sigma = np.zeros(MAX_N + 1, dtype=np.int64)
    for d in range(1, MAX_N + 1):
        sigma[d::d] += d

    # Group indices by sigma value
    # sigma[0] is 0, we ignore it.
    # We use a dictionary mapping sigma value to a list of indices.
    groups = {}
    # Iterating over numpy array is slow, but tolist() is fast.
    # However, creating a list of 10 million ints and grouping them in Python might be slow.
    # Let's try to optimize grouping.
    
    # Alternative: 
    # Since we need to search within groups, we can just build the groups.
    # 10 million iterations in Python is ~0.5-1s.
    
    sigma_list = sigma.tolist()
    
    # To speed up grouping, we can iterate and append.
    for i in range(1, MAX_N + 1):
        s = sigma_list[i]
        if s in groups:
            groups[s].append(i)
        else:
            groups[s] = [i]
            
    best_err = float('inf')
    best_pair = (1, 1)
    
    # Sort each group to enable binary search
    for key in groups:
        if len(groups[key]) > 1:
            groups[key].sort()

    # Search within groups
    # For each group, we look for a, b in group such that a/b ~ 1.5
    # a ~ 1.5 b
    # We iterate b in group, and find closest a in group to 1.5 b
    
    for key, lst in groups.items():
        if time.time() > deadline:
            break
        n = len(lst)
        if n < 2:
            continue
            
        # We can restrict b to be such that 1.5 b is within [min(lst), max(lst)]
        # But binary search handles out of bounds by checking edges.
        
        for i in range(n):
            b = lst[i]
            target = 1.5 * b
            
            # Find insertion point for target in lst
            pos = bisect.bisect_left(lst, target)
            
            # Check pos and pos-1
            # We need to check candidates around pos
            # Since we want a/b close to 1.5, a should be close to target.
            # The candidates are lst[pos-1] and lst[pos] if they exist.
            
            candidates = []
            if pos > 0:
                candidates.append(lst[pos-1])
            if pos < n:
                candidates.append(lst[pos])
            # Sometimes checking pos+1 or pos-2 might be safer if distribution is sparse?
            # No, bisect finds the closest in sorted list. The closest value to target in a sorted list
            # is either the predecessor or the successor.
            
            for a in candidates:
                if a == b:
                    continue
                # Calculate error exactly as rational to avoid float issues if possible?
                # Task says "score is the binary64 serialization of that rational".
                # So we just need the float value for comparison, but we should be careful.
                # err = abs(a/b - 1.5)
                # To avoid division by zero or precision issues, we can compare:
                # |a/b - 3/2| = |2a - 3b| / (2b)
                # We want to minimize this.
                # num = abs(2*a - 3*b)
                # den = 2*b
                # err = num / den
                
                num = abs(2*a - 3*b)
                if num == 0:
                    continue # Error is 0, invalid
                den = 2*b
                err = num / den
                
                if err < best_err:
                    best_err = err
                    best_pair = (a, b)

    a, b = best_pair
    return [int(a), int(b)]

# EVOLVE-BLOCK-END
