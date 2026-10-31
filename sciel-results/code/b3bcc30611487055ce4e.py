import time
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_score = 0
    best_set = [0, 1, 3] # Seed
    
    def calc_score(U):
        if not U:
            return 0
        max_u = max(U)
        if max_u == 0:
            return 1.0
        u_set = set(U)
        # Sumset
        sum_set = set()
        for a in u_set:
            for b in u_set:
                sum_set.add(a + b)
        # Diffset
        diff_set = set()
        for a in u_set:
            for b in u_set:
                diff_set.add(a - b)
        
        ratio = len(diff_set) / len(sum_set)
        if ratio <= 0:
            return 0
        denom = math.log(2 * max_u + 1)
        if denom == 0:
            return 0
        return 1 + math.log(ratio) / denom

    # Search subsets of {0, ..., K}
    # We fix 0 in the set.
    # We vary the other elements from {1, ..., K}.
    
    # Start with small K and increase
    # K=3: {0,1,2,3} subsets.
    # We can iterate over all masks of length K.
    
    max_K = 20
    for K in range(3, max_K + 1):
        # Check time
        if time.time() > deadline:
            break
            
        # Iterate over subsets of {1, ..., K}
        # There are 2^K subsets.
        # We can use bit manipulation.
        # Range of masks: 0 to 2^K - 1.
        
        # Optimization: 
        # If K is large, 2^K is huge.
        # For K=20, 2^20 = 1,000,000.
        # Each calc_score takes O(n^2). n <= 21.
        # 1M * 400 ops = 400M ops. Might be too slow in Python.
        
        # Let's be conservative with K or optimize.
        # We can skip sets that are clearly bad?
        # Or use a heuristic search?
        
        # Given the time limit, let's stick to K=15 or 16 first.
        # 2^15 = 32,768. Very fast.
        # 2^16 = 65,536. Fast.
        # 2^17 = 131,072.
        # 2^18 = 262,144.
        # 2^19 = 524,288.
        
        # Let's try up to K=18.
        if K > 18:
             continue # Skip larger K in this simple loop, or handle separately if time permits
        
        for mask in range(1 << K):
            # Construct U
            U = [0]
            for i in range(K):
                if mask & (1 << i):
                    U.append(i + 1)
            
            # Quick prune: 
            # If max U is small, denominator is small.
            # We want high ratio.
            
            s = calc_score(U)
            if s > best_score:
                best_score = s
                best_set = U[:]
                
    return best_set
