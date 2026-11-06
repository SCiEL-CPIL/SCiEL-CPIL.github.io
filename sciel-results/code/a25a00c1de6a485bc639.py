import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    # Generate all 4-subsets of [1, 16] as bitmasks (0-15)
    all_sets = []
    for combo in combinations(range(16), 4):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_sets.append(mask)
        
    n_sets = len(all_sets)
    indices = list(range(n_sets))
    
    best_family = []
    best_size = 0

    def greedy_pass(order):
        chosen = []
        # We use a list for chosen to allow O(1) append
        # To speed up the conflict check, we can't easily vectorize without NumPy overhead for small arrays
        # But we can reduce the constant factor.
        
        for idx in order:
            c = all_sets[idx]
            valid = True
            # Check against all pairs in chosen
            # Optimization: If len(chosen) is small, it's fast.
            # If len(chosen) is large, it's slow.
            for i in range(len(chosen)):
                a = chosen[i]
                # Quick check: if a & c == 0, then s must be 0 for b as well.
                # But we still need to check b.
                for j in range(i + 1, len(chosen)):
                    b = chosen[j]
                    s = a & b
                    # If s is 0, we need a&c==0 and b&c==0
                    # If s is not 0, we need a&c==s and b&c==s
                    if (a & c == s) and (b & c == s):
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                chosen.append(c)
        
        return chosen

    # 1. Baseline: Natural Order
    baseline = greedy_pass(indices)
    if len(baseline) > best_size:
        best_size = len(baseline)
        best_family = baseline[:]
        
    if time.time() > deadline:
        return [sorted([i+1 for i in range(16) if (m >> i) & 1]) for m in best_family]

    # 2. Randomized Greedy
    # Estimate time per pass.
    # If it's too slow, we reduce iterations.
    # Let's run as many as possible.
    
    # To ensure we don't time out, we check time in the loop.
    # We use a random seed for reproducibility? No, random is fine.
    # But for "improvement", we want diversity.
    
    # Optimization: Pre-shuffle indices once, then reverse/shuffle again?
    # Just shuffle.
    
    while time.time() < deadline - 5:
        random.shuffle(indices)
        current = greedy_pass(indices)
        if len(current) > best_size:
            best_size = len(current)
            best_family = current[:]
            
    # Final conversion
    result = []
    for m in best_family:
        s = []
        for i in range(16):
            if (m >> i) & 1:
                s.append(i + 1)
        result.append(sorted(s))
        
    return result
