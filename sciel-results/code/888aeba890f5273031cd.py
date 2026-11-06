import time
import sys
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    n = 256
    
    # Precompute for each v in [1, n], which sums in [2, 2n] are divisible by v
    # For v, the sums that are multiples of v are: v, 2v, 3v, ..., floor(2n/v)*v
    # We'll create a boolean array for each v indicating which sums conflict
    # Actually, let's think differently.
    
    # When we have set S and want to add v, we need to check:
    # For all b, c in S with b <= c: (b+c) % v != 0
    # Equivalently: no pair (b,c) in S has sum in {v, 2v, 3v, ...}
    
    # Strategy: For each v, precompute the set of "forbidden sums" F_v = {k*v : k*v <= 2n}
    # Then when adding v, check if any pair in S has sum in F_v
    
    # Better: use a bitset for S and for each v, check if any pair sums to a multiple of v
    # We can precompute for each v a bitmask of length 2n where bit s is set if s % v == 0
    # Then for current S, compute all pair sums as a bitmask, and check intersection
    
    # Let's use numpy for speed
    # Pair sums: for S with elements s1, s2, ..., sk, the pair sums (si + sj for i<=j) 
    # can be computed, but that's O(k^2)
    
    # Alternative: for each v, we can check if there exist b, c in S with b <= c and b+c = m*v
    # This is equivalent to: for some m, there exist b, c in S with b+c = m*v, b <= c
    # i.e., b is in S, c = m*v - b is in S, b <= c, b <= m*v/2
    
    # Let's implement the direct approach but with early termination and numpy for speed
    
    S = []  # sorted list of selected elements
    S_set = set()
    
    # Precompute conflict masks: for each v, which sums are multiples of v
    # We'll use this to quickly check conflicts
    # Actually, let's just do the direct check but optimized
    
    # For faster checking, note that if v is large, there are few multiples
    # For v > n/2, only one multiple: v itself (since 2v > n, but b+c can be up to 2n)
    # Wait, b and c are in [1, n], so b+c is in [2, 2n]
    # Multiples of v in [2, 2n]: v, 2v, 3v, ..., floor(2n/v)*v
    
    for v in range(n, 0, -1):
        if time.time() > deadline:
            break
        
        valid = True
        L = len(S)
        
        if L == 0:
            S.append(v)
            S_set.add(v)
            continue
        
        # Check all pairs b, c in S with b <= c
        # S is sorted in decreasing order (we add from n down to 1)
        # So S[0] > S[1] > ... > S[L-1]
        # For pairs, we need b <= c, so in our sorted list, b = S[j], c = S[i] with j >= i
        # Actually since S is decreasing, S[i] > S[j] for i < j
        # So b <= c means b = S[j], c = S[i] where j >= i
        
        # Let's just iterate all pairs
        for i in range(L):
            c = S[i]
            for j in range(i, L):
                b = S[j]
                if b > c:
                    b, c = c, b
                # Now b <= c
                if (b + c) % v == 0:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            # Insert v in correct position to keep sorted (decreasing)
            # Since we're going from n down to 1, v is smaller than all existing
            S.append(v)
            S_set.add(v)
    
    # Now try to improve with local search
    # Try adding elements that were skipped
    # Actually, we went from n down to 1, so we tried all elements
    # But maybe we can swap: remove some element and add a smaller one?
    # Or try a different order?
    
    # Let's try a local search: for each element in S, try removing it and see if we can add more
    # This is complex. Instead, let's try multiple random orders
    
    best_size = len(S)
    best_set = S_set.copy()
    
    # Try a few random permutations of the order we add elements
    import random
    random.seed(42)
    
    for trial in range(20):
        if time.time() > deadline:
            break
        
        order = list(range(1, n+1))
        random.shuffle(order)
        
        S_trial = []
        S_trial_set = set()
        
        for v in order:
            if time.time() > deadline:
                break
            
            valid = True
            L = len(S_trial)
            
            if L == 0:
                S_trial.append(v)
                S_trial_set.add(v)
                continue
            
            for i in range(L):
                c = S_trial[i]
                for j in range(i, L):
                    b = S_trial[j]
                    if b > c:
                        b, c = c, b
                    if (b + c) % v == 0:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                S_trial.append(v)
                S_trial_set.add(v)
        
        if len(S_trial) > best_size:
            best_size = len(S_trial)
            best_set = S_trial_set.copy()
    
    if not best_set:
        return [256]
        
    return sorted(best_set)

# EVOLVE-BLOCK-END
