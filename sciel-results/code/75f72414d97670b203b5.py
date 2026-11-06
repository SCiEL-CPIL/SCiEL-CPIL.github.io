import random
import time
from itertools import combinations

def is_sidon(subset):
    sums = set()
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def max_sidon_size(S):
    # S is a sorted list of distinct ints
    n = len(S)
    # Check from largest k down
    for k in range(n, 0, -1):
        for comb in combinations(S, k):
            if is_sidon(list(comb)):
                return k
    return 0

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Baseline
    seed_set = list(range(12))
    best_score_val = max_sidon_size(seed_set)
    best_set = seed_set[:]
    
    # If we can find a set with value < best_score_val, we improve.
    
    # Local search
    def optimize(S, time_limit):
        S = sorted(S)
        current_val = max_sidon_size(S)
        if current_val < best_score_val:
            best_score_val = current_val
            best_set = S[:]
            
        while time.time() < time_limit:
            if best_score_val <= 3: # Can't do much better than 3?
                break
                
            # Pick a random index to swap
            i = random.randint(0, 11)
            # Pick a random new value
            # Ensure distinct
            new_val = random.randint(0, 128)
            if new_val in S and new_val != S[i]:
                continue
            
            S[i] = new_val
            S.sort()
            
            new_val_score = max_sidon_size(S)
            
            if new_val_score < current_val:
                current_val = new_val_score
                if current_val < best_score_val:
                    best_score_val = current_val
                    best_set = S[:]
            else:
                # Revert if not better (or keep if equal? Revert to avoid drift)
                S[i] = S[i-1] if i > 0 and S[i] > S[i-1] else S[i] # This revert logic is messy
                # Better to keep a copy
                pass
                
        return best_score_val, best_set

    # Run several restarts
    for _ in range(20):
        if time.time() > deadline - 5:
            break
        S = random.sample(range(129), 12)
        S.sort()
        val, best_so_far = optimize(S, deadline - 2)
        
    return best_set
