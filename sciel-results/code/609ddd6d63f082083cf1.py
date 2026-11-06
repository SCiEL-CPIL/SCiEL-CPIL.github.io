"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

def _score_set(S):
    """Count distinct d > 0 such that a, a+d, a+2d all in S for some a."""
    # Build a bitset integer where bit (x-1) is 1 if x is in S
    B = 0
    for x in S:
        B |= (1 << (x - 1))
    
    distinct_d = 0
    # Max possible d is roughly (max(S) - min(S)) // 2
    # But we can just iterate up to 512//2 = 256
    # Actually, if a+2d <= 512, then 2d <= 512 - min(S) + min(S) ... 
    # The max d is bounded by 512//2 = 256.
    for d in range(1, 257):
        # Check if there is any a such that a, a+d, a+2d are in S
        # B has bit i for value i+1
        # B >> d shifts bits right by d. Bit i in (B>>d) is bit i+d in B.
        # So (B & (B >> d)) has bit i set if (i+1) in S and (i+1+d) in S.
        # Then & (B >> 2*d) checks if (i+1+2d) in S.
        if (B & (B >> d) & (B >> (2 * d))) != 0:
            distinct_d += 1
    return distinct_d

def _random_set(n=32, lo=1, hi=512):
    return random.sample(range(lo, hi+1), n)

def _local_search(S, time_limit):
    """Local search: try swaps to improve score."""
    start = time.time()
    best_S = list(S)
    best_score = _score_set(best_S)
    n = len(best_S)
    best_S_set = set(best_S)
    
    # Pre-compute list of values in S for faster swapping? 
    # We just need to replace one value.
    
    while time.time() - start < time_limit:
        improved = False
        # Try swapping each element with a random other value
        for i in range(n):
            if time.time() - start > time_limit:
                break
            old_val = best_S[i]
            # Try a few random replacements
            # We can try more replacements per element now that scoring is fast
            for _ in range(100):
                new_val = random.randint(1, 512)
                if new_val == old_val or new_val in best_S_set:
                    continue
                # Swap
                best_S[i] = new_val
                best_S_set.discard(old_val)
                best_S_set.add(new_val)
                new_score = _score_set(best_S)
                if new_score > best_score:
                    best_score = new_score
                    improved = True
                else:
                    # Revert
                    best_S[i] = old_val
                    best_S_set.discard(new_val)
                    best_S_set.add(old_val)
        
        if not improved:
            # Stuck, try random restart with perturbation
            for i in range(n):
                if random.random() < 0.3:
                    new_val = random.randint(1, 512)
                    if new_val not in best_S_set:
                        best_S_set.discard(best_S[i])
                        best_S[i] = new_val
                        best_S_set.add(new_val)
            best_score = _score_set(best_S)
    
    return best_S, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin: 110s out of 120s limit
    
    # Start with a greedy or random set
    random.seed(42)
    best_S = _random_set(32, 1, 512)
    best_score = _score_set(best_S)
    
    # Multiple restarts
    # Since scoring is O(1) effectively (bit ops), we can do many more restarts/steps
    restart_count = 0
    while time.time() < deadline:
        restart_count += 1
        
        # Random initial set
        S = _random_set(32, 1, 512)
        score = _score_set(S)
        if score > best_score:
            best_S = list(S)
            best_score = score
        
        # Local search with time budget
        remaining = deadline - time.time()
        if remaining < 2:
            break
        # Allocate a small chunk of time for local search
        # If we have lots of time, use more.
        time_budget = min(5.0, remaining * 0.5)
        
        S2, score2 = _local_search(S, time_budget)
        if score2 > best_score:
            best_S = list(S2)
            best_score = score2
        
        # Also try local search from best_S
        remaining = deadline - time.time()
        if remaining < 2:
            break
        time_budget = min(5.0, remaining * 0.5)
        S3, score3 = _local_search(best_S, time_budget)
        if score3 > best_score:
            best_S = list(S3)
            best_score = score3
            
    return best_S

# EVOLVE-BLOCK-END
