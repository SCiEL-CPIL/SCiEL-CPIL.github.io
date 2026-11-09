import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s margin
    
    # Candidates are 2..40
    # Primes > 20 are 23, 29, 31, 37
    # These are "safe" to include because they don't form products <= 40 with any other number >= 2
    # except themselves, and their products with others are > 40, so they can't collide with single elements.
    # Can they collide with products of multiple other elements?
    # Product must be multiple of 23. Only 23 is multiple of 23 in [2,40].
    # So any product divisible by 23 must include 23.
    # Thus, subsets with 23 are disjoint in product values from subsets without 23.
    # Same for 29, 31, 37.
    # And 23*29 > 40, so no interaction between them.
    
    safe_primes = [23, 29, 31, 37]
    other_candidates = [x for x in range(2, 41) if x not in safe_primes]
    
    best_set = list(safe_primes)
    
    # Initialize P with products of safe_primes
    P = {1}
    for x in safe_primes:
        new_P = set(P)
        for p in P:
            new_P.add(p * x)
        P = new_P
        
    # Now search in other_candidates
    # Sort candidates: try larger numbers first (less likely to cause conflicts)
    other_candidates.sort(reverse=True)
    
    def is_valid(S, P, x):
        # Check if adding x to S creates a collision
        # New products are {p * x for p in P}
        # Collision if any p * x is in P
        return len(P & {p * x for p in P}) == 0

    def dfs(index, S, P):
        nonlocal best_set
        if time.time() > deadline:
            return
            
        if len(S) > len(best_set):
            best_set = list(S)
            
        if index == len(other_candidates):
            return
            
        # Pruning: if we can't beat best_set, return
        # Max possible size = len(S) + (len(other_candidates) - index)
        if len(S) + (len(other_candidates) - index) <= len(best_set):
            return
            
        x = other_candidates[index]
        
        # Option 1: Exclude x (establish baseline first for better pruning)
        dfs(index + 1, S, P)
        
        # Option 2: Include x
        if is_valid(S, P, x):
            new_P = set(P)
            for p in P:
                new_P.add(p * x)
            dfs(index + 1, S + [x], new_P)

    dfs(0, list(safe_primes), P)
    
    # Final polish: try to add any remaining candidates that weren't in best_set
    best_set_set = set(best_set)
    remaining = [x for x in range(2, 41) if x not in best_set_set]
    
    # Build P from best_set
    P_best = {1}
    for x in best_set:
        new_P = set(P_best)
        for p in P_best:
            new_P.add(p * x)
        P_best = new_P
    
    # Try adding remaining candidates one by one
    for x in remaining:
        if time.time() > deadline:
            break
        if is_valid(best_set, P_best, x):
            new_P = set(P_best)
            for p in P_best:
                new_P.add(p * x)
            P_best = new_P
            best_set.append(x)
    
    # Ensure we return a list of ints
    return [int(x) for x in best_set]
