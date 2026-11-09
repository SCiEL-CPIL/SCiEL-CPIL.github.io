import time
import random

def _can_add(S, x):
    """
    Check if x can be added to S without creating a forbidden triple.
    Forbidden triple: distinct a, b, c in S_new such that a+b, a+c, b+c are in S_new.
    If we add x to S_old, the new triple must include x.
    So we look for a, b in S_old such that:
    1. a != b
    2. a + b in S_old
    3. a + x in S_new  => a + x in S_old (since x is new, a+x cannot be x unless a=0)
    4. b + x in S_new  => b + x in S_old
    """
    S_set = set(S)
    # Candidates for a must satisfy a + x in S_set
    # Candidates for b must satisfy b + x in S_set
    # And a != b
    # And a + b in S_set
    
    # Optimization: Precompute list of a such that a+x in S
    # Since S is small, we can just iterate.
    
    # To avoid O(N^2), we can note that if a+x in S, then a < x (since S contains positive ints, and max sum is 256, but S is subset of 1..128).
    # Actually, a+x <= 128 is required for a+x to be in S (since S subset of 1..128).
    # So a <= 128 - x.
    
    # Let's just iterate through S to find potential 'a's
    potential_as = []
    for a in S:
        if a + x in S_set:
            potential_as.append(a)
            
    # Now check if any pair (a, b) from potential_as (distinct) has a+b in S
    n_as = len(potential_as)
    for i in range(n_as):
        a = potential_as[i]
        for j in range(i + 1, n_as):
            b = potential_as[j]
            if a + b in S_set:
                return False
    return True

def construct():
    deadline = time.time() + 80.0
    n = 128
    best_size = 0
    best_set = set()
    
    # Initial seed: the known good solution from the "other" observation (score 1.265625)
    # That was likely a greedy add to {65..128}.
    # Let's start with the high-range set as a baseline.
    base = set(range(65, n + 1))
    best_set = base
    best_size = len(base)
    
    # Also try starting with empty or other bases?
    # The randomized greedy is powerful.
    
    rng = random.Random(42)
    
    while time.time() < deadline:
        # Randomized Greedy Construction
        S = set()
        # Shuffle the numbers
        candidates = list(range(1, n + 1))
        rng.shuffle(candidates)
        
        for x in candidates:
            if _can_add(S, x):
                S.add(x)
                
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
            # If we reach a very high size, we might stop early?
            # Theoretical max is unknown, but > 81.
            # Let's keep searching.
            
        # Occasionally, try a local search on the current best?
        # Or just rely on the random restarts.
        
    return list(best_set)
