import random
import time
from math import gcd

def construct():
    # Precompute GCDs
    N = 128
    gcd_mat = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            gcd_mat[i][j] = gcd(i, j)
            gcd_mat[j][i] = gcd(i, j)

    def count_triples(S):
        """Count forbidden triples in set S."""
        # S is a set of ints
        # To speed up, sort? No, just iterate.
        # We can use the property: 
        # A triple (a,b,c) is bad if gcd(a,b)==gcd(a,c)==gcd(b,c).
        # We can iterate pairs (a,b) and check c.
        # But O(N^3) is fine for N~50.
        # Let's just do simple triple iteration for clarity, it's fast enough.
        items = list(S)
        n = len(items)
        count = 0
        for i in range(n):
            a = items[i]
            for j in range(i + 1, n):
                b = items[j]
                dab = gcd_mat[a][b]
                for k in range(j + 1, n):
                    c = items[k]
                    if gcd_mat[a][c] == dab and gcd_mat[b][c] == dab:
                        count += 1
        return count

    def get_score(S):
        """Objective: Maximize size, penalize triples."""
        t = count_triples(S)
        # Penalty must be high enough to prevent accepting triples if size gain is small
        # Max size gain is 128. So penalty > 128 is enough to prefer 0 triples.
        # But we want to find 0-triple sets.
        # If we have 1 triple, score drops by 1000.
        # If we have 100 triples, score drops by 100000.
        # We want to drive T to 0.
        return len(S) - 1000 * t

    # Phase 1: Start with all numbers, remove random elements until valid
    S = set(range(1, N + 1))
    # This loop might take a while if we just remove one by one.
    # Better: Randomly remove chunks? Or just loop.
    # With N=128, T is large.
    # Let's run a quick greedy removal to get a valid start.
    # Actually, SA can handle starting from invalid states if penalty is high.
    # But starting valid is safer for maximizing size.
    
    # Let's try to find a valid set quickly.
    # Random removal
    while count_triples(S) > 0:
        # Remove a random element
        # To speed up, maybe remove element involved in many triples?
        # For now, random is simple.
        item_to_remove = random.choice(list(S))
        S.remove(item_to_remove)
        
    best_set = S.copy()
    best_size = len(S)
    best_score = get_score(S)

    start_time = time.time()
    time_limit = 55.0
    
    # Phase 2: Simulated Annealing
    # Temperature schedule
    T = 1.0
    cooling = 0.999
    
    while time.time() - start_time < time_limit:
        # Generate neighbor
        # 1. Remove random element from S
        # 2. Add random element from complement
        # Or swap
        
        # Let's do:
        # With prob 0.3: Add random v not in S
        # With prob 0.3: Remove random u in S
        # With prob 0.4: Swap (remove u, add v)
        
        r = random.random()
        S_new = S.copy()
        
        if r < 0.3:
            # Add
            complement = [x for x in range(1, N + 1) if x not in S]
            if complement:
                v = random.choice(complement)
                S_new.add(v)
        elif r < 0.6:
            # Remove
            if S:
                u = random.choice(list(S))
                S_new.remove(u)
        else:
            # Swap
            if S:
                u = random.choice(list(S))
                S_new.remove(u)
                complement = [x for x in range(1, N + 1) if x not in S_new]
                if complement:
                    v = random.choice(complement)
                    S_new.add(v)
        
        score_old = best_score if S == best_set else get_score(S) # Cache? No, just compute
        # Actually, we should maintain current_score
        # Let's restructure to maintain current_score
        
        # For simplicity in this edit, I will implement a cleaner SA loop below.
        pass
        
    # The above loop is messy. Let's rewrite the core logic properly in the replacement.
    
    return list(best_set)
