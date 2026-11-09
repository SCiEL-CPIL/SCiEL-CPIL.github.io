"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
from math import gcd
from itertools import combinations

def construct():
    # Precompute bad triples
    # A triple (a, b, c) is bad if gcd(a,b) == gcd(b,c) == gcd(c,a)
    N = 128
    numbers = list(range(1, N + 1))
    
    # We will use a local search / greedy approach with restarts.
    # To make it fast, we precompute for each element x, the list of pairs (y, z) 
    # such that {x, y, z} is a bad triple.
    # Or simpler: just check on the fly? 128 is small.
    # Precomputing bad triples list is better for counting.
    
    bad_triples = []
    # Map from element to list of indices of bad triples containing it
    # Or map from element to set of other elements that form bad triples with it?
    # Let's just store the list of triples.
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            g_ab = gcd(a, b)
            # c must be > b to avoid duplicates in generation, but we need all perms?
            # No, set {a,b,c}.
            for c in range(b + 1, N + 1):
                g_bc = gcd(b, c)
                if g_bc != g_ab:
                    continue
                g_ca = gcd(c, a)
                if g_ca == g_ab:
                    bad_triples.append((a, b, c))
                    
    # Optimization: For each element, store the indices of bad triples it belongs to?
    # Or for each element x, store the set of pairs (y, z) such that (x,y,z) is bad.
    # This allows O(1) or O(k) check when adding x to a set S.
    # If we add x to S, we need to check if there are y, z in S such that (x,y,z) is bad.
    # So for each x, let's store a list of (y, z) pairs.
    
    bad_pairs_for = {i: [] for i in range(1, N + 1)}
    for t in bad_triples:
        a, b, c = t
        bad_pairs_for[a].append((b, c))
        bad_pairs_for[b].append((a, c))
        bad_pairs_for[c].append((a, b))
        
    def is_valid_set(S):
        # S is a set of ints
        # Check if any bad triple is fully contained in S
        # This is slow if done from scratch.
        # Better to maintain a count of "active" bad triples.
        pass

    best_size = 0
    best_set = set()
    
    # Random restarts
    for _ in range(50): # 50 restarts
        # Start with a random subset
        # Heuristic: Start with all numbers, then remove greedily
        S = set(numbers)
        
        # Greedy removal
        # Calculate "weight" of each node: number of bad triples it is part of within S
        # We need to update this as we remove nodes.
        
        # Instead of complex updates, we can just iterate:
        # While there is a bad triple in S:
        #   Find one.
        #   Remove the element in the triple with the highest "degree" in bad triples within S.
        
        # To find a bad triple in S:
        # We can iterate through bad_triples and check if all 3 are in S.
        # This is O(|Bad| * 3). |Bad| is small enough?
        # Max triples is C(128,3) ~ 340k.
        # But most are not bad.
        # Let's estimate number of bad triples.
        # d=1: pairwise coprime triples.
        # d=2: multiples of 2, divided by 2 are pairwise coprime.
        # ...
        
        # Let's just run the removal loop.
        # To speed up "find a bad triple in S", we can precompute for each pair (y,z) in S, 
        # if there exists x in S such that (x,y,z) is bad.
        # Actually, just scanning bad_triples is probably fast enough for 128 elements.
        
        while True:
            found_bad = None
            for t in bad_triples:
                if t[0] in S and t[1] in S and t[2] in S:
                    found_bad = t
                    break
            
            if found_bad is None:
                break
                
            a, b, c = found_bad
            # Choose which one to remove.
            # Heuristic: Remove the one that is in the most bad triples currently in S.
            # Counting this exactly is expensive.
            # Approximation: Remove random one? Or the largest?
            # Let's try removing the one with the most "potential" conflicts.
            # Since we don't have dynamic counts, let's just pick the one with the highest value? 
            # Or random. Random is good for diversity in restarts.
            
            # Better heuristic: Remove the element that appears most frequently in the *precomputed* bad triples?
            # No, that's static.
            # Let's just pick the maximum element? Or random.
            # Random selection from the triple.
            victim = random.choice(found_bad)
            S.remove(victim)
            
        # Now S is valid.
        # Try to add back elements to maximize size.
        # Iterate through all numbers not in S.
        # Try to add them if they don't create a bad triple.
        
        # To check if adding x is safe:
        # Check if there exist y, z in S such that (x, y, z) is a bad triple.
        # We have bad_pairs_for[x] which is a list of (y, z).
        # We just need to check if any (y, z) in bad_pairs_for[x] has both y and z in S.
        
        for x in numbers:
            if x in S:
                continue
            # Check if x can be added
            can_add = True
            for y, z in bad_pairs_for[x]:
                if y in S and z in S:
                    can_add = False
                    break
            if can_add:
                S.add(x)
                
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
            
    # Convert to sorted list for consistent output if needed, but set is fine?
    # The task says "Return nonempty subset". A list is usually expected.
    return sorted(list(best_set))

# EVOLVE-BLOCK-END
