import time
import random
from itertools import combinations

V = 16
ALL_TRIPLES = list(combinations(range(V), 3))
# Precompute for each triple, the 7 other transversal triples of the K(2,2,2) it completes
# A K(2,2,2) is formed by 3 disjoint pairs (a,b), (c,d), (e,f).
# The 8 transversals are:
# (a,c,e), (a,c,f), (a,d,e), (a,d,f), (b,c,e), (b,c,f), (b,d,e), (b,d,f)
# For a given triple t = (x,y,z), it is part of a K(2,2,2) if there exist
# x', y', z' such that all 8 are present.
# Actually, it's easier to check: if we add t, does it complete a K(2,2,2)?
# A K(2,2,2) is a set of 8 triples. t is one of them.
# The other 7 must be in the current set.
# To speed this up, we can precompute for each triple t, the list of "complement sets"
# or simply iterate. Given the small size (560 triples), a smart check is feasible.

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Strategy: Randomized Greedy with Restart or Local Search
    # Let's try a simple randomized greedy first, then if time permits, local search.
    # But the seed is a specific structure. Let's try to build a large set.
    
    best_triples = []
    best_count = 0
    
    # We can try multiple random seeds/orders
    for seed in range(100):
        if time.time() > deadline:
            break
            
        random.seed(seed)
        triples_pool = ALL_TRIPLES[:]
        random.shuffle(triples_pool)
        
        current_set = set()
        
        # Fast check function
        def is_valid_add(t, current_set):
            x, y, z = t
            # Check if adding t creates a K(2,2,2)
            # A K(2,2,2) contains t. The other 7 triples must be in current_set.
            # The 8 triples are determined by the 3 pairs.
            # If t = (x,y,z), the pairs must be (x,x'), (y,y'), (z,z')
            # The 8 triples are all combinations of choosing one from each pair.
            # So we need to find x', y', z' such that
            # (x',y,z), (x',y',z), (x',y',z'), (x,y',z'), (x',y',z), ... wait.
            # The 8 transversals of (x,x'), (y,y'), (z,z') are:
            # (x,y,z) - this is t
            # (x,y,z')
            # (x,y',z)
            # (x,y',z')
            # (x',y,z)
            # (x',y,z')
            # (x',y',z)
            # (x',y',z')
            
            # So for t to be part of a K(2,2,2), there must exist x', y', z'
            # distinct from x,y,z and each other? No, x' != x, y' != y, z' != z.
            # And x', y', z' must be distinct? Yes, points in a hypergraph are distinct.
            # The pairs are disjoint, so x' != x, y' != y, z' != z.
            # Also x' != y', x' != z', y' != z' is not strictly required for the pairs to be disjoint
            # from each other? Wait.
            # Pairs are (x,x'), (y,y'), (z,z').
            # Disjoint means no point is shared between pairs.
            # So x' != y, x' != y', x' != z, x' != z' etc.
            # Basically {x,x'} ∩ {y,y'} = ∅ and {x,x'} ∩ {z,z'} = ∅ and {y,y'} ∩ {z,z'} = ∅.
            
            # Let's just iterate over possible x', y', z'
            # There are 15 choices for x', 14 for y', 13 for z' roughly.
            # 15*14*13 = 2730 checks. Each check is 7 set lookups.
            # 2730 * 7 = ~19k ops per triple.
            # 560 triples * 19k = ~10M ops. This might be slow in Python if done naively for every addition.
            
            # Optimization:
            # Precompute for each triple t, the list of "conflict triples" that would complete a K(2,2,2) with t?
            # No, a K(2,2,2) is 8 triples. t is one. The other 7 must be present.
            # We can precompute for each t, the set of "witnesses" (x',y',z') such that the other 7 are a specific set.
            # Actually, for a fixed t=(x,y,z), the other 7 triples are determined by the choice of x',y',z'.
            # Let's precompute for each t, a list of sets of 7 triples.
            # If ANY of these sets is a subset of current_set, then t cannot be added.
            
            # This precomputation is expensive to store but fast to check.
            # Number of (x',y',z') for a fixed t:
            # x' can be any of 15 points except x.
            # y' can be any of 14 points except y and x'.
            # z' can be any of 13 points except z, x', y'.
            # Total ~ 15*14*13 = 2730 sets per triple.
            # 560 triples * 2730 sets = ~1.5M sets of size 7.
            # Memory: 1.5M * 7 * 8 bytes (ints) ~ 84MB. Maybe too much or slow to build.
            
            # Alternative: On-the-fly check with pruning.
            # For t=(x,y,z):
            # We need x', y', z' such that:
            # (x',y,z), (x,y',z), (x,y,z'), (x',y',z), (x',y,z'), (x,y',z'), (x',y',z') are in current_set.
            
            # Let's try a different approach: Local Search starting from a good heuristic.
            # The seed program uses a star. Let's try to improve on a random greedy or a known dense structure.
            
            # Given time constraints, let's implement the on-the-fly check but optimize it.
            # If we iterate x', y', z' and check 7 triples, we can break early.
            
            for xp in range(V):
                if xp == x: continue
                if (xp, y, z) not in current_set: continue
                for yp in range(V):
                    if yp == y or yp == xp: continue
                    if (x, yp, z) not in current_set: continue
                    if (xp, yp, z) not in current_set: continue
                    for zp in range(V):
                        if zp == z or zp == xp or zp == yp: continue
                        if (x, y, zp) not in current_set: continue
                        if (xp, y, zp) not in current_set: continue
                        if (x, yp, zp) not in current_set: continue
                        if (xp, yp, zp) not in current_set: continue
                        return False # Conflict found
            return True

        for t in triples_pool:
            if is_valid_add(t, current_set):
                current_set.add(t)
                
        if len(current_set) > best_count:
            best_count = len(current_set)
            best_triples = list(current_set)
            
        # Early stopping if we find a very good solution?
        # Max possible is 560. K(2,2,2) free graphs are sparse.
        # Random greedy usually gets ~30-50% of max.
        
    return [list(t) for t in best_triples]
