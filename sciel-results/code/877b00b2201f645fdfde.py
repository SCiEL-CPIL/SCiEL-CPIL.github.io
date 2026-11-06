import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit
    
    n = 16
    all_triples = list(combinations(range(n), 3))
    
    # For each pair of pairs, track which triples are "transversal"
    # A forbidden config: 3 disjoint pairs (a,b),(c,d),(e,f) where all 8 transversal triples exist
    
    # Greedy: try to add triples, checking validity
    # To check if adding triple T creates a forbidden config:
    # T = {x,y,z}. We need to find two more disjoint pairs {a,b} and {c,d} 
    # from the remaining 13 points such that all 8 transversal triples of 
    # {x,y},{a,b},{c,d} are in our set (including T itself).
    
    # Actually, let me think differently. 
    # A forbidden configuration involves 6 distinct points forming 3 pairs, 
    # with all 8 transversal triples present.
    
    # Greedy approach: add triples one by one, checking if it completes a forbidden config.
    
    # For efficiency, let's use a different strategy:
    # Start with all triples, remove those that are "most conflicting"
    
    # Actually, let me try: random order greedy addition with validity check
    
    import random
    random.seed(42)
    
    def is_valid_config(triples_set, new_triple):
        """Check if adding new_triple to triples_set creates a forbidden K(2,2,2) config."""
        x, y, z = new_triple
        # The new triple uses points x, y, z
        # A forbidden config containing this triple must have x,y,z as one of the 3 pairs' transversal
        # Actually, the 8 transversal triples of pairs P1,P2,P3 each pick one from each pair
        # So new_triple = {p1, p2, p3} where pi is from pair Pi
        # We need to check: are there pairs P1,P2,P3 (disjoint, 6 distinct points)
        # such that new_triple is one of the 8 transversals, and all other 7 are in triples_set?
        
        # For new_triple {x,y,z}, x must be in P1, y in P2, z in P3 (or any permutation)
        # P1 = {x, x'}, P2 = {y, y'}, P3 = {z, z'} where x',y',z' are distinct and not in {x,y,z}
        # The 8 transversals: {x or x', y or y', z or z'}
        # We have {x,y,z} as one. The other 7 must be in triples_set.
        
        # x' can be any of the 13 other points (not x,y,z)
        # y' can be any of the remaining 12 (not x,y,z,x')
        # z' can be any of the remaining 11 (not x,y,z,x',y')
        
        # That's 13*12*11 = 1716 possibilities to check. Might be slow for many triples.
        
        # Let me optimize: for each choice of x' (13 options), y' (12), z' (11):
        # Check if all 7 other transversals are in triples_set
        
        other_points = [p for p in range(n) if p != x and p != y and p != z]
        
        for xi in other_points:
            for yi in [p for p in other_points if p != xi]:
                for zi in [p for p in other_points if p != xi and p != yi]:
                    # Pairs: {x,xi}, {y,yi}, {z,zi}
                    # 8 transversals:
                    # (x,y,z) - this is new_triple
                    # (x,y,zi), (x,yi,z), (x,yi,zi),
                    # (xi,y,z), (xi,y,zi), (xi,yi,z), (xi,yi,zi)
                    # Check if all 7 others are in triples_set
                    if (tuple(sorted((x, y, zi))) in triples_set and
                        tuple(sorted((x, yi, z))) in triples_set and
                        tuple(sorted((x, yi, zi))) in triples_set and
                        tuple(sorted((xi, y, z))) in triples_set and
                        tuple(sorted((xi, y, zi))) in triples_set and
                        tuple(sorted((xi, yi, z))) in triples_set and
                        tuple(sorted((xi, yi, zi))) in triples_set):
                        return False
        return True
    
    # This is too slow. Let me use a different approach.
    
    # Alternative: use the structure. 
    # Key insight: if we partition the 16 points into 4 groups of 4, and only allow triples 
    # that don't span all 4 groups in a certain way...
    
    # Actually, let me try a simpler greedy: random order, fast validity check.
    
    # Better approach: precompute for each triple, which other triples would complete a forbidden config with it.
    
    # Let me try: start empty, add triples in random order, check validity with a fast method.
    
    # Fast check: when adding triple T={x,y,z}, a forbidden config is created iff there exist
    # x',y',z' (distinct, not in {x,y,z}) such that all 7 other transversals of 
    # {x,x'},{y,y'},{z,z'} are already present.
    
    # To make this fast, I'll maintain for each pair of points, the count of triples containing them.
    # Actually that's not directly helpful.
    
    # Let me just do the brute force check but with early termination.
    # With ~200+ triples in the set, each check is 1716 * 7 set lookups = ~12k operations.
    # For 560 triples, that's ~6.7M operations. Should be fine in Python within time limit.
    
    triples_set = set()
    result = []
    
    # Try multiple random orders and keep the best
    best_count = 0
    best_result = []
    
    for attempt in range(20):
        if time.time() > deadline:
            break
        
        random.seed(42 + attempt)
        order = all_triples[:]
        random.shuffle(order)
        
        triples_set = set()
        result = []
        
        for t in order:
            if time.time() > deadline:
                break
            t_sorted = tuple(sorted(t))
            if t_sorted in triples_set:
                continue
            if is_valid_config(triples_set, t_sorted):
                triples_set.add(t_sorted)
                result.append(list(t_sorted))
        
        if len(result) > best_count:
            best_count = len(result)
            best_result = result[:]
    
    if best_count == 0:
        # Fallback to seed
        return [[0, a, b] for a in range(1, 16) for b in range(a + 1, 16)]
    
    return best_result
