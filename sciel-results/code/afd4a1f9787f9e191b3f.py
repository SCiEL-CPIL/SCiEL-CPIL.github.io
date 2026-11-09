import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    n = 16
    # All possible triples
    all_triples = list(combinations(range(n), 3))
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    num_triples = len(all_triples)

    # Precompute: for each pair of points, which triples contain both
    # pair_to_triples[(a,b)] = list of triple indices containing a and b
    pair_to_triples = {}
    for i, t in enumerate(all_triples):
        for a, b in combinations(t, 2):
            if a > b:
                a, b = b, a
            key = (a, b)
            if key not in pair_to_triples:
                pair_to_triples[key] = []
            pair_to_triples[key].append(i)

    # For each K(2,2,2) configuration (three disjoint pairs), precompute the 8 transversal triples
    # K(2,2,2) has vertices split into 3 parts of size 2. 
    # Let parts be (a1,a2), (b1,b2), (c1,c2)
    # The 8 transversal triples are: (a_i, b_j, c_k) for i,j,k in {1,2}
    
    # We need to find all such configurations and for each, the 8 triples
    # But checking all is expensive. Instead, we'll use a different approach.
    
    # Better approach: use the fact that a forbidden configuration is 3 disjoint pairs with all 8 transversals.
    # We'll use greedy + local search.
    
    # First, let's create a function to check if adding a triple creates a forbidden K(2,2,2)
    # We maintain a set of selected triples.
    # For a new triple t = {x, y, z}, we need to check if there exist two other points x', y', z' 
    # (disjoint from x,y,z and from each other) such that all 8 combinations are present.
    
    # Actually, the forbidden configuration is: three disjoint pairs P1, P2, P3 such that all 8 transversal triples are in our set.
    # So if we have triples selected, we check if any K(2,2,2) has all 8 of its transversal triples selected.
    
    # Let's precompute all K(2,2,2) configurations and their 8 transversal triples.
    # Number of ways to partition 6 points into 3 pairs: C(16,6) * 15 ways to pair them up... too many.
    
    # Alternative: use a different strategy. Let's use the observation that a star (all triples through one point) 
    # gives 105 triples. We can do better.
    
    # Let's try a greedy approach with random restarts and local search.
    
    def check_valid(selected_triple_set, triple_to_check_idx):
        """Check if adding triple_to_check_idx to selected_triple_set creates a forbidden K(2,2,2)."""
        t = all_triples[triple_to_check_idx]
        # The new triple t = {a, b, c}
        # A forbidden K(2,2,2) involving t would have t as one of the 8 transversals.
        # The other 7 transversals must also be in selected_triple_set.
        # The K(2,2,2) has 3 pairs. One pair contains exactly 2 of {a,b,c}? No.
        # Each transversal picks one from each pair. So t picks a from pair1, b from pair2, c from pair3 (or some permutation).
        # The other points in the pairs are a', b', c'.
        # The 8 transversals are: (a/b') x (b/b') x (c/c') combinations.
        # Specifically: (a,b,c), (a,b,c'), (a,b',c), (a,b',c'), (a',b,c), (a',b,c'), (a',b',c), (a',b',c')
        
        # For t=(a,b,c) to be a transversal, we need pairs {a,a'}, {b,b'}, {c,c'} where a',b',c' are distinct and not in {a,b,c}.
        # So we need to find a', b', c' such that all 8 triples are in selected_triple_set ∪ {t}.
        
        a, b, c = t
        
        # Try all possible a', b', c'
        # a' can be any point not in {a,b,c}
        remaining = [p for p in range(n) if p not in (a, b, c)]
        
        for a_prime in remaining:
            for b_prime in remaining:
                if b_prime == a_prime:
                    continue
                for c_prime in remaining:
                    if c_prime == a_prime or c_prime == b_prime:
                        continue
                    
                    # Check all 8 transversals
                    transversals = [
                        (a, b, c),
                        (a, b, c_prime),
                        (a, b_prime, c),
                        (a, b_prime, c_prime),
                        (a_prime, b, c),
                        (a_prime, b, c_prime),
                        (a_prime, b_prime, c),
                        (a_prime, b_prime, c_prime)
                    ]
                    
                    all_present = True
                    for trans in transversals:
                        if trans == t:
                            continue  # This is the one we're adding
                        idx = triple_to_idx.get(trans)
                        if idx is None or idx not in selected_triple_set:
                            all_present = False
                            break
                    if all_present:
                        return False  # Would create forbidden K(2,2,2)
        
        return True

    def greedy_construct(seed=42):
        random.seed(seed)
        selected = set()
        # Try random orderings
        candidates = list(range(num_triples))
        random.shuffle(candidates)
        
        for idx in candidates:
            if check_valid(selected, idx):
                selected.add(idx)
        
        return list(selected)

    def local_search(initial_selection, max_time=None):
        """Try to improve by swapping or adding triples."""
        selected_set = set(initial_selection)
        # Count current size
        current_size = len(selected_set)
        
        # Try to add triples
        # First pass: try to add any triple that doesn't violate
        improved = True
        while improved:
            improved = False
            for idx in range(num_triples):
                if idx not in selected_set:
                    if check_valid(selected_set, idx):
                        selected_set.add(idx)
                        improved = True
                        current_size += 1
        
        # Second pass: try swaps - remove one, add two?
        # This is complex. Let's just do the greedy add.
        
        return list(selected_set)

    # Start with a few greedy restarts
    best_result = []
    best_size = 0
    
    num_restarts = 50
    for i in range(num_restarts):
        elapsed = time.time() - start_time
        if elapsed > deadline - 10:
            break
        
        result = greedy_construct(seed=i)
        result = local_search(result)
        
        if len(result) > best_size:
            best_size = len(result)
            best_result = result
    
    # Convert to required format
    return [list(t) for t in best_result]
