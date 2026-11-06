import random
from itertools import combinations

def construct():
    n = 16
    points = range(n)
    all_triples = list(combinations(points, 3))
    num_triples = len(all_triples)
    triple_to_id = {t: i for i, t in enumerate(all_triples)}
    
    # Precompute partners for each triple
    # partner_sets[i] is a list of tuples, each tuple contains 7 triple IDs
    # that form a K(2,2,2) with triple i
    partner_sets = [[] for _ in range(num_triples)]
    
    for i, t in enumerate(all_triples):
        a, b, c = t
        t_set = {a, b, c}
        others = [x for x in points if x not in t_set]
        
        # We need to pick a', b', c' from others such that they are distinct
        # The K(2,2,2) is defined by pairs {a,a'}, {b,b'}, {c,c'}
        # The 8 triples are all combinations of one from each pair.
        # One is (a,b,c). The other 7 are the partners.
        
        for a_prime in others:
            for b_prime in others:
                if b_prime == a_prime: continue
                for c_prime in others:
                    if c_prime == a_prime or c_prime == b_prime: continue
                    
                    # Construct the 7 partner triples
                    # Points: a, b, c, a', b', c'
                    # Triples:
                    # (a, b, c) - current
                    # (a, b, c')
                    # (a, b', c)
                    # (a, b', c')
                    # (a', b, c)
                    # (a', b, c')
                    # (a', b', c)
                    # (a', b', c')
                    
                    p1 = (a, b, c_prime)
                    p2 = (a, b_prime, c)
                    p3 = (a, b_prime, c_prime)
                    p4 = (a_prime, b, c)
                    p5 = (a_prime, b, c_prime)
                    p6 = (a_prime, b_prime, c)
                    p7 = (a_prime, b_prime, c_prime)
                    
                    # Sort them to ensure they match all_triples keys
                    # Actually, all_triples are sorted tuples.
                    # We just need to check if these specific point sets are in the selected set.
                    # Using IDs is faster for set lookup if we map to ID.
                    
                    partner_ids = [
                        triple_to_id[tuple(sorted(p1))],
                        triple_to_id[tuple(sorted(p2))],
                        triple_to_id[tuple(sorted(p3))],
                        triple_to_id[tuple(sorted(p4))],
                        triple_to_id[tuple(sorted(p5))],
                        triple_to_id[tuple(sorted(p6))],
                        triple_to_id[tuple(sorted(p7))]
                    ]
                    partner_sets[i].append(tuple(partner_ids))
    
    best_count = 0
    best_set = set()
    
    # Run multiple random greedy constructions
    # Increase iterations since each check is now local and relatively fast
    for seed in range(100):
        random.seed(seed)
        shuffled_indices = list(range(num_triples))
        random.shuffle(shuffled_indices)
        
        current_set = set()
        
        for i in shuffled_indices:
            # Check if adding i is valid
            valid = True
            for partners in partner_sets[i]:
                # If all 7 partners are in current_set, adding i creates a K(2,2,2)
                # We can break early if we find one full set
                if all(p in current_set for p in partners):
                    valid = False
                    break
            
            if valid:
                current_set.add(i)
        
        if len(current_set) > best_count:
            best_count = len(current_set)
            best_set = current_set
            
    # Local search / Repair: 
    # Try to add any triple that was skipped
    # Then try to remove a triple that allows more to be added? 
    # Simple pass: try to add all remaining triples
    improved = True
    while improved:
        improved = False
        for i in range(num_triples):
            if i not in best_set:
                valid = True
                for partners in partner_sets[i]:
                    if all(p in best_set for p in partners):
                        valid = False
                        break
                if valid:
                    best_set.add(i)
                    improved = True
                    
    result = [list(all_triples[i]) for i in best_set]
    return result
