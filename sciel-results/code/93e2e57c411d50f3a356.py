import random
from itertools import combinations

def construct():
    n = 16
    points = range(n)
    all_triples = list(combinations(points, 3))
    triple_to_id = {t: i for i, t in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    # Precompute conflict groups
    # conflict_groups[i] is a list of tuples of 7 IDs
    # Each tuple represents a set of 7 triples that, if all present, 
    # prevent triple i from being added (because they form a K(2,2,2) with i)
    conflict_groups = [[] for _ in range(num_triples)]
    
    for i, t in enumerate(all_triples):
        a, b, c = t
        t_set = frozenset(t)
        # Find all disjoint triples t'
        # We can iterate over all other triples and check disjointness
        # Or generate them. There are 13 points left, choose 3.
        others = [x for x in points if x not in t_set]
        for t_prime in combinations(others, 3):
            a_prime, b_prime, c_prime = t_prime
            # The 7 neighbors are:
            # (a, b, c'), (a, b', c), (a', b, c), 
            # (a, b', c'), (a', b, c'), (a', b', c), (a', b', c')
            
            # Note: The pairs are {a, a'}, {b, b'}, {c, c'}
            # The 8 transversals are all combos.
            # t is (a, b, c).
            # The other 7 are:
            n1 = (a, b, c_prime)
            n2 = (a, b_prime, c)
            n3 = (a_prime, b, c)
            n4 = (a, b_prime, c_prime)
            n5 = (a_prime, b, c_prime)
            n6 = (a_prime, b_prime, c)
            n7 = (a_prime, b_prime, c_prime)
            
            # Get IDs
            ids = []
            valid = True
            for nt in (n1, n2, n3, n4, n5, n6, n7):
                if nt in triple_to_id:
                    ids.append(triple_to_id[nt])
                else:
                    # This shouldn't happen if we generated from valid points
                    valid = False
                    break
            if valid:
                conflict_groups[i].append(tuple(ids))

    best_size = 0
    best_result = []
    
    # Try multiple random seeds to find a better greedy solution
    for seed in range(20):
        random.seed(seed)
        indices = list(range(num_triples))
        random.shuffle(indices)
        
        present_ids = set()
        
        for idx in indices:
            # Check if adding idx is valid
            is_valid = True
            for group in conflict_groups[idx]:
                # Check if all 7 IDs in group are in present_ids
                # Using all() with a generator is efficient
                if all(g in present_ids for g in group):
                    is_valid = False
                    break
            
            if is_valid:
                present_ids.add(idx)
        
        if len(present_ids) > best_size:
            best_size = len(present_ids)
            best_result = [list(all_triples[i]) for i in present_ids]
            
    return best_result
