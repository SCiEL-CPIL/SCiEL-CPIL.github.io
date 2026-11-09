import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    n = 16
    points = range(n)
    all_triples = list(itertools.combinations(points, 3))
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    # Precompute conflicts
    # conflicts[i] is a list of lists of 7 triple indices
    # that form a K(2,2,2) with triple i
    conflicts = [[] for _ in range(num_triples)]
    
    # For each triple T, find all K(2,2,2) it belongs to
    # T = (a,b,c).
    # Other points a', b', c' from remaining 13.
    # Pairs (a,a'), (b,b'), (c,c').
    # The 8 transversals are:
    # (a,b,c), (a,b,c'), (a,b',c), (a,b',c'), (a',b,c), (a',b,c'), (a',b',c), (a',b',c')
    # We need the 7 others.
    
    for t in all_triples:
        a, b, c = t
        # Points not in t
        remaining = [p for p in points if p not in t]
        # Choose a', b', c' from remaining
        # They must be distinct
        for a_prime in remaining:
            for b_prime in remaining:
                if b_prime == a_prime: continue
                for c_prime in remaining:
                    if c_prime == a_prime or c_prime == b_prime: continue
                    
                    # Check if (a,a'), (b,b'), (c,c') are valid pairs
                    # Yes, by construction they are disjoint from each other and from t?
                    # a,b,c are distinct. a',b',c' are distinct.
                    # Are a',b',c' distinct from a,b,c? Yes, from 'remaining'.
                    # Are a',b',c' distinct from each other? Yes.
                    # So we have 3 disjoint pairs: (a,a'), (b,b'), (c,c').
                    
                    # Generate the 7 other transversals
                    transversals = [
                        (a, b, c_prime),
                        (a, b_prime, c),
                        (a, b_prime, c_prime),
                        (a_prime, b, c),
                        (a_prime, b, c_prime),
                        (a_prime, b_prime, c),
                        (a_prime, b_prime, c_prime)
                    ]
                    
                    # Convert to indices
                    idx_list = []
                    valid = True
                    for trans in transversals:
                        # Sort to match tuple key
                        trans_sorted = tuple(sorted(trans))
                        if trans_sorted in triple_to_idx:
                            idx_list.append(triple_to_idx[trans_sorted])
                        else:
                            valid = False
                            break
                    if valid:
                        conflicts[triple_to_idx[t]].append(idx_list)
                        
    best_system = []
    best_count = 0
    
    # Randomized Greedy
    # We can optimize by shuffling and picking
    while time.time() < deadline:
        random.shuffle(all_triples)
        current_set = set()
        current_list = []
        
        for t in all_triples:
            idx = triple_to_idx[t]
            # Check if any conflict set is fully in current_set
            is_valid = True
            for c_set in conflicts[idx]:
                # Check if all 7 are in current_set
                # Optimization: if any is missing, break
                # Using set intersection or all()
                if all(x in current_set for x in c_set):
                    is_valid = False
                    break
            
            if is_valid:
                current_set.add(idx)
                current_list.append(t)
                
        if len(current_list) > best_count:
            best_count = len(current_list)
            best_system = [list(t) for t in current_list]
            
    return best_system
