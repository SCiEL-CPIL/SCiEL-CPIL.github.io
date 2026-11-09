import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    
    n = 16
    all_triples = list(combinations(range(n), 3))
    triple_index = {t: i for i, t in enumerate(all_triples)}
    num_triples = len(all_triples)  # 560
    
    # Precompute for each triple, the list of K(2,2,2) configurations it belongs to
    # A K(2,2,2) is identified by the set of 8 triples it contains
    # For each triple, store which K(2,2,2)s it's in
    
    # For triple (a,b,c), find all K(2,2,2)s containing it
    # A K(2,2,2) containing (a,b,c) is determined by choosing a',b',c' distinct from a,b,c and each other
    # such that {a,a'},{b,b'},{c,c'} are the three pairs
    
    # Precompute: for each triple, list of (k222_id) where k222_id identifies the configuration
    # Actually, let's precompute for each triple, the set of "partner" triples that would complete a K(2,2,2)
    
    # Better: precompute for each pair of triples, whether they're in a common K(2,2,2)
    # This is too expensive (560^2 = 313600 pairs, each check is expensive)
    
    # Let's use a different strategy: greedy with efficient conflict checking
    
    # For each triple, precompute all K(2,2,2)s it belongs to
    # A K(2,2,2) is fully determined by its 8 triples
    # For triple t=(a,b,c), the K(2,2,2)s it's in are parameterized by (a',b',c') with all distinct
    
    triple_k222s = [set() for _ in range(num_triples)]
    
    for a, b, c in all_triples:
        ti = triple_index[(a,b,c)]
        used = {a, b, c}
        for a_prime in range(n):
            if a_prime in used:
                continue
            for b_prime in range(n):
                if b_prime in used or b_prime == a_prime:
                    continue
                for c_prime in range(n):
                    if c_prime in used or c_prime == a_prime or c_prime == b_prime:
                        continue
                    
                    # The 8 triples in this K(2,2,2)
                    k222_triples = [
                        (a, b, c),
                        (a, b, c_prime),
                        (a, b_prime, c),
                        (a, b_prime, c_prime),
                        (a_prime, b, c),
                        (a_prime, b, c_prime),
                        (a_prime, b_prime, c),
                        (a_prime, b_prime, c_prime),
                    ]
                    
                    # Create a frozenset to identify this K(2,2,2)
                    k222_id = frozenset(tuple(sorted(t)) for t in k222_triples)
                    
                    # Add this K(2,2,2) to each of the 8 triples
                    for t in k222_triples:
                        t_sorted = tuple(sorted(t))
                        t_idx = triple_index[t_sorted]
                        triple_k222s[t_idx].add(k222_id)
    
    # Now for greedy: maintain for each K(2,2,2), how many of its 8 triples are selected
    # When adding a triple, check if any K(2,2,2) it's in would reach count 8
    
    def greedy_with_k222_counting():
        k222_count = {}  # k222_id -> count of selected triples
        selected = set()
        
        # Order triples by number of K(2,2,2)s they're in (ascending)
        triple_order = sorted(range(num_triples), key=lambda i: len(triple_k222s[i]))
        
        for ti in triple_order:
            triple = all_triples[ti]
            
            # Check if adding this triple would complete any K(2,2,2)
            can_add = True
            for k222_id in triple_k222s[ti]:
                if k222_id not in k222_count:
                    k222_count[k222_id] = 0
                if k222_count[k222_id] >= 7:
                    can_add = False
                    break
            
            if can_add:
                selected.add(ti)
                for k222_id in triple_k222s[ti]:
                    k222_count[k222_id] += 1
        
        return [all_triples[ti] for ti in selected]
    
    # Run greedy with different orderings
    random.seed(42)
    best_result = greedy_with_k222_counting()
    best_size = len(best_result)
    
    # Try random orderings
    for _ in range(100):
        if time.time() > deadline - 10:
            break
        
        k222_count = {}
        selected = set()
        
        order = list(range(num_triples))
        random.shuffle(order)
        
        for ti in order:
            triple = all_triples[ti]
            
            can_add = True
            for k222_id in triple_k222s[ti]:
                if k222_id not in k222_count:
                    k222_count[k222_id] = 0
                if k222_count[k222_id] >= 7:
                    can_add = False
                    break
            
            if can_add:
                selected.add(ti)
                for k222_id in triple_k222s[ti]:
                    k222_count[k222_id] += 1
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_result = [all_triples[ti] for ti in selected]
    
    return best_result
