import itertools
import random
import time

def construct():
    n = 16
    points = list(range(n))
    all_triples = list(itertools.combinations(points, 3))
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    # Precompute conflicts
    # conflicts[i] is a list of frozensets of indices of the other 7 triples
    conflicts = [[] for _ in range(num_triples)]
    
    # To speed up precomputation, we can just iterate
    for i, t in enumerate(all_triples):
        a, b, c = t
        others = [p for p in points if p not in (a, b, c)]
        for a_prime in others:
            for b_prime in others:
                if b_prime == a_prime: continue
                for c_prime in others:
                    if c_prime in (a_prime, b_prime): continue
                    req_triples = [
                        (a, b, c_prime),
                        (a, b_prime, c),
                        (a, b_prime, c_prime),
                        (a_prime, b, c),
                        (a_prime, b, c_prime),
                        (a_prime, b_prime, c),
                        (a_prime, b_prime, c_prime)
                    ]
                    req_indices = frozenset(triple_to_idx[tuple(sorted(rt))] for rt in req_triples)
                    conflicts[i].append(req_indices)
                    
    best_set = set()
    start_time = time.time()
    time_limit = 145 # seconds
    
    # Initial run with a fixed order to get a baseline quickly?
    # Or just random.
    
    while time.time() - start_time < time_limit:
        current_set = set()
        indices = list(range(num_triples))
        random.shuffle(indices)
        
        for idx in indices:
            can_add = True
            for req_set in conflicts[idx]:
                if req_set.issubset(current_set):
                    can_add = False
                    break
            if can_add:
                current_set.add(idx)
                
        if len(current_set) > len(best_set):
            best_set = current_set
            
    return [list(all_triples[i]) for i in best_set]
