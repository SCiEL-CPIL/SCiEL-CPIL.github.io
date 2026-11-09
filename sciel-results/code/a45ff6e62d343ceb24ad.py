"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools

# EVOLVE-BLOCK-START
def _build_pair_index(triple_set):
    """Build index: for each pair (x,y), store the set of z such that (x,y,z) is in triple_set."""
    pair_index = {}
    for (a, b, c) in triple_set:
        for x, y, z in [(a, b, c), (a, c, b), (b, c, a)]:
            key = (min(x, y), max(x, y))
            if key not in pair_index:
                pair_index[key] = set()
            pair_index[key].add(z)
    return pair_index

def _is_valid_addition(triple, triple_set, pair_index):
    """Check if adding triple creates a K(2,2,2).
    
    K(2,2,2) has 3 disjoint pairs {a,a'}, {b,b'}, {c,c'} and all 8 transversals present.
    The new triple (a,b,c) is one of the 8. We need to check if there exist a',b',c'
    such that all 8 transversals are in triple_set ∪ {(a,b,c)}.
    
    The other 7 transversals must all be in triple_set.
    """
    a, b, c = triple
    
    for a_prime in range(16):
        if a_prime in (a, b, c):
            continue
        for b_prime in range(16):
            if b_prime in (a, b, c, a_prime):
                continue
            
            ab_prime_key = (min(a, b_prime), max(a, b_prime))
            a_prime_b_key = (min(a_prime, b), max(a_prime, b))
            a_prime_b_prime_key = (min(a_prime, b_prime), max(a_prime, b_prime))
            
            if c not in pair_index.get(ab_prime_key, set()):
                continue
            if c not in pair_index.get(a_prime_b_key, set()):
                continue
            if c not in pair_index.get(a_prime_b_prime_key, set()):
                continue
            
            ab_key = (min(a, b), max(a, b))
            candidates = pair_index.get(ab_key, set()).copy()
            candidates &= pair_index.get(ab_prime_key, set())
            candidates &= pair_index.get(a_prime_b_key, set())
            candidates &= pair_index.get(a_prime_b_prime_key, set())
            
            forbidden = {a, b, c, a_prime, b_prime}
            candidates -= forbidden
            
            if candidates:
                return False
    
    return True

def _greedy_construct(deadline, seed):
    """Greedy construction with randomization."""
    random.seed(seed)
    all_triples = [tuple(sorted((a, b, c))) for a in range(16) for b in range(a+1, 16) for c in range(b+1, 16)]
    random.shuffle(all_triples)
    
    triple_set = set()
    pair_index = {}
    
    for triple in all_triples:
        if time.time() > deadline:
            break
        if _is_valid_addition(triple, triple_set, pair_index):
            triple_set.add(triple)
            a, b, c = triple
            for x, y, z in [(a, b, c), (a, c, b), (b, c, a)]:
                key = (min(x, y), max(x, y))
                if key not in pair_index:
                    pair_index[key] = set()
                pair_index[key].add(z)
    
    return list(triple_set)

def _local_search(triples, deadline):
    """Local search: try to add more triples, and try swaps."""
    triple_set = set(tuple(t) for t in triples)
    pair_index = _build_pair_index(triple_set)
    all_triples = [tuple(sorted((a, b, c))) for a in range(16) for b in range(a+1, 16) for c in range(b+1, 16)]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Phase 1: Try to add any valid triple
        for t in all_triples:
            if t in triple_set:
                continue
            if _is_valid_addition(t, triple_set, pair_index):
                triple_set.add(t)
                a, b, c = t
                for x, y, z in [(a, b, c), (a, c, b), (b, c, a)]:
                    key = (min(x, y), max(x, y))
                    if key not in pair_index:
                        pair_index[key] = set()
                    pair_index[key].add(z)
                improved = True
                break
        
        if time.time() > deadline:
            break
        
        # Phase 2: Try swaps - remove one triple, add two
        if not improved:
            for t_remove in list(triple_set):
                if time.time() > deadline:
                    break
                
                triple_set.remove(t_remove)
                pair_index = _build_pair_index(triple_set)
                
                added_count = 0
                added_triples = []
                for t_add in all_triples:
                    if t_add in triple_set:
                        continue
                    if _is_valid_addition(t_add, triple_set, pair_index):
                        triple_set.add(t_add)
                        a, b, c = t_add
                        for x, y, z in [(a, b, c), (a, c, b), (b, c, a)]:
                            key = (min(x, y), max(x, y))
                            if key not in pair_index:
                                pair_index[key] = set()
                            pair_index[key].add(z)
                        added_count += 1
                        added_triples.append(t_add)
                
                if added_count >= 2:
                    improved = True
                    break
                else:
                    for t in added_triples:
                        triple_set.remove(t)
                    triple_set.add(t_remove)
                    pair_index = _build_pair_index(triple_set)
    
    return list(triple_set)

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill
    
    best_result = []
    num_attempts = 12
    
    for i in range(num_attempts):
        if time.time() > deadline:
            break
        result = _greedy_construct(deadline, seed=i * 7919 + 104729)
        if len(result) > len(best_result):
            best_result = result
    
    if time.time() < deadline:
        best_result = _local_search(best_result, deadline)
    
    return best_result
# EVOLVE-BLOCK-END
