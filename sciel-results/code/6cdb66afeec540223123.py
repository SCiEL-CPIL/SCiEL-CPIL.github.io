"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools

# EVOLVE-BLOCK-START
def _make_key(a, b, c):
    """Create a sorted tuple key for a triple."""
    return tuple(sorted((a, b, c)))

def _build_pair_index(triple_set):
    """Build index: for each pair (x,y), store the set of z such that (x,y,z) is in triple_set."""
    pair_index = {}
    for (a, b, c) in triple_set:
        for (i, j) in [(a, b), (a, c), (b, c)]:
            key = (min(i, j), max(i, j))
            if key not in pair_index:
                pair_index[key] = set()
            pair_index[key].add(c if key != (min(a,b), max(a,b)) or (min(a,b), max(a,b)) == (i, j) else 
                               (b if (i,j) == (a,c) else a))
    # Rebuild properly
    pair_index = {}
    for (a, b, c) in triple_set:
        pairs = [(a, b, c), (a, c, b), (b, c, a)]
        for x, y, z in pairs:
            key = (min(x, y), max(x, y))
            if key not in pair_index:
                pair_index[key] = set()
            pair_index[key].add(z)
    return pair_index

def _is_valid_addition_fast(triple, triple_set, pair_index):
    """Check if adding triple creates a K(2,2,2) using pair index."""
    a, b, c = triple
    
    # For K(2,2,2), we need 3 disjoint pairs {a,a'}, {b,b'}, {c,c'}
    # The 8 transversals are all combinations choosing one from each pair
    # We have (a,b,c). Need to check if there exist a',b',c' disjoint from {a,b,c}
    # such that all 8 transversals are in triple_set ∪ {(a,b,c)}
    
    # For each candidate a' (not in {a,b,c}), b' (not in {a,b,c,a'}), c' (not in {a,b,c,a',b'}):
    # Check if all 7 other transversals are in triple_set
    
    # Optimized: use pair_index to quickly check
    for a_prime in range(16):
        if a_prime in (a, b, c):
            continue
        for b_prime in range(16):
            if b_prime in (a, b, c, a_prime):
                continue
            # Check pair (a, a_prime) - need c and c' in pair_index
            # Check pair (b, b_prime) - need a and a' in pair_index
            # Check pair (c, c_prime) - need a and a' in pair_index
            
            # The 7 other transversals:
            # (a,b,c'), (a,b',c), (a,b',c'), (a',b,c), (a',b,c'), (a',b',c), (a',b',c')
            
            # For each c_prime candidate, check all 7
            # But we can be smarter: for fixed a', b', check which c' values would work
            
            # (a,b,c') requires c' in pair_index[(a,b)]
            # (a,b',c) requires c in pair_index[(a,b')]  
            # (a',b,c) requires c in pair_index[(a',b)]
            # (a,b',c') requires c' in pair_index[(a,b')]
            # (a',b,c') requires c' in pair_index[(a',b)]
            # (a',b',c) requires c in pair_index[(a',b')]
            # (a',b',c') requires c' in pair_index[(a',b')]
            
            # So for fixed a', b', we need:
            # c in pair_index[(a,b')] AND c in pair_index[(a',b)] AND c in pair_index[(a',b')]
            if c not in pair_index.get((min(a, b_prime), max(a, b_prime)), set()):
                continue
            if c not in pair_index.get((min(a_prime, b), max(a_prime, b)), set()):
                continue
            if c not in pair_index.get((min(a_prime, b_prime), max(a_prime, b_prime)), set()):
                continue
            
            # Now find valid c' candidates:
            # c' must be in: pair_index[(a,b)], pair_index[(a,b')], pair_index[(a',b)], pair_index[(a',b')]
            # and c' not in {a, b, c, a_prime, b_prime}
            
            candidates = None
            for pair in [(a, b), (a, b_prime), (a_prime, b), (a_prime, b_prime)]:
                s = pair_index.get((min(pair[0], pair[1]), max(pair[0], pair[1])), set())
                if candidates is None:
                    candidates = s.copy()
                else:
                    candidates &= s
            
            if candidates is None:
                continue
            
            # Remove forbidden points
            forbidden = {a, b, c, a_prime, b_prime}
            candidates -= forbidden
            
            if not candidates:
                continue
            
            # For any c' in candidates, all 7 transversals exist
            # So adding (a,b,c) would create K(2,2,2)
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
        if _is_valid_addition_fast(triple, triple_set, pair_index):
            triple_set.add(triple)
            # Update pair_index
            a, b, c = triple
            for x, y, z in [(a, b, c), (a, c, b), (b, c, a)]:
                key = (min(x, y), max(x, y))
                if key not in pair_index:
                    pair_index[key] = set()
                pair_index[key].add(z)
    
    return list(triple_set)

def _local_search(triples, deadline, max_iters=100):
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
            if _is_valid_addition_fast(t, triple_set, pair_index):
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
            # Try removing each triple and see if we can add more
            for t_remove in list(triple_set):
                if time.time() > deadline:
                    break
                
                # Temporarily remove
                triple_set.remove(t_remove)
                # Rebuild pair_index (expensive but necessary for correctness)
                pair_index = _build_pair_index(triple_set)
                
                # Try to add as many as possible
                added_count = 0
                added_triples = []
                for t_add in all_triples:
                    if t_add in triple_set:
                        continue
                    if _is_valid_addition_fast(t_add, triple_set, pair_index):
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
                    # Keep the swap
                    improved = True
                    break
                else:
                    # Revert: remove added, add back removed
                    for t in added_triples:
                        triple_set.remove(t)
                    triple_set.add(t_remove)
                    pair_index = _build_pair_index(triple_set)
    
    return list(triple_set)

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin
    
    # Try multiple random greedy constructions
    best_result = []
    num_attempts = 8
    
    for i in range(num_attempts):
        if time.time() > deadline:
            break
        result = _greedy_construct(deadline, seed=i * 7919 + 104729)
        if len(result) > len(best_result):
            best_result = result
    
    # Local search on best result
    if time.time() < deadline:
        best_result = _local_search(best_result, deadline, max_iters=50)
    
    return best_result
# EVOLVE-BLOCK-END
