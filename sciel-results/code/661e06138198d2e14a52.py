"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def _make_triple_set(triples):
    return set(tuple(sorted(t)) for t in triples)

def _has_k222(triple, triple_set):
    """
    Checks if adding `triple` to `triple_set` creates a K(2,2,2).
    A K(2,2,2) is defined by 3 disjoint pairs {a,a'}, {b,b'}, {c,c'}
    such that all 8 transversals (combinations of one from each pair) are present.
    
    If we add t=(a,b,c), we assume a is from pair 1, b from pair 2, c from pair 3.
    We need to find a', b', c' such that the other 7 transversals are in triple_set.
    The 7 transversals are:
    1. (a', b, c)
    2. (a, b', c)
    3. (a, b, c')
    4. (a', b', c)
    5. (a', b, c')
    6. (a, b', c')
    7. (a', b', c')
    """
    a, b, c = triple
    ts = triple_set
    
    # Precompute sorted tuples for the fixed parts to speed up lookups
    # We iterate a' and b', then derive c' or check existence.
    
    for a_prime in range(16):
        if a_prime in (a, b, c):
            continue
        # Check transversals involving a' and fixed b, c
        t_ab_c = tuple(sorted((a_prime, b, c)))
        if t_ab_c not in ts:
            continue
            
        for b_prime in range(16):
            if b_prime in (a, b, c, a_prime):
                continue
            # Check transversals involving a', b' and fixed c
            t_abp_c = tuple(sorted((a_prime, b_prime, c)))
            if t_abp_c not in ts:
                continue
            
            # Check transversals involving a and fixed b, c' is not fixed yet, but we can check
            # transversals that don't depend on c' first?
            # t_a_bp_c = (a, b', c)
            t_a_bp_c = tuple(sorted((a, b_prime, c)))
            if t_a_bp_c not in ts:
                continue
            
            # Now we have a', b'. We need to find c' such that:
            # (a, b, c'), (a', b, c'), (a, b', c'), (a', b', c') are in ts.
            # Note: c' must be distinct from a, b, c, a', b'.
            
            # Optimization: Iterate c'
            for c_prime in range(16):
                if c_prime in (a, b, c, a_prime, b_prime):
                    continue
                
                # Check the 4 remaining transversals
                t_a_b_cp = tuple(sorted((a, b, c_prime)))
                if t_a_b_cp not in ts:
                    continue
                t_ap_b_cp = tuple(sorted((a_prime, b, c_prime)))
                if t_ap_b_cp not in ts:
                    continue
                t_a_bp_cp = tuple(sorted((a, b_prime, c_prime)))
                if t_a_bp_cp not in ts:
                    continue
                t_ap_bp_cp = tuple(sorted((a_prime, b_prime, c_prime)))
                if t_ap_bp_cp not in ts:
                    continue
                
                # If we reach here, all 7 other transversals are present.
                return True
                
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    all_triples = [(a, b, c) for a in range(16) for b in range(a+1, 16) for c in range(b+1, 16)]
    random.seed(42)
    
    # Start with all triples through point 0
    current = set()
    for a in range(1, 16):
        for b in range(a+1, 16):
            current.add((0, a, b))
    
    # Iteratively try to add triples
    for _ in range(200):
        if time.time() > deadline:
            break
        added = False
        candidates = [t for t in all_triples if t not in current]
        random.shuffle(candidates)
        for t in candidates:
            if time.time() > deadline:
                break
            if not _has_k222(t, current):
                current.add(t)
                added = True
    
    # Local search: try swaps (remove 1, add up to 2)
    for _ in range(100):
        if time.time() > deadline:
            break
        if len(current) < 5:
            break
        # Pick a random triple to remove
        to_remove = random.choice(tuple(current))
        current.discard(to_remove)
        
        added_count = 0
        candidates = [t for t in all_triples if t not in current]
        random.shuffle(candidates)
        for t in candidates:
            if added_count >= 2:
                break
            if time.time() > deadline:
                break
            if not _has_k222(t, current):
                current.add(t)
                added_count += 1
        
        if added_count == 0:
            current.add(to_remove)
    
    return [list(t) for t in current]
# EVOLVE-BLOCK-END
