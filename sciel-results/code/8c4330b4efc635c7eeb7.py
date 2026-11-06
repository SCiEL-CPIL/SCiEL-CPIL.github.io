import time
import itertools
import random

def _independence_number(triples, vertices=None):
    if vertices is None:
        vertices = list(range(16))
    n = len(vertices)
    idx = {v: i for i, v in enumerate(vertices)}
    
    # Convert triples to bitmasks
    masks = []
    for t in triples:
        m = 0
        for p in t:
            m |= (1 << idx[p])
        masks.append(m)
        
    if not masks:
        return n

    # Check sizes from n down to 1
    # We can precompute combinations for each size to speed up multiple calls, 
    # but since we call this many times, we should be careful.
    # However, n=16 is small.
    
    for r in range(n, 0, -1):
        # Generate all subsets of size r
        for combo in itertools.combinations(range(n), r):
            s_mask = 0
            for j in combo:
                s_mask |= (1 << j)
            
            # Check if s_mask is independent
            independent = True
            for m in masks:
                if (s_mask & m) == m:
                    independent = False
                    break
            
            if independent:
                return r
    return 0

def _greedy_construct(seed):
    random.seed(seed)
    V = list(range(16))
    all_triples = list(itertools.combinations(V, 3))
    random.shuffle(all_triples)
    
    used_pairs = set()
    triples = []
    
    for t in all_triples:
        # Check if t is linear with current triples
        ok = True
        for a, b in itertools.combinations(t, 2):
            pair = (min(a, b), max(a, b))
            if pair in used_pairs:
                ok = False
                break
        if ok:
            triples.append(list(t))
            for a, b in itertools.combinations(t, 2):
                used_pairs.add((min(a, b), max(a, b)))
                
    return triples

def construct():
    start = time.time()
    DEADLINE = 115.0  # Safety margin
    
    best_alpha = 16
    best_sys = []
    
    # Initial guess: the system from the prompt or a simple one
    # Let's just run the search
    
    seed = 0
    while time.time() - start < DEADLINE:
        sys_candidate = _greedy_construct(seed)
        # Simple validation: ensure linearity (greedy ensures it, but good to be sure)
        # And ensure it's a list of lists
        alpha = _independence_number(sys_candidate)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_sys = sys_candidate
            # If we reach 4, it's very good. 3 is theoretical min for this density?
            # Actually, can we get 3?
            # If alpha=3, every 4-set has a triple.
            # Number of 4-sets is C(16,4) = 1820.
            # Each triple is contained in 13 4-sets (16-3=13 points to add).
            # So we need at least 1820/13 = 140 triples.
            # But max triples is 37.
            # So alpha=3 is IMPOSSIBLE for a linear 3-uniform hypergraph on 16 points.
            # Proof: If alpha <= 3, then every 4-subset contains a triple.
            # Let N be the number of triples.
            # Each triple covers C(16-3, 1) = 13 four-subsets.
            # Total four-subsets = C(16, 4) = 1820.
            # So 13 * N >= 1820 => N >= 140.
            # But max N for linear 3-graph on 16 vertices is 37.
            # Contradiction.
            # So minimum alpha is at least 4.
            
            if best_alpha == 4:
                # We found the theoretical minimum for this structure class?
                # Let's double check the bound.
                # Yes, alpha=4 is the best possible.
                # We can stop early if we want, but let's see if we can find it quickly.
                # Breaking early saves time, but we need to return a valid system.
                # Since 4 is the min, we can break.
                break
        
        seed += 1
        
    # Fallback if no system found (should not happen)
    if not best_sys:
        best_sys = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
        
    return best_sys
