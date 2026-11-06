import time
import random
from itertools import combinations

V = 16
ALL_TRIPLES = list(combinations(range(V), 3))
N_TRIPLES = len(ALL_TRIPLES)

# Map triple to index
triple_to_idx = {t: i for i, t in enumerate(ALL_TRIPLES)}

def construct():
    start = time.time()
    deadline = start + 115  # Safety margin
    
    best_count = 0
    best_trips = set()
    
    # Baseline: all triples containing 0
    baseline = {tuple(sorted((0, a, b))) for a in range(1, V) for b in range(a+1, V)}
    best_count = len(baseline)
    best_trips = set(baseline)
    
    # Precompute for each triple, the list of triples that would form K(2,2,2) with it
    # A K(2,2,2) on vertices {a,a',b,b',c,c'} has 8 transversal triples
    # For triple (a,b,c), a' != a,b,c; b' != a,b,c,a'; c' != a,b,c,a',b'
    # The 8 triples are: (a,b,c), (a,b,c'), (a,b',c), (a,b',c'), (a',b,c), (a',b,c'), (a',b',c), (a',b',c')
    
    # conflict_mask[i] = bitmask of triples j such that adding both i and j (plus others) could form K(2,2,2)
    # Actually, we need: for triple i=(a,b,c), which triples j conflict such that there exist a',b',c' making all 8 present
    # This is complex. Let's use a different approach.
    
    # For each triple (a,b,c), precompute the "conflict set": triples that share exactly 2 vertices with it
    # and are in the same "transversal structure"
    
    # Better approach: for each triple i, precompute all possible K(2,2,2) patterns that include i
    # A K(2,2,2) containing triple (a,b,c) requires:
    # - a' not in {a,b,c}
    # - b' not in {a,b,c,a'}  
    # - c' not in {a,b,c,a',b'}
    # - All 8 transversals present
    
    # Let's precompute for each triple i, the list of "other 7 triples" needed for each possible K(2,2,2)
    # This is too expensive to precompute all. Instead, let's optimize the greedy check.
    
    # Use numpy for faster operations
    import numpy as np
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        
        # Randomize the order
        order = list(range(N_TRIPLES))
        random.shuffle(order)
        
        # Use a boolean array for presence
        presence = bytearray(N_TRIPLES)
        count = 0
        
        # Track which triples are present for fast lookup
        for idx in order:
            if time.time() >= deadline:
                break
            
            a, b, c = ALL_TRIPLES[idx]
            
            # Check for K(2,2,2) conflict
            conflict = False
            
            # For K(2,2,2) with triple (a,b,c), we need:
            # a' not in {a,b,c}, b' not in {a,b,c,a'}, c' not in {a,b,c,a',b'}
            # All 8 transversals must be present (or being added)
            
            # Optimization: iterate over a' first
            for a_p in range(V):
                if a_p == a or a_p == b or a_p == c:
                    continue
                
                # Check if (a_p, b, c) is present
                t1 = (min(a_p, b), min(max(a_p, b), c), max(a_p, b, c))
                if not presence[triple_to_idx[t1]]:
                    continue
                
                # Now check for b' and c'
                # b' must be in {a, b', c} and {a_p, b', c} both present
                # c' must be in {a, b, c'} and {a_p, b, c'} both present
                # And {a, b', c'} and {a_p, b', c'} both present
                
                # Find candidate b' values
                b_candidates = []
                for b_p in range(V):
                    if b_p in (a, b, c, a_p):
                        continue
                    # Check {a, b_p, c}
                    t2 = (min(a, b_p), min(max(a, b_p), c), max(a, b_p, c))
                    if not presence[triple_to_idx[t2]]:
                        continue
                    # Check {a_p, b_p, c}
                    t3 = (min(a_p, b_p), min(max(a_p, b_p), c), max(a_p, b_p, c))
                    if not presence[triple_to_idx[t3]]:
                        continue
                    b_candidates.append(b_p)
                
                if not b_candidates:
                    continue
                
                # Find candidate c' values
                c_candidates = []
                for c_p in range(V):
                    if c_p in (a, b, c, a_p):
                        continue
                    # Check {a, b, c_p}
                    t4 = (min(a, b), min(max(a, b), c_p), max(a, b, c_p))
                    if not presence[triple_to_idx[t4]]:
                        continue
                    # Check {a_p, b, c_p}
                    t5 = (min(a_p, b), min(max(a_p, b), c_p), max(a_p, b, c_p))
                    if not presence[triple_to_idx[t5]]:
                        continue
                    c_candidates.append(c_p)
                
                if not c_candidates:
                    continue
                
                # Check pairs (b_p, c_p)
                found = False
                for b_p in b_candidates:
                    for c_p in c_candidates:
                        if b_p == c_p:
                            continue
                        # Check {a, b_p, c_p}
                        t6 = (min(a, b_p), min(max(a, b_p), c_p), max(a, b_p, c_p))
                        if not presence[triple_to_idx[t6]]:
                            continue
                        # Check {a_p, b_p, c_p}
                        t7 = (min(a_p, b_p), min(max(a_p, b_p), c_p), max(a_p, b_p, c_p))
                        if not presence[triple_to_idx[t7]]:
                            continue
                        
                        found = True
                        break
                    if found:
                        break
                if found:
                    conflict = True
                    break
            
            if not conflict:
                presence[idx] = 1
                count += 1
        
        if count > best_count:
            best_count = count
            best_trips = set(ALL_TRIPLES[i] for i in range(N_TRIPLES) if presence[i])
        
        # Try local improvement: try to add more triples
        if time.time() < deadline - 5:
            # Try to add triples that were skipped
            for idx in order:
                if time.time() >= deadline - 2:
                    break
                if presence[idx]:
                    continue
                a, b, c = ALL_TRIPLES[idx]
                conflict = False
                for a_p in range(V):
                    if a_p in (a, b, c):
                        continue
                    t1 = (min(a_p, b), min(max(a_p, b), c), max(a_p, b, c))
                    if not presence[triple_to_idx[t1]]:
                        continue
                    b_candidates = []
                    for b_p in range(V):
                        if b_p in (a, b, c, a_p):
                            continue
                        t2 = (min(a, b_p), min(max(a, b_p), c), max(a, b_p, c))
                        if not presence[triple_to_idx[t2]]:
                            continue
                        t3 = (min(a_p, b_p), min(max(a_p, b_p), c), max(a_p, b_p, c))
                        if not presence[triple_to_idx[t3]]:
                            continue
                        b_candidates.append(b_p)
                    if not b_candidates:
                        continue
                    c_candidates = []
                    for c_p in range(V):
                        if c_p in (a, b, c, a_p):
                            continue
                        t4 = (min(a, b), min(max(a, b), c_p), max(a, b, c_p))
                        if not presence[triple_to_idx[t4]]:
                            continue
                        t5 = (min(a_p, b), min(max(a_p, b), c_p), max(a_p, b, c_p))
                        if not presence[triple_to_idx[t5]]:
                            continue
                        c_candidates.append(c_p)
                    if not c_candidates:
                        continue
                    for b_p in b_candidates:
                        for c_p in c_candidates:
                            if b_p == c_p:
                                continue
                            t6 = (min(a, b_p), min(max(a, b_p), c_p), max(a, b_p, c_p))
                            if not presence[triple_to_idx[t6]]:
                                continue
                            t7 = (min(a_p, b_p), min(max(a_p, b_p), c_p), max(a_p, b_p, c_p))
                            if not presence[triple_to_idx[t7]]:
                                continue
                            conflict = True
                            break
                        if conflict:
                            break
                    if conflict:
                        break
                if not conflict:
                    presence[idx] = 1
                    count += 1
                    best_count = count
                    best_trips = set(ALL_TRIPLES[i] for i in range(N_TRIPLES) if presence[i])
    
    return [list(t) for t in best_trips]
