import numpy as np
import random
import itertools
import time

def construct():
    D = 11
    
    # Generate all 4-subsets (supports)
    supports = list(itertools.combinations(range(D), 4))
    n_sup = len(supports)
    
    # Precompute the triples for each support
    support_triples = []
    for s in supports:
        trs = list(itertools.combinations(s, 3))
        support_triples.append(trs)
        
    # Map triple to index for fast lookup
    all_triples = list(itertools.combinations(range(D), 3))
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    n_triples = len(all_triples)
    
    # Convert support_triples to lists of triple indices
    sup_triple_ids = []
    for trs in support_triples:
        ids = [triple_to_idx[t] for t in trs]
        sup_triple_ids.append(ids)

    best_count = 0
    best_selection = []
    
    start_time = time.time()
    max_time = 100.0
    
    # Greedy Set Packing with random restarts
    # We want to maximize the number of supports selected such that their triples are disjoint.
    
    # We can also try a "weighted" greedy or local search, but random greedy is robust.
    # To improve, we can try to fill gaps.
    
    for restart in range(5000):
        if time.time() - start_time > max_time:
            break
            
        order = list(range(n_sup))
        random.shuffle(order)
        
        used_triples = [False] * n_triples
        selected = []
        
        for idx in order:
            ids = sup_triple_ids[idx]
            # Check if all 4 triples are free
            # Optimization: check first, then the rest
            if not used_triples[ids[0]] and not used_triples[ids[1]] and not used_triples[ids[2]] and not used_triples[ids[3]]:
                selected.append(idx)
                used_triples[ids[0]] = True
                used_triples[ids[1]] = True
                used_triples[ids[2]] = True
                used_triples[ids[3]] = True
        
        if len(selected) > best_count:
            best_count = len(selected)
            best_selection = selected.copy()
            
            if best_count == n_triples // 4:
                break
                
    # Construct vectors
    vectors = []
    for idx in best_selection:
        sup = supports[idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for k, s in enumerate(sup):
                v[s] = signs[k]
            vectors.append(v)
            
    if not vectors:
        v = [0]*D
        v[0]=1; v[1]=1; v[2]=1; v[3]=1
        vectors.append(v)
        
    return np.array(vectors, dtype=np.int64)
