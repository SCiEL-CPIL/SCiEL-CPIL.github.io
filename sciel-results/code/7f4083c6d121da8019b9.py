import numpy as np
import itertools
import time
import random
from collections import defaultdict

def construct():
    """
    Constructs a set of nonzero integer vectors in R^11 such that the minimum 
    pairwise distance is at least the maximum vector norm.
    
    Strategy: 
    1. Generate all weight-4 vectors (5280 candidates).
    2. Use efficient greedy with pre-sorted orders and multiple restarts.
    3. Use swap-based local search to improve.
    4. Add compatible weight-2 vectors.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin for 120s limit
    
    D = 11
    
    # Generate all weight-4 vectors
    vectors = []
    vec_to_idx = {}
    
    for support in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            idx = len(vectors)
            vectors.append((support, signs))
            key = (support, signs)
            vec_to_idx[key] = idx
            
    n_vecs = len(vectors)
    
    # Build conflict graph
    triplet_map = defaultdict(list)
    
    for idx, (support, signs) in enumerate(vectors):
        for k in range(4):
            trip_sup = tuple(support[j] for j in range(4) if j != k)
            trip_sig = tuple(signs[j] for j in range(4) if j != k)
            key = (trip_sup, trip_sig)
            triplet_map[key].append(idx)
    
    neighbors = [set() for _ in range(n_vecs)]
    
    for idx, (support, signs) in enumerate(vectors):
        for k in range(4):
            trip_sup = tuple(support[j] for j in range(4) if j != k)
            trip_sig = tuple(signs[j] for j in range(4) if j != k)
            key = (trip_sup, trip_sig)
            for other_idx in triplet_map[key]:
                if other_idx != idx:
                    neighbors[idx].add(other_idx)
    
    degrees = [len(neighbors[i]) for i in range(n_vecs)]
    
    def greedy_sorted(order):
        """Greedy with pre-sorted order"""
        x = [0] * n_vecs
        remaining = set(range(n_vecs))
        
        for v in order:
            if v not in remaining:
                continue
            x[v] = 1
            remaining.discard(v)
            for u in neighbors[v]:
                remaining.discard(u)
        
        return x
    
    def swap_local_search(x, max_time=8.0):
        """Try swaps to improve independent set size"""
        x = x[:]
        selected = set(i for i in range(n_vecs) if x[i])
        unselected = set(range(n_vecs)) - selected
        
        start_ls = time.time()
        
        improved = True
        while improved and time.time() < start_ls + max_time and time.time() < deadline - 10.0:
            improved = False
            
            # Try to add an unselected vertex that has no conflicts with selected
            for u in list(unselected):
                if time.time() > deadline - 10.0:
                    break
                if not (neighbors[u] & selected):
                    x[u] = 1
                    selected.add(u)
                    unselected.discard(u)
                    improved = True
                    break
            
            if time.time() > deadline - 10.0:
                break
            
            # Try swaps: remove one selected, add two unselected
            if time.time() < deadline - 15.0 and len(unselected) > 1:
                for s in list(selected):
                    if time.time() > deadline - 10.0:
                        break
                    potential_adds = []
                    for u in unselected:
                        if u == s:
                            continue
                        if not (neighbors[u] & (selected - {s})):
                            potential_adds.append(u)
                    
                    added = []
                    for u in potential_adds:
                        if time.time() > deadline - 10.0:
                            break
                        if not (neighbors[u] & set(added)):
                            added.append(u)
                        if len(added) == 2:
                            break
                    
                    if len(added) >= 2:
                        x[s] = 0
                        selected.discard(s)
                        unselected.add(s)
                        for u in added:
                            x[u] = 1
                            selected.add(u)
                            unselected.discard(u)
                        improved = True
                        break
        
        return x
    
    # Multiple greedy restarts with different orderings
    best_x = None
    best_size = 0
    
    rng = random.Random(42)
    
    for restart in range(8):
        if time.time() > deadline - 25.0:
            break
        
        if restart == 0:
            order = sorted(range(n_vecs), key=lambda i: degrees[i])
        elif restart == 1:
            order = sorted(range(n_vecs), key=lambda i: -degrees[i])
        elif restart == 2:
            order = list(range(n_vecs))
            rng.shuffle(order)
        elif restart == 3:
            order = sorted(range(n_vecs), key=lambda i: (degrees[i], rng.random()))
        elif restart == 4:
            order = sorted(range(n_vecs), key=lambda i: (degrees[i], rng.random()))
        elif restart == 5:
            order = list(range(n_vecs))
            rng.shuffle(order)
        elif restart == 6:
            order = sorted(range(n_vecs), key=lambda i: (degrees[i], rng.random()))
        else:
            order = list(range(n_vecs))
            rng.shuffle(order)
        
        x = greedy_sorted(order)
        size = sum(x)
        if size > best_size:
            best_size = size
            best_x = x[:]
    
    # Local search on best solution
    if best_x is not None and time.time() < deadline - 15.0:
        best_x = swap_local_search(best_x, max_time=min(8.0, deadline - time.time() - 10.0))
    
    # Convert to vector list
    selected_indices = [i for i in range(n_vecs) if best_x[i]] if best_x else []
    
    # Build result vectors from selected weight-4 vectors
    result_vectors = []
    for idx in selected_indices:
        support, signs = vectors[idx]
        v = [0] * D
        for i, s in zip(support, signs):
            v[i] = s
        result_vectors.append(v)
    
    # Try to add weight-2 vectors
    if time.time() < deadline - 10.0:
        w2_vectors = []
        for support in itertools.combinations(range(D), 2):
            for signs in itertools.product([1, -1], repeat=2):
                v = [0] * D
                for i, s in zip(support, signs):
                    v[i] = s
                w2_vectors.append(v)
        
        selected_maps = []
        for idx in selected_indices:
            support, signs = vectors[idx]
            m = {}
            for i, s in zip(support, signs):
                m[i] = s
            selected_maps.append(m)
        
        compatible_w2 = []
        for u in w2_vectors:
            ok = True
            u_supp = [i for i in range(D) if u[i] != 0]
            for m in selected_maps:
                if len(u_supp) == 2:
                    i1, i2 = u_supp
                    if i1 in m and i2 in m:
                        if u[i1] == m[i1] and u[i2] == m[i2]:
                            ok = False
                            break
            if ok:
                compatible_w2.append(u)
        
        result_vectors.extend(compatible_w2)
    
    if not result_vectors:
        for support in itertools.combinations(range(D), 2):
            for signs in itertools.product([1, -1], repeat=2):
                v = [0] * D
                for i, s in zip(support, signs):
                    v[i] = s
                result_vectors.append(v)
                
    result = np.array(result_vectors, dtype=np.int64)
    return result
