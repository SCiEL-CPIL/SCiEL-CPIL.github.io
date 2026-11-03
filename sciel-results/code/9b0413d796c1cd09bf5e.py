import numpy as np
import itertools
import time
import sys

D = 11

def construct():
    # 1. Generate all vectors with exactly 4 non-zero entries (+/- 1)
    all_vectors = []
    support_map = {}
    
    idx = 0
    for comb in itertools.combinations(range(D), 4):
        support = comb
        support_map[support] = []
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for k, s in zip(support, signs):
                v[k] = s
            all_vectors.append((idx, support, signs, v))
            support_map[support].append(idx)
            idx += 1
            
    N = len(all_vectors)
    
    # 2. Build Conflict Graph
    # Conflict condition: Supports intersect in exactly 3, and signs match on those 3.
    neighbors = [set() for _ in range(N)]
    
    supports = list(support_map.keys())
    sup_to_vecs = {s: support_map[s] for s in supports}
    
    for i_s, s1 in enumerate(supports):
        for i_t, s2 in enumerate(supports):
            if i_t <= i_s: continue
            inter = set(s1) & set(s2)
            if len(inter) == 3:
                pos_in_s1 = [s1.index(x) for x in inter]
                pos_in_s2 = [s2.index(x) for x in inter]
                
                vecs_s1 = sup_to_vecs[s1]
                vecs_s2 = sup_to_vecs[s2]
                
                # Group s2 vectors by sign signature on intersection
                s2_by_sig = {}
                for idx2 in vecs_s2:
                    signs2 = all_vectors[idx2][2]
                    sig2 = tuple(signs2[p] for p in pos_in_s2)
                    if sig2 not in s2_by_sig:
                        s2_by_sig[sig2] = []
                    s2_by_sig[sig2].append(idx2)
                    
                for idx1 in vecs_s1:
                    signs1 = all_vectors[idx1][2]
                    sig1 = tuple(signs1[p] for p in pos_in_s1)
                    
                    if sig1 in s2_by_sig:
                        for idx2 in s2_by_sig[sig1]:
                            neighbors[idx1].add(idx2)
                            neighbors[idx2].add(idx1)
                            
    # Precompute static degrees
    static_degree = [len(n) for n in neighbors]
    
    best_size = 0
    best_mask = 0 # We will store the set of indices instead for easier reconstruction? Or bitmask.
    best_set = set()
    
    start_time = time.time()
    time_limit = 115.0

    def solve(candidates, current_count):
        nonlocal best_size, best_set
        
        if time.time() - start_time > time_limit:
            return
            
        if not candidates:
            if current_count > best_size:
                best_size = current_count
                best_set = set() # We need to track the actual set. 
                # This is a problem. We need to pass the selected set or reconstruct it.
                # Let's pass `selected` as a set.
            return
            
        if current_count + len(candidates) <= best_size:
            return
            
        # Heuristic: Pick vertex with max static degree
        v = max(candidates, key=lambda x: static_degree[x])
        
        # Branch 1: Include v
        new_cands_1 = candidates - {v} - neighbors[v]
        solve(new_cands_1, current_count + 1)
        
        # Branch 2: Exclude v
        new_cands_2 = candidates - {v}
        solve(new_cands_2, current_count)

    # We need to modify solve to track the selected set for reconstruction
    # Or, we can just return the size and then reconstruct?
    # Reconstructing from scratch is hard.
    # Let's pass `selected` set.
    
    def solve_with_set(candidates, current_count, selected):
        nonlocal best_size, best_set
        
        if time.time() - start_time > time_limit:
            return
            
        if not candidates:
            if current_count > best_size:
                best_size = current_count
                best_set = set(selected)
            return
            
        if current_count + len(candidates) <= best_size:
            return
            
        # Optional: Greedy lower bound to update best
        # This might be expensive. Let's skip for now.
        
        v = max(candidates, key=lambda x: static_degree[x])
        
        # Include v
        new_cands_1 = candidates - {v} - neighbors[v]
        selected.add(v)
        solve_with_set(new_cands_1, current_count + 1, selected)
        selected.remove(v)
        
        # Exclude v
        new_cands_2 = candidates - {v}
        solve_with_set(new_cands_2, current_count, selected)

    initial_candidates = set(range(N))
    solve_with_set(initial_candidates, 0, set())
    
    if best_size == 0:
        return np.array([], dtype=np.int64).reshape(0, D)
        
    result_vectors = []
    for i in best_set:
        result_vectors.append(all_vectors[i][3])
        
    return np.array(result_vectors, dtype=np.int64)
