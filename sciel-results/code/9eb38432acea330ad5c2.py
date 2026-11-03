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
    # adj[i] is a bitmask of j such that i conflicts with j
    adj = [0] * N
    
    # To speed up, we can iterate over pairs of supports that intersect in 3
    supports = list(support_map.keys())
    
    # Map support to list of indices
    sup_to_vecs = {s: support_map[s] for s in supports}
    
    # Pre-compute intersection size 3 pairs
    # There are 330 supports.
    # For each pair, check if intersection is 3.
    # If so, link vectors with matching signs on intersection.
    
    for i_s in range(len(supports)):
        s1 = supports[i_s]
        vecs_s1 = sup_to_vecs[s1]
        for i_t in range(i_s + 1, len(supports)):
            s2 = supports[i_t]
            inter = set(s1) & set(s2)
            if len(inter) != 3:
                continue
            
            # Find positions of intersection in s1 and s2
            pos_in_s1 = [s1.index(x) for x in inter]
            pos_in_s2 = [s2.index(x) for x in inter]
            
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
                        adj[idx1] |= (1 << idx2)
                        adj[idx2] |= (1 << idx1)

    # 3. Solve MIS using Branch and Bound with Heuristics
    
    best_size = 0
    best_mask = 0
    
    def popcount(x):
        return x.bit_count()

    def greedy_mis(candidates):
        selected = 0
        remaining = candidates
        while remaining:
            min_deg = float('inf')
            v = -1
            temp = remaining
            while temp:
                lsb = temp & -temp
                u = lsb.bit_length() - 1
                deg = popcount(adj[u] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    v = u
                temp ^= lsb
                
            if v == -1:
                break
                
            selected |= (1 << v)
            remaining &= ~(adj[v] | (1 << v))
            
        return selected, popcount(selected)

    # Initial greedy to set a good lower bound
    g_mask, g_count = greedy_mis((1 << N) - 1)
    best_size = g_count
    best_mask = g_mask

    start_time = time.time()
    time_limit = 115.0

    def solve(candidates, current_mask, current_count):
        nonlocal best_size, best_mask
        
        if time.time() - start_time > time_limit:
            return
            
        if not candidates:
            if current_count > best_size:
                best_size = current_count
                best_mask = current_mask
            return
            
        # Pruning
        if current_count + popcount(candidates) <= best_size:
            return
            
        # Heuristic: Select vertex with MAX degree in candidates
        # This is the "Most Constrained" variable in CSP terms (removes most options)
        max_deg = -1
        v = -1
        temp = candidates
        while temp:
            lsb = temp & -temp
            u = lsb.bit_length() - 1
            deg = popcount(adj[u] & candidates)
            if deg > max_deg:
                max_deg = deg
                v = u
            temp ^= lsb
            
        if v == -1:
            return
            
        v_bit = 1 << v
        
        # Branch 1: Exclude v
        # This keeps more candidates, likely to find larger sets
        new_candidates_excl = candidates & ~v_bit
        solve(new_candidates_excl, current_mask, current_count)
        
        # Branch 2: Include v
        new_candidates_incl = candidates & ~(adj[v] | v_bit)
        solve(new_candidates_incl, current_mask | v_bit, current_count + 1)

    initial_candidates = (1 << N) - 1
    solve(initial_candidates, 0, 0)
    
    if best_size == 0:
        return np.array([], dtype=np.int64).reshape(0, D)
        
    result_vectors = []
    temp = best_mask
    while temp:
        lsb = temp & -temp
        i = lsb.bit_length() - 1
        result_vectors.append(all_vectors[i][3])
        temp ^= lsb
        
    return np.array(result_vectors, dtype=np.int64)
