import numpy as np
import itertools
import time
import sys

D = 11

def construct():
    # Generate all vectors with exactly 4 non-zero entries (+/- 1)
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
    
    # Build Conflict Graph
    # Two vectors conflict (distance < 2) iff they share exactly 3 coordinates
    # with matching signs (so inner product = 3, dist^2 = 8+8-2*3=10 < 8? wait)
    # Actually: |v|^2 = 4, |w|^2 = 4. dist^2 = 8 - 2*v·w.
    # Need dist^2 >= 4 (norm^2) => 8 - 2*v·w >= 4 => v·w <= 2.
    # If supports intersect in 3 coords with matching signs: v·w = 3 -> dist^2 = 2 < 4. CONFLICT.
    # If supports intersect in 3 coords with 2 matching, 1 opposing: v·w = 1 -> dist^2 = 6 >= 4. OK.
    # If supports intersect in 3 coords with 1 matching, 2 opposing: v·w = -1 -> dist^2 = 10 >= 4. OK.
    # If supports intersect in 3 coords with 0 matching: v·w = -3 -> dist^2 = 14 >= 4. OK.
    # If supports intersect in 2 coords: |v·w| <= 2 -> dist^2 >= 4. OK.
    # If supports intersect in 4 coords (same support): v·w = 4,0,-4 -> dist^2 = 0,8,16.
    #   v·w=4 (identical): dist^2=0, but we have unique vectors so no self-conflict.
    #   v·w=0: dist^2=8>=4. OK.
    #   v·w=-4: dist^2=16>=4. OK.
    # So conflict iff: supports intersect in exactly 3 AND signs match on all 3.
    
    adj = [0] * N
    
    supports = list(support_map.keys())
    sup_to_vecs = {s: support_map[s] for s in supports}
    
    for i_s, s1 in enumerate(supports):
        for i_t, s2 in enumerate(supports):
            if i_t <= i_s:
                continue
            inter = set(s1) & set(s2)
            if len(inter) == 3:
                inter_list = list(inter)
                pos_in_s1 = [s1.index(x) for x in inter_list]
                pos_in_s2 = [s2.index(x) for x in inter_list]
                
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
                            adj[idx1] |= (1 << idx2)
                            adj[idx2] |= (1 << idx1)

    # Use bit_count if available
    try:
        (1).bit_count()
        def popcount(x):
            return x.bit_count()
    except AttributeError:
        def popcount(x):
            return bin(x).count('1')

    start_time = time.time()
    time_limit = 115.0

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

    best_size = 0
    best_mask = 0

    def solve(candidates, current_mask, current_count):
        nonlocal best_size, best_mask
        
        if time.time() - start_time > time_limit:
            return
            
        if not candidates:
            if current_count > best_size:
                best_size = current_count
                best_mask = current_mask
            return
            
        if current_count + popcount(candidates) <= best_size:
            return
            
        g_mask, g_count = greedy_mis(candidates)
        if current_count + g_count > best_size:
            best_size = current_count + g_count
            best_mask = current_mask | g_mask
            
        if current_count + popcount(candidates) <= best_size:
            return
            
        # Select vertex with maximum degree for branching
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
        
        # Branch 1: include v
        new_candidates_1 = candidates & ~(adj[v] | v_bit)
        solve(new_candidates_1, current_mask | v_bit, current_count + 1)
        
        # Branch 2: exclude v
        new_candidates_2 = candidates & ~v_bit
        solve(new_candidates_2, current_mask, current_count)

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
