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
    adj = [0] * N
    
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
                            adj[idx1] |= (1 << idx2)
                            adj[idx2] |= (1 << idx1)

    # 3. Solve MIS using Branch and Bound
    best_size = 0
    best_mask = 0
    
    # Use bit_count if available (Python 3.10+), else bin count
    try:
        (1).bit_count()
        def popcount(x):
            return x.bit_count()
    except AttributeError:
        def popcount(x):
            return bin(x).count('1')

    def get_color_class(remaining):
        selected = 0
        temp = remaining
        while temp:
            min_deg = float('inf')
            v = -1
            t = temp
            while t:
                lsb = t & -t
                u = lsb.bit_length() - 1
                deg = popcount(adj[u] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    v = u
                t ^= lsb
            if v == -1: break
            selected |= (1 << v)
            temp &= ~(adj[v] | (1 << v))
        return selected

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
            
        # Greedy Coloring Bound
        remaining = candidates
        color_classes = []
        while remaining:
            cc = get_color_class(remaining)
            if not cc: break
            color_classes.append(cc)
            remaining &= ~cc
            
        num_colors = len(color_classes)
        
        if current_count + num_colors <= best_size:
            return
            
        last_class = color_classes[-1]
        
        v_list = []
        temp = last_class
        while temp:
            lsb = temp & -temp
            u = lsb.bit_length() - 1
            v_list.append(u)
            temp ^= lsb
            
        for v in v_list:
            if time.time() - start_time > time_limit:
                return
                
            v_bit = 1 << v
            
            new_candidates_1 = candidates & ~(adj[v] | v_bit)
            solve(new_candidates_1, current_mask | v_bit, current_count + 1)
            
            candidates &= ~v_bit
            
            if current_count + popcount(candidates) <= best_size:
                break

    start_time = time.time()
    time_limit = 115.0
    
    def greedy_is(candidates):
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
            if v == -1: break
            selected |= (1 << v)
            remaining &= ~(adj[v] | (1 << v))
        return selected, popcount(selected)
        
    init_mask, init_size = greedy_is((1 << N) - 1)
    best_size = init_size
    best_mask = init_mask

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
