import numpy as np
import itertools
import time
import sys

D = 11

def construct():
    # 1. Generate all vectors and their indices
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
    # conflict_bits[i] will be an integer bitmask of indices j that conflict with i
    conflict_bits = [0] * N
    
    supports = list(support_map.keys())
    sup_to_vecs = {s: support_map[s] for s in supports}
    
    for i_s, s1 in enumerate(supports):
        for i_t in range(i_s + 1, len(supports)):
            s2 = supports[i_t]
            inter = set(s1) & set(s2)
            if len(inter) == 3:
                pos_in_s1 = [s1.index(x) for x in inter]
                pos_in_s2 = [s2.index(x) for x in inter]
                
                vecs_s1 = sup_to_vecs[s1]
                vecs_s2 = sup_to_vecs[s2]
                
                for idx1 in vecs_s1:
                    signs1 = all_vectors[idx1][2]
                    for idx2 in vecs_s2:
                        signs2 = all_vectors[idx2][2]
                        
                        match = True
                        for p1, p2 in zip(pos_in_s1, pos_in_s2):
                            if signs1[p1] != signs2[p2]:
                                match = False
                                break
                        if match:
                            conflict_bits[idx1] |= (1 << idx2)
                            conflict_bits[idx2] |= (1 << idx1)

    # 3. Build Compatibility Bitmasks
    all_mask = (1 << N) - 1
    compat_bits = [0] * N
    for i in range(N):
        compat_bits[i] = all_mask ^ (1 << i) ^ conflict_bits[i]

    # 4. Tomita Max Clique Algorithm
    best_size = 0
    best_clique = []
    start_time = time.time()
    time_limit = 110.0

    def get_lowest_bit(mask):
        if mask == 0:
            return -1
        return (mask & -mask).bit_length() - 1

    def color_sort(P_mask):
        if P_mask == 0:
            return [], []
        
        uncolored = P_mask
        order = []
        colors = []
        color = 1
        
        while uncolored:
            available = uncolored
            while available:
                v = get_lowest_bit(available)
                order.append(v)
                colors.append(color)
                
                bit_v = 1 << v
                available ^= bit_v
                uncolored ^= bit_v
                available &= ~compat_bits[v]
            
            color += 1
        
        return order, colors

    def expand(R, P_mask, X_mask, current_size):
        nonlocal best_size, best_clique, start_time, time_limit
        
        if time.time() - start_time > time_limit:
            return
        
        if P_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_clique = list(R)
            return
        
        order, colors = color_sort(P_mask)
        
        if current_size + colors[-1] <= best_size:
            return
            
        for i in range(len(order) - 1, -1, -1):
            if time.time() - start_time > time_limit:
                return
            
            v = order[i]
            
            if current_size + colors[i] <= best_size:
                break
                
            new_R = R + [v]
            new_P = P_mask & compat_bits[v]
            
            expand(new_R, new_P, X_mask | (1 << v), current_size + 1)
            
            P_mask ^= (1 << v)

    initial_P = all_mask
    initial_X = 0
    expand([], initial_P, initial_X, 0)
    
    if not best_clique:
        return np.array([], dtype=np.int64).reshape(0, D)
        
    result_vectors = [all_vectors[i][3] for i in best_clique]
    return np.array(result_vectors, dtype=np.int64)
