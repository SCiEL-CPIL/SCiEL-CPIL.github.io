import numpy as np
import time
from itertools import combinations, product

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    D = 11
    
    supports = list(combinations(range(D), 4))
    N_sup = len(supports)
    
    sup_masks = []
    for s in supports:
        mask = 0
        for i in s:
            mask |= (1 << i)
        sup_masks.append(mask)
    
    # Build incompatibility matrix
    incompat = np.zeros((N_sup, N_sup), dtype=np.uint8)
    for i in range(N_sup):
        if time.time() > deadline - 80.0:
            break
        for j in range(i + 1, N_sup):
            inter = sup_masks[i] & sup_masks[j]
            if inter.bit_count() >= 3:
                incompat[i, j] = 1
                incompat[j, i] = 1
    
    incompat_bool = incompat.astype(bool)
    
    # Precompute neighbor sets as frozensets for fast operations
    neighbors = []
    for i in range(N_sup):
        nbrs = set(np.where(incompat[i] == 1)[0].tolist())
        neighbors.append(nbrs)
    
    def greedy_mis(available, rng):
        selected = set()
        avail = set(available)
        
        while avail:
            if time.time() > deadline - 25.0:
                break
            
            if len(avail) > 150:
                avail_arr = np.array(list(avail), dtype=np.int32)
                deg_arr = incompat_bool[avail_arr][:, avail_arr].sum(axis=1)
                min_deg = deg_arr.min()
                candidates = avail_arr[deg_arr == min_deg]
                u = int(rng.choice(candidates))
            else:
                min_deg = float('inf')
                candidates = []
                for u in avail:
                    d = len(neighbors[u] & avail)
                    if d < min_deg:
                        min_deg = d
                        candidates = [u]
                    elif d == min_deg:
                        candidates.append(u)
                u = rng.choice(candidates)
            
            selected.add(u)
            avail -= {u} | (neighbors[u] & avail)
        
        return selected
    
    def local_search_add(selected):
        selected = set(selected)
        if not selected:
            return selected
        
        sel_arr = np.array(list(selected), dtype=np.int32)
        compatible_mask = ~incompat_bool[:, sel_arr].any(axis=1)
        compatible_mask[sel_arr] = False
        
        addable = np.where(compatible_mask)[0]
        for x in addable:
            if time.time() > deadline - 20.0:
                break
            selected.add(int(x))
        
        return selected
    
    def local_search_1_2_swap(selected, rng):
        selected = set(selected)
        improved = True
        
        while improved and time.time() < deadline - 20.0:
            improved = False
            sel_list = list(selected)
            
            for v in sel_list:
                if time.time() > deadline - 20.0:
                    break
                
                sel_minus_v = selected - {v}
                if not sel_minus_v:
                    compatible_all = np.ones(N_sup, dtype=bool)
                else:
                    smv_arr = np.array(list(sel_minus_v), dtype=np.int32)
                    compatible_all = ~incompat_bool[:, smv_arr].any(axis=1)
                compatible_all[v] = False
                compatible_all[list(selected)] = False
                
                compat_list = np.where(compatible_all)[0]
                n_compat = len(compat_list)
                
                found = False
                for i in range(n_compat):
                    if time.time() > deadline - 20.0:
                        break
                    u1 = compat_list[i]
                    for j in range(i + 1, n_compat):
                        u2 = compat_list[j]
                        if incompat[u1, u2] == 0:
                            selected.remove(v)
                            selected.add(int(u1))
                            selected.add(int(u2))
                            improved = True
                            found = True
                            break
                    if found:
                        break
                
                if improved:
                    break
        
        return selected
    
    rng = np.random.default_rng(42)
    best_iset = set()
    best_size = 0
    
    n_restarts = 200
    for r in range(n_restarts):
        if time.time() > deadline - 35.0:
            break
        
        if r == 0:
            iset = greedy_mis(set(range(N_sup)), rng)
        elif r < 50:
            iset = greedy_mis(set(range(N_sup)), rng)
            iset = local_search_add(iset)
            iset = local_search_1_2_swap(iset, rng)
        elif r < 100:
            avail = set(range(N_sup))
            n_remove = rng.integers(5, 60)
            to_remove = set(rng.choice(N_sup, size=min(int(n_remove), N_sup), replace=False).tolist())
            avail -= to_remove
            iset = greedy_mis(avail, rng)
            iset = local_search_add(iset)
            iset = local_search_1_2_swap(iset, rng)
        else:
            avail = set(range(N_sup))
            n_remove = rng.integers(30, 120)
            to_remove = set(rng.choice(N_sup, size=min(int(n_remove), N_sup), replace=False).tolist())
            avail -= to_remove
            iset = greedy_mis(avail, rng)
            iset = local_search_add(iset)
            iset = local_search_1_2_swap(iset, rng)
        
        size = len(iset)
        if size > best_size:
            best_size = size
            best_iset = set(iset)
    
    # Final polish
    if time.time() < deadline - 10.0:
        best_iset = local_search_add(best_iset)
        best_iset = local_search_1_2_swap(best_iset, rng)
        best_iset = local_search_add(best_iset)
    
    vectors = []
    for i in best_iset:
        sup = supports[i]
        for signs in product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int8)
            for idx, s in zip(sup, signs):
                v[idx] = s
            vectors.append(v)
    
    for i in range(D):
        for sign in [1, -1]:
            v = np.zeros(D, dtype=np.int8)
            v[i] = 2 * sign
            vectors.append(v)
    
    if not vectors:
        vectors = []
        for i in range(8):
            for j in range(i + 1, 8):
                for si in [1, -1]:
                    for sj in [1, -1]:
                        v = np.zeros(11, dtype=np.int8)
                        v[i] = 2 * si
                        v[j] = 2 * sj
                        vectors.append(v)
        for signs in product([1, -1], repeat=8):
            if sum(1 for s in signs if s == -1) % 2 == 0:
                v = np.zeros(11, dtype=np.int8)
                for i, s in enumerate(signs):
                    v[i] = s
                vectors.append(v)
        for zero_pos in range(3):
            for i in range(3):
                for j in range(3):
                    if i == j or i == zero_pos or j == zero_pos:
                        continue
                    for si in [1, -1]:
                        for sj in [1, -1]:
                            v = np.zeros(11, dtype=np.int8)
                            v[8 + i] = si
                            v[8 + j] = sj
                            vectors.append(v)
    
    return np.array(vectors, dtype=np.int8)
