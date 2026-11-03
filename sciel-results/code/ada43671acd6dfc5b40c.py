import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 5, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS  # 3, 5, 7
    
    # Generate all 2x2x2 blocks
    all_blocks = []
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k0 in range(p - 1):
                i1, j1, k1 = i0 + 1, j0 + 1, k0 + 1
                entries = frozenset((i, j, k) for i in (i0, i1) for j in (j0, j1) for k in (k0, k1))
                all_blocks.append((i0, i1, j0, j1, k0, k1, entries))
    
    num_blocks = len(all_blocks)
    
    # Build conflict graph using bitmasks for speed
    conflict_mask = [0] * num_blocks
    for i in range(num_blocks):
        for j in range(i + 1, num_blocks):
            if all_blocks[i][6] & all_blocks[j][6]:
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)
    
    best_set = []
    best_size = 0
    time_check_counter = [0]
    
    def get_bit_count(x):
        return x.bit_count() if hasattr(int, 'bit_count') else bin(x).count('1')

    def backtrack(available_mask, current_set):
        nonlocal best_set, best_size
        time_check_counter[0] += 1
        if time_check_counter[0] % 50 == 0 and time.time() > deadline:
            return
        
        avail_count = get_bit_count(available_mask)
        if len(current_set) + avail_count <= best_size:
            return
        
        if available_mask == 0:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = list(current_set)
            return
        
        # Choose the most constrained available block
        max_conf = -1
        best_idx = -1
        mask = available_mask
        while mask:
            lsb = mask & (-mask)
            idx = lsb.bit_length() - 1
            mask ^= lsb
            conf_in_avail = get_bit_count(conflict_mask[idx] & available_mask) - 1
            if conf_in_avail > max_conf:
                max_conf = conf_in_avail
                best_idx = idx
        
        if best_idx < 0:
            return
        
        remaining_mask = available_mask & ~(1 << best_idx)
        
        # Option 1: include block best_idx
        current_set.append(best_idx)
        new_available = remaining_mask & ~conflict_mask[best_idx]
        backtrack(new_available, current_set)
        current_set.pop()
        
        # Option 2: exclude block best_idx
        backtrack(remaining_mask, current_set)
        
    backtrack((1 << num_blocks) - 1, [])
    
    selected_indices = best_set
    covered = set()
    U_cols = []
    V_cols = []
    W_cols = []
    
    if not selected_indices:
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_cols.append({i * m + j: 1})
                    V_cols.append({j * p + k: 1})
                    W_cols.append({k * n + i: 1})
        rank = len(U_cols)
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        for c in range(rank):
            for idx, val in U_cols[c].items():
                U[idx, c] = val
            for idx, val in V_cols[c].items():
                V[idx, c] = val
            for idx, val in W_cols[c].items():
                W[idx, c] = val
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    for idx in selected_indices:
        i0, i1, j0, j1, k0, k1, _ = all_blocks[idx]
        for i in (i0, i1):
            for j in (j0, j1):
                for k in (k0, k1):
                    covered.add((i, j, k))
        
        a00 = i0 * m + j0
        a01 = i0 * m + j1
        a10 = i1 * m + j0
        a11 = i1 * m + j1
        
        b00 = j0 * p + k0
        b01 = j0 * p + k1
        b10 = j1 * p + k0
        b11 = j1 * p + k1
        
        c00 = k0 * n + i0
        c01 = k1 * n + i0
        c10 = k0 * n + i1
        c11 = k1 * n + i1
        
        U_cols.append({a00: 1, a11: 1})
        V_cols.append({b00: 1, b11: 1})
        W_cols.append({c00: 1, c11: 1})
        
        U_cols.append({a10: 1, a11: 1})
        V_cols.append({b00: 1})
        W_cols.append({c10: 1, c11: -1})
        
        U_cols.append({a00: 1})
        V_cols.append({b01: 1, b11: -1})
        W_cols.append({c01: 1, c11: 1})
        
        U_cols.append({a11: 1})
        V_cols.append({b10: 1, b00: -1})
        W_cols.append({c00: 1, c10: 1})
        
        U_cols.append({a00: 1, a01: 1})
        V_cols.append({b11: 1})
        W_cols.append({c00: -1, c01: 1})
        
        U_cols.append({a10: 1, a00: -1})
        V_cols.append({b00: 1, b01: 1})
        W_cols.append({c11: 1})
        
        U_cols.append({a01: 1, a11: -1})
        V_cols.append({b10: 1, b11: 1})
        W_cols.append({c00: 1})
        
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) in covered:
                    continue
                U_cols.append({i * m + j: 1})
                V_cols.append({j * p + k: 1})
                W_cols.append({k * n + i: 1})
                
    rank = len(U_cols)
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for c in range(rank):
        for idx, val in U_cols[c].items():
            U[idx, c] = val
        for idx, val in V_cols[c].items():
            V[idx, c] = val
        for idx, val in W_cols[c].items():
            W[idx, c] = val
            
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
                
    T_comp = np.einsum('ar,br,cr->abc', U, V, W)
    error = np.sum((T_comp - T) ** 2)
    
    if error > 1e-9:
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
                    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
