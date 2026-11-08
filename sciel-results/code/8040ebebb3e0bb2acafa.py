import time
import random
from collections import defaultdict

DIMENSIONS = [3, 5, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS  # 3, 5, 7
    U_rows = n * m  # 15
    V_rows = m * p  # 35
    W_rows = p * n  # 21
    
    # Generate all possible 2x2x2 Strassen blocks
    all_blocks = []
    for i1 in range(n - 1):
        for i2 in range(i1 + 1, n):
            for j1 in range(m - 1):
                for j2 in range(j1 + 1, m):
                    for k1 in range(p - 1):
                        for k2 in range(k1 + 1, p):
                            indices = frozenset()
                            for i in (i1, i2):
                                for j in (j1, j2):
                                    for k in (k1, k2):
                                        indices.add((i, j, k))
                            all_blocks.append(((i1, i2, j1, j2, k1, k2), indices))
    
    num_blocks = len(all_blocks)
    
    # Precompute overlap matrix as bitsets for fast checking
    # Map each (i,j,k) to an integer index
    total_cells = n * m * p  # 105
    cell_to_idx = {}
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                cell_to_idx[(i, j, k)] = idx
                idx += 1
    
    # Represent each block as a bitmask (use Python int as arbitrary-precision bitset)
    block_masks = []
    for _, indices in all_blocks:
        mask = 0
        for cell in indices:
            mask |= (1 << cell_to_idx[cell])
        block_masks.append(mask)
    
    # Greedy with random restarts and local search
    best_packing = []
    best_num = 0
    
    num_trials = 2000
    for t in range(num_trials):
        if time.time() > deadline - 10.0:
            break
        
        random.shuffle(all_blocks)
        used_mask = 0
        current_packing = []
        
        for b_idx, (block_params, block_indices) in enumerate(all_blocks):
            if (block_masks[b_idx] & used_mask) == 0:
                current_packing.append(b_idx)
                used_mask |= block_masks[b_idx]
        
        current_num = len(current_packing)
        
        # Local search: try to swap one block out for one in
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            for pos in range(len(current_packing)):
                if time.time() > deadline - 5.0:
                    break
                # Try removing current_packing[pos] and adding any other block
                removed_mask = block_masks[current_packing[pos]]
                new_mask = used_mask & ~removed_mask
                for other_idx in range(num_blocks):
                    if other_idx not in current_packing and (block_masks[other_idx] & new_mask) == 0:
                        # Can swap
                        current_packing[pos] = other_idx
                        used_mask = new_mask | block_masks[other_idx]
                        improved = True
                        break
                if improved:
                    break
        
        if len(current_packing) > best_num:
            best_num = len(current_packing)
            best_packing = current_packing[:]
            if best_num >= 10:  # Good enough
                break
    
    # Construct U, V, W
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_entries, v_entries, w_entries):
        U_cols.append([0] * U_rows)
        V_cols.append([0] * V_rows)
        W_cols.append([0] * W_rows)
        for idx, val in u_entries.items():
            U_cols[-1][idx] = val
        for idx, val in v_entries.items():
            V_cols[-1][idx] = val
        for idx, val in w_entries.items():
            W_cols[-1][idx] = val
    
    def add_strassen_block(i1, i2, j1, j2, k1, k2):
        a11 = i1 * m + j1
        a12 = i1 * m + j2
        a21 = i2 * m + j1
        a22 = i2 * m + j2
        
        b11 = j1 * p + k1
        b12 = j1 * p + k2
        b21 = j2 * p + k1
        b22 = j2 * p + k2
        
        c11 = k1 * n + i1
        c12 = k2 * n + i1
        c21 = k1 * n + i2
        c22 = k2 * n + i2
        
        add_term({a11: 1, a22: 1}, {b11: 1, b22: 1}, {c11: 1, c22: 1})
        add_term({a21: 1, a22: 1}, {b11: 1}, {c21: 1, c22: -1})
        add_term({a11: 1}, {b12: 1, b22: -1}, {c12: 1, c22: 1})
        add_term({a22: 1}, {b21: 1, b11: -1}, {c11: 1, c21: 1})
        add_term({a11: 1, a12: 1}, {b22: 1}, {c11: -1, c12: 1})
        add_term({a21: 1, a11: -1}, {b11: 1, b12: 1}, {c22: 1})
        add_term({a12: 1, a22: -1}, {b21: 1, b22: 1}, {c11: 1})
    
    covered = set()
    for b_idx in best_packing:
        i1, i2, j1, j2, k1, k2 = all_blocks[b_idx][0]
        add_strassen_block(i1, i2, j1, j2, k1, k2)
        for i in (i1, i2):
            for j in (j1, j2):
                for k in (k1, k2):
                    covered.add((i, j, k))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    add_term({i * m + j: 1}, {j * p + k: 1}, {k * n + i: 1})
    
    rank = len(U_cols)
    
    U = [[0] * rank for _ in range(U_rows)]
    V = [[0] * rank for _ in range(V_rows)]
    W = [[0] * rank for _ in range(W_rows)]
    
    for c in range(rank):
        for r in range(U_rows):
            U[r][c] = U_cols[c][r]
        for r in range(V_rows):
            V[r][c] = V_cols[c][r]
        for r in range(W_rows):
            W[r][c] = W_cols[c][r]
    
    return {"U": U, "V": V, "W": W}
