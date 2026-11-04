import time
import random
import numpy as np

DIMENSIONS = [3, 5, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS  # 3, 5, 7

    U_rows = n * m  # 15
    V_rows = m * p  # 35
    W_rows = p * n  # 21

    # Generate all possible 2x2x2 blocks
    all_blocks = []
    for i1 in range(n - 1):
        for i2 in range(i1 + 1, n):
            for j1 in range(m - 1):
                for j2 in range(j1 + 1, m):
                    for k1 in range(p - 1):
                        for k2 in range(k1 + 1, p):
                            indices = set()
                            for i in (i1, i2):
                                for j in (j1, j2):
                                    for k in (k1, k2):
                                        indices.add((i, j, k))
                            all_blocks.append(((i1, i2, j1, j2, k1, k2), indices))

    def local_refine(packing, used_indices):
        packing = list(packing)
        used_indices = set(used_indices)
        improved = True
        while improved:
            improved = False
            for idx, block in enumerate(packing):
                if time.time() > deadline - 5.0:
                    return packing, used_indices
                block_indices = None
                for bp, bi in all_blocks:
                    if bp == block:
                        block_indices = bi
                        break
                if block_indices is None:
                    continue
                new_used = used_indices - block_indices
                for other_block, other_indices in all_blocks:
                    if other_block in packing:
                        continue
                    if not (other_indices & new_used):
                        new_packing = packing[:idx] + packing[idx+1:] + [other_block]
                        new_used2 = new_used | other_indices
                        for bp3, bi3 in all_blocks:
                            if bp3 in new_packing:
                                continue
                            if not (bi3 & new_used2):
                                if len(new_packing) + 1 > len(packing):
                                    packing = new_packing + [bp3]
                                    used_indices = new_used2 | bi3
                                    improved = True
                                    break
                        if improved:
                            break
                if improved:
                    break
            if not improved:
                for bp, bi in all_blocks:
                    if bp in packing:
                        continue
                    if not (bi & used_indices):
                        packing.append(bp)
                        used_indices |= bi
                        improved = True
                        break
        return packing, used_indices

    best_num_blocks = 0
    best_packing = []
    
    num_trials = 200
    for t in range(num_trials):
        if time.time() > deadline - 5.0:
            break
            
        random.shuffle(all_blocks)
        used_indices = set()
        current_packing = []
        
        for block_params, block_indices in all_blocks:
            if not (block_indices & used_indices):
                current_packing.append(block_params)
                used_indices |= block_indices
        
        current_packing, used_indices = local_refine(current_packing, used_indices)
        
        if len(current_packing) > best_num_blocks:
            best_num_blocks = len(current_packing)
            best_packing = current_packing
            
            if best_num_blocks == 13:
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
    for block_params in best_packing:
        i1, i2, j1, j2, k1, k2 = block_params
        add_strassen_block(i1, i2, j1, j2, k1, k2)
        for i in (i1, i2):
            for j in (j1, j2):
                for k in (k1, k2):
                    covered.add((i, j, k))

    # Uncovered indices
    uncovered = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    uncovered.add((i, j, k))

    # Decompose uncovered set into rectangles
    while uncovered:
        if time.time() > deadline:
            break
        # Pick an index
        i, j, k = next(iter(uncovered))
        
        # Find maximal rectangle
        I = {i}
        J = {j}
        K = {k}
        
        changed = True
        while changed:
            changed = False
            # Expand I
            for i_new in range(n):
                if i_new in I:
                    continue
                new_I = I | {i_new}
                valid = True
                for i_test in new_I:
                    for j_test in J:
                        for k_test in K:
                            if (i_test, j_test, k_test) not in uncovered:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                if valid:
                    I = new_I
                    changed = True
            # Expand J
            for j_new in range(m):
                if j_new in J:
                    continue
                new_J = J | {j_new}
                valid = True
                for i_test in I:
                    for j_test in new_J:
                        for k_test in K:
                            if (i_test, j_test, k_test) not in uncovered:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                if valid:
                    J = new_J
                    changed = True
            # Expand K
            for k_new in range(p):
                if k_new in K:
                    continue
                new_K = K | {k_new}
                valid = True
                for i_test in I:
                    for j_test in J:
                        for k_test in new_K:
                            if (i_test, j_test, k_test) not in uncovered:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                if valid:
                    K = new_K
                    changed = True
        
        # Add term for this rectangle
        u_entries = {i * m + j: 1 for i in I for j in J}
        v_entries = {j * p + k: 1 for j in J for k in K}
        w_entries = {k * n + i: 1 for k in K for i in I}
        
        # Wait, the term is u \otimes v \otimes w.
        # u[i*m+j] = 1 for i in I, j in J.
        # v[j*p+k] = 1 for j in J, k in K.
        # w[k*n+i] = 1 for k in K, i in I.
        # This product is 1 for (i,j,k) in I x J x K.
        
        add_term(u_entries, v_entries, w_entries)
        
        # Remove rectangle from uncovered
        for i_test in I:
            for j_test in J:
                for k_test in K:
                    uncovered.discard((i_test, j_test, k_test))

    rank = len(U_cols)
    
    # Convert to matrix format
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
