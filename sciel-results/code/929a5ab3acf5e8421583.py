import time
import random

DIMENSIONS = [3, 5, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS

    U_rows = n * m
    V_rows = m * p
    W_rows = p * n

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
                            all_blocks.append(((i1, i2, j1, j2, k1, k2), frozenset(indices)))

    num_blocks = len(all_blocks)
    
    block_conflicts = [set() for _ in range(num_blocks)]
    for i in range(num_blocks):
        for j in range(i + 1, num_blocks):
            if all_blocks[i][1] & all_blocks[j][1]:
                block_conflicts[i].add(j)
                block_conflicts[j].add(i)

    best_num_blocks = 0
    best_packing = []
    
    random.seed(42)
    
    # Improved Search: Multiple random restarts + Local Search (1-1, 2-1, 1-2 swaps)
    num_trials = 2000
    for t in range(num_trials):
        if time.time() > deadline - 30.0:
            break
            
        indices = list(range(num_blocks))
        random.shuffle(indices)
        
        used_indices = set()
        current_packing = []
        
        for idx in indices:
            block_params, block_set = all_blocks[idx]
            if not (block_set & used_indices):
                current_packing.append(idx)
                used_indices |= block_set
                
        # Local Search
        improved = True
        while improved and time.time() < deadline - 30.0:
            improved = False
            current_set = set(current_packing)
            
            # Try 1->2 swap (remove 1, add 2)
            for remove_idx in current_packing:
                if time.time() > deadline - 30.0: break
                trial_packing = [x for x in current_packing if x != remove_idx]
                trial_set = set(trial_packing)
                
                candidates = []
                for add_idx in range(num_blocks):
                    if add_idx in trial_set or add_idx == remove_idx:
                        continue
                    ok = True
                    for p_idx in trial_packing:
                        if add_idx in block_conflicts[p_idx]:
                            ok = False
                            break
                    if ok:
                        candidates.append(add_idx)
                
                # Find 2 non-conflicting candidates
                found_pair = False
                for i_c in range(len(candidates)):
                    if found_pair: break
                    c1 = candidates[i_c]
                    for j_c in range(i_c + 1, len(candidates)):
                        c2 = candidates[j_c]
                        if c1 not in block_conflicts[c2]:
                            new_packing = trial_packing + [c1, c2]
                            current_packing = new_packing
                            improved = True
                            found_pair = True
                            break
                if found_pair:
                    break
                    
            if improved:
                continue

            # Try 1->0 (remove 1, add 1) - usually not helpful for size, but helps rearrange?
            # Actually, we want to INCREASE size. 1->2 increases size by 1.
            # If no 1->2 works, try 0->1 (add 1)
            for add_idx in range(num_blocks):
                if add_idx in current_set: continue
                conflict = False
                for p_idx in current_packing:
                    if add_idx in block_conflicts[p_idx]:
                        conflict = True
                        break
                if not conflict:
                    current_packing.append(add_idx)
                    improved = True
                    break
        
        if len(current_packing) > best_num_blocks:
            best_num_blocks = len(current_packing)
            best_packing = current_packing.copy()
            if best_num_blocks >= 10:
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
    for idx in best_packing:
        block_params, block_indices = all_blocks[idx]
        i1, i2, j1, j2, k1, k2 = block_params
        add_strassen_block(i1, i2, j1, j2, k1, k2)
        covered.update(block_indices)

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
