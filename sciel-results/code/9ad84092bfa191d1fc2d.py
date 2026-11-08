import time
import random
import itertools

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the exact tensor decomposition for multiplying a 3x5 matrix by a 5x7 matrix.
    Uses bitmask-accelerated greedy set-packing with local search to maximize disjoint 2x2x2 Strassen blocks.
    """
    start_time = time.time()
    deadline = start_time + 115.0

    n, m, p = DIMENSIONS  # 3, 5, 7
    U_rows = n * m  # 15
    V_rows = m * p  # 35
    W_rows = p * n  # 21
    total_cells = n * m * p  # 105

    # Generate all possible 2x2x2 Strassen blocks with bitmask representation
    all_blocks = []
    for i1 in range(n - 1):
        for i2 in range(i1 + 1, n):
            for j1 in range(m - 1):
                for j2 in range(j1 + 1, m):
                    for k1 in range(p - 1):
                        for k2 in range(k1 + 1, p):
                            mask = 0
                            for i in (i1, i2):
                                for j in (j1, j2):
                                    for k in (k1, k2):
                                        cell_idx = (i * m + j) * p + k
                                        mask |= (1 << cell_idx)
                            all_blocks.append(((i1, i2, j1, j2, k1, k2), mask))

    num_blocks_total = len(all_blocks)
    
    # Precompute conflict masks: for each block, which other blocks conflict with it
    # conflict_mask[i] is a bitmask over blocks (not cells) indicating which blocks conflict with block i
    # Actually, for greedy we just need cell-level conflict checking via masks
    
    best_num_blocks = 0
    best_packing = []
    
    # Phase 1: Random restart greedy
    num_trials = 3000
    for t in range(num_trials):
        if time.time() > deadline - 15.0:
            break
            
        order = list(range(num_blocks_total))
        random.shuffle(order)
        used_mask = 0
        current_packing = []
        
        for idx in order:
            params, mask = all_blocks[idx]
            if (used_mask & mask) == 0:
                current_packing.append(params)
                used_mask |= mask
        
        if len(current_packing) > best_num_blocks:
            best_num_blocks = len(current_packing)
            best_packing = current_packing[:]
            
            if best_num_blocks >= 10:  # Theoretical max for 3x5x7 is likely around 9-10
                break

    # Phase 2: Local search - try to swap one block for two
    if best_num_blocks < 10 and time.time() < deadline - 10.0:
        current_packing = best_packing[:]
        used_mask = 0
        for b in current_packing:
            i1, i2, j1, j2, k1, k2 = b
            for i in (i1, i2):
                for j in (j1, j2):
                    for k in (k1, k2):
                        cell_idx = (i * m + j) * p + k
                        used_mask |= (1 << cell_idx)
        
        # Build a set of currently used block params for quick lookup
        used_params = set(current_packing)
        
        improved = True
        max_local_iters = 50
        for iter_num in range(max_local_iters):
            if time.time() > deadline - 8.0:
                break
            improved = False
            
            # Try removing each block and see if we can add two non-conflicting blocks
            for idx_remove in range(len(current_packing)):
                if time.time() > deadline - 8.0:
                    break
                
                removed_block = current_packing[idx_remove]
                # Compute removed mask
                i1, i2, j1, j2, k1, k2 = removed_block
                removed_mask = 0
                for i in (i1, i2):
                    for j in (j1, j2):
                        for k in (k1, k2):
                            cell_idx = (i * m + j) * p + k
                            removed_mask |= (1 << cell_idx)
                
                temp_used = used_mask & ~removed_mask
                temp_packing = current_packing[:idx_remove] + current_packing[idx_remove+1:]
                temp_params = set(temp_packing)
                
                # Find candidate blocks that fit in the freed space
                candidates = []
                for idx in range(num_blocks_total):
                    if all_blocks[idx][0] in temp_params:
                        continue
                    if (all_blocks[idx][1] & temp_used) == 0:
                        candidates.append(idx)
                
                random.shuffle(candidates)
                
                # Try to find two disjoint candidates
                for c1 in candidates[:30]:
                    if time.time() > deadline - 8.0:
                        break
                    p1, m1 = all_blocks[c1]
                    new_used = temp_used | m1
                    for c2 in candidates[:30]:
                        if c2 == c1:
                            continue
                        p2, m2 = all_blocks[c2]
                        if (m2 & new_used) == 0:
                            # Found a valid swap: remove 1, add 2
                            new_packing = temp_packing + [p1, p2]
                            new_used_mask = new_used | m2
                            current_packing = new_packing
                            used_mask = new_used_mask
                            improved = True
                            if len(current_packing) > best_num_blocks:
                                best_num_blocks = len(current_packing)
                                best_packing = current_packing[:]
                            break
                    if improved:
                        break
                if improved:
                    break
            
            if not improved:
                break

    # Construct U, V, W based on the best packing found
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
        
        # M1 = (a11+a22)*(b11+b22)
        add_term({a11: 1, a22: 1}, {b11: 1, b22: 1}, {c11: 1, c22: 1})
        
        # M2 = (a21+a22)*b11
        add_term({a21: 1, a22: 1}, {b11: 1}, {c21: 1, c22: -1})
        
        # M3 = a11*(b12-b22)
        add_term({a11: 1}, {b12: 1, b22: -1}, {c12: 1, c22: 1})
        
        # M4 = a22*(b21-b11)
        add_term({a22: 1}, {b21: 1, b11: -1}, {c11: 1, c21: 1})
        
        # M5 = (a11+a12)*b22
        add_term({a11: 1, a12: 1}, {b22: 1}, {c11: -1, c12: 1})
        
        # M6 = (a21-a11)*(b11+b12)
        add_term({a21: 1, a11: -1}, {b11: 1, b12: 1}, {c22: 1})
        
        # M7 = (a12-a22)*(b21+b22)
        add_term({a12: 1, a22: -1}, {b21: 1, b22: 1}, {c11: 1})

    # Add the Strassen blocks from the best packing
    covered = set()
    for block_params in best_packing:
        i1, i2, j1, j2, k1, k2 = block_params
        add_strassen_block(i1, i2, j1, j2, k1, k2)
        for i in (i1, i2):
            for j in (j1, j2):
                for k in (k1, k2):
                    covered.add((i, j, k))

    # Add schoolbook terms for uncovered (i,j,k) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() > deadline:
                    break
                if (i, j, k) not in covered:
                    add_term({i * m + j: 1}, {j * p + k: 1}, {k * n + i: 1})

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
# EVOLVE-BLOCK-END
