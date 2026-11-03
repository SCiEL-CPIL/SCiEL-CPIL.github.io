"""Matrix multiplication 3x5x7 - rank-minimal exact tensor decomposition search."""
import time
import random

DIMENSIONS = [3, 5, 7]

def construct():
    """
    Constructs the exact tensor decomposition for multiplying a 3x5 matrix by a 5x7 matrix.
    Uses an optimized search to find a maximum packing of disjoint 2x2x2 Strassen blocks.
    Each Strassen block reduces the rank by 1 compared to schoolbook (7 vs 8 terms).
    Total rank = 7 * (num_blocks) + (105 - 8 * num_blocks) = 105 - num_blocks.
    """
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
                            indices = frozenset()
                            for i in (i1, i2):
                                for j in (j1, j2):
                                    for k in (k1, k2):
                                        indices = indices | frozenset([(i, j, k)])
                            all_blocks.append(((i1, i2, j1, j2, k1, k2), indices))

    # Precompute pairwise conflicts for faster checking
    # block_conflicts[i] = set of block indices that conflict with block i
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
    
    # Phase 1: Randomized greedy with multiple orderings
    num_trials = 2000
    for t in range(num_trials):
        if time.time() > deadline - 20.0:
            break
            
        random.shuffle(all_blocks)
        used_indices = set()
        current_packing = []
        
        for block_params, block_indices in all_blocks:
            if not (block_indices & used_indices):
                current_packing.append(block_params)
                used_indices |= block_indices
                
        if len(current_packing) > best_num_blocks:
            best_num_blocks = len(current_packing)
            best_packing = current_packing.copy()
            if best_num_blocks >= 10:
                break

    # Phase 2: Local search - try swapping blocks to improve packing
    if time.time() < deadline - 15.0:
        current_packing = best_packing.copy()
        current_used = set()
        for bp in current_packing:
            i1, i2, j1, j2, k1, k2 = bp
            for i in (i1, i2):
                for j in (j1, j2):
                    for k in (k1, k2):
                        current_used.add((i, j, k))
        
        improved = True
        max_local_iters = 50
        local_iter = 0
        while improved and local_iter < max_local_iters and time.time() < deadline - 10.0:
            improved = False
            local_iter += 1
            # Try adding a block not in current packing
            current_set = set(current_packing)
            for block_params, block_indices in all_blocks:
                if block_params in current_set:
                    continue
                if not (block_indices & current_used):
                    # Can add this block
                    current_packing.append(block_params)
                    current_used |= block_indices
                    improved = True
                    break
            if not improved:
                # Try removing one block and adding two others
                for idx_remove in range(len(current_packing)):
                    if time.time() > deadline - 10.0:
                        break
                    bp_remove = current_packing[idx_remove]
                    i1r, i2r, j1r, j2r, k1r, k2r = bp_remove
                    removed_indices = set()
                    for i in (i1r, i2r):
                        for j in (j1r, j2r):
                            for k in (k1r, k2r):
                                removed_indices.add((i, j, k))
                    
                    trial_used = current_used - removed_indices
                    trial_packing = current_packing[:idx_remove] + current_packing[idx_remove+1:]
                    
                    added = 0
                    for block_params, block_indices in all_blocks:
                        if block_params in set(trial_packing):
                            continue
                        if not (block_indices & trial_used):
                            trial_packing.append(block_params)
                            trial_used |= block_indices
                            added += 1
                            if added >= 2:
                                break
                    
                    if len(trial_packing) > len(current_packing):
                        current_packing = trial_packing
                        current_used = trial_used
                        improved = True
                        break
        
        if len(current_packing) > best_num_blocks:
            best_num_blocks = len(current_packing)
            best_packing = current_packing

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
