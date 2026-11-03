import time
import random
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the exact tensor decomposition for multiplying a 3x5 matrix by a 5x7 matrix.
    Uses Strassen 2x2x2 blocks with an improved packing search (greedy + local search).
    """
    start_time = time.time()
    deadline = start_time + 145.0

    n, m, p = DIMENSIONS  # 3, 5, 7

    U_rows = n * m  # 15
    V_rows = m * p  # 35
    W_rows = p * n  # 21

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

    # Generate all possible 2x2x2 blocks
    all_blocks = []
    for i1 in range(n - 1):
        for i2 in range(i1 + 1, n):
            for j1 in range(m - 1):
                for j2 in range(j1 + 1, m):
                    for k1 in range(p - 1):
                        for k2 in range(k1 + 1, p):
                            indices = frozenset((i, j, k) for i in (i1, i2) for j in (j1, j2) for k in (k1, k2))
                            all_blocks.append(((i1, i2, j1, j2, k1, k2), indices))
    
    # Precompute block indices as sets for faster intersection checks
    all_blocks_set = [(bp, set(ind)) for bp, ind in all_blocks]
    all_blocks_dict = {bp: set(ind) for bp, ind in all_blocks}

    def greedy_packing(block_list, used_indices=None):
        """Greedy: pick blocks in given order that don't conflict."""
        if used_indices is None:
            used_indices = set()
        packing = []
        for block_params, block_indices in block_list:
            if not (block_indices & used_indices):
                packing.append(block_params)
                used_indices |= block_indices
        return packing

    def local_search_improve(packing, max_iters=100):
        """Try to improve packing by swapping one block for another that allows more total blocks."""
        if not packing:
            return packing
        
        # Rebuild used_indices
        used_indices = set()
        for bp in packing:
            used_indices |= all_blocks_dict[bp]
        
        current_size = len(packing)
        best_packing = list(packing)
        
        # Precompute list of (block_params, set_indices) for iteration
        block_list = all_blocks_set
        
        for _ in range(max_iters):
            if time.time() > deadline - 10.0:
                break
            improved = False
            # Try removing each block and see if we can add more
            for i in range(len(best_packing)):
                if time.time() > deadline - 10.0:
                    break
                # Remove block i
                trial_packing = best_packing[:i] + best_packing[i+1:]
                trial_used = set()
                for bp in trial_packing:
                    trial_used |= all_blocks_dict[bp]
                
                # Try to greedily add blocks
                # We want to add blocks not in trial_packing that don't conflict with trial_used
                # To speed this up, we can just iterate all blocks
                for block_params, block_indices in block_list:
                    if block_params in trial_packing:
                        continue
                    if not (block_indices & trial_used):
                        trial_packing.append(block_params)
                        trial_used |= block_indices
                
                if len(trial_packing) > current_size:
                    best_packing = trial_packing
                    current_size = len(trial_packing)
                    improved = True
                    # Restart the loop with the new packing
                    break
            if not improved:
                break
        
        return best_packing

    # Phase 1: Greedy with random shuffles
    best_num_blocks = 0
    best_packing = []
    
    # Increase trials since the setup is cheap
    num_trials = 5000
    for t in range(num_trials):
        if time.time() > deadline - 30.0:
            break
        
        shuffled = list(all_blocks_set)
        random.shuffle(shuffled)
        packing = greedy_packing(shuffled)
        
        if len(packing) > best_num_blocks:
            best_num_blocks = len(packing)
            best_packing = packing
            if best_num_blocks >= 12:
                break

    # Phase 2: Local search improvement
    if best_num_blocks < 13:
        improved = local_search_improve(best_packing, max_iters=200)
        if len(improved) > best_num_blocks:
            best_num_blocks = len(improved)
            best_packing = improved

    # Construct U, V, W based on the best packing found
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

    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() > deadline:
                    break
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
# EVOLVE-BLOCK-END
