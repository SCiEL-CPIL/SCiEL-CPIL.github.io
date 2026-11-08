import time
import random
import itertools
from collections import defaultdict

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the exact tensor decomposition for multiplying a 3x5 matrix by a 5x7 matrix.
    Uses a Branch and Bound solver to find the maximum packing of disjoint 2x2x2 Strassen blocks.
    This minimizes the rank by maximizing the number of Strassen blocks used.
    """
    start_time = time.time()
    deadline = start_time + 110.0

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
                            indices = set()
                            for i in (i1, i2):
                                for j in (j1, j2):
                                    for k in (k1, k2):
                                        indices.add((i, j, k))
                            all_blocks.append(((i1, i2, j1, j2, k1, k2), indices))
    
    num_blocks_total = len(all_blocks)
    
    # Precompute conflict matrix
    conflict = [[False] * num_blocks_total for _ in range(num_blocks_total)]
    for i in range(num_blocks_total):
        for j in range(i + 1, num_blocks_total):
            if all_blocks[i][1] & all_blocks[j][1]:
                conflict[i][j] = True
                conflict[j][i] = True

    # Branch and Bound for Maximum Set Packing
    # State: (current_index, current_count, used_indices_set, selected_blocks_list)
    # Optimization: Sort blocks by "popularity" (number of conflicts) descending to prune earlier?
    # Or just use index order. Given small size, index order is fine.
    
    best_count = 0
    best_packing = []
    
    # Heuristic: Start with a greedy solution to get a good lower bound
    random.shuffle(all_blocks)
    used_indices = set()
    current_packing = []
    for block_params, block_indices in all_blocks:
        if not (block_indices & used_indices):
            current_packing.append(block_params)
            used_indices |= block_indices
    best_count = len(current_packing)
    best_packing = current_packing[:]

    # B&B Search
    # We can use a recursive function with pruning.
    # To avoid recursion depth issues or overhead, we can use a stack, but for 48 items, recursion is fine.
    # However, Python recursion limit might be an issue if depth is 48? No, limit is usually 1000.
    
    # Precompute index lists for faster access in B&B
    block_indices_list = [b[1] for b in all_blocks]
    block_params_list = [b[0] for b in all_blocks]
    
    # Sort blocks to improve pruning? 
    # Blocks that conflict with many others are "harder" to place, maybe put them first?
    # Or blocks with fewer conflicts?
    # Let's stick to original order or sort by number of conflicts descending.
    conflict_counts = [sum(conflict[i]) for i in range(num_blocks_total)]
    sorted_indices = sorted(range(num_blocks_total), key=lambda i: conflict_counts[i], reverse=True)
    
    # Re-map blocks to sorted order for B&B
    sorted_block_params = [block_params_list[i] for i in sorted_indices]
    sorted_block_indices = [block_indices_list[i] for i in sorted_indices]
    sorted_conflict = [[conflict[sorted_indices[i]][sorted_indices[j]] for j in range(num_blocks_total)] for i in range(num_blocks_total)]

    def solve_bnb(idx, count, used_indices, selected):
        nonlocal best_count, best_packing
        
        if time.time() > deadline - 5.0:
            return

        # Pruning: if we can't beat best_count, return
        # Upper bound: count + (num_blocks_total - idx)
        if count + (num_blocks_total - idx) <= best_count:
            return

        if idx == num_blocks_total:
            if count > best_count:
                best_count = count
                best_packing = selected[:]
            return

        # Option 1: Exclude block idx
        solve_bnb(idx + 1, count, used_indices, selected)

        # Option 2: Include block idx (if possible)
        current_indices = sorted_block_indices[idx]
        if not (current_indices & used_indices):
            new_used = used_indices | current_indices
            new_selected = selected + [sorted_block_params[idx]]
            solve_bnb(idx + 1, count + 1, new_used, new_selected)

    solve_bnb(0, 0, set(), [])

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
