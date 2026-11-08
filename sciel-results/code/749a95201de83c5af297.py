import time
import random
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = DIMENSIONS  # 3, 5, 7
    U_rows = n * m  # 15
    V_rows = m * p  # 35
    W_rows = p * n  # 21

    # Generate all possible 2x2x2 blocks
    # A block is defined by two i's, two j's, two k's
    # Since n=3, i-pairs are (0,1), (0,2), (1,2)
    # m=5, j-pairs are 10 combinations
    # p=7, k-pairs are 21 combinations
    # Total blocks = 3 * 10 * 21 = 630
    
    all_blocks = []
    block_indices = []
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
                            all_blocks.append((i1, i2, j1, j2, k1, k2))
                            block_indices.append(indices)

    num_blocks_total = len(all_blocks)
    
    # Pre-compute conflicts
    # Conflict if (I1 x J1 x K1) intersects (I2 x J2 x K2)
    # Since I1, I2 are subsets of {0,1,2} of size 2, they always intersect.
    # So conflict iff (J1 intersects J2) AND (K1 intersects K2).
    conflict = [[False] * num_blocks_total for _ in range(num_blocks_total)]
    for a in range(num_blocks_total):
        for b in range(a + 1, num_blocks_total):
            if block_indices[a] & block_indices[b]:
                conflict[a][b] = True
                conflict[b][a] = True

    # Backtracking search for maximum independent set
    # We want to find a set of blocks with no conflicts.
    # Target: 13 blocks.
    
    best_packing = []
    best_size = 0
    
    def backtrack(idx, current_packing, used):
        nonlocal best_packing, best_size
        
        # Pruning: if even taking all remaining blocks won't beat best, return
        if len(current_packing) + (num_blocks_total - idx) <= best_size:
            return
            
        if time.time() > deadline - 10.0:
            return

        # Try to add block idx
        if not any(conflict[idx][j] for j in current_packing):
            new_packing = current_packing + [idx]
            # Heuristic: if we hit 13, we can stop immediately as it's theoretical max
            if len(new_packing) == 13:
                best_packing = new_packing
                best_size = 13
                return
            backtrack(idx + 1, new_packing, used)
            
        # Try to skip block idx
        backtrack(idx + 1, current_packing, used)

    # To make backtracking faster, we can sort blocks or use a different strategy.
    # Given the small number (630), and high density of conflicts, simple DFS might be slow if not pruned well.
    # However, finding 13 is likely very fast if they exist.
    # Let's use a randomized greedy first to get a good lower bound, then backtrack.
    
    # Randomized greedy to find a good initial packing
    for _ in range(100):
        if time.time() > deadline - 50.0:
            break
        random.shuffle(all_blocks)
        # We need to map indices in all_blocks to original indices
        # Actually, let's just use the original list and shuffle a copy of indices
        indices_copy = list(range(num_blocks_total))
        random.shuffle(indices_copy)
        
        current_packing = []
        used = set()
        for b_idx in indices_copy:
            if not (block_indices[b_idx] & used):
                current_packing.append(b_idx)
                used |= block_indices[b_idx]
        
        if len(current_packing) > best_size:
            best_size = len(current_packing)
            best_packing = current_packing[:]
            
            if best_size == 13:
                break

    # If we didn't find 13, try backtracking from the best found? 
    # Or just run a limited backtracking search for 13.
    if best_size < 13:
        # Try to find 13 using DFS with a time limit
        # We can use the greedy result as a guide or just start from scratch
        # A simple DFS for 13 blocks:
        def find_13(idx, current_packing):
            nonlocal best_packing, best_size
            if best_size == 13:
                return
            if time.time() > deadline - 20.0:
                return
            if len(current_packing) + (num_blocks_total - idx) < 13:
                return
            if idx == num_blocks_total:
                return
            
            # Try adding
            if not any(conflict[idx][j] for j in current_packing):
                new_packing = current_packing + [idx]
                if len(new_packing) == 13:
                    best_packing = new_packing
                    best_size = 13
                    return
                find_13(idx + 1, new_packing)
                if best_size == 13:
                    return
            # Try skipping
            find_13(idx + 1, current_packing)
            
        # Sort blocks to improve pruning? Blocks with fewer conflicts first?
        # Degree of conflict
        degrees = [sum(conflict[i]) for i in range(num_blocks_total)]
        sorted_indices = sorted(range(num_blocks_total), key=lambda x: degrees[x])
        
        # Map sorted_indices to original indices for the DFS
        # The DFS above uses 'idx' as index into the sorted list? No, it uses 'idx' as index into 'all_blocks'.
        # Let's rewrite the DFS to iterate over sorted_indices.
        
        def find_13_sorted(pos, current_packing):
            nonlocal best_packing, best_size
            if best_size == 13:
                return
            if time.time() > deadline - 20.0:
                return
            if pos == len(sorted_indices):
                return
            if len(current_packing) + (len(sorted_indices) - pos) < 13:
                return
            
            b_idx = sorted_indices[pos]
            
            # Try adding b_idx
            if not any(conflict[b_idx][j] for j in current_packing):
                new_packing = current_packing + [b_idx]
                if len(new_packing) == 13:
                    best_packing = new_packing
                    best_size = 13
                    return
                find_13_sorted(pos + 1, new_packing)
                if best_size == 13:
                    return
            # Try skipping b_idx
            find_13_sorted(pos + 1, current_packing)

        find_13_sorted(0, [])

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
        i1, i2, j1, j2, k1, k2 = all_blocks[b_idx]
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
# EVOLVE-BLOCK-END
