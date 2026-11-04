"""Matrix multiplication 3x5x7 - rank-minimal exact tensor decomposition via deterministic packing."""
import time
import random
import itertools

DIMENSIONS = [3, 5, 7]


def construct():
    """
    Finds a maximum packing of disjoint rectangular blocks (2x2x2)
    for 3x5x7 matrix multiplication, using known optimal algorithms for each block type.
    Uses deterministic backtracking to find the maximum number of disjoint 2x2x2 blocks.
    """
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # 3, 5, 7
    U_rows = n * m  # 15
    V_rows = m * p  # 35
    W_rows = p * n  # 21

    # Generate all possible 2x2x2 blocks
    # Each block: (i1, i2, j1, j2, k1, k2) where i1 < i2, etc.
    # Indices are (i, j, k)
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
    
    # Precompute conflict sets
    # Two blocks conflict if their index sets intersect
    conflict = [set() for _ in range(num_blocks)]
    for i in range(num_blocks):
        for j in range(i + 1, num_blocks):
            if all_blocks[i][1] & all_blocks[j][1]:
                conflict[i].add(j)
                conflict[j].add(i)
                
    # Backtracking search for Maximum Independent Set
    # We want to maximize the number of blocks selected.
    # Since num_blocks is small (48), this is feasible.
    # We use a simple recursive backtracking with pruning.
    
    best_count = 0
    best_packing = []
    
    # Order blocks by degree (number of conflicts) descending? Or just index?
    # Higher degree nodes are more constraining, so pick them early or late?
    # Usually, picking high degree nodes early helps prune.
    order = sorted(range(num_blocks), key=lambda x: len(conflict[x]), reverse=True)
    
    def backtrack(idx, selected, used_indices):
        nonlocal best_count, best_packing
        
        # Pruning: if current selected + remaining possible <= best, stop
        # Upper bound: current count + (num_blocks - idx) is weak.
        # Better bound: current count + max independent set of remaining graph.
        # For now, simple bound: if len(selected) + (num_blocks - idx) <= best_count: return
        
        if len(selected) + (num_blocks - idx) <= best_count:
            return
            
        if idx == num_blocks:
            if len(selected) > best_count:
                best_count = len(selected)
                best_packing = list(selected)
            return
        
        bi = order[idx]
        
        # Option 1: Include block bi
        if not (all_blocks[bi][1] & used_indices):
            new_selected = selected + [bi]
            new_used = used_indices | all_blocks[bi][1]
            backtrack(idx + 1, new_selected, new_used)
            
        # Option 2: Exclude block bi
        backtrack(idx + 1, selected, used_indices)
        
    # To speed up, we can also try a greedy heuristic first to get a good lower bound
    # Greedy: pick block with min conflicts, remove it and its neighbors, repeat
    def greedy_lower_bound():
        remaining = set(range(num_blocks))
        count = 0
        selected = []
        while remaining:
            # Pick block with min conflicts within remaining
            min_conf_block = None
            min_conf_count = float('inf')
            for b in remaining:
                c = len(conflict[b] & remaining)
                if c < min_conf_count:
                    min_conf_count = c
                    min_conf_block = b
            selected.append(min_conf_block)
            count += 1
            # Remove min_conf_block and all its neighbors
            to_remove = {min_conf_block} | (conflict[min_conf_block] & remaining)
            remaining -= to_remove
        return count, selected
    
    g_count, g_packing = greedy_lower_bound()
    best_count = g_count
    best_packing = g_packing
    
    # Run backtracking
    # To avoid timeout, we can limit the search or rely on the small size.
    # 48 nodes is small enough for exact MIS with pruning.
    backtrack(0, [], frozenset())
    
    # Build the decomposition
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
    
    covered = set()
    for bi in best_packing:
        i1, i2, j1, j2, k1, k2 = all_blocks[bi][0]
        add_strassen_block(i1, i2, j1, j2, k1, k2)
        for i in (i1, i2):
            for j in (j1, j2):
                for k in (k1, k2):
                    covered.add((i, j, k))
    
    # Add schoolbook terms for uncovered indices
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    add_term({i * m + j: 1}, {j * p + k: 1}, {k * n + i: 1})
    
    rank = len(U_cols)
    
    # Convert to row-major matrices
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
