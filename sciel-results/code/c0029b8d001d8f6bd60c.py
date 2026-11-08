import time
import random

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

    # Find max independent set
    # Conflict graph: two blocks conflict if they share any index
    num_blocks = len(all_blocks)
    conflicts = [set() for _ in range(num_blocks)]
    for i in range(num_blocks):
        for j in range(i + 1, num_blocks):
            if all_blocks[i][1] & all_blocks[j][1]:
                conflicts[i].add(j)
                conflicts[j].add(i)

    # Backtracking to find max independent set
    best_set = []
    
    def backtrack(start, current_set, current_size):
        nonlocal best_set
        if current_size > len(best_set):
            best_set = current_set[:]
            
        # Pruning: if remaining nodes + current_size <= best_size, return
        if start + (num_blocks - start) <= len(best_set):
            return
            
        for i in range(start, num_blocks):
            if i not in current_set:
                # Check if i can be added
                if not any(x in current_set for x in conflicts[i]):
                    backtrack(i + 1, current_set + [i], current_size + 1)
                    # Optional: branch where i is NOT included
                    # backtrack(i + 1, current_set, current_size) 
                    # But we want max, so trying to include is good.
                    # However, excluding might lead to better packing later.
                    # Standard MIS backtracking:
                    # 1. Include i
                    # 2. Exclude i
                    # We do include, then recurse.
                    # Then we need to exclude i to explore other possibilities.
                    # But the loop structure `for i in range` handles exclusion naturally?
                    # No, if we include i, we skip it.
                    # If we don't include i, we move to i+1.
                    # The current structure only explores "include i" then "include j > i".
                    # It misses "exclude i, include j".
                    # We need a proper MIS solver.

    # Simpler: Greedy with random restarts is often good enough for small graphs.
    # But let's try a better heuristic.
    # Sort blocks by degree (conflicts) ascending.
    degrees = [len(conflicts[i]) for i in range(num_blocks)]
    order = sorted(range(num_blocks), key=lambda x: degrees[x])
    
    best_packing = []
    for _ in range(1000):
        if time.time() > deadline - 5.0:
            break
        random.shuffle(order)
        used = set()
        current = []
        for i in order:
            if not (all_blocks[i][1] & used):
                current.append(i)
                used |= all_blocks[i][1]
        if len(current) > len(best_packing):
            best_packing = current
            if len(best_packing) == 7: # Theoretical max?
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
        params, _ = all_blocks[idx]
        i1, i2, j1, j2, k1, k2 = params
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
