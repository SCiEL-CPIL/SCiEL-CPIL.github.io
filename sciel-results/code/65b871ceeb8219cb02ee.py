import numpy as np
import time
import random
from functools import lru_cache

DIMENSIONS = [3, 5, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 150s limit

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
    
    # Build conflict graph
    num_blocks = len(all_blocks)
    conflict = [[False] * num_blocks for _ in range(num_blocks)]
    for i in range(num_blocks):
        for j in range(i + 1, num_blocks):
            if all_blocks[i][1] & all_blocks[j][1]:
                conflict[i][j] = True
                conflict[j][i] = True

    # Backtracking search for maximum independent set
    best_set = []
    
    def backtrack(start, current_set, used_indices):
        nonlocal best_set
        if time.time() > deadline - 10.0:
            return
        
        if len(current_set) > len(best_set):
            best_set = current_set[:]
        
        # Pruning: if we can't beat best_set, return
        # Max possible additional blocks is roughly (num_blocks - start) / 2?
        # Simple bound: remaining blocks count
        if len(current_set) + (num_blocks - start) <= len(best_set):
            return

        for i in range(start, num_blocks):
            if conflict[i][start-1] if start > 0 else False:
                # Check conflict with last added
                if current_set and conflict[i][current_set[-1]]:
                    continue
            # Check conflict with all in current_set
            if any(conflict[i][idx] for idx in current_set):
                continue
                
            # Add block i
            current_set.append(i)
            backtrack(i + 1, current_set, used_indices)
            current_set.pop()

    # Start backtracking
    # Optimization: Sort blocks by degree (number of conflicts) to prune faster?
    # For now, simple backtracking. 48 nodes is small.
    backtrack(0, [], set())
    
    best_packing = [all_blocks[i][0] for i in best_set]

    # Build U, V, W columns
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

        # M1 = (a11+a22)*(b11+b22) -> c11 += M1, c22 += M1
        add_term({a11: 1, a22: 1}, {b11: 1, b22: 1}, {c11: 1, c22: 1})

        # M2 = (a21+a22)*b11 -> c21 += M2, c22 -= M2
        add_term({a21: 1, a22: 1}, {b11: 1}, {c21: 1, c22: -1})

        # M3 = a11*(b12-b22) -> c12 += M3, c22 += M3
        add_term({a11: 1}, {b12: 1, b22: -1}, {c12: 1, c22: 1})

        # M4 = a22*(b21-b11) -> c11 += M4, c21 += M4
        add_term({a22: 1}, {b21: 1, b11: -1}, {c11: 1, c21: 1})

        # M5 = (a11+a12)*b22 -> c11 -= M5, c12 += M5
        add_term({a11: 1, a12: 1}, {b22: 1}, {c11: -1, c12: 1})

        # M6 = (a21-a11)*(b11+b12) -> c22 += M6
        add_term({a21: 1, a11: -1}, {b11: 1, b12: 1}, {c22: 1})

        # M7 = (a12-a22)*(b21+b22) -> c11 += M7
        add_term({a12: 1, a22: -1}, {b21: 1, b22: 1}, {c11: 1})

    # Add Strassen blocks from the best packing
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
                if (i, j, k) not in covered:
                    add_term({i * m + j: 1}, {j * p + k: 1}, {k * n + i: 1})

    rank = len(U_cols)

    # Convert to numpy matrices and verify
    U = np.zeros((U_rows, rank), dtype=float)
    V = np.zeros((V_rows, rank), dtype=float)
    W = np.zeros((W_rows, rank), dtype=float)

    for c in range(rank):
        for r in range(U_rows):
            U[r, c] = U_cols[c][r]
        for r in range(V_rows):
            V[r, c] = V_cols[c][r]
        for r in range(W_rows):
            W[r, c] = W_cols[c][r]

    # Verify correctness
    T = np.zeros((U_rows, V_rows, W_rows))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    T_comp = np.einsum('ar,br,cr->abc', U, V, W)
    error = np.sum((T_comp - T) ** 2)

    if error > 1e-9:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((U_rows, rank), dtype=float)
        V = np.zeros((V_rows, rank), dtype=float)
        W = np.zeros((W_rows, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
