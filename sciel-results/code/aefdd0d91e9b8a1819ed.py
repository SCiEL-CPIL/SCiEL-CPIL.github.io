import time
import numpy as np
import random

DIMENSIONS = [3, 5, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    U_rows = n * m
    V_rows = m * p
    W_rows = p * n

    # Generate all blocks
    all_blocks = []
    for i1 in range(n):
        for i2 in range(i1 + 1, n):
            for j1 in range(m):
                for j2 in range(j1 + 1, m):
                    for k1 in range(p):
                        for k2 in range(k1 + 1, p):
                            indices = frozenset((i, j, k) for i in (i1, i2) for j in (j1, j2) for k in (k1, k2))
                            all_blocks.append(((i1, i2, j1, j2, k1, k2), indices))
                            
    # Sort blocks to help pruning? 
    # Blocks that use "central" indices might conflict more.
    # Let's just use the generated order.
    
    best_packing = []
    
    # 1. Greedy initialization
    rng = random.Random(42)
    for _ in range(500):
        if time.time() > deadline - 20.0:
            break
        order = list(range(len(all_blocks)))
        rng.shuffle(order)
        used = frozenset()
        current = []
        for idx in order:
            params, indices = all_blocks[idx]
            if not (indices & used):
                current.append(params)
                used |= indices
        if len(current) > len(best_packing):
            best_packing = current
            if len(best_packing) >= 9:
                break

    # 2. Backtracking to find 9
    def backtrack(idx, used, current):
        nonlocal best_packing
        if time.time() > deadline - 10.0:
            return
        if len(current) > len(best_packing):
            best_packing = list(current)
            if len(best_packing) >= 9:
                return
        if idx == len(all_blocks):
            return
        if len(current) + (len(all_blocks) - idx) <= len(best_packing):
            return
        
        # Skip
        backtrack(idx + 1, used, current)
        
        # Include
        params, indices = all_blocks[idx]
        if not (indices & used):
            current.append(params)
            backtrack(idx + 1, used | indices, current)
            current.pop()

    if len(best_packing) < 9:
        backtrack(0, frozenset(), [])

    # Construct U, V, W
    U_cols = []
    V_cols = []
    W_cols = []

    def add_term(u_entries, v_entries, w_entries):
        u = [0] * U_rows
        v = [0] * V_rows
        w = [0] * W_rows
        for idx, val in u_entries.items():
            u[idx] = val
        for idx, val in v_entries.items():
            v[idx] = val
        for idx, val in w_entries.items():
            w[idx] = val
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)

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
                if time.time() > deadline - 5.0:
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
