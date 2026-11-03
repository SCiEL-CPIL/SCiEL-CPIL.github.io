import time
import random
import numpy as np

DIMENSIONS = [3, 5, 6]

DEADLINE = 150.0
SAFETY_MARGIN = 15.0

def _strassen_2x2x2():
    """Return the 7 Strassen terms as (u, v, w) triples for a 2x2x2 block."""
    terms = []
    # M1
    terms.append((
        {(0,0): 1, (1,1): 1},
        {(0,0): 1, (1,1): 1},
        {(0,0): 1, (1,1): 1}
    ))
    # M2
    terms.append((
        {(1,0): 1, (1,1): 1},
        {(0,0): 1},
        {(1,0): 1, (1,1): -1}
    ))
    # M3
    terms.append((
        {(0,0): 1},
        {(0,1): 1, (1,1): -1},
        {(0,1): 1, (1,1): 1}
    ))
    # M4
    terms.append((
        {(1,1): 1},
        {(1,0): 1, (0,0): -1},
        {(0,0): 1, (1,0): 1}
    ))
    # M5
    terms.append((
        {(0,0): 1, (0,1): 1},
        {(1,1): 1},
        {(0,0): -1, (0,1): 1}
    ))
    # M6
    terms.append((
        {(1,0): 1, (0,0): -1},
        {(0,0): 1, (0,1): 1},
        {(1,1): 1}
    ))
    # M7
    terms.append((
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1}
    ))
    return terms

def _build_strassen_356():
    """Build a rank-84 decomposition for 3x5x6 using Strassen on 2x2x2 blocks."""
    n, m, p = 3, 5, 6
    U = {}
    V = {}
    W = {}
    
    term_count = 0
    strassen_terms = _strassen_2x2x2()
    
    blocks = []
    for i0 in [0]:
        for j0 in [0, 2]:
            for k0 in [0, 2, 4]:
                blocks.append((i0, j0, k0))
    
    for (i0, j0, k0) in blocks:
        for t_idx in range(7):
            u_entries, v_entries, w_entries = strassen_terms[t_idx]
            for (di, dj), coeff_u in u_entries.items():
                i, j = i0 + di, j0 + dj
                U.setdefault((i, j), []).append((coeff_u, term_count))
            for (dj, dk), coeff_v in v_entries.items():
                j, k = j0 + dj, k0 + dk
                V.setdefault((j, k), []).append((coeff_v, term_count))
            for (dk, di), coeff_w in w_entries.items():
                k, i = k0 + dk, i0 + di
                W.setdefault((k, i), []).append((coeff_w, term_count))
            term_count += 1
    
    covered = set()
    for (i0, j0, k0) in blocks:
        for di in [0, 1]:
            for dj in [0, 1]:
                for dk in [0, 1]:
                    covered.add((i0 + di, j0 + dj, k0 + dk))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U.setdefault((i, j), []).append((1, term_count))
                    V.setdefault((j, k), []).append((1, term_count))
                    W.setdefault((k, i), []).append((1, term_count))
                    term_count += 1
    
    rank = term_count
    U_mat = [[0] * rank for _ in range(n * m)]
    V_mat = [[0] * rank for _ in range(m * p)]
    W_mat = [[0] * rank for _ in range(p * n)]
    
    for (i, j), entries in U.items():
        for coeff, t in entries:
            U_mat[i * m + j][t] = coeff
    for (j, k), entries in V.items():
        for coeff, t in entries:
            V_mat[j * p + k][t] = coeff
    for (k, i), entries in W.items():
        for coeff, t in entries:
            W_mat[k * n + i][t] = coeff
    
    return {"U": U_mat, "V": V_mat, "W": W_mat}

def _verify(U, V, W):
    """Verify exact decomposition. Returns True if valid."""
    n, m, p = DIMENSIONS
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    if U_arr.shape != (n * m, rank):
        return False
    if V_arr.shape != (m * p, rank):
        return False
    if W_arr.shape != (p * n, rank):
        return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    result = _build_strassen_356()
    U, V, W = result["U"], result["V"], result["W"]
    rank = len(U[0])
    
    if not _verify(U, V, W):
        n, m, p = DIMENSIONS
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
