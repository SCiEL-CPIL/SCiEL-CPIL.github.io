import time
import numpy as np
from fractions import Fraction

def construct():
    t0 = time.time()
    deadline = t0 + 145.0

    def _elapsed_ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8
    
    # Fallback schoolbook rank 48
    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U[i * m + a][col] = 1
                    V[a * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def _validate(candidate):
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        rank = len(U[0])
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    acc = Fraction(0)
                    u_row = U[i * m + a]
                    v_row = V[a * p + k]
                    w_row = W[k * n + i]
                    for c in range(rank):
                        if u_row[c] == 0 or v_row[c] == 0 or w_row[c] == 0:
                            continue
                        acc += Fraction(u_row[c]) * Fraction(v_row[c]) * Fraction(w_row[c])
                    if acc != 1:
                        return False
        return True

    # Strassen 2x2x2 decomposition matrices
    # U factors for A1 (2x2), V factors for B (2x2), W factors for C (2x2)
    S_U = np.array([
        [1, 0, 0, 1], 
        [0, 0, 1, 1], 
        [1, 0, 0, 0], 
        [0, 0, 0, 1],
        [1, 1, 0, 0], 
        [-1, 0, 1, 0], 
        [0, 1, 0, -1]
    ])
    S_V = np.array([
        [1, 0, 0, 1], 
        [1, 0, 0, 0], 
        [0, 1, 0, -1], 
        [-1, 0, 1, 0],
        [0, 0, 0, 1], 
        [1, 1, 0, 0], 
        [0, 0, 1, 1]
    ])
    S_W = np.array([
        [1, 0, 0, 1], 
        [0, 1, 0, -1], 
        [0, 0, 1, 1], 
        [1, 1, 0, 0],
        [-1, 0, 1, 0], 
        [0, 0, 0, 1], 
        [1, 0, 0, 0]
    ])

    # Construct Rank 44 decomposition
    rank = 44
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    col = 0
    
    # Part 1: A1 (2x2) * B1 (2x8)
    # B1 is columns 0-1 of B. Split into 4 blocks of 2 columns.
    for block_idx in range(4):
        k_start = block_idx * 2
        
        for r in range(7):
            # Fill U part: indices correspond to A1 (i=0..1, a=0..1) -> i*m + a
            # S_U[r] is a row of 4 elements: [U00, U01, U10, U11]
            u_vals = S_U[r]
            for idx, val in enumerate(u_vals):
                if val != 0:
                    i_idx = idx // 2
                    a_idx = idx % 2
                    U[i_idx * m + a_idx, col] += val

            # Fill V part: indices correspond to B1 block (a=0..1, k=k_start..k_start+1)
            # S_V[r] is a row of 4 elements: [V00, V01, V10, V11]
            v_vals = S_V[r]
            for idx, val in enumerate(v_vals):
                if val != 0:
                    a_idx = idx // 2
                    k_idx = idx % 2
                    V[a_idx * p + (k_start + k_idx), col] += val

            # Fill W part: indices correspond to C1 block (k=k_start..k_start+1, i=0..1)
            # S_W[r] is a row of 4 elements: [W00, W01, W10, W11]
            # Note: W index is k*n + i. In Strassen standard, C is 2x2.
            # C[i, k] ? No, matrix mult C = A * B. C[i, k] = sum A[i, a] B[a, k].
            # Here C is 2x2. Indices i=0..1, k=0..1.
            # S_W[r] maps to C elements.
            w_vals = S_W[r]
            for idx, val in enumerate(w_vals):
                if val != 0:
                    k_idx = idx // 2
                    i_idx = idx % 2
                    W[(k_start + k_idx) * n + i_idx, col] += val

            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    # Part 2: A2 (2x1) * B2 (1x8)
    # A2 is column 2 of A. B2 is row 2 of B.
    # Product is outer product of A2 and B2.
    # A2 has elements A[i, 2] for i=0,1.
    # B2 has elements B[2, k] for k=0..7.
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 2  # a=2
            v_idx = 2 * p + k  # a=2
            w_idx = k * n + i
            U[u_idx, col] = 1
            V[v_idx, col] = 1
            W[w_idx, col] = 1
            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    # Convert to lists of integers (or half-integers if needed, but here integers)
    U_list = [[int(x) for x in row] for row in U]
    V_list = [[int(x) for x in row] for row in V]
    W_list = [[int(x) for x in row] for row in W]

    candidate = {"U": U_list, "V": V_list, "W": W_list}

    if _validate(candidate):
        return candidate
    else:
        return _schoolbook()
