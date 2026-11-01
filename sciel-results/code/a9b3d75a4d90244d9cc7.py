"""Tensor decomposition for 2x3 * 3x9 matrix multiplication."""
DIMENSIONS = [2, 3, 9]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s limit
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    U = []
    V = []
    W = []
    
    def add_term(u_entry, v_entry, w_entry):
        """Add a rank-1 term. u_entry: dict mapping A-index to coeff, etc."""
        u_vec = [0] * (n * m)
        v_vec = [0] * (m * p)
        w_vec = [0] * (p * n)
        for idx, coeff in u_entry.items():
            u_vec[idx] = coeff
        for idx, coeff in v_entry.items():
            v_vec[idx] = coeff
        for idx, coeff in w_entry.items():
            w_vec[idx] = coeff
        U.append(u_vec)
        V.append(v_vec)
        W.append(w_vec)
        
    # Part 1: A_2x1 * B_1x9 (rank 18)
    # A column 2 (index 2), B row 2 (index 2)
    # C[i,k] += A[i,2] * B[2,k]
    for i in range(n):
        for k in range(p):
            add_term(
                {i * m + 2: 1},
                {2 * p + k: 1},
                {k * n + i: 1}
            )
            
    # Part 2: A_2x2 * B_2x9
    # A_2x2: columns 0,1 of A (indices 0,1,3,4 in flat A)
    # B_2x9: rows 0,1 of B (indices 0..8, 9..17 in flat B)
    # Split 9 columns into groups: [0,1], [2,3], [4,5], [6,7], [8]
    
    # Strassen for 2x2 x 2x2 blocks
    for ko in [0, 2, 4, 6]:
        # A indices
        a00 = 0 * m + 0  # 0
        a01 = 0 * m + 1  # 1
        a10 = 1 * m + 0  # 3
        a11 = 1 * m + 1  # 4
        
        # B indices
        b00 = 0 * p + ko
        b01 = 0 * p + (ko + 1)
        b10 = 1 * p + ko
        b11 = 1 * p + (ko + 1)
        
        # C indices (W[k*n + i])
        c00 = ko * n + 0
        c01 = (ko + 1) * n + 0
        c10 = ko * n + 1
        c11 = (ko + 1) * n + 1
        
        # M1 = (a00+a11)(b00+b11) -> c00: +1, c11: +1
        add_term(
            {a00: 1, a11: 1},
            {b00: 1, b11: 1},
            {c00: 1, c11: 1}
        )
        # M2 = (a10+a11)b00 -> c10: +1, c11: -1
        add_term(
            {a10: 1, a11: 1},
            {b00: 1},
            {c10: 1, c11: -1}
        )
        # M3 = a00(b10-b11) -> c01: +1, c11: +1
        add_term(
            {a00: 1},
            {b10: 1, b11: -1},
            {c01: 1, c11: 1}
        )
        # M4 = a11(b01-b00) -> c00: +1, c10: +1
        add_term(
            {a11: 1},
            {b01: 1, b00: -1},
            {c00: 1, c10: 1}
        )
        # M5 = (a00+a01)b11 -> c01: +1, c00: -1
        add_term(
            {a00: 1, a01: 1},
            {b11: 1},
            {c01: 1, c00: -1}
        )
        # M6 = (a10-a00)(b00+b10) -> c11: +1
        add_term(
            {a10: 1, a00: -1},
            {b00: 1, b10: 1},
            {c11: 1}
        )
        # M7 = (a01-a11)(b01+b11) -> c00: +1, c01: +1
        add_term(
            {a01: 1, a11: -1},
            {b01: 1, b11: 1},
            {c00: 1, c01: 1}
        )
        
    # Part 3: Remaining column ko=8 (2x2 x 2x1, rank 4)
    ko = 8
    a00 = 0 * m + 0
    a01 = 0 * m + 1
    a10 = 1 * m + 0
    a11 = 1 * m + 1
    
    b00 = 0 * p + ko
    b10 = 1 * p + ko
    
    c00 = ko * n + 0
    c10 = ko * n + 1
    
    # Schoolbook for 2x2 x 2x1: 4 terms
    add_term({a00: 1}, {b00: 1}, {c00: 1})
    add_term({a01: 1}, {b10: 1}, {c00: 1})
    add_term({a00: 1}, {b10: 1}, {c10: 1})
    add_term({a01: 1}, {b00: 1}, {c10: 1})
    
    rank_structured = len(U)
    
    # Verify correctness using numpy
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Reconstruct tensor: T[i*m+j, j*p+k, k*n+i] = sum_r U[r][i*m+j] * V[r][j*p+k] * W[r][k*n+i]
    reconstructed = np.einsum('ra,rb,rc->abc', U_arr, V_arr, W_arr)
    
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
                
    if np.allclose(reconstructed, T, atol=1e-10):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook
        return best
# EVOLVE-BLOCK-END
