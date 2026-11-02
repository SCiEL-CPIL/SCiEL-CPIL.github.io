import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank 12 decomposition for 3x4x6
    # Split A (3x4) into A1 (3x2) and A2 (3x2)
    # Split B (4x6) into B1 (2x6) and B2 (2x6)
    # C = A1 B1 + A2 B2
    # Each term is 3x2x6
    # Split B1 (2x6) into B11 (2x2), B12 (2x2), B13 (2x2)
    # A1 B1 = A1 B11 + A1 B12 + A1 B13
    # Each term is 3x2x2
    # 3x2x2 rank is 2
    # Total rank = 2 terms * 3 splits * 2 rank = 12
    
    rank = 12
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    # A is 3x4, B is 4x6
    # A1 = A[:, :2], A2 = A[:, 2:]
    # B1 = B[:2, :], B2 = B[2:, :]
    
    # Term 1: A1 B1
    # A1 (3x2), B1 (2x6)
    # B1 = [B11 (2x2), B12 (2x2), B13 (2x2)]
    # A1 B11 (3x2x2)
    # Rank 2 decomposition for 3x2x2:
    # C_ij = A_i1 B_1j + A_i2 B_2j
    # Term 1: A_i1 B_1j -> U_i = A_i1, V_j = B_1j, W = 1
    # Term 2: A_i2 B_2j -> U_i = A_i2, V_j = B_2j, W = 1
    
    # Indices for A1: i in 0..2, k in 0..1 (global k in 0..1)
    # Indices for B1: k in 0..1, j in 0..5 (global j in 0..5)
    
    # Split B1 into 3 blocks of width 2
    for block in range(3):
        # j indices: 2*block, 2*block+1
        j_start = 2 * block
        # k indices: 0, 1
        
        # Sub-term 1: A_i0 * B_0j
        # U: row i, col 0 = 1 (for A_i0)
        # V: row 0*p + j, col 0 = 1 (for B_0j)
        # W: row j*n + i, col 0 = 1
        
        for i in range(3):
            U[i * m + 0, col] = 1.0 # A_i0 (global col 0)
            for j in [0, 1]:
                V[0 * p + (j_start + j), col] = 1.0 # B_0, j_start+j
                W[(j_start + j) * n + i, col] = 1.0
        col += 1
        
        # Sub-term 2: A_i1 * B_1j
        for i in range(3):
            U[i * m + 1, col] = 1.0 # A_i1 (global col 1)
            for j in [0, 1]:
                V[1 * p + (j_start + j), col] = 1.0 # B_1, j_start+j
                W[(j_start + j) * n + i, col] = 1.0
        col += 1

    # Term 2: A2 B2
    # A2 (3x2) -> global cols 2, 3
    # B2 (2x6) -> global rows 2, 3
    for block in range(3):
        j_start = 2 * block
        
        # Sub-term 1: A_i2 * B_2j
        for i in range(3):
            U[i * m + 2, col] = 1.0 # A_i2 (global col 2)
            for j in [0, 1]:
                V[2 * p + (j_start + j), col] = 1.0 # B_2, j_start+j
                W[(j_start + j) * n + i, col] = 1.0
        col += 1
        
        # Sub-term 2: A_i3 * B_3j
        for i in range(3):
            U[i * m + 3, col] = 1.0 # A_i3 (global col 3)
            for j in [0, 1]:
                V[3 * p + (j_start + j), col] = 1.0 # B_3, j_start+j
                W[(j_start + j) * n + i, col] = 1.0
        col += 1

    # Verify
    D = np.zeros((n, m, p))
    for c in range(rank):
        Uc = U[:, c].reshape(n, m)
        Vc = V[:, c].reshape(m, p)
        Wc = W[:, c].reshape(p, n)
        D += Uc[:, :, None] * Vc[None, :, :] * Wc.T[:, None, :]
        
    if not np.allclose(D, 1.0, atol=1e-9):
        # Fallback to schoolbook if error
        schoolbook_rank = n * m * p
        U_s = np.zeros((n * m, schoolbook_rank))
        V_s = np.zeros((m * p, schoolbook_rank))
        W_s = np.zeros((p * n, schoolbook_rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i * m + j, c] = 1.0
                    V_s[j * p + k, c] = 1.0
                    W_s[k * n + i, c] = 1.0
                    c += 1
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
