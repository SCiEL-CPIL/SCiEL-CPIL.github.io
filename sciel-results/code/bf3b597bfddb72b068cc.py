import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def _strassen_factors():
    """
    Returns (U, V, W) for 2x2x2 Strassen decomposition.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indexing:
    U row: i*2 + j
    V row: j*2 + k
    W row: k*2 + i
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)

    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1  # C00 += M1
    W[3, 0] = 1  # C11 += M1

    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1  # C10 += M2
    W[3, 1] = -1 # C11 -= M2

    # M3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1  # C01 += M3
    W[3, 2] = 1  # C11 += M3

    # M4 = A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1  # C00 += M4
    W[1, 3] = 1  # C10 += M4

    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1 # C00 -= M5
    W[2, 4] = 1  # C01 += M5

    # M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1  # C11 += M6

    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1  # C00 += M7

    return U, V, W

def _trivial_1x1x3_factors():
    """
    Returns (U, V, W) for 1x1x3 trivial decomposition.
    U: (1, 3), V: (3, 3), W: (3, 3)
    """
    U = np.zeros((1, 3), dtype=float)
    V = np.zeros((3, 3), dtype=float)
    W = np.zeros((3, 3), dtype=float)
    for k in range(3):
        U[0, k] = 1
        V[k, k] = 1
        W[k, k] = 1
    return U, V, W

def _build_rank42():
    """
    Builds rank-42 decomposition for 2x4x6.
    Structure: 2 blocks of (2x2x6).
    Each 2x2x6 is Kronecker product of 2x2x2 (Strassen) and 1x1x3 (Trivial).
    """
    n, m, p = 2, 4, 6
    U1, V1, W1 = _strassen_factors()
    U2, V2, W2 = _trivial_1x1x3_factors()
    
    rank1 = 7
    rank2 = 3
    block_rank = rank1 * rank2
    total_rank = 2 * block_rank # 42

    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)

    for block in range(2):
        # Block 0: A cols 0,1; B rows 0,1
        # Block 1: A cols 2,3; B rows 2,3
        
        for s1 in range(rank1):
            for s2 in range(rank2):
                s_local = s1 * rank2 + s2
                s = block * block_rank + s_local
                
                # U: A indices (i, j_local) -> global (i, j_global)
                # i in 0..1, j_local in 0..1
                # j_global = block * 2 + j_local
                for i in range(2):
                    for j_local in range(2):
                        j_global = block * 2 + j_local
                        u1_coeff = U1[i * 2 + j_local, s1]
                        u2_coeff = U2[0, s2]
                        if u1_coeff != 0 and u2_coeff != 0:
                            U[i * m + j_global, s] += u1_coeff * u2_coeff

                # V: B indices (j_local, k) -> global (j_global, k_global)
                # j_local in 0..1, k in 0..5
                # k_global = k1 * 3 + k2
                # k1 in 0..1 (from Strassen), k2 in 0..2 (from Trivial)
                # Wait, the 2x2x2 part handles k1 (0..1).
                # The 1x1x3 part handles k2 (0..2).
                # So k_global = k1 * 3 + k2.
                for j_local in range(2):
                    j_global = block * 2 + j_local
                    for k1 in range(2):
                        for k2 in range(3):
                            k_global = k1 * 3 + k2
                            v1_coeff = V1[j_local * 2 + k1, s1]
                            v2_coeff = V2[k2, s2]
                            if v1_coeff != 0 and v2_coeff != 0:
                                V[j_global * p + k_global, s] += v1_coeff * v2_coeff

                # W: C indices (k, i) -> global (k_global, i)
                # k_global = k1 * 3 + k2
                for k1 in range(2):
                    for k2 in range(3):
                        k_global = k1 * 3 + k2
                        for i in range(2):
                            w1_coeff = W1[k1 * 2 + i, s1]
                            w2_coeff = W2[k2, s2]
                            if w1_coeff != 0 and w2_coeff != 0:
                                W[k_global * n + i, s] += w1_coeff * w2_coeff

    return U, V, W

def _verify(U, V, W, n, m, p):
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for s in range(rank):
                    val += U[i * m + j, s] * V[j * p + k, s] * W[k * n + i, s]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _to_output(U, V, W):
    U_out = [[int(x) if abs(x - round(x)) < 1e-12 else float(x) for x in row] for row in U]
    V_out = [[int(x) if abs(x - round(x)) < 1e-12 else float(x) for x in row] for row in V]
    W_out = [[int(x) if abs(x - round(x)) < 1e-12 else float(x) for x in row] for row in W]
    return {"U": U_out, "V": V_out, "W": W_out}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0

    U, V, W = _build_rank42()
    
    # Quick verification
    if _verify(U, V, W, n, m, p):
        return _to_output(U, V, W)

    # Fallback: Schoolbook Rank 48
    if time.time() < deadline:
        rank = n * m * p
        U_fb = np.zeros((n * m, rank), dtype=float)
        V_fb = np.zeros((m * p, rank), dtype=float)
        W_fb = np.zeros((p * n, rank), dtype=float)
        col = 0
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    U_fb[ii * m + jj, col] = 1
                    V_fb[jj * p + kk, col] = 1
                    W_fb[kk * n + ii, col] = 1
                    col += 1
        return _to_output(U_fb, V_fb, W_fb)

    return {"U": [], "V": [], "W": []}
