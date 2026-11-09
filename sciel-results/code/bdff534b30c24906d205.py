"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Kronecker product decomposition for 3x4 x 4x8 matrix multiplication.
    
    Factor: (3x2x4) ⊗ (1x2x2)
    - Sub-problem 1: 3x2 x 2x4, rank 24 (schoolbook)
    - Sub-problem 2: 1x2 x 2x2, rank 2
    - Total rank: 24 * 2 = 48
    """
    import numpy as np
    
    # Sub-problem 2: 1x2 x 2x2, rank 2
    U2 = np.array([[1, 1],
                   [1, -1]], dtype=float)
    V2 = np.array([[1, 0],
                   [0, 1],
                   [0, 1],
                   [1, 0]], dtype=float)
    W2 = np.array([[1, 1],
                   [1, -1]], dtype=float)
    
    # Sub-problem 1: 3x2 x 2x4, schoolbook rank 24
    n1, m1, p1 = 3, 2, 4
    rank1 = n1 * m1 * p1
    U1 = np.zeros((n1 * m1, rank1))
    V1 = np.zeros((m1 * p1, rank1))
    W1 = np.zeros((p1 * n1, rank1))
    col = 0
    for i in range(n1):
        for j in range(m1):
            for k in range(p1):
                U1[i * m1 + j, col] = 1
                V1[j * p1 + k, col] = 1
                W1[k * n1 + i, col] = 1
                col += 1
    
    # Kronecker product
    n2, m2, p2 = 1, 2, 2
    rank2 = 2
    total_rank = rank1 * rank2
    
    U_full = np.zeros((n1 * n2 * m1 * m2, total_rank))
    V_full = np.zeros((m1 * m2 * p1 * p2, total_rank))
    W_full = np.zeros((p1 * p2 * n1 * n2, total_rank))
    
    for r1 in range(rank1):
        for r2 in range(rank2):
            r = r1 * rank2 + r2
            for i1 in range(n1):
                for j1 in range(m1):
                    for i2 in range(n2):
                        for j2 in range(m2):
                            idx_u = (i1 * n2 + i2) * (m1 * m2) + (j1 * m2 + j2)
                            U_full[idx_u, r] = U1[i1 * m1 + j1, r1] * U2[i2 * m2 + j2, r2]
            for j1 in range(m1):
                for k1 in range(p1):
                    for j2 in range(m2):
                        for k2 in range(p2):
                            idx_v = (j1 * m2 + j2) * (p1 * p2) + (k1 * p2 + k2)
                            V_full[idx_v, r] = V1[j1 * p1 + k1, r1] * V2[j2 * p2 + k2, r2]
            for k1 in range(p1):
                for i1 in range(n1):
                    for k2 in range(p2):
                        for i2 in range(n2):
                            idx_w = (k1 * p2 + k2) * (n1 * n2) + (i1 * n2 + i2)
                            W_full[idx_w, r] = W1[k1 * n1 + i1, r1] * W2[k2 * n2 + i2, r2]
    
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
# EVOLVE-BLOCK-END
