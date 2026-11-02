"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]
import time
import random

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Return the schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2():
    """
    Return the Strassen decomposition for 2x2 matrices.
    Rank is 7.
    U shape: (4, 7)  # 2*2 entries
    V shape: (4, 7)  # 2*2 entries
    W shape: (4, 7)  # 2*2 entries
    """
    n = 2
    rank = 7
    U = [[0] * rank for _ in range(n * n)]
    V = [[0] * rank for _ in range(n * n)]
    W = [[0] * rank for _ in range(n * n)]
    
    # Strassen steps for 2x2:
    steps = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: -1, 2: 1, 1: -1}), # M1
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),                     # M2
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),                     # M3
        ({3: 1}, {2: 1, 0: -1}, {0: -1, 2: 1}),                    # M4
        ({0: 1, 1: 1}, {3: 1}, {1: 1}),                            # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {0: 1}),                     # M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: -1, 3: 1}),              # M7
    ]
    
    for s in range(7):
        u_coeffs, v_coeffs, w_coeffs = steps[s]
        for idx, sign in u_coeffs.items():
            U[idx][s] += sign
        for idx, sign in v_coeffs.items():
            V[idx][s] += sign
        for idx, sign in w_coeffs.items():
            W[idx][s] += sign
            
    return {"U": U, "V": V, "W": W}

def _kronecker_product_decomp(D1, D2, n1, m1, p1, n2, m2, p2):
    """
    Compute the Kronecker product of two tensor decompositions.
    """
    U1, V1, W1 = D1["U"], D1["V"], D1["W"]
    U2, V2, W2 = D2["U"], D2["V"], D2["W"]
    
    r1 = len(U1[0])
    r2 = len(U2[0])
    total_rank = r1 * r2
    
    N = n1 * n2
    M = m1 * m2
    P = p1 * p2
    
    U = [[0] * total_rank for _ in range(N * M)]
    V = [[0] * total_rank for _ in range(M * P)]
    W = [[0] * total_rank for _ in range(P * N)]
    
    # U: row = (i1*n2 + i2) * M + (j1*m2 + j2), col = c1 * r2 + c2
    for i1 in range(n1):
        for j1 in range(m1):
            for c1 in range(r1):
                val_u1 = U1[i1 * m1 + j1][c1]
                if val_u1 == 0: continue
                for i2 in range(n2):
                    for j2 in range(m2):
                        for c2 in range(r2):
                            val_u2 = U2[i2 * m2 + j2][c2]
                            if val_u2 == 0: continue
                            row = (i1 * n2 + i2) * M + (j1 * m2 + j2)
                            col = c1 * r2 + c2
                            U[row][col] = val_u1 * val_u2
    
    # V: row = (j1*p2 + j2) * P + (k1*p2 + k2), col = c1 * r2 + c2
    for j1 in range(m1):
        for k1 in range(p1):
            for c1 in range(r1):
                val_v1 = V1[j1 * p1 + k1][c1]
                if val_v1 == 0: continue
                for j2 in range(m2):
                    for k2 in range(p2):
                        for c2 in range(r2):
                            val_v2 = V2[j2 * p2 + k2][c2]
                            if val_v2 == 0: continue
                            row = (j1 * m2 + j2) * P + (k1 * p2 + k2)
                            col = c1 * r2 + c2
                            V[row][col] = val_v1 * val_v2
    
    # W: row = (k1*n2 + k2) * N + (i1*n2 + i2), col = c1 * r2 + c2
    for k1 in range(p1):
        for i1 in range(n1):
            for c1 in range(r1):
                val_w1 = W1[k1 * n1 + i1][c1]
                if val_w1 == 0: continue
                for k2 in range(p2):
                    for i2 in range(n2):
                        for c2 in range(r2):
                            val_w2 = W2[k2 * n2 + i2][c2]
                            if val_w2 == 0: continue
                            row = (k1 * p2 + k2) * N + (i1 * n2 + i2)
                            col = c1 * r2 + c2
                            W[row][col] = val_w1 * val_w2
    
    return {"U": U, "V": V, "W": W}

def _verify(decomp, n, m, p, sample=False):
    """Verify that the decomposition is correct."""
    U = decomp["U"]
    V = decomp["V"]
    W = decomp["W"]
    rank = len(U[0]) if U else 0

    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if any(len(row) != rank for row in U) or any(len(row) != rank for row in V) or any(len(row) != rank for row in W):
        return False

    if sample:
        indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
        random.shuffle(indices)
        indices = indices[:min(200, len(indices))]
    else:
        indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]

    for i, j, k in indices:
        s = 0
        for c in range(rank):
            s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
        if abs(s - 1) > 1e-9:
            return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110

    # Strategy: Kronecker product of Strassen(2x2) and Schoolbook(3x3)
    # Rank = 7 * 27 = 189
    try:
        D_strassen = _strassen_2x2()
        D_schoolbook_3 = _schoolbook(3, 3, 3)
        
        D_kron = _kronecker_product_decomp(D_strassen, D_schoolbook_3, 2, 2, 2, 3, 3, 3)
        
        if _verify(D_kron, n, m, p, sample=False):
            return D_kron
    except Exception as e:
        pass

    # Fallback to schoolbook if anything fails
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
