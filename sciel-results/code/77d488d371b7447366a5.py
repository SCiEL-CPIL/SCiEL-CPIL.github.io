"""Combined program: block-structured Strassen for 6x6x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """
    Return Strassen decomposition for 2x2 matrix multiplication.
    Rank = 7
    
    For A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    
    M1 = (a+d)(e+h)
    M2 = (c+d)e
    M3 = a(f-h)
    M4 = d(g-e)
    M5 = (a+b)h
    M6 = (c-a)(e+f)
    M7 = (b-d)(g+h)
    
    C00 = M1 - M4 + M5 - M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    U = [[0] * 7 for _ in range(4)]
    V = [[0] * 7 for _ in range(4)]
    W = [[0] * 7 for _ in range(4)]
    
    # M1 = (a+d)(e+h)
    U[0][0] = 1  # a
    U[3][0] = 1  # d
    V[0][0] = 1  # e
    V[3][0] = 1  # h
    W[0][0] = 1  # c00
    W[3][0] = 1  # c11
    
    # M2 = (c+d)e
    U[2][1] = 1  # c
    U[3][1] = 1  # d
    V[0][1] = 1  # e
    W[2][1] = 1  # c10
    W[3][1] = -1  # c11
    
    # M3 = a(f-h)
    U[0][2] = 1  # a
    V[1][2] = 1  # f
    V[3][2] = -1  # h
    W[1][2] = 1  # c01
    W[3][2] = 1  # c11
    
    # M4 = d(g-e)
    U[3][3] = 1  # d
    V[2][3] = 1  # g
    V[0][3] = -1  # e
    W[0][3] = -1  # c00
    W[2][3] = 1  # c10
    
    # M5 = (a+b)h
    U[0][4] = 1  # a
    U[1][4] = 1  # b
    V[3][4] = 1  # h
    W[0][4] = 1  # c00
    W[1][4] = 1  # c01
    
    # M6 = (c-a)(e+f)
    U[2][5] = 1  # c
    U[0][5] = -1  # a
    V[0][5] = 1  # e
    V[1][5] = 1  # f
    W[3][5] = 1  # c11
    
    # M7 = (b-d)(g+h)
    U[1][6] = 1  # b
    U[3][6] = -1  # d
    V[2][6] = 1  # g
    V[3][6] = 1  # h
    W[0][6] = -1  # c00
    
    return U, V, W, 7

def _build_block_decomposition(bx, by, bz, local_U, local_V, local_W, local_rank):
    """
    Construct global U, V, W for 6x6 matrix multiplication using block structure.
    
    Args:
        bx, by, bz: block sizes for A, B, C (A is by x bx, B is bz x by, C is bz x bx)
        local_U: (by*bx, local_rank) tensor for local multiplication
        local_V: (bz*by, local_rank) tensor for local multiplication
        local_W: (bz*bx, local_rank) tensor for local multiplication
    
    Returns:
        (U, V, W) as lists of lists, rank
    """
    n, m, p = 6, 6, 6  # A is m x n, B is p x m, C is p x n
    
    rank = (6 // bx) * (6 // by) * (6 // bz) * local_rank
    
    U = [[0] * rank for _ in range(m * n)]  # 36 x rank
    V = [[0] * rank for _ in range(m * p)]  # 36 x rank
    W = [[0] * rank for _ in range(p * n)]  # 36 x rank
    
    col = 0
    for I in range(6 // bx):
        for J in range(6 // by):
            for K in range(6 // bz):
                for t in range(local_rank):
                    # U: for block (I, J), U is by x bx
                    for a in range(bx):
                        for b in range(by):
                            u_idx = (J * by + b) * n + (I * bx + a)
                            U[u_idx][col] = local_U[a * by + b][t]
                    # V: for block (J, K), V is bz x by
                    for b in range(by):
                        for c in range(bz):
                            v_idx = (K * bz + c) * m + (J * by + b)
                            V[v_idx][col] = local_V[b * bz + c][t]
                    # W: for block (K, I), W is bz x bx
                    for c in range(bz):
                        for a in range(bx):
                            w_idx = (K * bz + c) * n + (I * bx + a)
                            W[w_idx][col] = local_W[c * bx + a][t]
                    col += 1
    
    return U, V, W, rank

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that U, V, W correctly decompose the matrix multiplication tensor.
    C[i,k] = sum_{j,t} U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    """
    np.random.seed(42)
    A = np.random.rand(n, m).astype(float)
    B = np.random.rand(m, p).astype(float)
    C_true = A @ B
    
    C_comp = np.zeros((n, p), dtype=float)
    rank = len(U[0])
    for t in range(rank):
        U_col = np.array([U[i * m + j][t] for i in range(n) for j in range(m)]).reshape(n, m)
        V_col = np.array([V[j * p + k][t] for j in range(m) for k in range(p)]).reshape(m, p)
        M_t = U_col @ V_col
        W_col = np.array([W[k * n + i][t] for k in range(p) for i in range(n)]).reshape(p, n)
        C_comp += W_col.T * M_t
    
    if not np.allclose(C_comp, C_true, atol=1e-10):
        return False
    return True

def _schoolbook_6x6():
    """Fallback: schoolbook decomposition of rank 216"""
    n, m, p = 6, 6, 6
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
    return U, V, W, rank

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n, m, p = 6, 6, 6
    
    # Build 2x2 Strassen local decomposition
    U_local, V_local, W_local, local_rank = _strassen_2x2()
    
    # Verify the local Strassen decomposition
    np.random.seed(42)
    A_local = np.random.rand(2, 2)
    B_local = np.random.rand(2, 2)
    C_true_local = A_local @ B_local
    
    C_comp_local = np.zeros((2, 2))
    for t in range(local_rank):
        U_col = np.array([U_local[i * 2 + j][t] for i in range(2) for j in range(2)]).reshape(2, 2)
        V_col = np.array([V_local[j * 2 + k][t] for j in range(2) for k in range(2)]).reshape(2, 2)
        M_t = U_col @ V_col
        W_col = np.array([W_local[k * 2 + i][t] for k in range(2) for i in range(2)]).reshape(2, 2)
        C_comp_local += W_col.T * M_t
    
    if not np.allclose(C_comp_local, C_true_local, atol=1e-10):
        # Fallback to schoolbook if Strassen is incorrect
        U, V, W, rank = _schoolbook_6x6()
        return {"U": U, "V": V, "W": W}
    
    # Build global decomposition using 2x2x2 blocks with Strassen
    # Total rank = 3 * 3 * 3 * 7 = 189
    U, V, W, rank = _build_block_decomposition(2, 2, 2, U_local, V_local, W_local, local_rank)
    
    # Verify the full 6x6x6 decomposition
    if not _verify_decomposition(U, V, W, 6, 6, 6):
        # Fallback to schoolbook if verification fails
        U, V, W, rank = _schoolbook_6x6()
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
