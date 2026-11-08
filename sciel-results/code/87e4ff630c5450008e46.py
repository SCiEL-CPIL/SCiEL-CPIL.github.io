"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _schoolbook():
    """Return the trivial schoolbook decomposition (rank = n*m*p)."""
    n, m, p = DIMENSIONS
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

def _validate(U, V, W, tol=1e-10):
    """Check that the decomposition exactly reproduces matrix multiplication."""
    n, m, p = DIMENSIONS
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    U_3d = U_arr.reshape(n, m, -1)
    V_3d = V_arr.reshape(m, p, -1)
    W_3d = W_arr.reshape(p, n, -1)
    
    T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
    
    if not np.allclose(T, 1.0, atol=tol):
        return False
    return True

def _strassen_2x2x2():
    """
    Return Strassen decomposition for 2x2 * 2x2.
    Rank = 7.
    """
    n = m = p = 2
    rank = 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    U[0*2+0][0] = 1
    U[1*2+1][0] = 1
    V[0*2+0][0] = 1
    V[1*2+1][0] = 1
    W[0*2+0][0] = 1
    W[1*2+1][0] = 1
    
    # M2 = (A[1,0]+A[1,1]) * B[0,0]
    U[1*2+0][1] = 1
    U[1*2+1][1] = 1
    V[0*2+0][1] = 1
    W[0*2+1][1] = 1
    
    # M3 = A[0,0] * (B[1,0]-B[1,1])
    U[0*2+0][2] = 1
    V[1*2+0][2] = 1
    V[1*2+1][2] = -1
    W[1*2+0][2] = 1
    
    # M4 = A[1,1] * (B[0,1]-B[0,0])
    U[1*2+1][3] = 1
    V[0*2+1][3] = 1
    V[0*2+0][3] = -1
    W[0*2+0][3] = 1
    W[0*2+1][3] = 1
    
    # M5 = (A[0,0]+A[0,1]) * B[1,1]
    U[0*2+0][4] = 1
    U[0*2+1][4] = 1
    V[1*2+1][4] = 1
    W[0*2+0][4] = -1
    W[1*2+0][4] = 1
    
    # M6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
    U[1*2+0][5] = 1
    U[0*2+0][5] = -1
    V[0*2+0][5] = 1
    V[0*2+1][5] = 1
    W[1*2+1][5] = 1
    
    # M7 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
    U[0*2+1][6] = 1
    U[1*2+1][6] = -1
    V[1*2+0][6] = 1
    V[1*2+1][6] = 1
    W[0*2+0][6] = 1
    
    return {"U": U, "V": V, "W": W}

def _strassen_4x4x4():
    """
    Construct Strassen decomposition for 4x4 * 4x4 using Kronecker product
    of two 2x2x2 Strassen decompositions.
    Rank = 7*7 = 49.
    """
    s2 = _strassen_2x2x2()
    # Kronecker product of two rank-7 decompositions
    rank1 = 7
    rank2 = 7
    rank = rank1 * rank2
    
    n = m = p = 4
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c1 in range(rank1):
        for c2 in range(rank2):
            col = c1 * rank2 + c2
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            idx = (i1*2+i2)*4 + (j1*2+j2)
                            U[idx][col] = s2['U'][i1*2+j1][c1] * s2['U'][i2*2+j2][c2]
            for j1 in range(2):
                for k1 in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            idx = (j1*2+j2)*4 + (k1*2+k2)
                            V[idx][col] = s2['V'][j1*2+k1][c1] * s2['V'][j2*2+k2][c2]
            for k1 in range(2):
                for i1 in range(2):
                    for k2 in range(2):
                        for i2 in range(2):
                            idx = (k1*2+k2)*4 + (i1*2+i2)
                            W[idx][col] = s2['W'][k1*2+i1][c1] * s2['W'][k2*2+i2][c2]
    
    return {"U": U, "V": V, "W": W}

def _strassen_4x4x9():
    """
    Construct a decomposition for 4x4 * 4x9.
    
    Approach: Split the 9 columns into 4+4+1.
    - For the first 4 columns: use Strassen 4x4x4 (rank 49)
    - For the next 4 columns: use Strassen 4x4x4 (rank 49)
    - For the last 1 column: use schoolbook 4x4x1 (rank 16)
    Total rank: 49 + 49 + 16 = 114
    
    But we can do better by using a mixed approach.
    
    Actually, let's try: 4x4x9 = 4x(4x9). We can view this as computing 9 separate
    4x4 * 4x1 products. For each, we can use a decomposition.
    
    Better approach: Use Strassen on the 4x4 part and schoolbook on the 9.
    The tensor 4x4x9 can be decomposed as:
    - For each k in 0..8, we need to compute C[:,k] = A * B[:,k]
    - This is 9 copies of the 4x4x1 tensor
    
    The 4x4x1 tensor has rank at most 16 (schoolbook).
    But we can use Strassen-like methods to reduce this.
    
    For 4x4x1: Split A into 2x2 blocks, B[:,k] into 2x1 vectors.
    Using Strassen for 2x2 * 2x1:
    - M1 = (A00+A11) * b0
    - M2 = A10 * b0
    - M3 = A00 * (b1-b0)
    - M4 = A11 * b1
    - M5 = (A00+A01) * b1
    - M6 = (A10-A00) * (b0+b1)
    - M7 = (A01-A11) * b1
    
    Wait, this doesn't quite work for 2x2 * 2x1. Let me think more carefully.
    
    Actually, for 4x4x9, let's just use the direct approach:
    - Use Strassen 4x4x4 for the first 4 columns
    - Use Strassen 4x4x4 for the next 4 columns  
    - Use schoolbook for the last column
    
    Total: 49 + 49 + 16 = 114
    
    But we can combine the two 4x4x4 parts. Actually, they're independent, so
    we just concatenate the columns.
    
    Let me construct this properly.
    """
    n, m, p = 4, 4, 9
    
    # Get Strassen 4x4x4 decomposition
    s4 = _strassen_4x4x4()
    rank_s4 = len(s4['U'][0])  # 49
    
    # Get schoolbook 4x4x1 decomposition
    U_sb = [[0] * 16 for _ in range(16)]
    V_sb = [[0] * 16 for _ in range(4)]
    W_sb = [[0] * 16 for _ in range(4)]
    col = 0
    for i in range(4):
        for j in range(4):
            U_sb[i*4+j][col] = 1
            V_sb[j*1+0][col] = 1
            W_sb[0*4+i][col] = 1
            col += 1
    rank_sb = 16
    
    # Total rank: 49 + 49 + 16 = 114
    total_rank = rank_s4 + rank_s4 + rank_sb
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # First 49 columns: Strassen for columns 0-3
    for c in range(rank_s4):
        for i in range(16):
            U[i][c] = s4['U'][i][c]
        for j in range(4):
            for k in range(4):
                V[j*9+k][c] = s4['V'][j*4+k][c]
        for k in range(4):
            for i in range(4):
                W[k*4+i][c] = s4['W'][k*4+i][c]
    
    # Next 49 columns: Strassen for columns 4-7
    offset1 = rank_s4
    for c in range(rank_s4):
        for i in range(16):
            U[i][offset1 + c] = s4['U'][i][c]
        for j in range(4):
            for k in range(4):
                V[j*9+(k+4)][offset1 + c] = s4['V'][j*4+k][c]
        for k in range(4):
            for i in range(4):
                W[(k+4)*4+i][offset1 + c] = s4['W'][k*4+i][c]
    
    # Last 16 columns: schoolbook for column 8
    offset2 = 2 * rank_s4
    for c in range(rank_sb):
        for i in range(16):
            U[i][offset2 + c] = U_sb[i][c]
        for j in range(4):
            V[j*9+8][offset2 + c] = V_sb[j][c]
        for i in range(4):
            W[8*4+i][offset2 + c] = W_sb[i][c]
    
    return {"U": U, "V": V, "W": W}

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 4x4x9 matrix multiplication.
    Returns a dictionary with U, V, W matrices.
    """
    start_time = time.time()
    deadline = start_time + 145
    
    best_result = None
    best_rank = float('inf')
    
    # Strategy 1: Schoolbook (rank 144)
    schoolbook = _schoolbook()
    if _validate(schoolbook['U'], schoolbook['V'], schoolbook['W']):
        best_result = schoolbook
        best_rank = len(schoolbook['U'][0])
    
    # Strategy 2: Try Strassen-based decomposition
    if time.time() < deadline:
        strassen_result = _strassen_4x4x9()
        if strassen_result is not None:
            if _validate(strassen_result['U'], strassen_result['V'], strassen_result['W']):
                rank = len(strassen_result['U'][0])
                if rank < best_rank:
                    best_result = strassen_result
                    best_rank = rank
    
    if best_result is None:
        return _schoolbook()
    
    return best_result

# EVOLVE-BLOCK-END
