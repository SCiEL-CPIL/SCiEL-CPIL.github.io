"""Matrix multiplication 2x5x6 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START

def _strassen_222():
    """Return U, V, W for 2x2x2 Strassen decomposition (rank 7).
    U shape (4,7), V shape (4,7), W shape (4,7).
    Convention: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    # Strassen's 7 multiplications for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1: (A00+A11)(B00+B11)
    # U: A00=U[0], A11=U[3] -> both get 1
    # V: B00=V[0], B11=V[3] -> both get 1
    # W: C00=W[0] +=1, C11=W[3] +=1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2: (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3: A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4: A11(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5: (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6: (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7: (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W


def _block_216():
    """Return U, V, W for 2x1x6 multiplication (rank 12).
    A is 2x1, B is 1x6, C is 2x6.
    U shape (2,12), V shape (6,12), W shape (12,12).
    Convention: U[i*1+j, r]=U[i, r], V[j*6+k, r]=V[k, r], W[k*2+i, r]
    Schoolbook: rank = 2*1*6 = 12
    """
    rank = 12
    U = [[0]*rank for _ in range(2)]
    V = [[0]*rank for _ in range(6)]
    W = [[0]*rank for _ in range(12)]
    col = 0
    for i in range(2):
        for k in range(6):
            U[i][col] = 1
            V[k][col] = 1
            W[k*2 + i][col] = 1
            col += 1
    return U, V, W


def _block_226():
    """Return U, V, W for 2x2x6 multiplication using 3 Strassen 2x2x2 blocks (rank 21).
    A is 2x2, B is 2x6, C is 2x6.
    U shape (4,21), V shape (12,21), W shape (12,21).
    Convention: U[i*2+j, r], V[j*6+k, r], W[k*2+i, r]
    
    Split B into three 2x2 blocks: B[:, 0:2], B[:, 2:4], B[:, 4:6]
    Each 2x2x2 uses Strassen rank 7, total rank 21.
    """
    rank = 21
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(12)]
    W = [[0]*rank for _ in range(12)]
    
    U_s, V_s, W_s = _strassen_222()
    
    for block_idx in range(3):
        base = block_idx * 7
        for i in range(2):
            for j in range(2):
                U[i*2 + j][base + r_idx] if False else None
        # Copy Strassen U
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    U[i*2 + j][base + r] = U_s[i*2 + j][r]
        # Copy Strassen V: V_s[j*2+k, r] maps to V[(j)*6 + (block_idx*2 + k), r]
        for r in range(7):
            for j in range(2):
                for k in range(2):
                    V[j*6 + block_idx*2 + k][base + r] = V_s[j*2 + k][r]
        # Copy Strassen W: W_s[k*2+i, r] maps to W[(block_idx*2 + k)*2 + i, r]
        for r in range(7):
            for k in range(2):
                for i in range(2):
                    W[(block_idx*2 + k)*2 + i][base + r] = W_s[k*2 + i][r]
    
    return U, V, W


def _build_54():
    """Build the full 2x5x6 decomposition with rank 54.
    Split: 2x1x6 (rank 12) + 2x2x6 (rank 21) + 2x2x6 (rank 21) = 54
    """
    n, m, p = 2, 5, 6
    rank = 54
    
    U = [[0]*rank for _ in range(n*m)]   # (10, 54)
    V = [[0]*rank for _ in range(m*p)]   # (30, 54)
    W = [[0]*rank for _ in range(p*n)]   # (12, 54)
    
    # Part 1: 2x1x6, j=0, rank offset 0
    U_216, V_216, W_216 = _block_216()
    for i in range(2):
        for r in range(12):
            U[i*5 + 0][r] = U_216[i][r]
    for k in range(6):
        for r in range(12):
            V[0*6 + k][r] = V_216[k][r]
    for k in range(6):
        for i in range(2):
            for r in range(12):
                W[k*2 + i][r] = W_216[k*2 + i][r]
    
    # Part 2: 2x2x6, j=1,2, rank offset 12
    U_226a, V_226a, W_226a = _block_226()
    base1 = 12
    for i in range(2):
        for j in range(2):
            for r in range(21):
                U[i*5 + 1 + j][base1 + r] = U_226a[i*2 + j][r]
    for j in range(2):
        for k in range(6):
            for r in range(21):
                V[(1+j)*6 + k][base1 + r] = V_226a[j*6 + k][r]
    for k in range(6):
        for i in range(2):
            for r in range(21):
                W[k*2 + i][base1 + r] = W_226a[k*2 + i][r]
    
    # Part 3: 2x2x6, j=3,4, rank offset 33
    U_226b, V_226b, W_226b = _block_226()
    base2 = 33
    for i in range(2):
        for j in range(2):
            for r in range(21):
                U[i*5 + 3 + j][base2 + r] = U_226b[i*2 + j][r]
    for j in range(2):
        for k in range(6):
            for r in range(21):
                V[(3+j)*6 + k][base2 + r] = V_226b[j*6 + k][r]
    for k in range(6):
        for i in range(2):
            for r in range(21):
                W[k*2 + i][base2 + r] = W_226b[k*2 + i][r]
    
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W):
    """Verify the decomposition is exact for matrix multiplication."""
    n, m, p = 2, 5, 6
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # For random A (n x m) and B (m x p), check C = A @ B
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    C_computed = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            s = 0.0
            for j in range(m):
                for r in range(rank):
                    s += U_arr[i*m + j, r] * V_arr[j*p + k, r] * W_arr[k*n + i, r]
            C_computed[i, k] = s
    
    return np.allclose(C_computed, C_expected, atol=1e-10)


def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 110  # safety margin before 120s/150s limit
    
    # Build the rank-54 decomposition
    result = _build_54()
    
    # Quick verification
    if _verify(result["U"], result["V"], result["W"]):
        return result
    
    # Fallback: schoolbook
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
