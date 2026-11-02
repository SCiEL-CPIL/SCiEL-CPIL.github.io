"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W):
    """Verify that the decomposition exactly matches the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = len(W[0])
    rng = np.random.default_rng(12345)
    A = rng.standard_normal((n, m))
    B = rng.standard_normal((m, p))
    C_expected = A @ B
    C_computed = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            for r in range(rank):
                lA = sum(U[i*m+j, r] * A[i, j] for j in range(m))
                lB = sum(V[j*p+k, r] * B[j, k] for j in range(m))
                C_computed[i, k] += W[k*n+i, r] * lA * lB
    return np.allclose(C_computed, C_expected, atol=1e-8)

def _rank_11_solution():
    """Return the known rank-11 decomposition (Strassen + schoolbook for col 2)."""
    n, m, p = DIMENSIONS
    rank = 11
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen part (terms 0-6) for 2x2 * 2x2 (cols 0,1 of B)
    # Indices for A: i*m+j -> 0:00, 1:01, 2:10, 3:11
    # Indices for B: j*p+k -> 0:00, 1:01, 2:10, 3:11, 4:02, 5:12
    # Indices for C: k*n+i -> 0:00, 1:10, 2:01, 3:11, 4:02, 5:12
    
    # M1 = (A00+A11)(B00+B11)
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1
    # M2 = (A10+A11)B00
    U[2,1]=1; U[3,1]=1; V[0,1]=1
    # M3 = A00(B01-B11)
    U[0,2]=1; V[1,2]=1; V[3,2]=-1
    # M4 = A11(B10-B00)
    U[3,3]=1; V[2,3]=1; V[0,3]=-1
    # M5 = (A00+A01)B11
    U[0,4]=1; U[1,4]=1; V[3,4]=1
    # M6 = (A10-A00)(B00+B01)
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1
    # M7 = (A01-A11)(B10+B11)
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1
    
    # Combine for C00, C10, C01, C11
    # C00 = M1 - M4 - M6 + M7
    W[0,0]=1; W[0,3]=-1; W[0,5]=-1; W[0,6]=1
    # C10 = M2 + M4
    W[1,1]=1; W[1,3]=1
    # C01 = M3 + M5
    W[2,2]=1; W[2,4]=1
    # C11 = M1 - M3 - M5 + M6
    W[3,0]=1; W[3,2]=-1; W[3,4]=-1; W[3,5]=1
    
    # Schoolbook for col 2 (terms 7-10)
    # C02 = A00*B02 + A01*B12
    # C12 = A10*B02 + A11*B12
    
    # Term 7: A00 * B02
    U[0,7]=1; V[4,7]=1; W[4,7]=1
    # Term 8: A01 * B12
    U[1,8]=1; V[5,8]=1; W[4,8]=1
    # Term 9: A10 * B02
    U[2,9]=1; V[4,9]=1; W[5,9]=1
    # Term 10: A11 * B12
    U[3,10]=1; V[5,10]=1; W[5,10]=1
    
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    
    # Return the optimal rank-11 solution
    U, V, W = _rank_11_solution()
    
    # Verify the decomposition
    if not _verify(U, V, W):
        # Fallback to schoolbook if verification fails (should not happen)
        rank = 12
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
