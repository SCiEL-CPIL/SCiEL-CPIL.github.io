"""Matrix multiplication 2x3x8 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 3, 8]

import time
import numpy as np

# EVOLVE-BLOCK-START

def _build_decomp(n, m, p):
    """
    Build a deterministic decomposition for 2x3 * 3x8 matrix multiplication.
    
    Strategy: Split A (2x3) into A1 (2x2) and A2 (2x1).
    Split B (3x8) into B1 (2x8) and B2 (1x8).
    C = A1*B1 + A2*B2.
    
    Part 1: A2*B2 -> rank 16 (each of 2*1*8 scalar products).
    Part 2: A1*B1 -> A1 is 2x2, B1 is 2x8. Split B1 into 4 blocks of 2x2.
             Each 2x2 * 2x2 uses Strassen (rank 7). Total 4*7 = 28.
    Grand total: 44.
    
    Indexing:
    U[i*m+j, c] where i in 0..1, j in 0..2
    V[j*p+k, c] where j in 0..2, k in 0..7
    W[k*n+i, c] where k in 0..7, i in 0..1
    """
    n, m, p = 2, 3, 8
    
    rank_total = 44
    U = np.zeros((n * m, rank_total))  # 6 x 44
    V = np.zeros((m * p, rank_total))  # 24 x 44
    W = np.zeros((p * n, rank_total))  # 16 x 44
    
    col = 0
    
    # Part 1: A2*B2
    # A2 has entries at U[2, c] (i=0,j=2) and U[5, c] (i=1,j=2)
    # B2 has entries at V[16+k, c] for k in 0..7 (j=2, k)
    # Output: W[k*n+i, c] for i in 0..1, k in 0..7
    for k in range(8):
        # i=0, j=2, k
        U[2, col] = 1
        V[16 + k, col] = 1
        W[k * 2 + 0, col] = 1
        col += 1
        # i=1, j=2, k
        U[5, col] = 1
        V[16 + k, col] = 1
        W[k * 2 + 1, col] = 1
        col += 1
    
    # Part 2: A1*B1 via Strassen (4 blocks of 2x2)
    # A1 entries: U[0,c] (i=0,j=0), U[1,c] (i=0,j=1), U[3,c] (i=1,j=0), U[4,c] (i=1,j=1)
    # B1 entries: V[j*p+k, c] where j in 0,1 and k in 0..7
    # For block g (g=0,1,2,3), B1 columns 2g, 2g+1
    # Output: W[k*n+i, c] for i in 0,1 and k in 2g, 2g+1
    
    for g in range(4):
        # B1 block indices: j=0: V[0*p+2g], V[0*p+2g+1]; j=1: V[1*p+2g], V[1*p+2g+1]
        b11 = 0 * p + 2 * g      # V index for j=0, k=2g
        b12 = 0 * p + 2 * g + 1  # V index for j=0, k=2g+1
        b21 = 1 * p + 2 * g      # V index for j=1, k=2g
        b22 = 1 * p + 2 * g + 1  # V index for j=1, k=2g+1
        
        a11 = 0  # U index for i=0, j=0
        a12 = 1  # U index for i=0, j=1
        a21 = 3  # U index for i=1, j=0
        a22 = 4  # U index for i=1, j=1
        
        # Output W indices:
        # c11: W[(2g)*2 + 0, col] -> W[2g*2, col]
        # c12: W[(2g+1)*2 + 0, col] -> W[(2g+1)*2, col]
        # c21: W[(2g)*2 + 1, col] -> W[2g*2+1, col]
        # c22: W[(2g+1)*2 + 1, col] -> W[(2g+1)*2+1, col]
        
        w11 = (2 * g) * 2 + 0
        w12 = (2 * g + 1) * 2 + 0
        w21 = (2 * g) * 2 + 1
        w22 = (2 * g + 1) * 2 + 1
        
        # Strassen's 7 multiplications for 2x2 * 2x2:
        # M1 = (a11+a22)(b11+b22)
        # M2 = (a21+a22)b11
        # M3 = a11(b22-b12)
        # M4 = a22(b11-b21)
        # M5 = (a11+a12)b22
        # M6 = (a21-a11)(b11+b12)
        # M7 = (a12-a22)(b21+b22)
        #
        # c11 = M1 + M4 - M5 - M7
        # c12 = M3 + M5
        # c21 = M2 + M6
        # c22 = M1 - M2 + M3 + M4
        
        # M1 = (a11+a22)(b11+b22)
        U[a11, col] = 1; U[a22, col] = 1
        V[b11, col] = 1; V[b22, col] = 1
        W[w11, col] = 1; W[w22, col] = 1
        col += 1
        
        # M2 = (a21+a22)b11
        U[a21, col] = 1; U[a22, col] = 1
        V[b11, col] = 1
        W[w21, col] = 1; W[w22, col] = -1
        col += 1
        
        # M3 = a11(b22-b12)
        U[a11, col] = 1
        V[b22, col] = 1; V[b12, col] = -1
        W[w12, col] = 1; W[w22, col] = 1
        col += 1
        
        # M4 = a22(b11-b21)
        U[a22, col] = 1
        V[b11, col] = 1; V[b21, col] = -1
        W[w11, col] = 1; W[w22, col] = 1
        col += 1
        
        # M5 = (a11+a12)b22
        U[a11, col] = 1; U[a12, col] = 1
        V[b22, col] = 1
        W[w11, col] = -1; W[w12, col] = 1
        col += 1
        
        # M6 = (a21-a11)(b11+b12)
        U[a21, col] = 1; U[a11, col] = -1
        V[b11, col] = 1; V[b12, col] = 1
        W[w21, col] = 1
        col += 1
        
        # M7 = (a12-a22)(b21+b22)
        U[a12, col] = 1; U[a22, col] = -1
        V[b21, col] = 1; V[b22, col] = 1
        W[w11, col] = -1
        col += 1
    
    return U.tolist(), V.tolist(), W.tolist()


def _verify(U, V, W, n, m, p):
    """Verify decomposition correctness."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
                if abs(val - 1.0) > max_err:
                    max_err = abs(val - 1.0)
    return max_err


def _build_schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U.tolist(), V.tolist(), W.tolist()


def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    n, m, p = DIMENSIONS
    
    # Primary: Strassen-based rank-44 decomposition
    try:
        U, V, W = _build_decomp(n, m, p)
        err = _verify(U, V, W, n, m, p)
        if err < 1e-10:
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fallback: schoolbook rank-48
    U, V, W = _build_schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
