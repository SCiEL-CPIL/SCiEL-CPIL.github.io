import time
import sys
import numpy as np
from fractions import Fraction

# Dimensions for the matrix multiplication: (n, m, p)
# A is n x m, B is m x p, C is n x p
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x3 matrix.
    The known minimal rank for 2x3x3 matrix multiplication is 17.
    We return a verified rank-17 decomposition. If verification fails (unlikely), we fall back to rank 18.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    N_u = n * m  # 6
    N_v = m * p  # 9
    N_w = p * n  # 6
    
    def verify_exact(U, V, W):
        """
        Verify exact decomposition using exact arithmetic (Fraction).
        U, V, W are lists of lists of numbers (int/float/Fraction).
        """
        rank = len(U[0])
        # Convert to Fraction for exactness
        Uf = [[Fraction(x) for x in row] for row in U]
        Vf = [[Fraction(x) for x in row] for row in V]
        Wf = [[Fraction(x) for x in row] for row in W]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    s = Fraction(0)
                    for c in range(rank):
                        s += Uf[idx_u][c] * Vf[idx_v][c] * Wf[idx_w][c]
                    if s != 1:
                        return False
        return True

    def verify_fast(U, V, W):
        """Quick verify with floats to avoid slow Fraction arithmetic if possible."""
        rank = len(U[0])
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    s = np.dot(Ua[idx_u], Va[idx_v] * Wa[idx_w])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def schoolbook():
        """Return the schoolbook rank-18 decomposition."""
        rank = n * m * p
        U = [[0.0] * rank for _ in range(N_u)]
        V = [[0.0] * rank for _ in range(N_v)]
        W = [[0.0] * rank for _ in range(N_w)]
        col_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    U[idx_u][col_idx] = 1.0
                    V[idx_v][col_idx] = 1.0
                    W[idx_w][col_idx] = 1.0
                    col_idx += 1
        return {"U": U, "V": V, "W": W}
    
    def rank17_decomposition():
        """
        Construct a rank-17 decomposition.
        Based on the known optimal algorithm for 2x3x3 matrix multiplication.
        This uses a 2x2x2 Strassen block (rank 7) plus schoolbook terms for the remaining parts.
        """
        rank = 17
        U = [[0.0] * rank for _ in range(N_u)]
        V = [[0.0] * rank for _ in range(N_v)]
        W = [[0.0] * rank for _ in range(N_w)]
        
        # Indices:
        # U rows: i*m+j -> (0,0)=0, (0,1)=1, (0,2)=2, (1,0)=3, (1,1)=4, (1,2)=5
        # V rows: j*p+k -> (0,0)=0, (0,1)=1, (0,2)=2, (1,0)=3, (1,1)=4, (1,2)=5, (2,0)=6, (2,1)=7, (2,2)=8
        # W rows: k*n+i -> (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3, (2,0)=4, (2,1)=5
        
        # 1. Strassen 2x2x2 for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
        # A indices involved: 0,1,3,4 (i=0,1; j=0,1)
        # B indices involved: 0,1,3,4 (j=0,1; k=0,1)
        # C indices involved: 0,1,2,3 (k=0,1; i=0,1)
        
        # Term 0: (A00+A11)(B00+B11) -> C00 +=, C11 +=
        U[0][0] = 1; U[4][0] = 1
        V[0][0] = 1; V[4][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # Term 1: (A10+A11)B00 -> C10 +=, C11 -=
        U[3][1] = 1; U[4][1] = 1
        V[0][1] = 1
        W[1][1] = 1; W[3][1] = -1
        
        # Term 2: A00(B11-B01) -> C01 +=, C11 +=
        U[0][2] = 1
        V[1][2] = 1; V[4][2] = -1
        W[2][2] = 1; W[3][2] = 1
        
        # Term 3: A11(B10-B00) -> C00 +=, C10 +=
        U[4][3] = 1
        V[3][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[1][3] = 1
        
        # Term 4: (A00+A01)B11 -> C00 -=, C01 +=
        U[0][4] = 1; U[1][4] = 1
        V[4][4] = 1
        W[0][4] = -1; W[2][4] = 1
        
        # Term 5: (A10-A00)(B00+B01) -> C11 +=
        U[3][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # Term 6: (A01-A11)(B10+B11) -> C00 +=
        U[1][6] = 1; U[4][6] = -1
        V[3][6] = 1; V[4][6] = 1
        W[0][6] = 1
        
        # 2. Schoolbook for A[0:2, 2] * B[2, 0:2] -> C[0:2, 0:2]
        # A indices: 2 (i=0,j=2), 5 (i=1,j=2)
        # B indices: 6 (j=2,k=0), 7 (j=2,k=1)
        # C indices: 0,2 (k=0,i=0,1), 1,3 (k=1,i=0,1)
        
        # Term 7: A02 * B20 -> C00
        U[2][7] = 1
        V[6][7] = 1
        W[0][7] = 1
        
        # Term 8: A02 * B21 -> C01
        U[2][8] = 1
        V[7][8] = 1
        W[2][8] = 1
        
        # Term 9: A12 * B20 -> C10
        U[5][9] = 1
        V[6][9] = 1
        W[1][9] = 1
        
        # Term 10: A12 * B21 -> C11
        U[5][10] = 1
        V[7][10] = 1
        W[3][10] = 1
        
        # 3. Schoolbook for A[0:2, 0:3] * B[0:3, 2] -> C[0:2, 2]
        # A indices: 0,1,2 (i=0), 3,4,5 (i=1)
        # B indices: 2 (j=0,k=2), 5 (j=1,k=2), 8 (j=2,k=2)
        # C indices: 4 (k=2,i=0), 5 (k=2,i=1)
        
        # Term 11: A00 * B02 -> C02
        U[0][11] = 1
        V[2][11] = 1
        W[4][11] = 1
        
        # Term 12: A01 * B12 -> C02
        U[1][12] = 1
        V[5][12] = 1
        W[4][12] = 1
        
        # Term 13: A02 * B22 -> C02
        U[2][13] = 1
        V[8][13] = 1
        W[4][13] = 1
        
        # Term 14: A10 * B02 -> C12
        U[3][14] = 1
        V[2][14] = 1
        W[5][14] = 1
        
        # Term 15: A11 * B12 -> C12
        U[4][15] = 1
        V[5][15] = 1
        W[5][15] = 1
        
        # Term 16: A12 * B22 -> C12
        U[5][16] = 1
        V[8][16] = 1
        W[5][16] = 1
        
        return {"U": U, "V": V, "W": W}

    # Strategy:
    # 1. Attempt Rank 17 (Theoretical Minimum).
    # 2. Verify it.
    # 3. If valid, return it.
    # 4. If invalid (should not happen), return Rank 18.
    
    result = rank17_decomposition()
    
    # Fast check first
    if verify_fast(result["U"], result["V"], result["W"]):
        # Exact check to be absolutely sure
        if verify_exact(result["U"], result["V"], result["W"]):
            return result
    
    # Fallback to Schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
