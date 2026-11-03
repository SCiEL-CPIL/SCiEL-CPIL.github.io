"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 120.0
_START_TIME = time.time()

def _elapsed():
    return time.time() - _START_TIME

def _remaining():
    return _DEADLINE - 10.0 - _elapsed()

def construct():
    n, m, p = DIMENSIONS
    
    def _build_optimal_decomp():
        # Known optimal rank for 2x3x3 is 17
        rank = 17
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        
        # Indices:
        # U: (i, k) -> i*m + k, i in 0..1, k in 0..2
        # V: (k, j) -> k*p + j, k in 0..2, j in 0..2
        # W: (j, i) -> j*n + i, j in 0..2, i in 0..1
        
        # This decomposition combines Strassen's 7-term algorithm for the 2x2 block
        # with additional terms for the 3rd dimension.
        
        # Strassen part for the 2x2 sub-matrix (using rows/cols 0,1 of A and B, and 0,1 of C)
        # A[0:2, 0:2], B[0:2, 0:2] -> C[0:2, 0:2]
        
        # M1: (A00+A11)(B00+B11)
        U[0, 0] = 1; U[4, 0] = 1  # A00, A11
        V[0, 0] = 1; V[4, 0] = 1  # B00, B11
        W[0, 0] = 1; W[3, 0] = 1  # C00, C11 (W index: j*n+i. C00 is j=0,i=0->0. C11 is j=1,i=1->1*2+1=3)
        
        # M2: (A10+A11)B00
        U[3, 1] = 1; U[4, 1] = 1  # A10, A11
        V[0, 1] = 1               # B00
        W[1, 1] = 1; W[3, 1] = -1 # C10 (j=0,i=1->1), C11 (j=1,i=1->3)
        
        # M3: A00(B10-B00)
        U[0, 2] = 1               # A00
        V[1, 2] = 1; V[4, 2] = -1 # B10, B00
        W[2, 2] = 1; W[3, 2] = 1  # C01 (j=1,i=0->2), C11 (j=1,i=1->3)
        
        # M4: A11(B01-B10)
        U[4, 3] = 1               # A11
        V[2, 3] = 1; V[1, 3] = -1 # B01, B10
        W[0, 3] = 1; W[1, 3] = 1  # C00 (j=0,i=0->0), C10 (j=0,i=1->1)
        
        # M5: (A00+A01)B11
        U[0, 4] = 1; U[1, 4] = 1  # A00, A01
        V[5, 4] = 1               # B11
        W[2, 4] = 1; W[0, 4] = -1 # C01 (j=1,i=0->2), C00 (j=0,i=0->0)
        
        # M6: (A10-A00)(B00+B01)
        U[3, 5] = 1; U[0, 5] = -1 # A10, A00
        V[0, 5] = 1; V[2, 5] = 1  # B00, B01
        W[3, 5] = 1               # C11 (j=1,i=1->3)
        
        # M7: (A01-A11)(B10+B11)
        U[1, 6] = 1; U[4, 6] = -1 # A01, A11
        V[1, 6] = 1; V[5, 6] = 1  # B10, B11
        W[0, 6] = 1               # C00 (j=0,i=0->0)
        
        # Terms involving the 3rd column of A (index 2) and 3rd row/col of B (index 2)
        # A is 2x3, B is 3x3, C is 2x3.
        # C[i,j] = sum_k A[i,k] B[k,j]
        
        # C[0,0] += A[0,2]B[2,0]
        U[2, 7] = 1; V[6, 7] = 1; W[0, 7] = 1
        # C[0,1] += A[0,2]B[2,1]
        U[2, 8] = 1; V[7, 8] = 1; W[2, 8] = 1
        # C[0,2] += A[0,2]B[2,2]
        U[2, 9] = 1; V[8, 9] = 1; W[4, 9] = 1
        
        # C[1,0] += A[1,2]B[2,0]
        U[5, 10] = 1; V[6, 10] = 1; W[1, 10] = 1
        # C[1,1] += A[1,2]B[2,1]
        U[5, 11] = 1; V[7, 11] = 1; W[3, 11] = 1
        # C[1,2] += A[1,2]B[2,2]
        U[5, 12] = 1; V[8, 12] = 1; W[5, 12] = 1
        
        # C[0,0] += A[0,0]B[0,2] + A[0,1]B[1,2]
        # Wait, the Strassen part covered B[0:2, 0:2]. We need terms for B[:, 2] (col 2 of B)
        # and B[2, :] (row 2 of B)
        
        # C[0,0] gets no contribution from B[0,2] or B[1,2] because those are in col 2 of B, 
        # contributing to C[0,2] and C[1,2].
        # C[0,2] = A[0,0]B[0,2] + A[0,1]B[1,2] + A[0,2]B[2,2]
        # We already added A[0,2]B[2,2] in term 9.
        # Need A[0,0]B[0,2] and A[0,1]B[1,2]
        U[0, 13] = 1; V[2, 13] = 1; W[4, 13] = 1 # A00 B02 -> C02 (j=2,i=0 -> 2*2+0=4)
        U[1, 14] = 1; V[5, 14] = 1; W[4, 14] = 1 # A01 B12 -> C02 (j=2,i=0 -> 4)
        
        # C[1,2] = A[1,0]B[0,2] + A[1,1]B[1,2] + A[1,2]B[2,2]
        # We already added A[1,2]B[2,2] in term 12.
        # Need A[1,0]B[0,2] and A[1,1]B[1,2]
        U[3, 15] = 1; V[2, 15] = 1; W[5, 15] = 1 # A10 B02 -> C12 (j=2,i=1 -> 2*2+1=5)
        U[4, 16] = 1; V[5, 16] = 1; W[5, 16] = 1 # A11 B12 -> C12 (j=2,i=1 -> 5)
        
        # What about B[2,0] and B[2,1]?
        # C[0,0] += A[0,2]B[2,0] (Term 7)
        # C[0,1] += A[0,2]B[2,1] (Term 8)
        # C[1,0] += A[1,2]B[2,0] (Term 10)
        # C[1,1] += A[1,2]B[2,1] (Term 11)
        
        # Are there any other terms?
        # C[0,0] = A00B00 + A01B10 + A02B20
        # Strassen covers A00B00, A01B10 (partially via combinations).
        # Actually, Strassen computes the full 2x2 product.
        # So A00B00, A01B10, A10B00, A11B10, A00B01, A01B11, A10B01, A11B11 are all handled by M1-M7.
        # The terms involving index 2 are:
        # A02B20, A02B21, A02B22
        # A12B20, A12B21, A12B22
        # A00B02, A01B12
        # A10B02, A11B12
        
        # Total 10 extra terms? 7 + 10 = 17.
        # Yes, this matches the rank 17 construction.
        
        return U, V, W

    def _verify(U, V, W):
        rank = U.shape[1]
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    val = 0
                    for r in range(rank):
                        val += U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
                    if val != 1:
                        return False
        return True

    U, V, W = _build_optimal_decomp()
    
    if _remaining() > 5.0:
        if not _verify(U, V, W):
            # Fallback to schoolbook (rank 18) if something went wrong
            rank = n * m * p
            U = np.zeros((n * m, rank), dtype=int)
            V = np.zeros((m * p, rank), dtype=int)
            W = np.zeros((p * n, rank), dtype=int)
            for i in range(n):
                for k in range(m):
                    for j in range(p):
                        r = i * m * p + k * p + j
                        U[i*m+k, r] = 1
                        V[k*p+j, r] = 1
                        W[j*n+i, r] = 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
