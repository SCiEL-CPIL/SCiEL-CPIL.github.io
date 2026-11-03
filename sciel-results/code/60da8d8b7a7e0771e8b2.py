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
    n, m, p = DIMENSIONS  # 2, 3, 3

    def _build_rank_17():
        """
        Construct a rank-17 exact decomposition for 2x3 x 3x3 matrix multiplication.
        
        Indexing convention:
        - U row index: i*m + k  (i = row of A, k = inner index)
        - V row index: k*p + j  (k = inner index, j = col of B)
        - W row index: j*n + i  (j = col of C, i = row of C)
        
        The tensor equation is:
        For all i in [0,n), k in [0,m), j in [0,p):
            sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] = 1
        """
        rank = 17
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        
        # We decompose the 2x3x3 multiplication into:
        # 1) A 2x2x2 Strassen-like block for the first 2 columns of B (j=0,1) 
        #    and first 2 rows of A (i=0,1), using inner index k=0,1
        # 2) Schoolbook terms for the remaining cases involving j=2 or k=2
        
        # The full set of (i,k,j) tuples: i in {0,1}, k in {0,1,2}, j in {0,1,2}
        # Total 12 terms.
        
        # Let's use a hybrid approach:
        # For the submatrix where i in {0,1}, k in {0,1}, j in {0,1}: use 7 Strassen terms
        # For the remaining terms (involving k=2 or j=2): use schoolbook
        
        # Strassen terms for i in {0,1}, k in {0,1}, j in {0,1}:
        # These contribute to C[i,j] for i in {0,1}, j in {0,1}
        
        # M1: (A00+A11)(B00+B11) -> contributes to C00, C11
        # Actually, let me use the standard 2x2 Strassen adapted to our indexing.
        
        # Standard 2x2 Strassen for A=[[a,b],[c,d]], B=[[e,f],[g,h]]:
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        # C00 = M1+M4-M5+M7
        # C01 = M3+M5
        # C10 = M2+M4
        # C11 = M1-M2+M3+M6
        
        # In our case, for the 2x2 block (i in {0,1}, k in {0,1}, j in {0,1}):
        # A block: [[A[0,0], A[0,1]], [A[1,0], A[1,1]]]
        # B block: [[B[0,0], B[0,1]], [B[1,0], B[1,1]]]
        
        # Let me define the 7 Strassen terms:
        
        # r=0: M1 = (A00+A11)(B00+B11)
        # U: A00 and A11 -> U[0*3+0, 0]=1, U[1*3+1, 0]=1
        # V: B00 and B11 -> V[0*3+0, 0]=1, V[1*3+1, 0]=1
        # W: contributes to C00 and C11 -> W[0*2+0, 0]=1 (j=0,i=0), W[1*2+1, 0]=1 (j=1,i=1)
        U[0*3+0, 0] = 1; U[1*3+1, 0] = 1
        V[0*3+0, 0] = 1; V[1*3+1, 0] = 1
        W[0*2+0, 0] = 1; W[1*2+1, 0] = 1
        
        # r=1: M2 = (A10+A11)B00
        # U: A10, A11 -> U[1*3+0, 1]=1, U[1*3+1, 1]=1
        # V: B00 -> V[0*3+0, 1]=1
        # W: contributes to C10 and C11 with signs
        # C10 gets +M2, C11 gets -M2
        # W[0*2+1, 1]=1 (j=0,i=1), W[1*2+1, 1]=-1 (j=1,i=1)
        U[1*3+0, 1] = 1; U[1*3+1, 1] = 1
        V[0*3+0, 1] = 1
        W[0*2+1, 1] = 1; W[1*2+1, 1] = -1
        
        # r=2: M3 = A00(B01-B11)
        # U: A00 -> U[0*3+0, 2]=1
        # V: B01, B11 -> V[0*3+1, 2]=1, V[1*3+1, 2]=-1
        # W: contributes to C01 and C11
        # C01 gets +M3, C11 gets +M3
        W[1*2+0, 2] = 1 (j=1,i=0); W[1*2+1, 2] = 1 (j=1,i=1)
        U[0*3+0, 2] = 1
        V[0*3+1, 2] = 1; V[1*3+1, 2] = -1
        W[1*2+0, 2] = 1; W[1*2+1, 2] = 1
        
        # r=3: M4 = A11(B10-B00)
        # U: A11 -> U[1*3+1, 3]=1
        # V: B10, B00 -> V[1*3+0, 3]=1, V[0*3+0, 3]=-1
        # W: contributes to C00 and C10
        # C00 gets +M4, C10 gets +M4
        U[1*3+1, 3] = 1
        V[1*3+0, 3] = 1; V[0*3+0, 3] = -1
        W[0*2+0, 3] = 1; W[0*2+1, 3] = 1
        
        # r=4: M5 = (A00+A01)B11
        # U: A00, A01 -> U[0*3+0, 4]=1, U[0*3+1, 4]=1
        # V: B11 -> V[1*3+1, 4]=1
        # W: contributes to C01 and C00
        # C01 gets +M5, C00 gets -M5
        U[0*3+0, 4] = 1; U[0*3+1, 4] = 1
        V[1*3+1, 4] = 1
        W[1*2+0, 4] = 1; W[0*2+0, 4] = -1
        
        # r=5: M6 = (A10-A00)(B00+B01)
        # U: A10, A00 -> U[1*3+0, 5]=1, U[0*3+0, 5]=-1
        # V: B00, B01 -> V[0*3+0, 5]=1, V[0*3+1, 5]=1
        # W: contributes to C11
        # C11 gets +M6
        U[1*3+0, 5] = 1; U[0*3+0, 5] = -1
        V[0*3+0, 5] = 1; V[0*3+1, 5] = 1
        W[1*2+1, 5] = 1
        
        # r=6: M7 = (A01-A11)(B10+B11)
        # U: A01, A11 -> U[0*3+1, 6]=1, U[1*3+1, 6]=-1
        # V: B10, B11 -> V[1*3+0, 6]=1, V[1*3+1, 6]=1
        # W: contributes to C00
        # C00 gets +M7
        U[0*3+1, 6] = 1; U[1*3+1, 6] = -1
        V[1*3+0, 6] = 1; V[1*3+1, 6] = 1
        W[0*2+0, 6] = 1
        
        # Now for the remaining terms involving k=2 or j=2:
        # We need to handle:
        # (i,k,j) where k=2 or j=2 (and not both already covered)
        
        # Terms with j=2 (any i, k):
        # (0,0,2): A00*B02 -> C02
        # (0,1,2): A01*B12 -> C02
        # (0,2,2): A02*B22 -> C02
        # (1,0,2): A10*B02 -> C12
        # (1,1,2): A11*B12 -> C12
        # (1,2,2): A12*B22 -> C12
        
        # Terms with k=2, j in {0,1}:
        # (0,2,0): A02*B20 -> C00
        # (0,2,1): A02*B21 -> C01
        # (1,2,0): A12*B20 -> C10
        # (1,2,1): A12*B21 -> C11
        
        # That's 6 + 4 = 10 more terms, total 17.
        
        # r=7: (0,0,2) A00*B02 -> C02
        U[0*3+0, 7] = 1; V[0*3+2, 7] = 1; W[2*2+0, 7] = 1
        # r=8: (0,1,2) A01*B12 -> C02
        U[0*3+1, 8] = 1; V[1*3+2, 8] = 1; W[2*2+0, 8] = 1
        # r=9: (0,2,2) A02*B22 -> C02
        U[0*3+2, 9] = 1; V[2*3+2, 9] = 1; W[2*2+0, 9] = 1
        # r=10: (1,0,2) A10*B02 -> C12
        U[1*3+0, 10] = 1; V[0*3+2, 10] = 1; W[2*2+1, 10] = 1
        # r=11: (1,1,2) A11*B12 -> C12
        U[1*3+1, 11] = 1; V[1*3+2, 11] = 1; W[2*2+1, 11] = 1
        # r=12: (1,2,2) A12*B22 -> C12
        U[1*3+2, 12] = 1; V[2*3+2, 12] = 1; W[2*2+1, 12] = 1
        # r=13: (0,2,0) A02*B20 -> C00
        U[0*3+2, 13] = 1; V[2*3+0, 13] = 1; W[0*2+0, 13] = 1
        # r=14: (0,2,1) A02*B21 -> C01
        U[0*3+2, 14] = 1; V[2*3+1, 14] = 1; W[1*2+0, 14] = 1
        # r=15: (1,2,0) A12*B20 -> C10
        U[1*3+2, 15] = 1; V[2*3+0, 15] = 1; W[0*2+1, 15] = 1
        # r=16: (1,2,1) A12*B21 -> C11
        U[1*3+2, 16] = 1; V[2*3+1, 16] = 1; W[1*2+1, 16] = 1
        
        return U, V, W

    def _verify(U, V, W):
        """Verify the decomposition."""
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

    U, V, W = _build_rank_17()
    
    if _remaining() > 5.0:
        if not _verify(U, V, W):
            # Fallback to schoolbook rank 18
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
