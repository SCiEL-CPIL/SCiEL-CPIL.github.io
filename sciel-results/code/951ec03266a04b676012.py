"""
Rank-minimal exact tensor decomposition for 2x5 by 5x6 matrix multiplication.

The matrix multiplication tensor for C = A * B where A is (n x m) and B is (m x p)
has a schoolbook decomposition of rank n*m*p. For n=2, m=5, p=6 this is 60.

We search for lower-rank decompositions using structured ansatz based on
block decomposition and Strassen-like ideas.

Key insight: We can decompose the 2x5 matrix as [A1 | A2] where A1 is 2x2 and A2 is 2x3,
and the 5x6 matrix as [[B1],[B2]] where B1 is 2x6 and B2 is 3x6.
Then C = A1*B1 + A2*B2.

For the 2x2 by 2x6 multiplication, we can use a rank-9 decomposition per 2x2 block
(giving 9*3=27 for three 2x2 blocks if we partition the 6 columns into 3 pairs),
but that gives 27 + 3*2*6 = 27 + 36 = 63, which is worse.

Alternative approach: Use the fact that for 2xN by Nxp, the rank can be less than 2Np.
For 2x5 by 5x6, we try a direct rank-50 decomposition using a structured approach.

Actually, let me implement a generic search: start with schoolbook (rank 60) and try
to find a lower rank decomposition by solving for integer/half-integer entries.

Given the time constraint, I'll implement a structured approach:
1. Start with schoolbook rank 60.
2. Try to reduce by using the structure: the 2x5 * 5x6 can be seen as 
   computing 12 outputs, each is a sum of 5 products.
   
   For each output C[i,k] = sum_j A[i,j] * B[j,k].
   
   A rank-r decomposition means we find r "traces" where each trace computes 
   a product of a scalar from A-space, a scalar from B-space, and contributes 
   to the output.

Let me implement a rank-54 decomposition using a structured ansatz:
- Group the 5 middle indices into {0,1,2,3,4}
- Use 3 Strassen-like products for indices {0,1,2} (rank 3 instead of 4... no, Strassen is 7 for 2x2x2)

Actually, for a 2x3 by 3x6 multiplication, the schoolbook rank is 36.
For a 2x2 by 2x6 multiplication, the schoolbook rank is 24.
Total: 60.

To beat this, we need a more clever decomposition. Let me try:
- Use rank 9 for 2x2 by 2x2 (Strassen) per column pair
- 2x2 by 2x6: treat as 3 independent 2x2 by 2x2, each rank 7 -> 21
- 2x3 by 3x6: schoolbook 36
Total: 21 + 36 = 57

That's better! Let me verify:
A = [A1 | A2] where A1 is 2x2, A2 is 2x3
B = [[B1],[B2]] where B1 is 2x6, B2 is 3x6

C = A1*B1 + A2*B2

For A1*B1: partition B1 into 3 columns of width 2: B1 = [B1a | B1b | B1c]
Each A1*B1i is 2x2 * 2x2, use Strassen rank 7.
So A1*B1 has rank 3*7 = 21.

For A2*B2: 2x3 * 3x6, schoolbook rank = 36.

Total rank = 21 + 36 = 57.

Let me implement this.

Strassen for 2x2 * 2x2:
M1 = (a+d)(e+h)
M2 = (c+d)(e)
M3 = (a-b)(h)
M4 = (a+d)(h-f)
M5 = (a+c)(g-e)
M6 = (b+d)(f-g)
M7 = (a-c)(e+h)

Where A = [[a,b],[c,d]], B = [[e,f],[g,h]]
C11 = M1 - M4 + M5 + M7
C12 = M4 + M6
C21 = M2 + M5
C22 = M1 - M2 + M3 + M6
"""

import time
import random
from fractions import Fraction

DIMENSIONS = [2, 5, 6]

def _strassen_2x2(A, B):
    """
    Strassen decomposition for 2x2 * 2x2 -> rank 7.
    Returns (U, V, W) where U is (4, 7), V is (4, 7), W is (4, 7)
    such that U[i,k]*V[j,k]*W[k,l] sums to (A*B)[i,l]... 
    
    Actually, the tensor decomposition is:
    T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
    where T[i,j,k] = A[i,j] * B[j,k]... no.
    
    The matrix multiplication tensor is T[i,j,k] such that C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
    In a rank-r decomposition, T = sum_c U[i,c] * V[j,c] * W[k,c]
    
    For 2x2 * 2x2:
    A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    
    Strassen:
    M1 = (a+d)(e+h) -> contributes to C11, C22
    M2 = (c+d)(e)   -> contributes to C21, C22
    M3 = (a-b)(h)   -> contributes to C22
    M4 = (a+d)(h-f) -> contributes to C11, C12
    M5 = (a+c)(g-e) -> contributes to C11, C21
    M6 = (b+d)(f-g) -> contributes to C12, C22
    M7 = (a-c)(e+h) -> contributes to C11
    
    C11 = M1 - M4 + M5 + M7
    C12 = M4 + M6
    C21 = M2 + M5
    C22 = M1 - M2 + M3 + M6
    """
    # U[i, c] where i indexes A's entries (i: 0=a, 1=b, 2=c, 3=d)
    # V[j, c] where j indexes B's entries (j: 0=e, 1=f, 2=g, 3=h)
    # W[k, c] where k indexes C's entries (k: 0=C11, 1=C12, 2=C21, 3=C22)
    
    # c=0: M1 = (a+d)(e+h)
    # c=1: M2 = (c+d)(e)
    # c=2: M3 = (a-b)(h)
    # c=3: M4 = (a+d)(h-f)
    # c=4: M5 = (a+c)(g-e)
    # c=5: M6 = (b+d)(f-g)
    # c=6: M7 = (a-c)(e+h)
    
    U = [[0]*7, [0]*7, [0]*7, [0]*7]  # A: a,b,c,d
    V = [[0]*7, [0]*7, [0]*7, [0]*7]  # B: e,f,g,h
    W = [[0]*7, [0]*7, [0]*7, [0]*7]  # C: C11,C12,C21,C22
    
    # M1 = (a+d)(e+h)
    U[0][0] = 1  # a
    U[3][0] = 1  # d
    V[0][0] = 1  # e
    V[3][0] = 1  # h
    W[0][0] = 1  # C11
    W[3][0] = 1  # C22
    
    # M2 = (c+d)(e)
    U[2][1] = 1  # c
    U[3][1] = 1  # d
    V[0][1] = 1  # e
    W[2][1] = 1  # C21
    W[3][1] = -1 # C22
    
    # M3 = (a-b)(h)
    U[0][2] = 1  # a
    U[1][2] = -1 # b
    V[3][2] = 1  # h
    W[3][2] = 1  # C22
    
    # M4 = (a+d)(h-f)
    U[0][3] = 1  # a
    U[3][3] = 1  # d
    V[3][3] = 1  # h
    V[1][3] = -1 # f
    W[0][3] = -1 # C11
    W[1][3] = 1  # C12
    
    # M5 = (a+c)(g-e)
    U[0][4] = 1  # a
    U[2][4] = 1  # c
    V[2][4] = 1  # g
    V[0][4] = -1 # e
    W[0][4] = 1  # C11
    W[2][4] = 1  # C21
    
    # M6 = (b+d)(f-g)
    U[1][5] = 1  # b
    U[3][5] = 1  # d
    V[1][5] = 1  # f
    V[2][5] = -1 # g
    W[1][5] = 1  # C12
    W[3][5] = 1  # C22
    
    # M7 = (a-c)(e+h)
    U[0][6] = 1  # a
    U[2][6] = -1 # c
    V[0][6] = 1  # e
    V[3][6] = 1  # h
    W[0][6] = 1  # C11
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    
    # Strategy: Split middle index j into two groups: {0,1} and {2,3,4}
    # A = [A1 | A2] where A1 is 2x2, A2 is 2x3
    # B = [[B1],[B2]] where B1 is 2x6, B2 is 3x6
    # C = A1*B1 + A2*B2
    #
    # A1*B1: 2x2 * 2x6. Partition B1 into 3 blocks of 2 columns.
    #   Each 2x2 * 2x2 uses Strassen (rank 7). Total: 3*7 = 21.
    # A2*B2: 2x3 * 3x6. Schoolbook: 2*3*6 = 36.
    # Total: 21 + 36 = 57.
    
    # Indices:
    # A has indices (i, j) where i in {0,1}, j in {0,1,2,3,4}
    #   A1: j in {0,1}, A2: j in {2,3,4}
    # B has indices (j, k) where j in {0,1,2,3,4}, k in {0,1,2,3,4,5}
    #   B1: j in {0,1}, B2: j in {2,3,4}
    # C has indices (i, k) where i in {0,1}, k in {0,1,2,3,4,5}
    #   In W, index is k*n + i = 2k + i
    
    # U has shape (n*m, rank) = (10, rank)
    #   U[i*m + j, c] where i in {0,1}, j in {0,1,2,3,4}
    # V has shape (m*p, rank) = (30, rank)
    #   V[j*p + k, c] where j in {0,1,2,3,4}, k in {0,1,2,3,4,5}
    # W has shape (p*n, rank) = (12, rank)
    #   W[k*n + i, c] = W[2k + i, c] where k in {0,1,2,3,4,5}, i in {0,1}
    
    # Strassen part: 21 terms for A1*B1
    # A1*B1 = A1 * [B1a | B1b | B1c] where each B1x is 2x2
    # B1a: k in {0,1}, B1b: k in {2,3}, B1c: k in {4,5}
    
    sU, sV, sW = _strassen_2x2(None, None)
    
    rank = 57
    U = [[0] * rank for _ in range(10)]
    V = [[0] * rank for _ in range(30)]
    W = [[0] * rank for _ in range(12)]
    
    col = 0
    # For each block of 2 columns (col_block = 0, 1, 2)
    for col_block in range(3):
        # Column offset for this block: 2*col_block, 2*col_block+1
        for c in range(7):
            # Strassen term c for this column block
            # U: A1 entries (i in {0,1}, j in {0,1})
            #   A1[i,j] -> U[i*5 + j, col]
            # V: B1 entries (j in {0,1}, k in {2*col_block, 2*col_block+1})
            #   B1[j,k] -> V[j*6 + k, col]
            # W: C entries (i in {0,1}, k in {2*col_block, 2*col_block+1})
            #   C[i,k] -> W[2k + i, col]
            
            # sU has 4 rows: index 0=a(A[0,0]), 1=b(A[0,1]), 2=c(A[1,0]), 3=d(A[1,1])
            # sV has 4 rows: index 0=e(B[0,0]), 1=f(B[0,1]), 2=g(B[1,0]), 3=h(B[1,1])
            # sW has 4 rows: index 0=C11, 1=C12, 2=C21, 3=C22
            
            for a_idx in range(4):
                if sU[a_idx][c] != 0:
                    # Map a_idx to (i, j) in A1
                    i = a_idx // 2  # 0 or 1
                    j = a_idx % 2   # 0 or 1
                    U[i * 5 + j][col] = sU[a_idx][c]
            
            for b_idx in range(4):
                if sV[b_idx][c] != 0:
                    # Map b_idx to (j, k_offset) in B1
                    j = b_idx // 2   # 0 or 1
                    k_off = b_idx % 2 # 0 or 1
                    k = 2 * col_block + k_off
                    V[j * 6 + k][col] = sV[b_idx][c]
            
            for w_idx in range(4):
                if sW[w_idx][c] != 0:
                    # Map w_idx to (i, k_offset) in C
                    i = w_idx // 2    # 0 or 1
                    k_off = w_idx % 2 # 0 or 1
                    k = 2 * col_block + k_off
                    W[2 * k + i][col] = sW[w_idx][c]
            
            col += 1
    
    # Schoolbook part: A2*B2, 2x3 * 3x6, rank 36
    # A2: i in {0,1}, j in {2,3,4} -> U[i*5+j, col]
    # B2: j in {2,3,4}, k in {0,1,2,3,4,5} -> V[j*6+k, col]
    # C: i in {0,1}, k in {0,1,2,3,4,5} -> W[2k+i, col]
    
    for i in range(2):
        for j in range(3):  # j=0,1,2 corresponds to actual j=2,3,4
            actual_j = j + 2
            for k in range(6):
                U[i * 5 + actual_j][col] = 1
                V[actual_j * 6 + k][col] = 1
                W[2 * k + i][col] = 1
                col += 1
    
    assert col == rank, f"Column mismatch: {col} vs {rank}"
    
    return {"U": U, "V": V, "W": W}


if __name__ == "__main__":
    import numpy as np
    t0 = time.time()
    result = construct()
    U = np.array(result["U"], dtype=float)
    V = np.array(result["V"], dtype=float)
    W = np.array(result["W"], dtype=float)
    
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    
    # Verify: For random A (n x m) and B (m x p), check C = A*B
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Decomposition: C[i,k] = sum_c U[i*m+j, c] * A[i,j] * V[j*p+k, c] * B[j,k] * W[k*n+i, c]
    # Actually: C[i,k] = sum_c (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k]) * W[k*n+i, c]
    
    # Wait, the decomposition is:
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # And C[i,k] = sum_j A[i,j] * B[j,k] * T[i,j,k]
    #           = sum_j A[i,j] * B[j,k] * sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    #           = sum_c W[k*n+i, c] * (sum_j A[i,j] * U[i*m+j, c]) * (sum_j B[j,k] * V[j*p+k, c])
    
    # Let's verify this
    np.random.seed(42)
    for trial in range(10):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_true = A @ B
        
        C_decomp = np.zeros((n, p))
        for c in range(rank):
            # sum_j A[i,j] * U[i*m+j, c] for each i
            Au = np.zeros(n)
            for i in range(n):
                for j in range(m):
                    Au[i] += A[i, j] * U[i * m + j, c]
            
            # sum_j B[j,k] * V[j*p+k, c] for each k
            Bv = np.zeros(p)
            for k in range(p):
                for j in range(m):
                    Bv[k] += B[j, k] * V[j * p + k, c]
            
            for i in range(n):
                for k in range(p):
                    C_decomp[i, k] += W[k * n + i, c] * Au[i] * Bv[k]
        
        diff = np.max(np.abs(C_true - C_decomp))
        if diff > 1e-10:
            print(f"Trial {trial}: max diff = {diff}")
            break
    else:
        print(f"All trials passed. Rank = {rank}, time = {time.time() - t0:.3f}s")
