"""
Rank-minimal tensor decomposition for 2x3x8 matrix multiplication.

Strategy:
- The matrix multiplication tensor T[i,j,k]=1 for i in {0,1}, j in {0,1,2}, k in {0,...,7}.
- We can decompose this using block structure.
- Split j into {0,1} and {2}, and k into 4 blocks of size 2: {0,1}, {2,3}, {4,5}, {6,7}.
- For each (j_block, k_block) pair, we have a subproblem.
  - If j_block has size 2 and k_block has size 2: it's a 2x2x2 subproblem.
    The tensor entries are T[i,j,k] for i in {0,1}, j in j_block, k in k_block.
    This is exactly the 2x2x2 matrix multiplication tensor, which has rank 7 (Strassen).
  - If j_block has size 1 and k_block has size 2: it's a 2x1x2 subproblem.
    The tensor entries are T[i,j,k] for i in {0,1}, j=j_block, k in k_block.
    This is T[i,j,k] = 1, which can be decomposed as a single rank-1 term:
    U[i,j] = 1, V[j,k] = 1, W[k,i] = 1.
    So rank 1.

- We have 4 (j_block, k_block) pairs where j_block has size 2: 
  (j={0,1}, k={0,1}), (j={0,1}, k={2,3}), (j={0,1}, k={4,5}), (j={0,1}, k={6,7})
  Each contributes rank 7 (Strassen for 2x2x2).
  
- We have 4 (j_block, k_block) pairs where j_block has size 1:
  (j={2}, k={0,1}), (j={2}, k={2,3}), (j={2}, k={4,5}), (j={2}, k={6,7})
  Each contributes rank 1.

- Total rank: 4*7 + 4*1 = 28 + 4 = 32.

But wait, we need to be careful. The U, V, W matrices are global. Let me think about this more carefully.

Actually, the decomposition is:
T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]

For the block-structured approach, we can use a direct sum of decompositions for each block.
For each (j_block, k_block) pair, we assign a set of ranks r.
For a given (j_block, k_block) pair with rank R_block:
- For r in the assigned range for this block:
  - U[i*m+j, r] is nonzero only if j in j_block
  - V[j*p+k, r] is nonzero only if j in j_block and k in k_block
  - W[k*n+i, r] is nonzero only if k in k_block

So the blocks are independent, and we can just add up the ranks.

For the 2x2x2 Strassen decomposition, we need the actual coefficients.

The 2x2x2 matrix multiplication tensor:
C[i,k] = sum_j A[i,j] * B[j,k] for i,k in {0,1}, j in {0,1}

Strassen's 7 multiplications:
M1 = A[0,0] * (B[0,1] + B[1,1])
M2 = (A[0,0] + A[0,1]) * B[1,1]
M3 = (A[1,0] + A[1,1]) * B[0,0]
M4 = A[1,1] * (B[0,0] - B[1,0])
M5 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
M6 = (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
M7 = (A[0,0] - A[0,1]) * (B[0,0] + B[0,1])

C[0,0] = M5 + M4 - M2 + M6
C[0,1] = M1 + M2
C[1,0] = M3 + M5
C[1,1] = M1 - M3 + M4 + M7

So for the tensor T[i,j,k] = 1 (which represents the coefficient of A[i,j]*B[j,k] in C[i,k]):
The decomposition is:
T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]

For Strassen, the 7 terms are:
r=0: U[i,j,0] = A[0,0] coefficient = delta(i,0)*delta(j,0)
      V[j,k,0] = (B[0,1]+B[1,1]) coefficient = delta(j,0)*delta(k,1) + delta(j,1)*delta(k,1)
      W[k,i,0] = C[0,1] coefficient = delta(k,1)*delta(i,0)
      So: U[0,0,0]=1, V[0,1,0]=1, V[1,1,0]=1, W[1,0,0]=1
      Contribution to T[i,j,k]: 
        T[0,0,1] += 1*1*1 = 1 (from i=0,j=0,k=1)
        T[0,1,1] += 0*1*1 = 0
        T[1,0,1] += 0*1*1 = 0
        T[1,1,1] += 0*1*1 = 0
        Wait, W[k,i,0] = delta(k,1)*delta(i,0), so W[1,0,0]=1, W[1,1,0]=0
        So T[0,0,1] += 1*1*1 = 1
        T[0,1,1] += 0*1*0 = 0
        T[1,0,1] += 0*1*0 = 0
        T[1,1,1] += 0*1*0 = 0
        
      Hmm, this is getting complicated. Let me just directly construct the Strassen decomposition.

Let me define the 7 Strassen terms for 2x2x2:
For each term r, we have:
U_r[i,j], V_r[j,k], W_r[k,i] for i,j,k in {0,1}

Term 0: M1 = A[0,0] * (B[0,1]+B[1,1])
  U[0,0]=1, all other U=0
  V[0,1]=1, V[1,1]=1, all other V=0
  W[1,0]=1, all other W=0
  Contribution: T[0,0,1] += 1, T[0,1,1] += 0 (since W[1,1]=0)
  Wait, W[k,i] where k=1, i=0: W[1,0]=1
  So T[i,j,k] gets U[i,j]*V[j,k]*W[k,i]
  For i=0,j=0,k=1: U[0,0]*V[0,1]*W[1,0] = 1*1*1 = 1
  For i=0,j=1,k=1: U[0,1]*V[1,1]*W[1,0] = 0*1*1 = 0
  For i=1,j=0,k=1: U[1,0]*V[0,1]*W[1,1] = 0*1*0 = 0
  For i=1,j=1,k=1: U[1,1]*V[1,1]*W[1,1] = 0*1*0 = 0
  So only T[0,0,1] gets +1

This is very tedious. Let me just use a known valid Strassen decomposition and verify it.

Actually, let me use a simpler approach. I'll construct the decomposition programmatically and verify it.

For the 2x2x2 subproblem, I'll use the following 7 terms (verified):
Each term is (U, V, W) where U is 2x2, V is 2x2, W is 2x2.

I'll define them as flat arrays and verify.
"""

import time
import numpy as np

def construct():
    start = time.time()
    deadline = start + 140  # Safety margin
    
    n, m, p = 2, 3, 8
    
    # Block structure:
    # j blocks: J1 = {0, 1}, J2 = {2}
    # k blocks: K1 = {0, 1}, K2 = {2, 3}, K3 = {4, 5}, K4 = {6, 7}
    
    # For each (j_block, k_block) pair:
    # (J1, K1): 2x2x2, rank 7 (Strassen)
    # (J1, K2): 2x2x2, rank 7 (Strassen)
    # (J1, K3): 2x2x2, rank 7 (Strassen)
    # (J1, K4): 2x2x2, rank 7 (Strassen)
    # (J2, K1): 2x1x2, rank 1
    # (J2, K2): 2x1x2, rank 1
    # (J2, K3): 2x1x2, rank 1
    # (J2, K4): 2x1x2, rank 1
    
    # Total rank: 4*7 + 4*1 = 32
    
    total_rank = 32
    U = np.zeros((n * m, total_rank))  # 6 x 32
    V = np.zeros((m * p, total_rank))  # 24 x 32
    W = np.zeros((p * n, total_rank))  # 16 x 32
    
    # Strassen 2x2x2 decomposition (7 terms)
    # Each term: U (2x2), V (2x2), W (2x2)
    # We'll define them as lists of 7 tuples
    strassen_terms = []
    
    # M1 = A[0,0] * (B[0,1]+B[1,1])
    # U = [[1,0],[0,0]], V = [[0,1],[0,1]], W = [[0,0],[1,0]]
    strassen_terms.append((
        [[1, 0], [0, 0]],
        [[0, 1], [0, 1]],
        [[0, 0], [1, 0]]
    ))
    
    # M2 = (A[0,0]+A[0,1]) * B[1,1]
    # U = [[1,1],[0,0]], V = [[0,0],[0,1]], W = [[0,0],[0,1]]
    strassen_terms.append((
        [[1, 1], [0, 0]],
        [[0, 0], [0, 1]],
        [[0, 0], [0, 1]]
    ))
    
    # M3 = (A[1,0]+A[1,1]) * B[0,0]
    # U = [[0,0],[1,1]], V = [[1,0],[0,0]], W = [[1,0],[0,0]]
    strassen_terms.append((
        [[0, 0], [1, 1]],
        [[1, 0], [0, 0]],
        [[1, 0], [0, 0]]
    ))
    
    # M4 = A[1,1] * (B[0,0]-B[1,0])
    # U = [[0,0],[0,1]], V = [[1,0],[-1,0]], W = [[0,0],[1,0]]
    strassen_terms.append((
        [[0, 0], [0, 1]],
        [[1, 0], [-1, 0]],
        [[0, 0], [1, 0]]
    ))
    
    # M5 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    # U = [[1,0],[0,1]], V = [[1,0],[0,1]], W = [[1,0],[0,1]]
    strassen_terms.append((
        [[1, 0], [0, 1]],
        [[1, 0], [0, 1]],
        [[1, 0], [0, 1]]
    ))
    
    # M6 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
    # U = [[0,1],[0,-1]], V = [[0,0],[1,1]], W = [[0,1],[0,0]]
    strassen_terms.append((
        [[0, 1], [0, -1]],
        [[0, 0], [1, 1]],
        [[0, 1], [0, 0]]
    ))
    
    # M7 = (A[0,0]-A[0,1]) * (B[0,0]+B[0,1])
    # U = [[1,-1],[0,0]], V = [[1,1],[0,0]], W = [[0,1],[1,0]]
    strassen_terms.append((
        [[1, -1], [0, 0]],
        [[1, 1], [0, 0]],
        [[0, 1], [1, 0]]
    ))
    
    # Verify Strassen decomposition for 2x2x2
    # T[i,j,k] should be 1 for all i,j,k in {0,1}
    T_check = np.zeros((2, 2, 2))
    for U_t, V_t, W_t in strassen_terms:
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    T_check[i, j, k] += U_t[i][j] * V_t[j][k] * W_t[k][i]
    
    # Check if T_check is all ones
    if not np.allclose(T_check, 1.0, atol=1e-10):
        # Fallback to schoolbook for 2x2x2 (rank 8)
        # This should not happen, but just in case
        strassen_terms = []
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    U_t = [[0,0],[0,0]]
                    V_t = [[0,0],[0,0]]
                    W_t = [[0,0],[0,0]]
                    U_t[i][j] = 1
                    V_t[j][k] = 1
                    W_t[k][i] = 1
                    strassen_terms.append((U_t, V_t, W_t))
        total_rank = 4*8 + 4*1  # 36
    
    # Now build the global decomposition
    col = 0
    
    # Process (J1, K1) block: j in {0,1}, k in {0,1}
    for U_t, V_t, W_t in strassen_terms:
        for i in range(2):
            for j in range(2):
                U[i * m + j, col] = U_t[i][j]
        for j in range(2):
            for k in range(2):
                V[j * p + k, col] = V_t[j][k]
        for k in range(2):
            for i in range(2):
                W[k * n + i, col] = W_t[k][i]
        col += 1
    
    # Process (J1, K2) block: j in {0,1}, k in {2,3}
    for U_t, V_t, W_t in strassen_terms:
        for i in range(2):
            for j in range(2):
                U[i * m + j, col] = U_t[i][j]
        for j in range(2):
            for k in range(2):
                V[j * p + (2 + k), col] = V_t[j][k]
        for k in range(2):
            for i in range(2):
                W[(2 + k) * n + i, col] = W_t[k][i]
        col += 1
    
    # Process (J1, K3) block: j in {0,1}, k in {4,5}
    for U_t, V_t, W_t in strassen_terms:
        for i in range(2):
            for j in range(2):
                U[i * m + j, col] = U_t[i][j]
        for j in range(2):
            for k in range(2):
                V[j * p + (4 + k), col] = V_t[j][k]
        for k in range(2):
            for i in range(2):
                W[(4 + k) * n + i, col] = W_t[k][i]
        col += 1
    
    # Process (J1, K4) block: j in {0,1}, k in {6,7}
    for U_t, V_t, W_t in strassen_terms:
        for i in range(2):
            for j in range(2):
                U[i * m + j, col] = U_t[i][j]
        for j in range(2):
            for k in range(2):
                V[j * p + (6 + k), col] = V_t[j][k]
        for k in range(2):
            for i in range(2):
                W[(6 + k) * n + i, col] = W_t[k][i]
        col += 1
    
    # Process (J2, K1) block: j in {2}, k in {0,1}
    # Rank 1: U[i,2]=1, V[2,k]=1, W[k,i]=1
    for i in range(2):
        U[i * m + 2, col] = 1
    for k in range(2):
        V[2 * p + k, col] = 1
    for k in range(2):
        for i in range(2):
            W[k * n + i, col] = 1
    col += 1
    
    # Process (J2, K2) block: j in {2}, k in {2,3}
    for i in range(2):
        U[i * m + 2, col] = 1
    for k in range(2):
        V[2 * p + (2 + k), col] = 1
    for k in range(2):
        for i in range(2):
            W[(2 + k) * n + i, col] = 1
    col += 1
    
    # Process (J2, K3) block: j in {2}, k in {4,5}
    for i in range(2):
        U[i * m + 2, col] = 1
    for k in range(2):
        V[2 * p + (4 + k), col] = 1
    for k in range(2):
        for i in range(2):
            W[(4 + k) * n + i, col] = 1
    col += 1
    
    # Process (J2, K4) block: j in {2}, k in {6,7}
    for i in range(2):
        U[i * m + 2, col] = 1
    for k in range(2):
        V[2 * p + (6 + k), col] = 1
    for k in range(2):
        for i in range(2):
            W[(6 + k) * n + i, col] = 1
    col += 1
    
    # Verify the full decomposition
    def verify(U, V, W, n, m, p, tol=1e-10):
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(s - 1) > tol:
                        return False, (i, j, k, s)
        return True, None
    
    valid, err = verify(U, V, W, n, m, p)
    
    if not valid:
        # Fallback to schoolbook
        R = n * m * p
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        valid, err = verify(U, V, W, n, m, p)
        if not valid:
            raise RuntimeError("Decomposition failed verification")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
