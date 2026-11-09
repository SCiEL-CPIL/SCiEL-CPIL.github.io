"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import combinations

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                # Indices:
                # U[i*m + j] = 1
                # V[j*p + k] = 1
                # W[k*n + i] = 1
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # Target Tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T_target = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i * m + j, j * p + k, k * n + i] = 1.0

    # We want to find a rank-17 decomposition.
    # The schoolbook has rank 18.
    # A rank-17 decomposition can be viewed as 16 standard schoolbook terms 
    # plus 1 "merged" term that accounts for the remaining 2 schoolbook contributions,
    # OR a more complex mix.
    
    # However, a simpler and more robust strategy for small dimensions:
    # We know the rank is 17.
    # Let's try to construct it by taking the schoolbook 18 terms and "merging" two of them
    # into a single rank-1 term if possible? 
    # Merging two rank-1 terms into one rank-1 term is generally impossible unless they are identical or one is zero.
    # So we cannot just merge two schoolbook terms into one.
    
    # Instead, we can replace a *subset* of schoolbook terms with a different set of terms.
    # For example, replace 2 schoolbook terms with 1 term (impossible as above) 
    # or replace 3 schoolbook terms with 2 terms (possible if the sub-tensor of those 3 has rank 2).
    # Or replace 4 with 3, etc.
    
    # The known rank-17 solution for <2,3,3> often uses Strassen on a 2x2x2 sub-block.
    # Let's systematically try all partitions of the indices (i,j,k) into a "Strassen block" and "rest".
    
    # Actually, let's just hardcode the known valid rank-17 decomposition derived from Strassen + Schoolbook,
    # but with a rigorous verification step. If the hardcoded one fails (due to index errors), 
    # we will fall back to a search.
    
    # Given the previous attempt had bugs, let's re-derive the Strassen part carefully.
    # Strassen for 2x2 * 2x2:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = [[ae+cg, af+dh], [ce+dg, cf+gh]]
    
    # M1 = a(e+h)
    # M2 = (a+c)h
    # M3 = d(e+f)
    # M4 = b(g+h)
    # M5 = (a+b)e
    # M6 = (d-b)(e+f)
    # M7 = (c-a)(g+h)
    
    # C11 = M1 - M2 + M4
    # C12 = M3 + M4
    # C21 = M1 + M5
    # C22 = M1 - M5 + M2
    
    # We apply this to the sub-block A[0:2, 0:2] and B[0:2, 0:2] to compute C[0:2, 0:2].
    # This uses 7 terms.
    # The remaining terms involve index 2 in i, j, or k.
    # i is in {0,1}, so i=2 is not an index for A. A is 2x3.
    # j is in {0,1,2}. k is in {0,1,2}.
    
    # The Strassen block uses j in {0,1} and k in {0,1}.
    # So it computes C[i, k] for i in {0,1}, k in {0,1} using A[i, 0:2] and B[0:2, 0:2].
    # It does NOT use A[i, 2] or B[2, k].
    
    # We need to add contributions from A[i, 2] and B[2, k].
    # C[i, k] = (Sum_{j=0,1} A[i,j]B[j,k]) + A[i,2]B[2,k]
    
    # For k in {0,1}:
    # C[i, 0] = Strassen_C[i,0] + A[i,2]B[2,0]
    # C[i, 1] = Strassen_C[i,1] + A[i,2]B[2,1]
    
    # For k = 2:
    # C[i, 2] = A[i,0]B[0,2] + A[i,1]B[1,2] + A[i,2]B[2,2]
    
    # Let's count the additional terms needed.
    # Terms for A[i,2]B[2,0] for i=0,1: 2 terms.
    # Terms for A[i,2]B[2,1] for i=0,1: 2 terms.
    # Terms for C[i,2]:
    #   C[0,2] = A[0,0]B[0,2] + A[0,1]B[1,2] + A[0,2]B[2,2]
    #   C[1,2] = A[1,0]B[0,2] + A[1,1]B[1,2] + A[1,2]B[2,2]
    #   This is 6 terms if schoolbook.
    
    # Total = 7 (Strassen) + 2 + 2 + 6 = 17.
    # This matches the known rank.
    
    # Let's construct the matrices U, V, W for these 17 terms.
    # U indices: i*m + j. i in {0,1}, j in {0,1,2}.
    #   A[0,0]->0, A[0,1]->1, A[0,2]->2
    #   A[1,0]->3, A[1,1]->4, A[1,2]->5
    # V indices: j*p + k. j in {0,1,2}, k in {0,1,2}.
    #   B[0,0]->0, B[0,1]->1, B[0,2]->2
    #   B[1,0]->3, B[1,1]->4, B[1,2]->5
    #   B[2,0]->6, B[2,1]->7, B[2,2]->8
    # W indices: k*n + i. k in {0,1,2}, i in {0,1}.
    #   C[0,0]->0, C[1,0]->1
    #   C[0,1]->2, C[1,1]->3
    #   C[0,2]->4, C[1,2]->5
    
    U = np.zeros((6, 17), dtype=np.float64)
    V = np.zeros((9, 17), dtype=np.float64)
    W = np.zeros((6, 17), dtype=np.float64)
    
    # 1. Strassen Terms (0-6)
    # M1 = A[0,0] * (B[0,0] + B[1,1])
    # U[0] (A[0,0]), V[0] (B[0,0]), V[4] (B[1,1])
    # W[0] (C[0,0]), W[1] (C[1,0])
    U[0, 0] = 1
    V[0, 0] = 1
    V[4, 0] = 1
    W[0, 0] = 1
    W[1, 0] = 1
    
    # M2 = (A[0,0] + A[1,0]) * B[1,1]
    # U[0] (A[0,0]), U[3] (A[1,0]), V[4] (B[1,1])
    # W[0] (C[0,0]) coeff -1, W[3] (C[1,1]) coeff 1
    U[0, 1] = 1
    U[3, 1] = 1
    V[4, 1] = 1
    W[0, 1] = -1
    W[3, 1] = 1
    
    # M3 = A[1,1] * (B[0,0] + B[0,1])
    # U[4] (A[1,1]), V[0] (B[0,0]), V[1] (B[0,1])
    # W[2] (C[0,1]) coeff 1, W[3] (C[1,1]) coeff 1
    U[4, 2] = 1
    V[0, 2] = 1
    V[1, 2] = 1
    W[2, 2] = 1
    W[3, 2] = 1
    
    # M4 = A[0,1] * (B[1,0] + B[1,1])
    # U[1] (A[0,1]), V[3] (B[1,0]), V[4] (B[1,1])
    # W[0] (C[0,0]) coeff 1, W[2] (C[0,1]) coeff 1
    U[1, 3] = 1
    V[3, 3] = 1
    V[4, 3] = 1
    W[0, 3] = 1
    W[2, 3] = 1
    
    # M5 = (A[0,0] + A[0,1]) * B[0,0]
    # U[0] (A[0,0]), U[1] (A[0,1]), V[0] (B[0,0])
    # W[1] (C[1,0]) coeff 1, W[3] (C[1,1]) coeff -1
    U[0, 4] = 1
    U[1, 4] = 1
    V[0, 4] = 1
    W[1, 4] = 1
    W[3, 4] = -1
    
    # M6 = (A[1,1] - A[0,1]) * (B[0,0] + B[0,1])
    # U[4] (A[1,1]), U[1] (A[0,1]), V[0] (B[0,0]), V[1] (B[0,1])
    # W[3] (C[1,1]) coeff 1
    U[4, 5] = 1
    U[1, 5] = -1
    V[0, 5] = 1
    V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A[0,0] - A[1,0]) * (B[1,0] + B[1,1])
    # U[0] (A[0,0]), U[3] (A[1,0]), V[3] (B[1,0]), V[4] (B[1,1])
    # W[1] (C[1,0]) coeff 1
    U[0, 6] = 1
    U[3, 6] = -1
    V[3, 6] = 1
    V[4, 6] = 1
    W[1, 6] = 1
    
    # 2. Terms for A[:,2] * B[2, 0:2] (7-10)
    # C[0,0] += A[0,2]B[2,0]
    U[2, 7] = 1
    V[6, 7] = 1
    W[0, 7] = 1
    
    # C[1,0] += A[1,2]B[2,0]
    U[5, 8] = 1
    V[6, 8] = 1
    W[1, 8] = 1
    
    # C[0,1] += A[0,2]B[2,1]
    U[2, 9] = 1
    V[7, 9] = 1
    W[2, 9] = 1
    
    # C[1,1] += A[1,2]B[2,1]
    U[5, 10] = 1
    V[7, 10] = 1
    W[3, 10] = 1
    
    # 3. Terms for C[:,2] (11-16)
    # C[0,2] = A[0,0]B[0,2] + A[0,1]B[1,2] + A[0,2]B[2,2]
    # Term 11: A[0,0]B[0,2]
    U[0, 11] = 1
    V[2, 11] = 1
    W[4, 11] = 1
    
    # Term 12: A[0,1]B[1,2]
    U[1, 12] = 1
    V[5, 12] = 1
    W[4, 12] = 1
    
    # Term 13: A[0,2]B[2,2]
    U[2, 13] = 1
    V[8, 13] = 1
    W[4, 13] = 1
    
    # C[1,2] = A[1,0]B[0,2] + A[1,1]B[1,2] + A[1,2]B[2,2]
    # Term 14: A[1,0]B[0,2]
    U[3, 14] = 1
    V[2, 14] = 1
    W[5, 14] = 1
    
    # Term 15: A[1,1]B[1,2]
    U[4, 15] = 1
    V[5, 15] = 1
    W[5, 15] = 1
    
    # Term 16: A[1,2]B[2,2]
    U[5, 16] = 1
    V[8, 16] = 1
    W[5, 16] = 1
    
    # Verify the decomposition
    # T_recon[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
    # Note: U index is i*m+j, V index is j*p+k, W index is k*n+i.
    # The tensor T is defined such that T[i*m+j, j*p+k, k*n+i] = 1.
    # So we check if np.einsum('ar,br,cr->abc', U, V, W) matches T_target.
    
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    
    if np.allclose(T_recon, T_target, atol=1e-9):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # If verification fails, fall back to schoolbook
        # This should not happen if the logic above is correct.
        return schoolbook()

# EVOLVE-BLOCK-END
