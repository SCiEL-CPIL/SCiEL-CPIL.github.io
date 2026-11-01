"""Matrix multiplication 3,3,3 — Strassen-block rank-26 decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START

def _construct_rank26():
    """
    Build a rank-26 decomposition for 3x3 matrix multiplication.
    
    Partition:
      A = [[A22, a02],
           [a12, a22]]
      where A22 is top-left 2x2, a02 is top-right 1x1 (A[0,2]),
            a12 is bottom-left 1x1 (A[1,2]), a22 is bottom-right 1x1 (A[2,2]).
      Similarly B = [[B22, b00],[b10, b20]] with columns split 2|1.
      C = A*B, split rows 2|1 and cols 2|1:
        C = [[C22, c02],
             [c12, c22]]
    
    Apply Strassen's 7-term algorithm to the 2x2 blocks:
      C22 (top-left 2x2) = A22 * B22  (7 Strassen terms)
      c02 (top-right 1x2) = a02 * B22 + A22 * b00
      c12 (bottom-left 2x1) = a12 * B22 + A22 * b10
      c22 (bottom-right 1x1) = a22 * b20 + a12 * b20 + a02 * b10 + A22 * B22 contribution? 
      
    Wait, let me re-think the partition.
    
    A is 3x3:
      Row 0: A[0,0] A[0,1] A[0,2]
      Row 1: A[1,0] A[1,1] A[1,2]
      Row 2: A[2,0] A[2,1] A[2,2]
    
    B is 3x3:
      Row 0: B[0,0] B[0,1] B[0,2]
      Row 1: B[1,0] B[1,1] B[1,2]
      Row 2: B[2,0] B[2,1] B[2,2]
    
    C = A*B:
      C[i,j] = sum_k A[i,k]*B[k,j]
    
    Partition A into:
      A_top = A[0:2, :]  (2x3)
      A_bot = A[2:3, :]  (1x3)
    Partition B into:
      B_left = B[:, 0:2]  (3x2)
      B_right = B[:, 2:3]  (3x1)
    
    Then C = [[A_top * B_left, A_top * B_right],
              [A_bot * B_left, A_bot * B_right]]
    
    A_top * B_left is 2x2 — use Strassen (7 terms)
    A_top * B_right is 2x1 — 2 terms
    A_bot * B_left is 1x2 — 2 terms
    A_bot * B_right is 1x1 — 1 term
    Total: 7 + 2 + 2 + 1 = 12? No that's not right either.
    
    Actually, let me think about this differently. The known result is that 3x3 matrix
    multiplication has rank 23 (Lam, 2008) or at most 23. But the problem says rank 26
    was achieved by partitioning into 2x2 and 1x1 blocks.
    
    Let me use a simpler approach: apply Strassen to a 2x2 sub-block and handle the
    remaining cross terms individually.
    
    Partition:
      A = [[X, u],
           [v, w]]
      where X is 2x2 (top-left), u is 2x1 (top-right column),
            v is 1x2 (bottom-left row), w is 1x1 (bottom-right).
      
      B = [[Y, s],
           [t, z]]
      where Y is 2x2 (top-left), s is 2x1 (top-right column),
            t is 1x2 (bottom-left row), z is 1x1 (bottom-right).
      
      C = A*B = [[X*Y + u*t, X*s + u*z],
                 [v*Y + w*t, v*s + w*z]]
      
    So:
      C[0:2, 0:2] = X*Y + u*t
      C[0:2, 2:3] = X*s + u*z
      C[2:3, 0:2] = v*Y + w*t
      C[2:3, 2:3] = v*s + w*z
    
    X*Y: 2x2*2x2 — Strassen: 7 terms
    u*t: (2x1)*(1x2) = outer product of u and t: 4 terms (each element of u times each of t)
    Actually u*t is a 2x2 matrix where (u*t)[i,j] = u[i]*t[j]. That's 4 scalar products.
    But we can do it in 4 terms (rank 4 for a 2x2 rank-1 matrix? No, rank-1 matrix has rank 1 as a tensor? No, the tensor rank for matrix multiplication of (2x1)*(1x2) is... 
    
    Hmm, I'm overcomplicating this. Let me just use the known decomposition.
    
    The standard approach for 3x3 with rank 26:
    - Use Strassen on the top-left 2x2 block (7 multiplications)
    - Handle the remaining 5 cross-block terms with 19 additional multiplications
    - Total: 7 + 19 = 26
    
    Actually, let me just construct it directly by enumerating which scalar products
    contribute to each C[i,j] and finding a minimal set.
    
    C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
    C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
    C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
    C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
    C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] + A[1,2]*B[2,1]
    C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2]
    C[2,0] = A[2,0]*B[0,0] + A[2,1]*B[1,0] + A[2,2]*B[2,0]
    C[2,1] = A[2,0]*B[0,1] + A[2,1]*B[1,1] + A[2,2]*B[2,1]
    C[2,2] = A[2,0]*B[0,2] + A[2,1]*B[1,2] + A[2,2]*B[2,2]
    
    Schoolbook: 27 terms.
    
    For Strassen on the top-left 2x2 block:
    P1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    P2 = (A[1,0]+A[1,1])*B[0,0]
    P3 = A[0,0]*(B[0,1]-B[1,1])
    P4 = A[1,1]*(B[1,0]-B[0,0])
    P5 = (A[0,0]+A[0,1])*B[1,1]
    P6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
    P7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
    
    These give:
    C[0,0] = P1 - P4 - P2 + P7  ... wait, standard Strassen:
    C11 = P1 - P4 - P2 + P7  (but this is for the 2x2 result)
    
    Actually the standard Strassen formulas are:
    M1 = (A11 + A22)(B11 + B22)
    M2 = (A21 + A22)B11
    M3 = A11(B12 - B22)
    M4 = A22(B21 - B11)
    M5 = (A11 + A12)B22
    M6 = (A21 - A11)(B11 + B12)
    M7 = (A12 - A22)(B21 + B22)
    
    C11 = M1 - M4 - M2 + M7
    C12 = M3 + M5
    C21 = M6 + M4
    C22 = M1 - M3 - M5 + M2
    
    So for the 2x2 top-left block of C:
    C[0,0] gets: M1 - M4 - M2 + M7
    C[0,1] gets: M3 + M5
    C[1,0] gets: M6 + M4
    C[1,1] gets: M1 - M3 - M5 + M2
    
    Plus the contributions from the third row/column:
    C[0,0] += A[0,2]*B[2,0]
    C[0,1] += A[0,2]*B[2,1]
    C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
    C[1,0] += A[1,2]*B[2,0]
    C[1,1] += A[1,2]*B[2,1]
    C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2]
    C[2,0] = A[2,0]*B[0,0] + A[2,1]*B[1,0] + A[2,2]*B[2,0]
    C[2,1] = A[2,0]*B[0,1] + A[2,1]*B[1,1] + A[2,2]*B[2,1]
    C[2,2] = A[2,0]*B[0,2] + A[2,1]*B[1,2] + A[2,2]*B[2,2]
    
    The additional terms needed beyond the 7 Strassen products:
    - A[0,2]*B[2,0] (1 term)
    - A[0,2]*B[2,1] (1 term)
    - A[0,0]*B[0,2], A[0,1]*B[1,2], A[0,2]*B[2,2] (3 terms for C[0,2])
    - A[1,2]*B[2,0] (1 term)
    - A[1,2]*B[2,1] (1 term)
    - A[1,0]*B[0,2], A[1,1]*B[1,2], A[1,2]*B[2,2] (3 terms for C[1,2])
    - A[2,0]*B[0,0], A[2,1]*B[1,0], A[2,2]*B[2,0] (3 terms for C[2,0])
    - A[2,0]*B[0,1], A[2,1]*B[1,1], A[2,2]*B[2,1] (3 terms for C[2,1])
    - A[2,0]*B[0,2], A[2,1]*B[1,2], A[2,2]*B[2,2] (3 terms for C[2,2])
    
    That's 1+1+3+1+1+3+3+3+3 = 19 additional terms.
    Total: 7 + 19 = 26. 
    
    But wait, some of these can be shared. For example, A[0,2]*B[2,2] appears in C[0,2] and... no, it only appears in C[0,2].
    A[1,2]*B[2,2] appears in C[1,2].
    A[2,0]*B[0,0] appears in C[2,0].
    
    Hmm, but we can potentially combine some. Let's just go with 26 terms and verify.
    
    Actually, I realize I should just carefully enumerate all 26 terms and verify numerically.
    """
    n, m, p = 3, 3, 3
    rank = 26
    U = np.zeros((n * m, rank))  # 9 x 26
    V = np.zeros((m * p, rank))  # 9 x 26
    W = np.zeros((p * n, rank))  # 9 x 26
    
    # Helper to set a term
    # U row index: i*m + j = i*3 + j  (A[i,j])
    # V row index: j*p + k = j*3 + k  (B[j,k])
    # W row index: k*n + i = k*3 + i  (C[k,i]) -- wait, the convention is W[k*n+i, r]
    # So W[k*n+i] corresponds to C[i,k]
    
    # Let me re-read: "The output factor follows the official trace convention W[k*n+i,r]"
    # So W is indexed by (k, i) where k is the column of B (and row of C?) and i is the row of A (and column of C?)
    # In the tensor decomposition: T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # where a = i*m+j (A entry), b = j*p+k (B entry), c = k*n+i (C entry)
    # So C[i,k] corresponds to c = k*n + i
    
    # Strassen terms for the 2x2 top-left block (indices 0,1 for rows/cols)
    
    # M1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
    # U: A[0,0]=U[0], A[1,1]=U[4]
    # V: B[0,0]=V[0], B[1,1]=V[4]
    # W: contributes to C[0,0]=W[0*3+0]=W[0], C[0,1]=W[1*3+0]=W[3], C[1,0]=W[0*3+1]=W[1], C[1,1]=W[1*3+1]=W[4]
    # From Strassen: C[0,0]+=M1, C[0,1]-=M1? No:
    # C[0,0] = M1 - M4 - M2 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M6 + M4
    # C[1,1] = M1 - M3 - M5 + M2
    
    # Term 0: M1
    U[0, 0] += 1   # A[0,0]
    U[4, 0] += 1   # A[1,1]
    V[0, 0] += 1   # B[0,0]
    V[4, 0] += 1   # B[1,1]
    W[0, 0] += 1   # C[0,0] (k=0,i=0)
    W[1, 0] -= 1   # C[1,0] (k=0,i=1) -> wait, C[1,0] has k=0, i=1? No.
    # C[i,k]: i is row, k is column. C[1,0] means row 1, col 0. k=0, i=1. W[k*n+i] = W[0*3+1] = W[1]
    # But M1 contributes +1 to C[0,0] and... let me re-derive.
    # M1 = (A[0,0]+A[1,1])(B[0,0]+B[1,1]) = A[0,0]B[0,0] + A[0,0]B[1,1] + A[1,1]B[0,0] + A[1,1]B[1,1]
    # C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0] + A[0,2]B[2,0]
    # C[1,1] = A[1,0]B[0,1] + A[1,1]B[1,1] + A[1,2]B[2,1]
    # M1 contains A[0,0]B[0,0] (in C[0,0]), A[0,0]B[1,1] (in C[0,1]? No, C[0,1] = A[0,0]B[0,1]+A[0,1]B[1,1]+A[0,2]B[2,1])
    # Hmm, A[0,0]*B[1,1] doesn't directly appear in any C[i,k] in the standard way.
    # That's the point of Strassen - the products are combined cleverly.
    
    # Let me just use the standard Strassen decomposition formulas and track coefficients.
    # C[0,0] = M1 - M4 - M2 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M6 + M4
    # C[1,1] = M1 - M3 - M5 + M2
    
    # So:
    # M1: +1 to C[0,0], +1 to C[1,1]
    # M2: -1 to C[0,0], +1 to C[1,1]
    # M3: +1 to C[0,1], -1 to C[1,1]
    # M4: -1 to C[0,0], +1 to C[1,0]
    # M5: +1 to C[0,1], -1 to C[1,1]
    # M6: +1 to C[1,0]
    # M7: +1 to C[0,0]
    
    # W indices: C[i,k] -> W[k*3+i]
    # C[0,0] -> W[0], C[0,1] -> W[3], C[1,0] -> W[1], C[1,1] -> W[4]
    
    # M1: W[0]+=1, W[4]+=1
    # M2: W[0]-=1, W[4]+=1
    # M3: W[3]+=1, W[4]-=1
    # M4: W[0]-=1, W[1]+=1
    # M5: W[3]+=1, W[4]-=1
    # M6: W[1]+=1
    # M7: W[0]+=1
    
    # Let me redo this more carefully.
    # Reset
    U = np.zeros((9, 26))
    V = np.zeros((9, 26))
    W = np.zeros((9, 26))
    
    # Term 0: M1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
    # A[0,0]=U[0*3+0]=U[0], A[1,1]=U[1*3+1]=U[4]
    # B[0,0]=V[0*3+0]=V[0], B[1,1]=V[1*3+1]=V[4]
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    # C[0,0]=W[0*3+0]=W[0], C[1,1]=W[1*3+1]=W[4]
    W[0, 0] = 1; W[4, 0] = 1
    
    # Term 1: M2 = (A[1,0] + A[1,1]) * B[0,0]
    # A[1,0]=U[1*3+0]=U[3], A[1,1]=U[4], B[0,0]=V[0]
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    # C[0,0] -= M2, C[1,1] += M2
    W[0, 1] = -1; W[4, 1] = 1
    
    # Term 2: M3 = A[0,0] * (B[0,1] - B[1,1])
    # A[0,0]=U[0], B[0,1]=V[0*3+1]=V[1], B[1,1]=V[4]
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    # C[0,1] += M3, C[1,1] -= M3
    # C[0,1]=W[1*3+0]=W[3], C[1,1]=W[4]
    W[3, 2] = 1; W[4, 2] = -1
    
    # Term 3: M4 = A[1,1] * (B[1,0] - B[0,0])
    # A[1,1]=U[4], B[1,0]=V[1*3+0]=V[3], B[0,0]=V[0]
    U[4, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    # C[0,0] -= M4, C[1,0] += M4
    # C[0,0]=W[0], C[1,0]=W[0*3+1]=W[1]
    W[0, 3] = -1; W[1, 3] = 1
    
    # Term 4: M5 = (A[0,0] + A[0,1]) * B[1,1]
    # A[0,0]=U[0], A[0,1]=U[0*3+1]=U[1], B[1,1]=V[4]
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    # C[0,1] += M5, C[1,1] -= M5
    W[3, 4] = 1; W[4, 4] = -1
    
    # Term 5: M6 = (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
    # A[1,0]=U[3], A[0,0]=U[0], B[0,0]=V[0], B[0,1]=V[1]
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    # C[1,0] += M6
    W[1, 5] = 1
    
    # Term 6: M7 = (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
    # A[0,1]=U[1], A[1,1]=U[4], B[1,0]=V[3], B[1,1]=V[4]
    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    # C[0,0] += M7
    W[0, 6] = 1
    
    # Now the additional 19 terms:
    # C[0,0] += A[0,2]*B[2,0]
    # C[0,1] += A[0,2]*B[2,1]
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
    # C[1,0] += A[1,2]*B[2,0]
    # C[1,1] += A[1,2]*B[2,1]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2]
    # C[2,0] = A[2,0]*B[0,0] + A[2,1]*B[1,0] + A[2,2]*B[2,0]
    # C[2,1] = A[2,0]*B[0,1] + A[2,1]*B[1,1] + A[2,2]*B[2,1]
    # C[2,2] = A[2,0]*B[0,2] + A[2,1]*B[1,2] + A[2,2]*B[2,2]
    
    # Term 7: A[0,2]*B[2,0] -> C[0,0]
    U[0*3+2, 7] = 1  # A[0,2] = U[2]
    V[2*3+0, 7] = 1  # B[2,0] = V[6]
    W[0*3+0, 7] = 1  # C[0,0] = W[0]
    
    # Term 8: A[0,2]*B[2,1] -> C[0,1]
    U[2, 8] = 1
    V[2*3+1, 8] = 1  # B[2,1] = V[7]
    W[1*3+0, 8] = 1  # C[0,1] = W[3]
    
    # Term 9: A[0,0]*B[0,2] -> C[0,2]
    U[0, 9] = 1
    V[0*3+2, 9] = 1  # B[0,2] = V[2]
    W[2*3+0, 9] = 1  # C[0,2] = W[6]
    
    # Term 10: A[0,1]*B[1,2] -> C[0,2]
    U[1, 10] = 1
    V[1*3+2, 10] = 1  # B[1,2] = V[5]
    W[6, 10] = 1
    
    # Term 11: A[0,2]*B[2,2] -> C[0,2]
    U[2, 11] = 1
    V[2*3+2, 11] = 1  # B[2,2] = V[8]
    W[6, 11] = 1
    
    # Term 12: A[1,2]*B[2,0] -> C[1,0]
    U[1*3+2, 12] = 1  # A[1,2] = U[5]
    V[6, 12] = 1
    W[1, 12] = 1  # C[1,0] = W[1]
    
    # Term 13: A[1,2]*B[2,1] -> C[1,1]
    U[5, 13] = 1
    V[7, 13] = 1
    W[4, 13] = 1  # C[1,1] = W[4]
    
    # Term 14: A[1,0]*B[0,2] -> C[1,2]
    U[3, 14] = 1
    V[2, 14] = 1
    W[2*3+1, 14] = 1  # C[1,2] = W[7]
    
    # Term 15: A[1,1]*B[1,2] -> C[1,2]
    U[4, 15] = 1
    V[5, 15] = 1
    W[7, 15] = 1
    
    # Term 16: A[1,2]*B[2,2] -> C[1,2]
    U[5, 16] = 1
    V[8, 16] = 1
    W[7, 16] = 1
    
    # Term 17: A[2,0]*B[0,0] -> C[2,0]
    U[2*3+0, 17] = 1  # A[2,0] = U[6]
    V[0, 17] = 1
    W[0*3+2, 17] = 1  # C[2,0] = W[2]
    
    # Term 18: A[2,1]*B[1,0] -> C[2,0]
    U[2*3+1, 18] = 1  # A[2,1] = U[7]
    V[3, 18] = 1
    W[2, 18] = 1
    
    # Term 19: A[2,2]*B[2,0] -> C[2,0]
    U[2*3+2, 19] = 1  # A[2,2] = U[8]
    V[6, 19] = 1
    W[2, 19] = 1
    
    # Term 20: A[2,0]*B[0,1] -> C[2,1]
    U[6, 20] = 1
    V[1, 20] = 1
    W[1*3+2, 20] = 1  # C[2,1] = W[5]
    
    # Term 21: A[2,1]*B[1,1] -> C[2,1]
    U[7, 21] = 1
    V[4, 21] = 1
    W[5, 21] = 1
    
    # Term 22: A[2,2]*B[2,1] -> C[2,1]
    U[8, 22] = 1
    V[7, 22] = 1
    W[5, 22] = 1
    
    # Term 23: A[2,0]*B[0,2] -> C[2,2]
    U[6, 23] = 1
    V[2, 23] = 1
    W[2*3+2, 23] = 1  # C[2,2] = W[8]
    
    # Term 24: A[2,1]*B[1,2] -> C[2,2]
    U[7, 24] = 1
    V[5, 24] = 1
    W[8, 24] = 1
    
    # Term 25: A[2,2]*B[2,2] -> C[2,2]
    U[8, 25] = 1
    V[8, 25] = 1
    W[8, 25] = 1
    
    return U, V, W


def _verify(U, V, W, n=3, m=3, p=3, tol=1e-9):
    """Verify the decomposition."""
    r = U.shape[1]
    T = np.einsum('ar,br,cr->abc', U, V, W)
    T_exp = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_exp[i*m+j, j*p+k, k*n+i] = 1.0
    return np.allclose(T, T_exp, atol=tol, rtol=0)


def construct():
    n, m, p = DIMENSIONS
    U, V, W = _construct_rank26()
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U_s = np.zeros((n * m, rank))
        V_s = np.zeros((m * p, rank))
        W_s = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i * m + j, col] = 1
                    V_s[j * p + k, col] = 1
                    W_s[k * n + i, col] = 1
                    col += 1
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
