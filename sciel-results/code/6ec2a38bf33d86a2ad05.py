import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 2, 9]

def _strassen_u_v_w():
    """
    Return rank-7 decomposition for 2x2 * 2x2 matrix multiplication.
    U: shape (4, 7), V: shape (4, 7), W: shape (4, 7)
    Convention: U[i*2+j, c] = coefficient of a_{i,j} in rank-c term
                V[j*2+k, c] = coefficient of b_{j,k} in rank-c term
                W[k*2+i, c] = coefficient of (c)_{k,i} in rank-c term
    """
    # Strassen's 7 rank-1 terms for 2x2 matrix multiplication
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # C = A*B, C[i,j] = sum_k A[i,k] B[k,j]
    
    # m1 = (a11 + a22)(b11 + b22)
    # m2 = (a21 + a22)b11
    # m3 = a11(b12 - b22)
    # m4 = a22(b21 - b11)
    # m5 = (a11 + a12)b22
    # m6 = (a21 - a11)(b11 + b12)
    # m7 = (a12 - a22)(b21 + b22)
    #
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    
    U = np.zeros((4, 7), dtype=Fraction)
    V = np.zeros((4, 7), dtype=Fraction)
    W = np.zeros((4, 7), dtype=Fraction)
    
    # Index mapping: row in U is i*2+j for A[i,j], row in V is j*2+k for B[j,k], row in W is k*2+i for C[k,i]
    
    # Term 1: m1 = (a11 + a22)(b11 + b22)
    U[0, 0] = 1  # a11
    U[3, 0] = 1  # a22
    V[0, 0] = 1  # b11
    V[3, 0] = 1  # b22
    W[0, 0] = 1  # C11 (k=0, i=0)
    W[3, 0] = 1  # C22 (k=1, i=1)
    
    # Term 2: m2 = (a21 + a22)b11
    U[2, 1] = 1  # a21
    U[3, 1] = 1  # a22
    V[0, 1] = 1  # b11
    W[2, 1] = 1  # C21 (k=0, i=1)
    W[3, 1] = -1 # C22 (k=1, i=1)
    
    # Term 3: m3 = a11(b12 - b22)
    U[0, 2] = 1  # a11
    V[1, 2] = 1  # b12
    V[3, 2] = -1 # b22
    W[1, 2] = 1  # C12 (k=0, i=0)... wait, let me re-check the indexing
    
    # Let me redo this more carefully.
    # A is 2x2: indices (i,j) where i in {0,1}, j in {0,1}
    # B is 2x2: indices (j,k) where j in {0,1}, k in {0,1}
    # C is 2x2: C[i,j] = sum_k A[i,k] B[k,j]
    # Output index: k*n + i = k*2 + i
    
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    
    # W index: k*2 + i where C[k,i]... no wait.
    # The convention says W[k*n+i, r]. So for C[i,j], we need... 
    # Actually, let me re-read: "output factor follows the official trace convention W[k*n+i,r]"
    # So W has p*n = 2*2 = 4 rows, indexed by k*n+i = k*2+i
    # This means W[k*2+i, r] gives the coefficient for C[i,k]... 
    # Hmm, let me think. C = A*B, C[i,j] = sum_k A[i,k]*B[k,j]
    # If W[k*2+i, r] is the coefficient, then the output at position (i,j) would be...
    # Actually, I think the convention is that the (i,j)-entry of the product is:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]... no that doesn't make sense dimensionally.
    
    # Let me re-read the task: "U of shape (n*m, rank), V of shape (m*p, rank), W of shape (p*n, rank)"
    # n=2, m=2, p=2. So U is (4, rank), V is (4, rank), W is (4, rank).
    # The tensor is T[i,j,k] = delta[i,k] (for matrix mult A(i,j) * B(j,k) -> C(i,k))
    # Actually for matrix multiplication: C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor is T[i,j,k] which equals 1 if we're computing the (i,k) entry and j is the intermediate.
    # Wait, the standard tensor for matrix multiplication M_{n,m,p} is:
    # (A * B)[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor has indices (i, j, k) and T[i,j,k] = 1 (it's the identity in j for each (i,k) pair... no)
    # Actually T[i,j,k] is defined such that sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k] * W[k,n+i]... 
    # I think the decomposition is: (A*B)[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over j?
    
    # Let me just use the standard schoolbook to verify my understanding:
    # For each (i, j, k) triple, we have a term. U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
    # So the product at (i, k) is: sum_col U[i*m+j, col] * V[j*p+k, col] * W[k*n+i, col] for some j?
    # No, it's sum over r of (sum_j U[i*m+j, r] * A[i,j]) * (sum_k V[j*p+k, r] * B[j,k]) * ... 
    
    # I think the correct interpretation is:
    # (A*B)[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_k' V[?]*B[?,?]) ...
    # This is getting confusing. Let me just trust the schoolbook construction and make Strassen work with the same indexing.
    
    # In schoolbook: for triple (i,j,k), U[i*m+j, c]=1, V[j*p+k, c]=1, W[k*n+i, c]=1
    # This means the r-th term contributes A[i,j]*B[j,k] to position... and W[k*n+i, c]=1 means it contributes to output at (i,k)? 
    # Since W is indexed by k*n+i, and the output C has entries C[i,k], the index k*n+i corresponds to (i,k) with i as second index.
    # So C[i,k] gets contribution from terms where W[k*n+i, r] is nonzero.
    
    # OK so for Strassen, I need:
    # C[i,k] = sum_r (linear combo of A entries via U) * (linear combo of B entries via V) * W[k*2+i, r]
    # where U[i*2+j, r] picks out A[i,j] contributions, V[j*2+k, r] picks out B[j,k] contributions.
    
    # Let me redo Strassen with this understanding:
    # C[0,0] = m1 + m4 - m5 + m7
    # C[0,1] = m3 + m5
    # C[1,0] = m2 + m4
    # C[1,1] = m1 - m2 + m3 + m6
    
    # W[k*2+i, r]: W[0*2+0]=W[0] for C[0,0], W[0*2+1]=W[1] for C[1,0], W[1*2+0]=W[2] for C[0,1], W[1*2+1]=W[3] for C[1,1]
    
    U2 = np.zeros((4, 7), dtype=Fraction)
    V2 = np.zeros((4, 7), dtype=Fraction)
    W2 = np.zeros((4, 7), dtype=Fraction)
    
    # m1 = (a11+a22)(b11+b22)
    U2[0, 0] = 1  # a11
    U2[3, 0] = 1  # a22
    V2[0, 0] = 1  # b11
    V2[3, 0] = 1  # b22
    W2[0, 0] = 1  # C[0,0]
    W2[3, 0] = 1  # C[1,1]
    
    # m2 = (a21+a22)*b11
    U2[2, 1] = 1  # a21
    U2[3, 1] = 1  # a22
    V2[0, 1] = 1  # b11
    W2[1, 1] = 1  # C[1,0]
    W2[3, 1] = -1 # C[1,1]
    
    # m3 = a11*(b12-b22)
    U2[0, 2] = 1  # a11
    V2[1, 2] = 1  # b12
    V2[3, 2] = -1 # b22
    W2[2, 2] = 1  # C[0,1]
    W2[3, 2] = 1  # C[1,1]
    
    # m4 = a22*(b21-b11)
    U2[3, 3] = 1  # a22
    V2[2, 3] = 1  # b21
    V2[0, 3] = -1 # b11
    W2[0, 3] = 1  # C[0,0]
    W2[1, 3] = 1  # C[1,0]
    
    # m5 = (a11+a12)*b22
    U2[0, 4] = 1  # a11
    U2[1, 4] = 1  # a12
    V2[3, 4] = 1  # b22
    W2[2, 4] = 1  # C[0,1]
    W2[3, 4] = -1 # C[1,1]
    
    # m6 = (a21-a11)*(b11+b12)
    U2[2, 5] = 1  # a21
    U2[0, 5] = -1 # a11
    V2[0, 5] = 1  # b11
    V2[1, 5] = 1  # b12
    W2[3, 5] = 1  # C[1,1]
    
    # m7 = (a12-a22)*(b21+b22)
    U2[1, 6] = 1  # a12
    U2[3, 6] = -1 # a22
    V2[2, 6] = 1  # b21
    V2[3, 6] = 1  # b22
    W2[0, 6] = 1  # C[0,0]
    W2[2, 6] = 1  # C[0,1]
    
    return U2, V2, W2

def construct():
    """
    Rank-32 decomposition: 4 blocks of Strassen (rank 7 each) for the 2x2x8 part,
    plus 1 schoolbook term for the 2x2x1 part.
    Total rank = 4*7 + 1*4 = 32.
    
    Wait, that's not right. We have 2x2 * 2x9. We can split the 9 columns into 4 groups of 2 plus 1 single.
    For each group of 2 columns, we use Strassen (rank 7). For the single column, we use schoolbook (rank 4).
    Total rank = 4*7 + 4 = 32.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=9
    
    total_rank = 32
    U = np.zeros((n * m, total_rank), dtype=Fraction)  # (4, 32)
    V = np.zeros((m * p, total_rank), dtype=Fraction)  # (18, 32)
    W = np.zeros((p * n, total_rank), dtype=Fraction)  # (18, 32)
    
    U_str, V_str, W_str = _strassen_u_v_w()
    
    col_offset = 0
    # 4 blocks of 2 columns each (columns 0-1, 2-3, 4-5, 6-7)
    for block in range(4):
        for s in range(7):
            c = col_offset + s
            # U: same for all blocks
            for row in range(4):
                U[row, c] = U_str[row, s]
            
            # V: rows are j*p+k where j in {0,1}, k in {0,...,8}
            # For this block, the 2 columns are at positions 2*block and 2*block+1
            # V[j*p+k, c] = V_str[j*2+k_local, s] where k_local is 0 or 1 within the block
            for j in range(2):
                for k_local in range(2):
                    k_global = 2 * block + k_local
                    V[j * p + k_global, c] = V_str[j * 2 + k_local, s]
            
            # W: rows are k*n+i where k in {0,...,8}, i in {0,1}
            # For this block, output columns k_global = 2*block, 2*block+1
            for k_local in range(2):
                for i in range(2):
                    k_global = 2 * block + k_local
                    W[k_global * n + i, c] = W_str[k_local * 2 + i, s]
        
        col_offset += 7
    
    # 1 block of 1 column (column 8), schoolbook rank 4
    # For a single column k=8:
    # C[i, 8] = sum_j A[i,j] * B[j, 8]
    # 4 terms: (i,j) pairs
    for i in range(2):
        for j in range(2):
            c = col_offset  # just one column per (i,j), but we need 4 terms
            # Actually for schoolbook on a single column, we need m=2 terms? No.
            # C[i,8] = A[i,0]*B[0,8] + A[i,1]*B[1,8]
            # So for fixed i, we need 2 terms (one for j=0, one for j=1).
            # Total 4 terms for i in {0,1}, j in {0,1}.
            pass
    
    # Let me redo the last part properly:
    col_offset = 28  # after 4*7 = 28 columns
    for i in range(2):
        for j in range(2):
            c = col_offset
            col_offset += 1
            U[i * m + j, c] = 1
            V[j * p + 8, c] = 1
            W[8 * n + i, c] = 1
    
    # Convert to lists of lists
    U_list = [[float(U[i, j]) if U[i, j] == U[i, j] else 0 for j in range(total_rank)] for i in range(n * m)]
    V_list = [[float(V[i, j]) if V[i, j] == V[i, j] else 0 for j in range(total_rank)] for i in range(m * p)]
    W_list = [[float(W[i, j]) if W[i, j] == W[i, j] else 0 for j in range(total_rank)] for i in range(p * n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
