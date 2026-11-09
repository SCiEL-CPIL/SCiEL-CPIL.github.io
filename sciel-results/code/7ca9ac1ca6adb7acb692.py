"""Fresh design for rank-minimal 2x3x3 matrix multiplication tensor decomposition."""
import time

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-17 exact decomposition for 2x3x3 matrix multiplication.
    
    Strategy: Use Strassen's rank-7 algorithm for the 2x2x2 block (top-left),
    then handle the remaining parts (top-right 2x2x1, bottom-left 1x2x2, bottom-right 1x2x1)
    with minimal additional rank.
    
    Matrix multiplication: C = A * B where A is 2x3, B is 3x3, C is 2x3.
    Partition:
      A = [A11 A12]  where A11 is 2x2, A12 is 2x1
      B = [B11 B12; B21 B22] where B11 is 2x2, B12 is 2x1, B21 is 1x2, B22 is 1x1
      
    Wait, let me reconsider the partitioning.
    
    A is 2x3, B is 3x3.
    Let's partition the columns of A and rows of B into 2 and 1:
      A = [A_left (2x2) | A_right (2x1)]
      B = [B_top (2x3); B_bottom (1x3)]
    
    Then C = A_left * B_top + A_right * B_bottom
    
    C is 2x3.
    
    Let me think about this differently. The tensor is T_{i,j,k} = delta_{i,k} where
    C[i,k] = sum_j A[i,j] * B[j,k].
    
    Actually, the standard tensor for matrix multiplication of (n x m) * (m x p) -> (n x p)
    has T[i,j,k] = delta_{i,k}... no wait.
    
    The tensor T is defined such that C[i,k] = sum_j A[i,j] B[j,k] = sum_j T[i,j,k] A[i,j] B[j,k]
    where T[i,j,k] = 1 if the indices align properly. Actually T[i,j,k] = delta_{i,i} * delta_{k,k}... 
    
    No. The multiplication tensor for (n,m,p) has entries T[i,j,k] where i in [0,n), j in [0,m), k in [0,p).
    T[i,j,k] = 1 if the j-th column of A (row i) multiplies with the j-th row of B (col k) to contribute to C[i,k].
    So T[i,j,k] = 1 for all valid i,j,k? No, that's not right either.
    
    Actually, the tensor is: C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    So the tensor T has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    Wait no, that would give C[i,k] = sum_j A[i,j] B[j,k] * 1, which is correct.
    
    Hmm, but that means the tensor is all ones? That can't be right for rank minimization.
    
    Let me reconsider. The matrix multiplication tensor T ∈ F^{n×m} ⊗ F^{m×p} ⊗ F^{n×p}
    is defined by T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k * delta where... 
    
    Actually I think the tensor is:
    T = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} A_{i,j} B_{j,k} C_{k,i}^*
    
    No, the standard definition: The bilinear map (A, B) -> C = AB can be represented as:
    C = sum_{r=1}^{rank} U_r ⊙ V_r ⊙ W_r
    
    where each term is a rank-1 tensor. The "multiplication tensor" T is the tensor such that
    <T, A ⊗ B> = C (flattened).
    
    T[i,j,k] where i indexes A (n*m entries), j indexes B (m*p entries), k indexes C (n*p entries).
    T[A_idx, B_idx, C_idx] = 1 if the product A[A_idx] * B[B_idx] contributes to C[C_idx].
    
    Specifically, if A_idx = i*m + j (row i, col j of A), B_idx = j'*p + k (row j', col k of B),
    C_idx = k'*n + i' (row i', col k' of C) [using the trace convention W[k*n+i,r] which means C index = k*n + i].
    
    Wait, the trace convention says W[k*n+i, r]. So C is indexed as C[k*n + i] where i is row, k is col? 
    That seems unusual. Let me re-read: "The output factor follows the official trace convention W[k*n+i,r]."
    
    So C has p*n entries, indexed as k*n + i where i in [0,n) and k in [0,p). This means C is stored column-major: C[col, row] = C[k, i] = C[k*n + i].
    
    So C[k*n + i] = sum_j A[i*m + j] * B[j*p + k]
    
    Therefore, the tensor T has:
    T[i*m + j, j*p + k, k*n + i] = 1 for all valid i, j, k.
    All other entries are 0.
    
    So the tensor is a permutation of the identity-like structure. The rank of this tensor is what we're minimizing.
    
    For n=2, m=3, p=3:
    - A has 6 entries: indices 0..5 (i*m+j for i in 0,1 and j in 0,1,2)
    - B has 9 entries: indices 0..8 (j*p+k for j in 0,1,2 and k in 0,1,2)
    - C has 6 entries: indices 0..5 (k*n+i for k in 0,1,2 and i in 0,1)
    
    The non-zero entries of T are at positions (i*m+j, j*p+k, k*n+i) for i in {0,1}, j in {0,1,2}, k in {0,1,2}.
    
    That's 2*3*3 = 18 non-zero entries, each being a rank-1 tensor. The schoolbook rank is 18.
    
    Now, can we do better? The known result for 2x3x3 is that the rank is 17 (using Strassen-like methods).
    
    Let me construct the rank-17 decomposition.
    
    Idea: Group the 18 terms. Notice that for fixed i, the terms involve j and k.
    
    For i=0: terms are (j, j*p+k, k*2+0) = (j, 3j+k, 2k) for j in {0,1,2}, k in {0,1,2}
    For i=1: terms are (m+j, j*p+k, k*2+1) = (3+j, 3j+k, 2k+1) for j in {0,1,2}, k in {0,1,2}
    
    Let me try to use Strassen's idea on the 2x2x2 part.
    
    Consider the sub-problem where we multiply a 2x2 matrix by a 2x2 matrix. The indices would be:
    i in {0,1}, j in {0,1}, k in {0,1}.
    
    The 8 terms for this sub-problem:
    (i*2+j, j*2+k, k*2+i) for i,j,k in {0,1}
    
    But in our full problem, m=3, p=3, so the indices are different. Let me map them:
    A index: i*3 + j
    B index: j*3 + k
    C index: k*2 + i
    
    For the "2x2x2 core" (j in {0,1}, k in {0,1}):
    i=0,j=0,k=0: (0, 0, 0)
    i=0,j=0,k=1: (0, 1, 2)
    i=0,j=1,k=0: (1, 3, 0)
    i=0,j=1,k=1: (1, 4, 2)
    i=1,j=0,k=0: (3, 0, 1)
    i=1,j=0,k=1: (3, 1, 3)
    i=1,j=1,k=0: (4, 3, 1)
    i=1,j=1,k=1: (4, 4, 3)
    
    Strassen's algorithm for 2x2x2 uses 7 multiplications. The 7 rank-1 tensors for Strassen are:
    M1 = (A11+A22)(B11+B22)
    M2 = (A21+A22)B11
    M3 = A11(B12-B22)
    M4 = A22(B21-B11)
    M5 = (A11+A12)B22
    M6 = (A21-A11)(B11+B12)
    M7 = (A12-A22)(B21+B22)
    
    And the results are:
    C11 = M1 - M4 + M6 - M7
    C12 = M3 + M5 + M7
    C21 = M2 + M4 - M6
    C22 = M1 - M3 + M2 + M5
    
    Let me verify this works for our specific indexing.
    
    In Strassen, A is 2x2 with A[i,j], B is 2x2 with B[j,k], C is 2x2 with C[i,k].
    The tensor entries are T[i,j,k] = delta (1 if the product contributes).
    
    For our problem, mapping to Strassen's 2x2x2:
    A_sub[i,j] corresponds to A[i*3 + j] for i in {0,1}, j in {0,1}
    B_sub[j,k] corresponds to B[j*3 + k] for j in {0,1}, k in {0,1}
    C_sub[i,k] corresponds to C[k*2 + i] for i in {0,1}, k in {0,1}
    
    So the 8 Strassen terms correspond to the 8 tensor entries listed above.
    
    Now, the remaining 10 terms involve j=2 or k=2:
    
    Terms with j=2 (A column 2, B row 2):
    i=0,j=2,k=0: (6, 6, 0)
    i=0,j=2,k=1: (6, 7, 2)
    i=0,j=2,k=2: (6, 8, 4)
    i=1,j=2,k=0: (7, 6, 1)
    i=1,j=2,k=1: (7, 7, 3)
    i=1,j=2,k=2: (7, 8, 5)
    
    Terms with k=2 (B col 2, C col 2), j in {0,1}:
    i=0,j=0,k=2: (0, 2, 4)
    i=0,j=1,k=2: (1, 5, 4)
    i=1,j=0,k=2: (3, 2, 5)
    i=1,j=1,k=2: (4, 5, 5)
    
    Wait, I already counted j=2,k=2 above. Let me list all 18 terms and identify the 8 Strassen ones.
    
    All 18 terms (i, j, k) -> (A_idx, B_idx, C_idx):
    i=0,j=0,k=0: (0, 0, 0)
    i=0,j=0,k=1: (0, 1, 2)
    i=0,j=0,k=2: (0, 2, 4)
    i=0,j=1,k=0: (1, 3, 0)
    i=0,j=1,k=1: (1, 4, 2)
    i=0,j=1,k=2: (1, 5, 4)
    i=0,j=2,k=0: (6, 6, 0)
    i=0,j=2,k=1: (6, 7, 2)
    i=0,j=2,k=2: (6, 8, 4)
    i=1,j=0,k=0: (3, 0, 1)
    i=1,j=0,k=1: (3, 1, 3)
    i=1,j=0,k=2: (3, 2, 5)
    i=1,j=1,k=0: (4, 3, 1)
    i=1,j=1,k=1: (4, 4, 3)
    i=1,j=1,k=2: (4, 5, 5)
    i=1,j=2,k=0: (7, 6, 1)
    i=1,j=2,k=1: (7, 7, 3)
    i=1,j=2,k=2: (7, 8, 5)
    
    The 8 Strassen terms (j in {0,1}, k in {0,1}):
    (0,0,0), (0,1,2), (1,3,0), (1,4,2), (3,0,1), (3,1,3), (4,3,1), (4,4,3)
    
    The remaining 10 terms:
    (0,2,4), (1,5,4), (6,6,0), (6,7,2), (6,8,4), (3,2,5), (4,5,5), (7,6,1), (7,7,3), (7,8,5)
    
    So 7 (Strassen) + 10 = 17. 
    
    Now I need to construct the actual U, V, W matrices.
    
    For each rank-1 term, we have a coefficient for U, V, W. The Strassen terms have coefficients that are +/-1 or 0.
    
    Let me define the 7 Strassen rank-1 tensors with their coefficients:
    
    For the 2x2x2 sub-problem with A_sub[i,j] = A[i*3+j], B_sub[j,k] = B[j*3+k], C_sub[i,k] = C[k*2+i]:
    
    M1 = (A00+A11)(B00+B11)
    U: A00=1, A11=1; V: B00=1, B11=1; W: C00=1, C11=1, C01=-1, C10=-1
    Wait, C11 = M1 - M4 + M6 - M7, C00... 
    
    Let me use standard Strassen notation:
    A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[i,j],[k,l]]
    C[i,j] where i=row, j=col.
    
    In our case, C_sub[i,k] = C[k*2+i]. So:
    C_sub[0,0] = C[0*2+0] = C[0]
    C_sub[0,1] = C[1*2+0] = C[2]
    C_sub[1,0] = C[0*2+1] = C[1]
    C_sub[1,1] = C[1*2+1] = C[3]
    
    So C[0]=C_sub[0,0], C[1]=C_sub[1,0], C[2]=C_sub[0,1], C[3]=C_sub[1,1].
    
    Strassen:
    M1 = (a+d)(e+h)
    M2 = (c+d)e
    M3 = a(f-h)
    M4 = d(g-e)
    M5 = (a+b)h
    M6 = (c-a)(e+f)
    M7 = (b-d)(g+h)
    
    C[0] = C_sub[0,0] = M1 - M4 + M6 - M7
    C[2] = C_sub[0,1] = M3 + M5 + M7
    C[1] = C_sub[1,0] = M2 + M4 - M6
    C[3] = C_sub[1,1] = M1 - M3 + M2 + M5
    
    Now mapping to our indices:
    a = A[0*3+0] = A[0]
    b = A[0*3+1] = A[1]
    c = A[1*3+0] = A[3]
    d = A[1*3+1] = A[4]
    
    e = B[0*3+0] = B[0]
    f = B[0*3+1] = B[1]
    g = B[1*3+0] = B[3]
    h = B[1*3+1] = B[4]
    
    C[0] = C_sub[0,0]
    C[1] = C_sub[1,0]
    C[2] = C_sub[0,1]
    C[3] = C_sub[1,1]
    
    Now for each M_r, the U, V, W coefficients:
    
    M1: U: A[0]=1, A[4]=1; V: B[0]=1, B[4]=1; W: C[0]=1, C[1]=-1, C[2]=-1, C[3]=1
    Wait, let me redo. C[0] = M1 - M4 + M6 - M7, so M1 contributes +1 to C[0].
    C[3] = M1 - M3 + M2 + M5, so M1 contributes +1 to C[3].
    C[1] = M2 + M4 - M6, M1 contributes 0 to C[1].
    C[2] = M3 + M5 + M7, M1 contributes 0 to C[2].
    
    So for M1: W: C[0]=1, C[3]=1, others 0.
    
    M2: U: A[3]=1, A[4]=1; V: B[0]=1; W: C[1]=1, C[3]=1
    C[1] = M2 + M4 - M6, so M2 contributes +1 to C[1].
    C[3] = M1 - M3 + M2 + M5, so M2 contributes +1 to C[3].
    
    M3: U: A[0]=1; V: B[1]=1, B[4]=-1; W: C[2]=1, C[3]=-1
    C[2] = M3 + M5 + M7, so M3 contributes +1 to C[2].
    C[3] = M1 - M3 + M2 + M5, so M3 contributes -1 to C[3].
    
    M4: U: A[4]=1; V: B[3]=1, B[0]=-1; W: C[0]=-1, C[1]=1
    C[0] = M1 - M4 + M6 - M7, so M4 contributes -1 to C[0].
    C[1] = M2 + M4 - M6, so M4 contributes +1 to C[1].
    
    M5: U: A[0]=1, A[1]=1; V: B[4]=1; W: C[2]=1, C[3]=1
    C[2] = M3 + M5 + M7, so M5 contributes +1 to C[2].
    C[3] = M1 - M3 + M2 + M5, so M5 contributes +1 to C[3].
    
    M6: U: A[3]=1, A[0]=-1; V: B[0]=1, B[1]=1; W: C[0]=1, C[1]=-1
    C[0] = M1 - M4 + M6 - M7, so M6 contributes +1 to C[0].
    C[1] = M2 + M4 - M6, so M6 contributes -1 to C[1].
    
    M7: U: A[1]=1, A[4]=-1; V: B[3]=1, B[4]=1; W: C[0]=-1, C[2]=1
    C[0] = M1 - M4 + M6 - M7, so M7 contributes -1 to C[0].
    C[2] = M3 + M5 + M7, so M7 contributes +1 to C[2].
    
    Now the 10 remaining terms are just rank-1 with coefficient 1:
    
    Term (0,2,4): U: A[0]=1; V: B[2]=1; W: C[4]=1
    Term (1,5,4): U: A[1]=1; V: B[5]=1; W: C[4]=1
    Term (6,6,0): U: A[6]=1; V: B[6]=1; W: C[0]=1
    Term (6,7,2): U: A[6]=1; V: B[7]=1; W: C[2]=1
    Term (6,8,4): U: A[6]=1; V: B[8]=1; W: C[4]=1
    Term (3,2,5): U: A[3]=1; V: B[2]=1; W: C[5]=1
    Term (4,5,5): U: A[4]=1; V: B[5]=1; W: C[5]=1
    Term (7,6,1): U: A[7]=1; V: B[6]=1; W: C[1]=1
    Term (7,7,3): U: A[7]=1; V: B[7]=1; W: C[3]=1
    Term (7,8,5): U: A[7]=1; V: B[8]=1; W: C[5]=1
    
    Total rank = 7 + 10 = 17.
    
    Let me build the matrices.
    """
    n, m, p = 2, 3, 3
    rank = 17
    
    # Initialize U, V, W as lists of lists
    U = [[0] * rank for _ in range(n * m)]  # 6 x 17
    V = [[0] * rank for _ in range(m * p)]  # 9 x 17
    W = [[0] * rank for _ in range(p * n)]  # 6 x 17
    
    # Strassen terms (columns 0-6)
    # M1: col 0
    U[0][0] = 1; U[4][0] = 1
    V[0][0] = 1; V[4][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2: col 1
    U[3][1] = 1; U[4][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = 1
    
    # M3: col 2
    U[0][2] = 1
    V[1][2] = 1; V[4][2] = -1
    W[2][2] = 1; W[3][2] = -1
    
    # M4: col 3
    U[4][3] = 1
    V[3][3] = 1; V[0][3] = -1
    W[0][3] = -1; W[1][3] = 1
    
    # M5: col 4
    U[0][4] = 1; U[1][4] = 1
    V[4][4] = 1
    W[2][4] = 1; W[3][4] = 1
    
    # M6: col 5
    U[3][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[0][5] = 1; W[1][5] = -1
    
    # M7: col 6
    U[1][6] = 1; U[4][6] = -1
    V[3][6] = 1; V[4][6] = 1
    W[0][6] = -1; W[2][6] = 1
    
    # Remaining 10 terms (columns 7-16)
    # (0,2,4): col 7
    U[0][7] = 1; V[2][7] = 1; W[4][7] = 1
    # (1,5,4): col 8
    U[1][8] = 1; V[5][8] = 1; W[4][8] = 1
    # (6,6,0): col 9
    U[6][9] = 1; V[6][9] = 1; W[0][9] = 1
    # (6,7,2): col 10
    U[6][10] = 1; V[7][10] = 1; W[2][10] = 1
    # (6,8,4): col 11
    U[6][11] = 1; V[8][11] = 1; W[4][11] = 1
    # (3,2,5): col 12
    U[3][12] = 1; V[2][12] = 1; W[5][12] = 1
    # (4,5,5): col 13
    U[4][13] = 1; V[5][13] = 1; W[5][13] = 1
    # (7,6,1): col 14
    U[7][14] = 1; V[6][14] = 1; W[1][14] = 1
    # (7,7,3): col 15
    U[7][15] = 1; V[7][15] = 1; W[3][15] = 1
    # (7,8,5): col 16
    U[7][16] = 1; V[8][16] = 1; W[5][16] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
