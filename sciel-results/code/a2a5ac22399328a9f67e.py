"""Matrix multiplication 6x6x6 via Kronecker product of Strassen (2x2) and schoolbook (3x3)."""

def construct():
    """
    Construct a rank-189 decomposition for 6x6 matrix multiplication.
    
    The 6x6 multiplication tensor is decomposed as a Kronecker product of:
    - Strassen's rank-7 algorithm for 2x2 matrix multiplication
    - Schoolbook rank-9 algorithm for 3x3 matrix multiplication
    
    Total rank = 7 * 9 = 63? No wait, let me reconsider.
    
    Actually, for 6x6 = 2*3 x 2*3, we can use a block structure where:
    A is partitioned into 2x2 blocks of size 3x3
    B is partitioned into 2x2 blocks of size 3x3
    
    Then C = A*B uses Strassen's 7 multiplications of 3x3 blocks.
    Each 3x3 multiplication uses rank 9 (schoolbook).
    Total rank = 7 * 9 = 63.
    
    Wait, but that's not quite right either. Let me think more carefully.
    
    If we view 6x6 multiplication as:
    A (6x6) = [A11 A12; A21 A22] where each block is 3x3
    B (6x6) = [B11 B12; B21 B22] where each block is 3x3
    
    Then using Strassen:
    M1 = (A11+A22)(B11+B22)
    M2 = A22(B21-B11)
    M3 = (A11+A12)B22
    M4 = (A21+A22)B11
    M5 = (A11-B22)(B12+B22)
    M6 = (A12-A22)(B11+B21)
    M7 = (A11-A12)(B12-B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    Each Mi is a 3x3 matrix product. Using schoolbook for 3x3, each has rank 9.
    Total rank = 7 * 9 = 63.
    
    Now I need to construct U, V, W matrices.
    
    U has shape (6*6, 63) = (36, 63)
    V has shape (6*6, 63) = (36, 63)
    W has shape (6*6, 63) = (36, 63)
    
    The index convention:
    U[i*m+j, c] where i,j are 0..5 (row,col of A), c is the rank index
    V[j*p+k, c] where j,k are 0..5 (row,col of B), c is the rank index
    W[k*n+i, c] where k,i are 0..5, c is the rank index; W[k*n+i, c] corresponds to C[k,i]... 
    Wait, the task says W[k*n+i, r]. Let me re-read.
    
    "The output factor follows the official trace convention W[k*n+i,r]"
    
    So W has shape (p*n, rank) = (36, 63), and W[k*n+i, r] is the coefficient for C element at position... 
    Let me think about the tensor decomposition.
    
    The matrix multiplication tensor T[i,j,k] = A[i,j] * B[j,k] contributes to C[i,k].
    
    In the decomposition: C = sum_r U[:,r] * V[:,r] * W[:,r] (outer product structure)
    
    More precisely, if we vectorize A as a (n*m)-vector, B as (m*p)-vector, C as (p*n)-vector:
    vec(C)[k*n+i] = sum_r U[i*m+j... 
    
    Hmm, let me be more careful with the indexing.
    
    Actually, for the standard tensor decomposition of matrix multiplication:
    T is a tensor of shape (n*m, m*p, p*n) where T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k, else 0.
    
    Wait, that's not quite right either. Let me think again.
    
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    So the tensor is: for each (i,j,k), there's a term A[i,j] * B[j,k] that contributes to C[i,k].
    
    If we index:
    - A has index (i,j) -> flat index i*m + j (row-major)
    - B has index (j,k) -> flat index j*p + k (row-major)  
    - C has index (i,k) -> flat index... 
    
    The task says W[k*n+i, r]. So C's flat index is k*n + i. That means C is stored with k (column of C, which is row of B) as the major index? That's unusual.
    
    Actually wait, let me re-read: "W[k*n+i,r]" - so the row index of W is k*n+i where k is 0..p-1 and i is 0..n-1. This means C is indexed as C[k,i] in flat form, i.e., column-major if we think of C as n x p, or it's just a specific convention.
    
    Let me just follow the convention: W row = k*n + i, where k comes from B's column and i comes from A's row.
    
    So for the decomposition:
    vec(C)[k*n + i] = sum_j A[i,j] * B[j,k]
    
    Now for the Kronecker product approach:
    
    Let's partition:
    i = i1 * 3 + i2, where i1 in {0,1} (block row), i2 in {0,1,2} (within block)
    j = j1 * 3 + j2, where j1 in {0,1} (block col of A / block row of B), j2 in {0,1,2}
    k = k1 * 3 + k2, where k1 in {0,1} (block col of B / block col of C), k2 in {0,1,2}
    
    Strassen's 7 products:
    M1 = (A11+A22)(B11+B22)
    M2 = A22(B21-B11)
    M3 = (A11+A12)B22
    M4 = (A21+A22)B11
    M5 = (A11-B22)(B12+B22)
    M6 = (A12-A22)(B11+B21)
    M7 = (A11-A12)(B12-B22)
    
    For each M_t, it's a product of two 3x3 matrices.
    
    For M_t = X * Y where X and Y are 3x3:
    X[i2, j2] = sum of certain A[i1*3+i2, j1*3+j2] terms
    Y[j2, k2] = sum of certain B[j1*3+j2, k1*3+k2] terms
    
    Then M_t[i2, k2] = sum_j2' X[i2, j2'] * Y[j2', k2]
    
    And C[i1*3+i2, k1*3+k2] = sum of contributions from various M_t.
    
    Let me define the U, V, W matrices.
    
    The rank index c ranges from 0 to 62 (63 total).
    c = t * 9 + q, where t in 0..6 (Strassen product index), q in 0..8 (schoolbook index within 3x3)
    
    For each t and q, we have:
    q = j2' * 3 + (some index)... 
    
    For schoolbook 3x3: C3[i2, k2] = sum_j2' X3[i2, j2'] * Y3[j2', k2]
    The rank-9 decomposition has q = i2 * 3 + k2? No.
    
    Schoolbook: for each (i2, j2', k2) with i2, j2', k2 in 0..2, but that's 27 terms.
    Rank 9: for each (j2', i2, k2)? No.
    
    Actually, schoolbook 3x3 has rank 9: for each pair (i2, k2), C3[i2,k2] = sum_j2' X3[i2,j2'] Y3[j2',k2].
    The rank-9 decomposition: q indexes (j2', i2, k2)? No, that's 27.
    
    Wait, schoolbook for n x n matrix multiplication has rank n^3. For 3x3, rank 27, not 9.
    
    I made an error. Schoolbook 3x3 has rank 3^3 = 27, not 9.
    
    So total rank = 7 * 27 = 189. That matches the observation in the prompt!
    
    OK so let me redo this with rank 189.
    
    c = t * 27 + q, where t in 0..6, q in 0..26
    q = i2 * 9 + j2' * 3 + k2? Let's define q = (i2 * 3 + j2') * 3 + k2 = i2*9 + j2'*3 + k2.
    
    Hmm, let me just define q as a linear index for the 27 terms in schoolbook 3x3.
    
    For schoolbook 3x3 multiplication:
    C3[i2, k2] = sum_{j2'} X3[i2, j2'] * Y3[j2', k2]
    
    Rank-27 decomposition: for each (i2, j2', k2):
    U3[i2, q] contributes X3[i2, j2']
    V3[j2', q] contributes Y3[j2', k2]
    W3[k2, q] contributes 1 (to C3[i2, k2])
    
    Wait, that's not quite right. Let me be more precise.
    
    For schoolbook n x n:
    T[i,j,k] = A[i,j] * B[j,k] contributes to C[i,k]
    
    Rank n^3 decomposition:
    For each (i, j, k):
    U[i, c] = 1 (selects A[i,j] where c encodes (i,j,k))
    V[j, c] = 1 (selects B[j,k])
    W[k, c] = 1 (contributes to C[i,k])
    
    So U has shape (n*n, n^3), V has shape (n*n, n^3), W has shape (n*n, n^3).
    U[i, c] where c = i*n^2 + j*n + k? No, c should encode which (i,j,k) triple.
    
    Let's say c = i * n^2 + j * n + k.
    Then U[i * n + j... 
    
    Hmm, I'm getting confused with the flattening. Let me just build it step by step.
    
    For the 6x6 case with 2x3 block structure:
    
    A[i, j] where i = i1*3+i2, j = j1*3+j2
    B[j, k] where j = j1*3+j2, k = k1*3+k2
    C[i, k] where i = i1*3+i2, k = k1*3+k2
    
    Flat indices:
    A_flat[i*6+j] = A[i, j]
    B_flat[j*6+k] = B[j, k]
    C_flat[k*6+i] = C[i, k]  (following W[k*n+i] convention)
    
    Strassen block products (t = 0..6):
    For each t, we compute M_t = X_t * Y_t where X_t and Y_t are 3x3 matrices
    formed by linear combinations of the 3x3 blocks of A and B.
    
    Let me define the Strassen formulas:
    t=0: M0 = (A11+A22)(B11+B22)
    t=1: M1 = A22(B21-B11)
    t=2: M2 = (A11+A12)B22
    t=3: M3 = (A21+A22)B11
    t=4: M4 = (A11-B22)(B12+B22)
    t=5: M5 = (A12-A22)(B11+B21)
    t=6: M6 = (A11-A12)(B12-B22)
    
    Where Aab is the block with row-block a, col-block b (a,b in {0,1}), similarly for B.
    
    For each t, X_t[i2, j2] = sum over (a, j1) of coeff_X[t][a, j1, i2, j2] * A[a*3+i2, j1*3+j2]
    Y_t[j2, k2] = sum over (j1, b) of coeff_Y[t][j1, b, j2, k2] * B[j1*3+j2, b*3+k2]
    
    Then M_t[i2, k2] = sum_{j2'} X_t[i2, j2'] * Y_t[j2', k2]
    
    And C[i1*3+i2, k1*3+k2] = sum_t coeff_C[t][i1, k1] * M_t[i2, k2]
    
    Let me define the coefficient tensors.
    
    For X_t (left factor):
    t=0: X0 = A11 + A22, so X0[i2,j2] = A[0*3+i2, 0*3+j2] + A[1*3+i2, 1*3+j2]
        coeff: (i1=0, j1=0, +1) and (i1=1, j1=1, +1)
    t=1: X1 = A22, so X1[i2,j2] = A[1*3+i2, 1*3+j2]
        coeff: (i1=1, j1=1, +1)
    t=2: X2 = A11 + A12, so X2[i2,j2] = A[0*3+i2, 0*3+j2] + A[0*3+i2, 1*3+j2]
        coeff: (i1=0, j1=0, +1) and (i1=0, j1=1, +1)
    t=3: X3 = A21 + A22, so X3[i2,j2] = A[1*3+i2, 0*3+j2] + A[1*3+i2, 1*3+j2]
        coeff: (i1=1, j1=0, +1) and (i1=1, j1=1, +1)
    t=4: X4 = A11 - A22, so X4[i2,j2] = A[0*3+i2, 0*3+j2] - A[1*3+i2, 1*3+j2]
        coeff: (i1=0, j1=0, +1) and (i1=1, j1=1, -1)
    t=5: X5 = A12 - A22, so X5[i2,j2] = A[0*3+i2, 1*3+j2] - A[1*3+i2, 1*3+j2]
        coeff: (i1=0, j1=1, +1) and (i1=1, j1=1, -1)
    t=6: X6 = A11 - A12, so X6[i2,j2] = A[0*3+i2, 0*3+j2] - A[0*3+i2, 1*3+j2]
        coeff: (i1=0, j1=0, +1) and (i1=0, j1=1, -1)
    
    For Y_t (right factor):
    t=0: Y0 = B11 + B22, Y0[j2,k2] = B[0*3+j2, 0*3+k2] + B[1*3+j2, 1*3+k2]
        coeff: (j1=0, k1=0, +1) and (j1=1, k1=1, +1)
    t=1: Y1 = B21 - B11, Y1[j2,k2] = B[1*3+j2, 0*3+k2] - B[0*3+j2, 0*3+k2]
        coeff: (j1=1, k1=0, +1) and (j1=0, k1=0, -1)
    t=2: Y2 = B22, Y2[j2,k2] = B[1*3+j2, 1*3+k2]
        coeff: (j1=1, k1=1, +1)
    t=3: Y3 = B11, Y3[j2,k2] = B[0*3+j2, 0*3+k2]
        coeff: (j1=0, k1=0, +1)
    t=4: Y4 = B12 + B22, Y4[j2,k2] = B[0*3+j2, 1*3+k2] + B[1*3+j2, 1*3+k2]
        coeff: (j1=0, k1=1, +1) and (j1=1, k1=1, +1)
    t=5: Y5 = B11 + B21, Y5[j2,k2] = B[0*3+j2, 0*3+k2] + B[1*3+j2, 0*3+k2]
        coeff: (j1=0, k1=0, +1) and (j1=1, k1=0, +1)
    t=6: Y6 = B12 - B22, Y6[j2,k2] = B[0*3+j2, 1*3+k2] - B[1*3+j2, 1*3+k2]
        coeff: (j1=0, k1=1, +1) and (j1=1, k1=1, -1)
    
    For C reconstruction:
    C11 = M0 + M3 - M4 + M6
    C12 = M2 + M4
    C21 = M1 + M3
    C22 = M0 - M1 + M2 + M5
    
    So coeff_C[t][i1][k1]:
    t=0: C11+=1, C22+=1 -> (i1=0,k1=0):1, (i1=1,k1=1):1
    t=1: C21+=1, C22-=1 -> (i1=1,k1=0):1, (i1=1,k1=1):-1
    t=2: C12+=1, C22+=1 -> (i1=0,k1=1):1, (i1=1,k1=1):1
    t=3: C11+=1, C21+=1 -> (i1=0,k1=0):1, (i1=1,k1=0):1
    t=4: C11-=1, C12+=1 -> (i1=0,k1=0):-1, (i1=0,k1=1):1
    t=5: C22+=1 -> (i1=1,k1=1):1
    t=6: C11+=1 -> (i1=0,k1=0):1
    
    Now, for the full decomposition:
    Rank = 7 * 27 = 189
    c = t * 27 + q, where t in 0..6, q in 0..26
    
    For schoolbook 3x3 within each t:
    q = i2 * 9 + j2' * 3 + k2 where i2, j2', k2 in 0..2
    
    U[A_flat_index, c]:
    A_flat_index = i*6+j = (i1*3+i2)*6 + (j1*3+j2)
    For given t and q (which encodes i2, j2', k2):
    U[(i1*3+i2)*6 + (j1*3+j2), t*27+q] = coeff_X[t][i1, j1]
    (j2 = j2', the middle index)
    
    V[B_flat_index, c]:
    B_flat_index = j*6+k = (j1*3+j2)*6 + (k1*3+k2)
    V[(j1*3+j2)*6 + (k1*3+k2), t*27+q] = coeff_Y[t][j1, k1]
    (j2 = j2')
    
    W[C_flat_index, c]:
    C_flat_index = k*6+i = (k1*3+k2)*6 + (i1*3+i2)
    W[(k1*3+k2)*6 + (i1*3+i2), t*27+q] = coeff_C[t][i1, k1]
    
    Let me code this up.
    """
    import time
    start_time = time.time()
    
    # Deadline guard
    deadline = start_time + 110  # 110 seconds, leaving margin
    
    n, m, p = 6, 6, 6
    rank = 189
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define Strassen coefficients
    # coeff_X[t][i1][j1] = coefficient for A[i1*3+i2, j1*3+j2] in X_t
    coeff_X = [
        # t=0: A11 + A22
        {(0,0): 1, (1,1): 1},
        # t=1: A22
        {(1,1): 1},
        # t=2: A11 + A12
        {(0,0): 1, (0,1): 1},
        # t=3: A21 + A22
        {(1,0): 1, (1,1): 1},
        # t=4: A11 - A22
        {(0,0): 1, (1,1): -1},
        # t=5: A12 - A22
        {(0,1): 1, (1,1): -1},
        # t=6: A11 - A12
        {(0,0): 1, (0,1): -1},
    ]
    
    # coeff_Y[t][j1][k1] = coefficient for B[j1*3+j2, k1*3+k2] in Y_t
    coeff_Y = [
        # t=0: B11 + B22
        {(0,0): 1, (1,1): 1},
        # t=1: B21 - B11
        {(1,0): 1, (0,0): -1},
        # t=2: B22
        {(1,1): 1},
        # t=3: B11
        {(0,0): 1},
        # t=4: B12 + B22
        {(0,1): 1, (1,1): 1},
        # t=5: B11 + B21
        {(0,0): 1, (1,0): 1},
        # t=6: B12 - B22
        {(0,1): 1, (1,1): -1},
    ]
    
    # coeff_C[t][i1][k1] = coefficient for M_t in C[i1*3+i2, k1*3+k2]
    coeff_C = [
        # t=0: C11+=1, C22+=1
        {(0,0): 1, (1,1): 1},
        # t=1: C21+=1, C22-=1
        {(1,0): 1, (1,1): -1},
        # t=2: C12+=1, C22+=1
        {(0,1): 1, (1,1): 1},
        # t=3: C11+=1, C21+=1
        {(0,0): 1, (1,0): 1},
        # t=4: C11-=1, C12+=1
        {(0,0): -1, (0,1): 1},
        # t=5: C22+=1
        {(1,1): 1},
        # t=6: C11+=1
        {(0,0): 1},
    ]
    
    # Build U, V, W
    for t in range(7):
        for i2 in range(3):
            for j2p in range(3):  # j2' (middle index in schoolbook)
                for k2 in range(3):
                    q = i2 * 9 + j2p * 3 + k2
                    c = t * 27 + q
                    
                    # U: for each (i1, j1) in coeff_X[t]
                    for (i1, j1), val in coeff_X[t].items():
                        a_flat = (i1 * 3 + i2) * 6 + (j1 * 3 + j2p)
                        U[a_flat][c] = val
                    
                    # V: for each (j1, k1) in coeff_Y[t]
                    for (j1, k1), val in coeff_Y[t].items():
                        b_flat = (j1 * 3 + j2p) * 6 + (k1 * 3 + k2)
                        V[b_flat][c] = val
                    
                    # W: for each (i1, k1) in coeff_C[t]
                    for (i1, k1), val in coeff_C[t].items():
                        c_flat = (k1 * 3 + k2) * 6 + (i1 * 3 + i2)
                        W[c_flat][c] = val
    
    # Time check
    if time.time() > deadline:
        pass  # Already computed, just return
    
    return {"U": U, "V": V, "W": W}
