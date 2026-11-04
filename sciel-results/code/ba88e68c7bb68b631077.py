import numpy as np
import time

DIMENSIONS = [2, 3, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # We will use the structural approach from Inspiration 2:
    # 1. Decompose the 2x3x7 multiplication into blocks.
    #    - The 2x3 matrix A can be seen as two 2x2 blocks (A_left, A_right) plus a 2x1 column.
    #    - The 3x7 matrix B can be seen as three 2x2 blocks (B_0, B_1, B_2) plus a 1x7 row.
    #    Wait, the dimensions are A(2x3) * B(3x7).
    #    Let's split the inner dimension m=3 into 2+1.
    #    A = [A_0 (2x2), A_1 (2x1)]
    #    B = [B_0 (2x7), B_1 (1x7)]
    #    C = A_0 B_0 + A_1 B_1
    #    
    #    We can apply Strassen to A_0 B_0 (2x2 * 2x7).
    #    This requires splitting the 7 columns of B_0 into 2x2 blocks?
    #    B_0 is 2x7. We can split columns into 2+2+2+1.
    #    So B_0 = [B_00 (2x2), B_01 (2x2), B_02 (2x2), B_03 (2x1)].
    #    
    #    Then A_0 B_0 = A_0 B_00 + A_0 B_01 + A_0 B_02 + A_0 B_03.
    #    Each A_0 B_0i is 2x2 * 2x2 (for i=0,1,2) which can be done with rank 7 (Strassen).
    #    A_0 B_03 is 2x2 * 2x1 which can be done with rank 4 (schoolbook) or potentially lower?
    #    2x2 * 2x1 is equivalent to multiplying 2x2 by a vector.
    #    Rank of 2x2x1 tensor:
    #    (A00+A11)(B0+B1), (A10+A11)B0, A00(B1-B0), A11(B0-B1)... 
    #    Actually, 2x2x1 is just 4 multiplications in schoolbook. Can we do better?
    #    The rank of the 2x2x1 tensor is 4. (It's just the bilinear map (a,b,c,d) -> 4 outputs).
    #    Wait, the tensor T[i,j,k] for C = AB.
    #    For 2x2 * 2x1, C is 2x1.
    #    C0 = A00 B0 + A01 B1
    #    C1 = A10 B0 + A11 B1
    #    This is 4 multiplications. Rank is 4.
    #    
    #    So for A_0 B_0 part:
    #    3 blocks of 2x2x2 (rank 7 each) -> 21
    #    1 block of 2x2x1 (rank 4) -> 4
    #    Total for A_0 B_0 = 25.
    #    
    #    Now for A_1 B_1 part:
    #    A_1 is 2x1, B_1 is 1x7.
    #    C_add = A_1 B_1.
    #    C_add is 2x7.
    #    C_add[0, k] = A_1[0] * B_1[0, k]
    #    C_add[1, k] = A_1[1] * B_1[0, k]
    #    This requires 2 * 7 = 14 multiplications?
    #    Or can we do better?
    #    The tensor is 2x1x7.
    #    Rank of 2x1x7 tensor?
    #    It's just outer product of vector a (2x1) and b (1x7).
    #    This is rank 1 in terms of matrix rank, but tensor rank?
    #    T[i, j, k] = A[i, 0] * B[0, k]. Note j is always 0.
    #    Wait, the index j in the middle dimension is fixed to 0 for this block.
    #    So we have terms A[i] B[k].
    #    This is a single rank-1 tensor? No, it's a sum of products?
    #    Actually, the multiplication is sum_j A[i,j] B[j,k].
    #    Here j is fixed to the 3rd column (index 2).
    #    So C[i,k] += A[i,2] B[2,k].
    #    This is 2 * 7 = 14 scalar multiplications.
    #    Can we combine them?
    #    The tensor for this part has non-zero entries at (i, 2, k) for all i, k.
    #    Value is 1.
    #    This is a tensor of shape (2, 3, 7) with support only on slice j=2.
    #    It is the outer product of e_2 (in dim 3) with the 2x7 all-ones matrix?
    #    No, it's sum over r of U[i,r] V[2,r] W[k,r].
    #    We need U[i,r] V[2,r] W[k,r] = 1 for all i,k when r corresponds to this block.
    #    If we use rank 14, we set U[i,r]=1, V[2,r]=1, W[k,r]=1 for r=1..14.
    #    Can we do it with lower rank?
    #    The matrix M[i,k] = 1 is rank 1.
    #    The tensor T[i,j,k] = delta_{j,2} * 1.
    #    This tensor is the outer product of (delta_{j,2}) and (1_{i,k}).
    #    The 1_{i,k} matrix is rank 1, so it can be decomposed as u v^T.
    #    So T = delta_{j,2} \otimes (u v^T) = (delta_{j,2} \otimes u) (v^T).
    #    This suggests rank 1?
    #    Let's check.
    #    T[i,j,k] = U[i,r] V[j,r] W[k,r].
    #    We want T[i,2,k] = 1 and T[i,j,k]=0 otherwise.
    #    If we set r=0:
    #    V[2,0] = 1, V[j,0] = 0 for j!=2.
    #    U[i,0] = 1 for all i.
    #    W[k,0] = 1 for all k.
    #    Then sum_r U[i,r]V[j,r]W[k,r] = 1*1*1 = 1 if j=2, else 0.
    #    Yes! The rank of the "column times row" part is 1?
    #    Wait, U is (n*m, rank) = (6, rank).
    #    V is (m*p, rank) = (21, rank).
    #    W is (p*n, rank) = (14, rank).
    #    In the block A_1 B_1, the indices are:
    #    A index: i*3 + 2. (i=0,1). So rows 2 and 5 of U.
    #    B index: 2*7 + k. (k=0..6). So rows 14..20 of V.
    #    C index: k*2 + i.
    #    
    #    If we use 1 term for this:
    #    U[2, r] = 1, U[5, r] = 1.
    #    V[14, r] = 1, ..., V[20, r] = 1.
    #    W[0, r] = 1, ..., W[13, r] = 1.
    #    Then for any i, k:
    #    Sum_r U[i*3+2, r] V[2*7+k, r] W[k*2+i, r]
    #    = 1 * 1 * 1 = 1.
    #    This contributes 1 to C[i,k].
    #    The term is A[i,2] * B[2,k].
    #    Yes, this works with rank 1.
    #    
    #    So total rank = 25 (for A_0 B_0) + 1 (for A_1 B_1) = 26.
    #    
    #    Let's verify the A_0 B_0 decomposition.
    #    A_0 is 2x2 (rows 0,1 of A, cols 0,1). Indices in U: 0,1,3,4.
    #    B_0 is 2x7 (rows 0,1 of B, cols 0..6). Indices in V: 0..6, 7..13.
    #    We split B_0 into B_00 (cols 0,1), B_01 (cols 2,3), B_02 (cols 4,5), B_03 (cols 6).
    #    
    #    Block 0 (cols 0,1): 2x2 * 2x2. Rank 7.
    #    Block 1 (cols 2,3): 2x2 * 2x2. Rank 7.
    #    Block 2 (cols 4,5): 2x2 * 2x2. Rank 7.
    #    Block 3 (col 6): 2x2 * 2x1. Rank 4.
    #    
    #    Total rank = 7+7+7+4+1 = 26.
    #    
    #    This is a significant improvement over 42 (schoolbook) and 39 (previous best).
    #    
    #    Let's construct the matrices.

    rank = 26
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    def add_strassen_2x2x2(u_base, v_base, w_base):
        """
        Adds 7 terms for 2x2 * 2x2 multiplication.
        u_base: list of 4 indices [u00, u01, u10, u11]
        v_base: list of 4 indices [v00, v01, v10, v11]
        w_base: list of 4 indices [w00, w01, w10, w11]
        """
        nonlocal col
        # M1: (A00+A11)(B00+B11) -> C00, C11
        U[u_base[0], col] += 1
        U[u_base[3], col] += 1
        V[v_base[0], col] += 1
        V[v_base[3], col] += 1
        W[w_base[0], col] += 1
        W[w_base[3], col] += 1
        col += 1
        
        # M2: (A10+A11)B00 -> C10, -C11
        U[u_base[2], col] += 1
        U[u_base[3], col] += 1
        V[v_base[0], col] += 1
        W[w_base[2], col] += 1
        W[w_base[3], col] -= 1
        col += 1
        
        # M3: A00(B01-B11) -> C01, C11
        U[u_base[0], col] += 1
        V[v_base[1], col] += 1
        V[v_base[3], col] -= 1
        W[w_base[1], col] += 1
        W[w_base[3], col] += 1
        col += 1
        
        # M4: A11(B10-B00) -> C00, C10
        U[u_base[3], col] += 1
        V[v_base[2], col] += 1
        V[v_base[0], col] -= 1
        W[w_base[0], col] += 1
        W[w_base[2], col] += 1
        col += 1
        
        # M5: (A00+A01)B11 -> -C00, C01
        U[u_base[0], col] += 1
        U[u_base[1], col] += 1
        V[v_base[3], col] += 1
        W[w_base[0], col] -= 1
        W[w_base[1], col] += 1
        col += 1
        
        # M6: (A10-A00)(B00+B01) -> C11
        U[u_base[2], col] += 1
        U[u_base[0], col] -= 1
        V[v_base[0], col] += 1
        V[v_base[1], col] += 1
        W[w_base[3], col] += 1
        col += 1
        
        # M7: (A01-A11)(B10+B11) -> C00
        U[u_base[1], col] += 1
        U[u_base[3], col] -= 1
        V[v_base[2], col] += 1
        V[v_base[3], col] += 1
        W[w_base[0], col] += 1
        col += 1

    def add_2x2x1(u_base, v_base, w_base):
        """
        Adds 4 terms for 2x2 * 2x1 multiplication.
        u_base: list of 4 indices [u00, u01, u10, u11]
        v_base: list of 2 indices [v0, v1]
        w_base: list of 2 indices [w0, w1]
        """
        nonlocal col
        # C0 = A00 v0 + A01 v1
        U[u_base[0], col] = 1
        V[v_base[0], col] = 1
        W[w_base[0], col] = 1
        col += 1
        
        U[u_base[1], col] = 1
        V[v_base[1], col] = 1
        W[w_base[0], col] = 1
        col += 1
        
        # C1 = A10 v0 + A11 v1
        U[u_base[2], col] = 1
        V[v_base[0], col] = 1
        W[w_base[1], col] = 1
        col += 1
        
        U[u_base[3], col] = 1
        V[v_base[1], col] = 1
        W[w_base[1], col] = 1
        col += 1

    # A indices for 2x2 block (rows 0,1; cols 0,1)
    # U indices: i*3 + j
    u_2x2 = [0*3+0, 0*3+1, 1*3+0, 1*3+1] # [0, 1, 3, 4]
    
    # B indices for 2x7 block (rows 0,1; cols 0..6)
    # V indices: j*7 + k
    
    # Block 0: B cols 0,1
    v_b0 = [0*7+0, 0*7+1, 1*7+0, 1*7+1] # [0, 1, 7, 8]
    # W indices for C cols 0,1 (k=0,1)
    # W indices: k*2 + i
    w_c0 = [0*2+0, 0*2+1, 1*2+0, 1*2+1] # [0, 1, 2, 3]
    add_strassen_2x2x2(u_2x2, v_b0, w_c0)
    
    # Block 1: B cols 2,3
    v_b1 = [0*7+2, 0*7+3, 1*7+2, 1*7+3] # [2, 3, 9, 10]
    w_c1 = [2*2+0, 2*2+1, 3*2+0, 3*2+1] # [4, 5, 6, 7]
    add_strassen_2x2x2(u_2x2, v_b1, w_c1)
    
    # Block 2: B cols 4,5
    v_b2 = [0*7+4, 0*7+5, 1*7+4, 1*7+5] # [4, 5, 11, 12]
    w_c2 = [4*2+0, 4*2+1, 5*2+0, 5*2+1] # [8, 9, 10, 11]
    add_strassen_2x2x2(u_2x2, v_b2, w_c2)
    
    # Block 3: B col 6
    v_b3 = [0*7+6, 1*7+6] # [6, 13]
    w_c3 = [6*2+0, 6*2+1] # [12, 13]
    add_2x2x1(u_2x2, v_b3, w_c3)
    
    # Block 4: A col 2, B row 2
    # A indices: i*3 + 2 -> [2, 5]
    # B indices: 2*7 + k -> [14, 15, 16, 17, 18, 19, 20]
    # W indices: k*2 + i -> all 0..13
    
    # Rank 1 term for this block
    U[2, col] = 1
    U[5, col] = 1
    for k in range(p):
        V[2*7 + k, col] = 1
    for k in range(p):
        for i in range(n):
            W[k*2 + i, col] = 1
    col += 1
    
    # Verify
    # T[alpha, beta, gamma]
    # alpha = i*m + j
    # beta = j*p + k
    # gamma = k*n + i
    
    # Check time
    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
