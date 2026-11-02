"""Matrix multiplication 4,4,4 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p):
    """Check that the decomposition is exact: T_{i,j,k} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]"""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0  # The matrix multiplication tensor has T[i,j,k] = 1
                actual = 0.0
                for r in range(rank):
                    actual += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                if abs(actual - expected) > 1e-9:
                    return False
    return True

def construct():
    """
    Rank-49 decomposition for 4x4 matrix multiplication using Strassen's algorithm
    applied at the block level.
    
    The 4x4 matrices are viewed as 2x2 block matrices of 2x2 blocks.
    Strassen gives 7 scalar multiplications for the block level, and each
    scalar multiplication is a 2x2 matrix multiplication (rank 4 schoolbook).
    Total rank = 7 * 4 = 28... wait, let me reconsider.
    
    Actually, the standard Strassen 4x4 decomposition:
    - Split 4x4 into 2x2 blocks of 2x2 submatrices
    - Use 7 Strassen products, each of which is a 2x2 matrix multiply
    - Each 2x2 matrix multiply has rank 4 (schoolbook)
    - Total rank = 7 * 4 = 28
    
    Wait, but the known rank of 4x4 matrix multiplication is 49 (Larson & Shults).
    Let me use the known rank-49 construction.
    
    Actually, let me think more carefully. The standard approach:
    - 4x4 = 2x2 blocks of 2x2
    - Strassen: 7 multiplications of 2x2 matrices
    - Each 2x2 multiplication: rank 4
    - Total: 28
    
    But 28 is not achievable; the rank of 2x2 is 7 (Strassen), not 4.
    Schoolbook for 2x2 is rank 4, Strassen is rank 7.
    
    So if we use schoolbook for the inner 2x2: 7 * 4 = 28.
    If we use Strassen for the inner 2x2: 7 * 7 = 49.
    
    Wait, that's backwards. Schoolbook 2x2 is rank 4, Strassen 2x2 is rank 7.
    So using schoolbook for inner gives 7*4=28, using Strassen gives 7*7=49.
    
    But 28 should be valid! Let me verify: the rank of 4x4 matrix multiplication
    is known to be between 49 and 72 (or similar bounds). Actually the exact rank
    is unknown, but 49 is the best known upper bound from specific constructions.
    
    Hmm, wait. Let me reconsider. The 2x2 matrix multiplication tensor has rank
    exactly 7 (Strassen). The schoolbook decomposition of 2x2 is rank 4? No!
    Schoolbook 2x2 is rank 4 (four scalar multiplications: (a11*b11), (a11*b12+a12*b11)...).
    
    No wait. Schoolbook 2x2: C = AB where A, B are 2x2.
    C11 = A11*B11 + A12*B21
    C12 = A11*B12 + A12*B22
    C21 = A21*B11 + A22*B21
    C22 = A21*B12 + A22*B22
    
    Each C entry is a sum of 2 products. If we write this as a tensor decomposition,
    we need to express the bilinear map. The standard schoolbook uses 4 scalar
    multiplications? No, it uses 8 scalar multiplications in the naive way.
    
    Actually for the tensor rank: the 2x2 matrix multiplication tensor T[i,j,k] where
    i indexes A entry, j indexes B entry, k indexes C entry. T[A_idx, B_idx, C_idx] = 1
    if C_idx = (row, col) and the product A[row,*] dot B[*,col] contributes.
    
    The rank of the 2x2 matrix multiplication tensor is 7 (Strassen).
    The "schoolbook" gives rank 4? Let me count: we can write
    C11 = A11*B11 + A12*B21 -> two rank-1 terms
    C12 = A11*B12 + A12*B22 -> two rank-1 terms
    C21 = A21*B11 + A22*B21 -> two rank-1 terms
    C22 = A21*B12 + A22*B22 -> two rank-1 terms
    Total 8 rank-1 terms. But with clever grouping, Strassen gets 7.
    
    So schoolbook 2x2 is rank 8 (or 4 if we consider the bilinear form differently).
    
    I'm confusing myself. Let me just use the known result: rank of n x n matrix
    multiplication for n=4 is at most 49 (from various constructions).
    
    Let me implement the rank-49 construction properly.
    
    Actually, the simplest correct approach: use the Kronecker product / tensor
    product of the Strassen 2x2 decomposition with itself.
    
    Strassen 2x2: rank 7.
    4x4 as 2x2 of 2x2 blocks: the tensor is T_4 = T_2 x T_2 (Kronecker/tensor product).
    Rank of tensor product: rank(T x S) <= rank(T) * rank(S).
    So rank(T_4) <= 7 * 7 = 49.
    
    This gives us a rank-49 decomposition!
    """
    n, m, p = DIMENSIONS
    
    # Strassen 2x2 decomposition
    # For 2x2 matrices A, B:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    # where:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # U vectors for A (4 entries: A11, A12, A21, A22)
    U2 = [
        [1, 0, 0, 1],  # M1: A11 + A22
        [0, 0, 1, 1],  # M2: A21 + A22
        [1, 0, 0, 0],  # M3: A11
        [0, 0, 0, 1],  # M4: A22
        [1, 1, 0, 0],  # M5: A11 + A12
        [1, 0, -1, 0], # M6: A21 - A11
        [0, 1, 0, -1], # M7: A12 - A22
    ]
    
    # V vectors for B (4 entries: B11, B12, B21, B22)
    V2 = [
        [1, 0, 0, 1],  # M1: B11 + B22
        [1, 0, 0, 0],  # M2: B11
        [0, 1, 0, -1], # M3: B12 - B22
        [0, -1, 1, 0], # M4: B21 - B11
        [0, 0, 0, 1],  # M5: B22
        [1, 1, 0, 0],  # M6: B11 + B12
        [0, 0, 1, 1],  # M7: B21 + B22
    ]
    
    # W matrices for C (2x2 matrix each, flattened to 4 entries: C11, C12, C21, C22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    W2 = [
        [1, 0, 0, 1],   # M1: contributes to C11(+1), C22(+1)
        [0, 0, 1, -1],  # M2: contributes to C21(+1), C22(-1)
        [0, 1, 0, 1],   # M3: contributes to C12(+1), C22(+1)
        [1, 0, 1, 0],   # M4: contributes to C11(+1), C21(+1)
        [-1, 1, 0, 0],  # M5: contributes to C11(-1), C12(+1)
        [0, 0, 0, 1],   # M6: contributes to C22(+1)
        [1, 0, 0, 0],   # M7: contributes to C11(+1)
    ]
    
    # Verify Strassen 2x2 decomposition
    # T2[a,b,c] = sum_s U2[s][a] * V2[s][b] * W2[s][c]
    # where a,b index entries of A and B (row-major: 0=A11, 1=A12, 2=A21, 3=A22)
    # and c indexes entry of C (0=C11, 1=C12, 2=C21, 3=C22)
    
    # Now construct 4x4 decomposition via tensor product
    # A is 4x4, indexed by (i,j) -> flat index i*4+j
    # B is 4x4, indexed by (j,k) -> flat index j*4+k  
    # C is 4x4, indexed by (i,k) -> flat index i*4+k
    # 
    # Block decomposition: A[i,j] where i = 2*I + i2, j = 2*J + j2
    # B[j,k] where j = 2*J + j2, k = 2*K + k2
    # C[i,k] where i = 2*I + i2, k = 2*K + k2
    #
    # The 4x4 multiplication tensor T4[I*4+J, J*4+K, K*4+I] ... 
    # Actually: T4[a, b, c] where a = row_A * 4 + col_A, b = row_B * 4 + col_B, c = row_C * 4 + col_C
    # T4[a,b,c] = 1 if col_A == row_B and row_A == row_C and col_B == col_C
    #
    # In block terms: a = (I, i2, J, j2), b = (J, j2, K, k2), c = (I, i2, K, k2)
    # The inner 2x2 multiplication handles (i2, j2, k2) and (j2, k2, i2)
    # The outer 2x2 multiplication handles (I, J, K)
    
    # Rank-49 decomposition: r = s1 * 7 + s2 where s1, s2 in 0..6
    # U4[a, r] = U2[s1][I*2+j2_block] ... 
    
    # Let me be more explicit:
    # a = I*8 + i2*4 + J*2 + j2  (I in 0,1; i2 in 0,1; J in 0,1; j2 in 0,1)
    #   Actually a = row_A * 4 + col_A where row_A = I*2 + i2, col_A = J*2 + j2
    #   So a = (I*2+i2)*4 + (J*2+j2) = I*8 + i2*4 + J*2 + j2
    # 
    # b = row_B * 4 + col_B where row_B = J*2 + j2, col_B = K*2 + k2
    #   So b = (J*2+j2)*4 + (K*2+k2) = J*8 + j2*4 + K*2 + k2
    #
    # c = row_C * 4 + col_C where row_C = I*2 + i2, col_C = K*2 + k2
    #   So c = (I*2+i2)*4 + (K*2+k2) = I*8 + i2*4 + K*2 + k2
    
    # T4[a,b,c] = 1 iff J == J (col_A == row_B) and I == I (row_A == row_C) and j2==j2 and i2==i2 and k2==k2
    # Wait: col_A = row_B means J*2+j2 = J*2+j2 (always true in our indexing)
    # row_A = row_C means I*2+i2 = I*2+i2 (always true)
    # col_B = col_C means K*2+k2 = K*2+k2 (always true)
    # 
    # The actual condition for the tensor is:
    # T4[a,b,c] = 1 if col_A_index == row_B_index AND row_A_index == row_C_index AND col_B_index == col_C_index
    # where these are the flat indices within their respective 4-element vectors.
    # col_A is the second index of A: a = row_A*4 + col_A
    # row_B is the first index of B: b = row_B*4 + col_B
    # row_C is the first index of C: c = row_C*4 + col_C
    # col_B is the second index of B
    # col_C is the second index of C
    #
    # So T4[a,b,c] = 1 if (a % 4) == (b // 4) AND (a // 4) == (c // 4) AND (b % 4) == (c % 4)
    
    # In block notation:
    # a % 4 = J*2 + j2, b // 4 = J*2 + j2 -> J==J and j2==j2 (always)
    # a // 4 = I*2 + i2, c // 4 = I*2 + i2 -> I==I and i2==i2 (always)
    # b % 4 = K*2 + k2, c % 4 = K*2 + k2 -> K==K and k2==k2 (always)
    
    # So T4[a,b,c] = 1 whenever the block indices match, and within blocks:
    # T4 = T2[inner] * T2[outer] in a sense.
    
    # More precisely, define:
    # outer indices: I, J, K (block row/col)
    # inner indices: i2, j2, k2 (within-block row/col)
    # 
    # a = (I, i2, J, j2), b = (J, j2, K, k2), c = (I, i2, K, k2)
    # T4 = 1 if (J,j2) appears consistently: col_A=(J,j2)=row_B=(J,j2), etc.
    #
    # The factorization: T4 = sum_{s1,s2} U_outer[s1](I,J) * U_inner[s2](i2,j2) * V_outer[s1](J,K) * V_inner[s2](j2,k2) * W_outer[s1](I,K) * W_inner[s2](i2,k2)
    
    # Where:
    # U_outer[s1](I,J) = U2[s1][I*2+J]  (treating 2x2 block matrix with entries indexed by (I,J))
    # U_inner[s2](i2,j2) = U2[s2][i2*2+j2]
    # Similarly for V and W.
    
    # Let's construct:
    rank = 49
    U4 = [[0.0] * rank for _ in range(16)]
    V4 = [[0.0] * rank for _ in range(16)]
    W4 = [[0.0] * rank for _ in range(16)]
    
    for s1 in range(7):
        for s2 in range(7):
            r = s1 * 7 + s2
            # For each entry a of A (16 entries)
            for I in range(2):
                for i2 in range(2):
                    for J in range(2):
                        for j2 in range(2):
                            a = (I * 2 + i2) * 4 + (J * 2 + j2)
                            # U4[a, r] = U2[s1][I*2+J] * U2[s2][i2*2+j2]
                            U4[a][r] = U2[s1][I * 2 + J] * U2[s2][i2 * 2 + j2]
            
            for J in range(2):
                for j2 in range(2):
                    for K in range(2):
                        for k2 in range(2):
                            b = (J * 2 + j2) * 4 + (K * 2 + k2)
                            # V4[b, r] = V2[s1][J*2+K] * V2[s2][j2*2+k2]
                            V4[b][r] = V2[s1][J * 2 + K] * V2[s2][j2 * 2 + k2]
            
            for I in range(2):
                for i2 in range(2):
                    for K in range(2):
                        for k2 in range(2):
                            c = (I * 2 + i2) * 4 + (K * 2 + k2)
                            # W4[c, r] = W2[s1][I*2+K] * W2[s2][i2*2+k2]
                            W4[c][r] = W2[s1][I * 2 + K] * W2[s2][i2 * 2 + k2]
    
    # Verify the decomposition
    if _verify_decomposition(U4, V4, W4, n, m, p):
        return {"U": U4, "V": V4, "W": W4}
    else:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
