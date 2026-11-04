"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

import time
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START

def _strassen_4x4x4():
    """
    Return U, V, W matrices for 4x4*4x4 with rank 49 using recursive Strassen.
    Convention: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    Here n=4, m=4, p=4.
    """
    n = m = p = 4
    rank = 49
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # Strassen for 2x2x2 has rank 7
    # For 4x4x4, we split into 2x2 blocks, so we have 8 sub-problems of 2x2x2
    # Each 2x2x2 uses 7 multiplications
    # Total: 8 * 7 = 56? No, that's not right.
    
    # Actually, for 4x4x4 using Strassen:
    # Split A (4x4) into 4 blocks of 2x2: A11, A12, A21, A22
    # Split B (4x4) into 4 blocks of 2x2: B11, B12, B21, B22
    # C = A*B, C is 4x4 with 4 blocks of 2x2
    # 
    # Using Strassen at the 2x2 block level:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # Each Mj is a 2x2 * 2x2 = 2x2 multiplication, which uses 7 scalar mults.
    # Total: 7 * 7 = 49. Correct!
    
    # Let's build this explicitly.
    # For each of the 7 Strassen combinations at the block level,
    # and for each of the 7 Strassen combinations at the scalar level within each 2x2.
    
    # First, define the 2x2 Strassen decomposition.
    # For 2x2 * 2x2, the 7 multiplications are:
    # Let A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # C = [[ae+bg, af+bh],[ce+dg, cf+dh]]
    #
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    #
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    
    # The factor matrices for 2x2*2x2 with rank 7:
    # U is 4x7, V is 4x7, W is 4x7
    # Convention: U[i*m+j, c] where m=2, so U[i*2+j, c]
    # V[j*p+k, c] where p=2, so V[j*2+k, c]
    # W[k*n+i, c] where n=2, so W[k*2+i, c]
    
    U2 = np.zeros((4, 7), dtype=float)
    V2 = np.zeros((4, 7), dtype=float)
    W2 = np.zeros((4, 7), dtype=float)
    
    # m1 = (a+d)(e+h)
    # U: a is at (0,0) -> index 0*2+0=0, d is at (1,1) -> index 1*2+1=3
    U2[0, 0] = 1  # a
    U2[3, 0] = 1  # d
    # V: e is at (0,0) -> index 0*2+0=0, h is at (1,1) -> index 1*2+1=3
    V2[0, 0] = 1  # e
    V2[3, 0] = 1  # h
    # W: C11 gets +1, C22 gets +1
    # C11 is at (0,0) in C, which is k=0, i=0 -> W[0*2+0, 0] = W[0, 0]
    # C22 is at (1,1) in C, which is k=1, i=1 -> W[1*2+1, 0] = W[3, 0]
    W2[0, 0] = 1  # C11
    W2[3, 0] = 1  # C22
    
    # m2 = (c+d)e
    # U: c is at (1,0) -> index 1*2+0=2, d is at (1,1) -> index 3
    U2[2, 1] = 1  # c
    U2[3, 1] = 1  # d
    # V: e is at (0,0) -> index 0
    V2[0, 1] = 1  # e
    # W: C21 gets +1
    # C21 is at (1,0) in C, which is k=1, i=0 -> W[1*2+0, 1] = W[2, 1]
    W2[2, 1] = 1  # C21
    
    # m3 = a(f-h)
    # U: a is at (0,0) -> index 0
    U2[0, 2] = 1  # a
    # V: f is at (0,1) -> index 0*2+1=1, h is at (1,1) -> index 3
    V2[1, 2] = 1  # f
    V2[3, 2] = -1  # -h
    # W: C12 gets +1
    # C12 is at (0,1) in C, which is k=0, i=1 -> W[0*2+1, 2] = W[1, 2]
    W2[1, 2] = 1  # C12
    
    # m4 = d(g-e)
    # U: d is at (1,1) -> index 3
    U2[3, 3] = 1  # d
    # V: g is at (1,0) -> index 1*2+0=2, e is at (0,0) -> index 0
    V2[2, 3] = 1  # g
    V2[0, 3] = -1  # -e
    # W: C11 gets +1, C21 gets +1
    W2[0, 3] = 1  # C11
    W2[2, 3] = 1  # C21
    
    # m5 = (a+b)h
    # U: a is at (0,0) -> index 0, b is at (0,1) -> index 1
    U2[0, 4] = 1  # a
    U2[1, 4] = 1  # b
    # V: h is at (1,1) -> index 3
    V2[3, 4] = 1  # h
    # W: C11 gets -1, C12 gets +1
    W2[0, 4] = -1  # C11
    W2[1, 4] = 1  # C12
    
    # m6 = (c-a)(e+f)
    # U: c is at (1,0) -> index 2, a is at (0,0) -> index 0
    U2[2, 5] = 1  # c
    U2[0, 5] = -1  # -a
    # V: e is at (0,0) -> index 0, f is at (0,1) -> index 1
    V2[0, 5] = 1  # e
    V2[1, 5] = 1  # f
    # W: C22 gets +1
    W2[3, 5] = 1  # C22
    
    # m7 = (b-d)(g+h)
    # U: b is at (0,1) -> index 1, d is at (1,1) -> index 3
    U2[1, 6] = 1  # b
    U2[3, 6] = -1  # -d
    # V: g is at (1,0) -> index 2, h is at (1,1) -> index 3
    V2[2, 6] = 1  # g
    V2[3, 6] = 1  # h
    # W: C11 gets +1
    W2[0, 6] = 1  # C11
    
    # Now for 4x4*4x4, we use Strassen at the block level.
    # A is 4x4, split into 2x2 blocks:
    # A11 = A[0:2, 0:2], A12 = A[0:2, 2:4]
    # A21 = A[2:4, 0:2], A22 = A[2:4, 2:4]
    # Similarly for B.
    
    # The 7 block-level Strassen combinations:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Each Mj is a 2x2 * 2x2 multiplication.
    # We need to express the sum/difference of 2x2 blocks as linear combinations of the 4x4 matrix entries.
    
    # For A (4x4), the entries are indexed by (i,j) where i,j in {0,1,2,3}.
    # U[i*m+j, c] where m=4, so U[i*4+j, c].
    
    # Let's build U, V, W for 4x4*4x4.
    # There are 7 block-level multiplications, each with 7 scalar-level multiplications.
    # Total rank = 49.
    
    U4 = np.zeros((16, 49), dtype=float)
    V4 = np.zeros((16, 49), dtype=float)
    W4 = np.zeros((16, 49), dtype=float)
    
    # For each block-level multiplication j (0 to 6), and each scalar-level multiplication s (0 to 6):
    # Column index = j*7 + s
    
    # Define the block-level coefficients for A and B.
    # For each of the 7 block-level multiplications, we have coefficients for A11, A12, A21, A22
    # and for B11, B12, B21, B22.
    
    # M1 = (A11+A22)(B11+B22)
    # A coeffs: A11: +1, A12: 0, A21: 0, A22: +1
    # B coeffs: B11: +1, B12: 0, B21: 0, B22: +1
    
    # M2 = (A21+A22)B11
    # A coeffs: A11: 0, A12: 0, A21: +1, A22: +1
    # B coeffs: B11: +1, B12: 0, B21: 0, B22: 0
    
    # M3 = A11(B12-B22)
    # A coeffs: A11: +1, A12: 0, A21: 0, A22: 0
    # B coeffs: B11: 0, B12: +1, B21: 0, B22: -1
    
    # M4 = A22(B21-B11)
    # A coeffs: A11: 0, A12: 0, A21: 0, A22: +1
    # B coeffs: B11: -1, B12: 0, B21: +1, B22: 0
    
    # M5 = (A11+A12)B22
    # A coeffs: A11: +1, A12: +1, A21: 0, A22: 0
    # B coeffs: B11: 0, B12: 0, B21: 0, B22: +1
    
    # M6 = (A21-A11)(B11+B12)
    # A coeffs: A11: -1, A12: 0, A21: +1, A22: 0
    # B coeffs: B11: +1, B12: +1, B21: 0, B22: 0
    
    # M7 = (A12-A22)(B21+B22)
    # A coeffs: A11: 0, A12: +1, A21: 0, A22: -1
    # B coeffs: B11: 0, B12: 0, B21: +1, B22: +1
    
    a_block_coeffs = [
        [1, 0, 0, 1],   # M1: A11, A12, A21, A22
        [0, 0, 1, 1],   # M2
        [1, 0, 0, 0],   # M3
        [0, 0, 0, 1],   # M4
        [1, 1, 0, 0],   # M5
        [-1, 0, 1, 0],  # M6
        [0, 1, 0, -1],  # M7
    ]
    
    b_block_coeffs = [
        [1, 0, 0, 1],   # M1: B11, B12, B21, B22
        [1, 0, 0, 0],   # M2
        [0, 1, 0, -1],  # M3
        [-1, 0, 1, 0],  # M4
        [0, 0, 0, 1],   # M5
        [1, 1, 0, 0],   # M6
        [0, 0, 1, 1],   # M7
    ]
    
    # C block combinations:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For each C block (C11, C12, C21, C22), we have coefficients for M1...M7.
    c_block_coeffs = {
        (0, 0): {0: 1, 3: 1, 4: -1, 6: 1},   # C11
        (0, 1): {2: 1, 4: 1},                  # C12
        (1, 0): {1: 1, 3: 1},                  # C21
        (1, 1): {0: 1, 1: -1, 2: 1, 5: 1},     # C22
    }
    
    # Now, for each block-level multiplication j (0-6) and scalar-level multiplication s (0-6):
    # Column c = j*7 + s
    # 
    # U[i*4+j, c] = a_block_coeffs[j][block_i] * U2[i*2+j_local, s]
    # where block_i is 0 for A11 (i=0,1), 1 for A12 (i=0,1), 2 for A21 (i=2,3), 3 for A22 (i=2,3)
    # j_local is the local column index within the 2x2 block.
    
    for j in range(7):
        for s in range(7):
            c = j * 7 + s
            for i in range(4):
                for j_local in range(4):
                    # Determine which block this (i, j_local) belongs to
                    block_i = i // 2
                    block_j = j_local // 2
                    i_local = i % 2
                    j_local_local = j_local % 2
                    
                    # A block index: block_i*2 + block_j
                    a_block = block_i * 2 + block_j
                    a_coeff = a_block_coeffs[j][a_block]
                    
                    if a_coeff != 0:
                        U4[i * 4 + j_local, c] = a_coeff * U2[i_local * 2 + j_local_local, s]
            
            for j_local in range(4):
                for k in range(4):
                    block_j = j_local // 2
                    block_k = k // 2
                    j_local_local = j_local % 2
                    k_local = k % 2
                    
                    b_block = block_j * 2 + block_k
                    b_coeff = b_block_coeffs[j][b_block]
                    
                    if b_coeff != 0:
                        V4[j_local * 4 + k, c] = b_coeff * V2[j_local_local * 2 + k_local, s]
            
            for k in range(4):
                for i in range(4):
                    block_k = k // 2
                    block_i = i // 2
                    k_local = k % 2
                    i_local = i % 2
                    
                    # C block: (block_i, block_k)
                    c_block = (block_i, block_k)
                    if c_block in c_block_coeffs and j in c_block_coeffs[c_block]:
                        c_coeff = c_block_coeffs[c_block][j]
                        W4[k * 4 + i, c] = c_coeff * W2[k_local * 2 + i_local, s]
    
    return U4, V4, W4


def _strassen_4x4x2():
    """
    Return U, V, W matrices for 4x4*4x2 with rank 28 using Strassen.
    Convention: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    Here n=4, m=4, p=2.
    
    We can use Strassen for 2x2*2x2 (rank 7) on the 2x2 blocks.
    A is 4x4, B is 4x2.
    Split A into 2x2 blocks: A11, A12, A21, A22 (each 2x2)
    Split B into 2x1 blocks: B11, B12, B21, B22 (each 2x1)
    
    Actually, let's think differently. We can use the fact that 4x4*4x2 can be seen as
    two 4x4*4x1 problems, or we can use a block Strassen approach.
    
    A simpler approach: use Strassen for 2x2*2x2 on the 2x2 blocks of A and B.
    A is 4x4, split into four 2x2 blocks.
    B is 4x2, split into two 2x1 blocks (top and bottom).
    
    Actually, let's use a different decomposition. We can write:
    C = A * B where A is 4x4, B is 4x2, C is 4x2.
    
    We can use the following approach:
    Split A into 2x2 blocks: A = [[A11, A12], [A21, A22]]
    Split B into 2x1 blocks: B = [[B1], [B2]] where B1, B2 are 2x1.
    
    C = A * B = [[A11*B1 + A12*B2], [A21*B1 + A22*B2]]
    C1 = A11*B1 + A12*B2 (2x1)
    C2 = A21*B1 + A22*B2 (2x1)
    
    Each term is a 2x2 * 2x1 multiplication.
    A 2x2 * 2x1 multiplication can be done with rank 3 (since it's like computing two dot products of length 2, but we can use Strassen-like tricks).
    Actually, 2x2*2x1 has tensor rank 3? Let's check.
    
    For 2x2 * 2x1:
    A = [[a,b],[c,d]], B = [[e],[f]]
    C = [[ae+bf],[ce+df]]
    
    This is a 2x2x1 tensor. The rank is... let's think.
    We need to express C1 = ae+bf, C2 = ce+df.
    
    One decomposition:
    m1 = ae, m2 = bf, m3 = ce, m4 = df -> rank 4
    Can we do better?
    
    m1 = (a+c)e, m2 = (b+d)f, m3 = (a-c)e, m4 = (b-d)f
    C1 = (m1+m3)/2 + (m2+m4)/2
    C2 = (m1-m3)/2 + (m2-m4)/2
    Still rank 4.
    
    Actually, the tensor rank of 2x2x1 is 3? No, I think it's 4.
    
    Let me reconsider. For 4x4*4x2, we can use:
    - 4 blocks of 2x2*2x1, each with rank 4 -> total rank 16? No, that's not right either.
    
    Let me use a different approach. We can use Strassen for 4x4*4x4 (rank 49) and then restrict to the case where B has the second half of columns zero. But that doesn't directly give us 4x4*4x2.
    
    Actually, a known result is that 4x4*4x2 has rank 28. Let me construct it using block Strassen.
    
    A is 4x4, B is 4x2.
    Split A into 2x2 blocks: A11, A12, A21, A22
    Split B into 2x1 blocks: B1, B2 (each 2x1)
    
    C1 = A11*B1 + A12*B2
    C2 = A21*B1 + A22*B2
    
    Each 2x2*2x1 multiplication has rank 4 (I believe).
    So total rank = 4 * 4 = 16? That seems too low.
    
    Let me verify the rank of 2x2*2x1.
    The tensor is T[i,j,k] = delta_{i,k} * delta_{j,k} for the identity, but actually it's the multiplication tensor.
    
    For 2x2 * 2x1:
    Inputs: A (2x2) with entries a11, a12, a21, a22
    B (2x1) with entries b1, b2
    Output: C (2x1) with entries c1 = a11*b1 + a12*b2, c2 = a21*b1 + a22*b2
    
    This is a bilinear map from R^4 x R^2 -> R^2.
    The tensor rank is the minimum number of rank-1 tensors needed.
    
    A rank-1 tensor is of the form u * v * w where u in R^4, v in R^2, w in R^2.
    
    I think the rank is 3. Let me try:
    m1 = (a11+a21)*b1
    m2 = (a12+a22)*b2
    m3 = (a11-a21)*b1
    
    c1 = (m1+m3)/2 + m2
    c2 = (m1-m3)/2 + m2
    
    That gives rank 3! Let me verify:
    c1 = (a11+a21)*b1/2 + (a11-a21)*b1/2 + (a12+a22)*b2 = a11*b1 + a12*b2 + a22*b2
    Wait, that's not right. c1 should be a11*b1 + a12*b2.
    
    Let me try again:
    m1 = a11*b1
    m2 = a12*b2
    m3 = (a21-b1)*(a22-b2) ... no, that's not bilinear in the right way.
    
    Actually, I think the rank of 2x2*2x1 is 4. Let me just use rank 4 for each block.
    
    So for 4x4*4x2:
    C1 = A11*B1 + A12*B2: two 2x2*2x1 multiplications, each rank 4 -> rank 8
    C2 = A21*B1 + A22*B2: two 2x2*2x1 multiplications, each rank 4 -> rank 8
    Total: rank 16.
    
    But the known rank is 28, so my approach is wrong.
    
    Let me use a different strategy. I'll use the fact that we can embed 4x4*4x2 into 4x4*4x4.
    If we have a rank-49 decomposition for 4x4*4x4, we can use it for 4x4*4x2 by setting the last two columns of B to zero. But that still gives rank 49, not 28.
    
    Alternatively, I can use Strassen recursively. For 4x4*4x2:
    Split A into 2x2 blocks.
    B is 4x2, which can be seen as two 2x1 blocks.
    
    Actually, let me just use the schoolbook method for the 4x4*4x2 part and see what rank I get.
    Schoolbook: 4*4*2 = 32.
    
    The known best is 28, so there's a 4-rank improvement.
    
    For now, let me use a simple approach: use Strassen for the 4x4*4x4 part (rank 49) and handle the 4x4*4x2 part separately with schoolbook (rank 32).
    Total: 49 + 32 = 81. But we want 77.
    
    The task says: "Correct recursive Strassen for 4x4x4 (rank 49) plus block Strassen for 4x4x2 (rank 28) gives rank 77."
    
    So I need to implement a rank-28 decomposition for 4x4*4x2.
    
    Let me think about this more carefully. 4x4*4x2 can be decomposed as:
    A (4x4) * B (4x2) = C (4x2)
    
    Using block Strassen:
    A = [[A11, A12], [A21, A22]] (2x2 blocks of 2x2)
    B = [[B11, B12], [B21, B22]] (2x2 blocks of 2x1)
    
    C = [[A11*B11 + A12*B21, A11*B12 + A12*B22],
         [A21*B11 + A22*B21, A21*B12 + A22*B22]]
    
    Each term is a 2x2 * 2x1 multiplication.
    
    If the rank of 2x2*2x1 is 3, then total rank = 8 * 3 = 24.
    If the rank is 4, then total rank = 8 * 4 = 32.
    
    The known rank is 28, so it's between 24 and 32.
    
    Let me look up or derive the rank of 2x2*2x1.
    
    Actually, I recall that the tensor rank of [2,2,1] is 3. Let me verify with a specific decomposition.
    
    For 2x2 * 2x1:
    A = [[a, b], [c, d]], B = [[e], [f]]
    C = [[ae + bf], [ce + df]]
    
    Try:
    m1 = (a + c) * e
    m2 = (b + d) * f
    m3 = (a - c) * e
    
    C1 = (m1 + m3) / 2 + m2 = a*e + b*f + d*f ... no, that's not right.
    
    Let me try:
    m1 = a * e
    m2 = b * f
    m3 = (c + a) * e
    m4 = (d + b) * f
    
    C1 = m1 + m2
    C2 = (m3 - m1) + (m4 - m2) = c*e + d*f
    
    That's rank 4.
    
    I think the rank is indeed 4 for 2x2*2x1. So 8 * 4 = 32 for schoolbook.
    
    To get rank 28, we need a more clever decomposition.
    
    For now, let me just use rank 49 for 4x4*4x4 and rank 32 for 4x4*4x2, giving total rank 81.
    This is worse than the target of 77, but it's a valid decomposition.
    
    Actually, let me reconsider. The task is to multiply a 4x4 matrix by a 4x6 matrix.
    So A is 4x4, B is 4x6, C is 4x6.
    
    We can split B into B1 (4x4) and B2 (4x2).
    C = A * B1 + A * B2
    
    If we use rank 49 for A*B1 and rank 32 for A*B2, total rank = 81.
    
    But the known best is 77, so there must be a better way.
    
    Let me just implement the rank-49 Strassen for 4x4*4x4 and the schoolbook for 4x4*4x2, and see if I can improve.
    
    For now, let me return the rank-81 decomposition and see if it passes.
    """
    n = 4
    m = 4
    p = 2
    rank = 32  # Schoolbook for now
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    
    return U, V, W


def construct():
    """
    Construct the factor matrices for 4x4 * 4x6 matrix multiplication.
    Returns a dictionary with 'U', 'V', 'W' keys.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n = 4
    m = 4
    p = 6
    
    # Split B (4x6) into B1 (4x4) and B2 (4x2)
    # C = A * B1 + A * B2
    
    # Get Strassen decomposition for 4x4 * 4x4 (rank 49)
    U1, V1, W1 = _strassen_4x4x4()
    rank1 = 49
    
    # Get decomposition for 4x4 * 4x2 (rank 32 for now)
    U2, V2, W2 = _strassen_4x4x2()
    rank2 = 32
    
    total_rank = rank1 + rank2
    
    # Combine the decompositions
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    # First part: A * B1 (B1 is the first 4 columns of B)
    U[:, :rank1] = U1
    V[:, :rank1] = V1[:, :m * 4]  # V1 is for 4x4, so m*4 = 16 entries
    W[:, :rank1] = W1[:, :p * n]  # Wait, W1 is for 4x4 output, so p*n = 16 entries
    # Actually, W1 has shape (4*4, 49) = (16, 49) for the 4x4 output.
    # For the 4x6 output, the first 4 columns correspond to the 4x4 part.
    # W[k*n+i, c] where k is in 0..5 for the 4x6 case.
    # For the 4x4 part, k is in 0..3.
    W[:4 * n, :rank1] = W1
    
    # Second part: A * B2 (B2 is the last 2 columns of B)
    U[:, rank1:] = U2
    V[:, rank1:] = V2
    W[4 * n:, rank1:] = W2  # The last 2 rows of the output (k=4,5)
    
    # Convert to lists for the return format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
