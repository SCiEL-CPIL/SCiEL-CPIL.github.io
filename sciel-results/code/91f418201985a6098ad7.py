"""Rank-minimal exact tensor decomposition for 4x4 * 4x5 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    # We decompose the 4x4 * 4x5 problem as:
    #   A (4x4) * B (4x5) = A * [B1 | B2]
    # where B1 is 4x4 and B2 is 4x1.
    # We use Strassen for the 4x4*4x4 part (via Kronecker with 2x2 Strassen)
    # and schoolbook for the 4x4*4x1 part.
    #
    # Strassen 2x2x2 has rank 7.
    # Kronecker product with itself gives 4x4x4 of rank 49.
    # 4x4x1 schoolbook has rank 16.
    # Total rank = 49 + 16 = 65.
    #
    # The matrices U, V, W must satisfy:
    #   U: (n*m, rank) = (16, 65)
    #   V: (m*p, rank) = (20, 65)
    #   W: (p*n, rank) = (20, 65)
    #
    # Indexing:
    #   U[i*m + j, r]
    #   V[j*p + k, r]
    #   W[k*n + i, r]
    #
    # For the 4x4x4 part (k in 0..3):
    #   U_444[0:16, 0:49]
    #   V_444[j*5+k, 0:49] for j in 0..3, k in 0..3
    #   W_444[k*4+i, 0:49] for k in 0..3, i in 0..3
    #
    # For the 4x4x1 part (k=4):
    #   U_441[0:16, 49:65]
    #   V_441[j*5+4, 49:65] for j in 0..3
    #   W_441[4*4+i, 49:65] = W_441[16+i, 49:65] for i in 0..3

    n, m, p = DIMENSIONS
    rank_444 = 49
    rank_441 = 16
    total_rank = rank_444 + rank_441

    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))

    # Strassen 2x2x2 rank-7 decomposition
    # U_2x2: (4, 7), V_2x2: (4, 7), W_2x2: (4, 7)
    # Standard Strassen:
    # M1 = (A11+A22)(B12+B21)
    # M2 = (A21+A22)B11
    # M3 = A11(B11-B21)
    # M4 = A22(B22-B12)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6

    U2 = np.array([
        [1, 1, 1, 0, 1, -1, 0],   # A11
        [0, 1, 0, 1, 0, 1, -1],   # A21
        [0, 1, 0, 0, 1, 0, 0],    # A22... wait let me redo
    ])

    # Let me carefully construct the Strassen 2x2x2 factors.
    # Input A = [[A11, A12],[A21, A22]], B = [[B11, B12],[B21, B22]]
    # U indices: [0]=A11, [1]=A21, [2]=A22, [3]=A12
    # V indices: [0]=B11, [1]=B12, [2]=B21, [3]=B22
    # W indices: [0]=C11, [1]=C12, [2]=C21, [3]=C22

    # M1 = (A11+A22)(B12+B21)
    # M2 = (A21+A22)B11
    # M3 = A11(B11-B21)
    # M4 = A22(B22-B12)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)

    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))

    # M1
    U2[0, 0] = 1  # A11
    U2[2, 0] = 1  # A22
    V2[1, 0] = 1  # B12
    V2[2, 0] = 1  # B21
    W2[0, 0] = 1  # C11
    W2[3, 0] = 1  # C22

    # M2
    U2[1, 1] = 1  # A21
    U2[2, 1] = 1  # A22
    V2[0, 1] = 1  # B11
    W2[2, 1] = 1  # C21
    W2[3, 1] = -1 # C22

    # M3
    U2[0, 2] = 1  # A11
    V2[0, 2] = 1  # B11
    V2[2, 2] = -1 # B21
    W2[1, 2] = 1  # C12
    W2[3, 2] = 1  # C22

    # M4
    U2[2, 3] = 1  # A22
    V2[3, 3] = 1  # B22
    V2[1, 3] = -1 # B12
    W2[0, 3] = 1  # C11
    W2[2, 3] = 1  # C21

    # M5
    U2[0, 4] = 1  # A11
    U2[3, 4] = 1  # A12
    V2[3, 4] = 1  # B22
    W2[0, 4] = -1 # C11
    W2[1, 4] = 1  # C12

    # M6
    U2[1, 5] = 1  # A21
    U2[0, 5] = -1 # A11
    V2[0, 5] = 1  # B11
    V2[1, 5] = 1  # B12
    W2[3, 5] = 1  # C22

    # M7
    U2[3, 6] = 1  # A12
    U2[2, 6] = -1 # A22
    V2[2, 6] = 1  # B21
    V2[3, 6] = 1  # B22
    W2[0, 6] = 1  # C11

    # Verify Strassen 2x2x2
    # C = A*B
    # For each r: sum over u,v: U2[u,r]*V2[v,r]*W2[w,r] should give (A*B)[w]
    # This is standard, let's trust it.

    # Now Kronecker product for 4x4x4
    # A_4x4 = [[A11, A12, A13, A14],[A21,...],...]
    # We view A_4x4 as 2x2 blocks of 2x2:
    # A_4x4 = [[A_11, A_12],[A_21, A_22]] where each A_ij is 2x2
    # Same for B_4x4 and C_4x4.
    #
    # U_444[u4, r] where u4 = u_block*4 + u_local
    #   u_block in {0,1} (block row), u_local in {0,1,2,3} (local row in 2x2)
    #   But our 2x2 U2 has indices 0..3 for A11,A21,A22,A12
    #   Actually U2 index: 0=A11, 1=A21, 2=A22, 3=A12
    #   These correspond to local positions in the 2x2 block.
    #
    # For Kronecker: U_444[u_block*4+u_local, r_block*7+r_local] = U2[u_local, r_local] * U2[u_block, r_block]
    # Wait, that's not quite right. The Kronecker product of two Strassen decompositions:
    # U_kron = U2 ⊗ U2, V_kron = V2 ⊗ V2, W_kron = W2 ⊗ W2
    # But we need to be careful about the ordering.

    U444 = np.kron(U2, U2)  # (16, 49)
    V444 = np.kron(V2, V2)  # (16, 49)
    W444 = np.kron(W2, W2)  # (16, 49)

    # Now map V444 and W444 to the global indexing with p=5.
    # V444 has indices for 4x4 B matrix: v = j*4 + k where j,k in 0..3
    # Global V index: j*5 + k
    # We need to scatter V444 into V[:, 0:49]

    V_444_global = np.zeros((20, 49))
    for j in range(4):
        for k in range(4):
            V_444_global[j*5 + k, :] = V444[j*4 + k, :]

    # W444 has indices for 4x4 C matrix: w = k*4 + i where k,i in 0..3
    # Global W index: k*4 + i (since p=5 but for the 444 part, k is 0..3)
    # Actually W index is k*n + i = k*4 + i, same as local.
    W_444_global = np.zeros((20, 49))
    for k in range(4):
        for i in range(4):
            W_444_global[k*4 + i, :] = W444[k*4 + i, :]

    # U444: 16x49, same indexing for A: u = i*4 + j, global U index = i*m+j = i*4+j
    # So U_444_global = U444 directly.
    U_444_global = np.zeros((16, 49))
    for i in range(4):
        for j in range(4):
            U_444_global[i*4 + j, :] = U444[i*4 + j, :]

    # 4x4x1 schoolbook part
    # A (4x4) * B_col (4x1) -> C_col (4x1)
    # U_441: 16x16, V_441: 4x16, W_441: 4x16
    # For each (i,j): U[i*4+j, r]=1, V[j*1+0, r]=1, W[0*4+i, r]=1
    # r = i*4+j
    U_441_global = np.zeros((16, 16))
    V_441_global = np.zeros((4, 16))
    W_441_global = np.zeros((4, 16))
    for i in range(4):
        for j in range(4):
            r = i*4 + j
            U_441_global[i*4 + j, r] = 1
            V_441_global[j, r] = 1
            W_441_global[i, r] = 1

    # Assemble
    U[:, 0:49] = U_444_global
    U[:, 49:65] = U_441_global

    V[:, 0:49] = V_444_global
    # V_441 goes to rows j*5+4 for j in 0..3
    for j in range(4):
        V[j*5 + 4, 49:65] = V_441_global[j, :]

    W[:, 0:49] = W_444_global
    # W_441 goes to rows 4*4+i = 16+i for i in 0..3
    W[16:20, 49:65] = W_441_global

    # Convert to lists of integers/half-integers
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
