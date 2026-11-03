"""Program for matrix multiplication 2x4 by 4x8 using Kronecker product decomposition."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We view the multiplication as a 2x2 matrix multiplication over the ring of 2x4 matrices.
    # A (2x4) is split into two 2x2 blocks: A_0 = A[:, 0:2], A_1 = A[:, 2:4]
    # B (4x8) is split into four 2x8 blocks: B_0 = B[0:2, :], B_1 = B[2:4, :]
    # Wait, the middle dimension is 4. So A is 2x4, B is 4x8.
    # Let's re-partition.
    # A (2x4) -> A_0 (2x2), A_1 (2x2) where A_0 = A[:, 0:2], A_1 = A[:, 2:4]
    # B (4x8) -> B_0 (2x8), B_1 (2x8) where B_0 = B[0:2, :], B_1 = B[2:4, :]
    # C = A_0 B_0 + A_1 B_1
    # This is a sum of two 2x2 * 2x8 multiplications.
    # This is not a 2x2 matrix multiplication over a ring directly.
    
    # Alternative: View as 2x2 matrix multiplication over the ring of 1x4 matrices?
    # A (2x4) -> A_0 (1x4), A_1 (1x4) where A_0 = A[0, :], A_1 = A[1, :]
    # B (4x8) -> B_0 (4x8)
    # This doesn't help.
    
    # Let's try the Kronecker product approach explicitly.
    # The tensor T_{2,4,8} can be seen as T_{2,2,2} \otimes T_{1,2,4} ?
    # No, T_{n, m, p} = T_{n1, m1, p1} \otimes T_{n2, m2, p2} if n=n1*n2, m=m1*m2, p=p1*p2.
    # Here 2=1*2, 4=2*2, 8=4*2.
    # So T_{2,4,8} = T_{1,2,4} \otimes T_{2,2,2}.
    # Rank(T_{1,2,4}) = 1*2*4 = 8.
    # Rank(T_{2,2,2}) = 7 (Strassen).
    # Total rank = 8 * 7 = 56.
    
    # Can we do better?
    # T_{2,4,8} = T_{2,2,4} \otimes T_{1,2,2}?
    # 2=2*1, 4=2*2, 8=4*2.
    # Rank(T_{2,2,4}) = 2*2*4 = 16.
    # Rank(T_{1,2,2}) = 1*2*2 = 4.
    # Total rank = 16 * 4 = 64. (Schoolbook)
    
    # What about T_{2,4,8} = T_{2,4,2} \otimes T_{1,1,4}?
    # 2=2*1, 4=4*1, 8=2*4.
    # Rank(T_{2,4,2}) = 2*4*2 = 16.
    # Rank(T_{1,1,4}) = 1*1*4 = 4.
    # Total rank = 16 * 4 = 64.
    
    # So 56 is the best we can get from simple Kronecker products.
    # Let's implement the rank 56 decomposition.
    
    # T_{2,4,8} = T_{1,2,4} \otimes T_{2,2,2}
    # n=2, m=4, p=8
    # n1=1, m1=2, p1=4
    # n2=2, m2=2, p2=2
    
    # We need to construct the Kronecker product of the two tensors.
    # Let T1 be the tensor for 1x2 * 2x4 multiplication.
    # Let T2 be the tensor for 2x2 * 2x2 multiplication (Strassen).
    
    # The Kronecker product of tensors T1 and T2 has rank rank(T1) * rank(T2).
    # The indices are combined.
    
    # For T1 (1x2 * 2x4):
    # U1 has shape (1*2, 8) = (2, 8)
    # V1 has shape (2*4, 8) = (8, 8)
    # W1 has shape (4*1, 8) = (4, 8)
    
    # For T2 (2x2 * 2x2):
    # U2 has shape (2*2, 7) = (4, 7)
    # V2 has shape (2*2, 7) = (4, 7)
    # W2 has shape (2*2, 7) = (4, 7)
    
    # The Kronecker product T = T1 \otimes T2 has:
    # U has shape (2*4, 8*7) = (8, 56)
    # V has shape (8*4, 8*7) = (32, 56)
    # W has shape (4*4, 8*7) = (16, 56)
    
    # But our target is:
    # U has shape (n*m, rank) = (2*4, 56) = (8, 56)
    # V has shape (m*p, rank) = (4*8, 56) = (32, 56)
    # W has shape (p*n, rank) = (8*2, 56) = (16, 56)
    
    # This matches!
    
    # Now we need to construct U, V, W from U1, V1, W1, U2, V2, W2.
    # The Kronecker product of tensors is defined as:
    # T(i, j, k) = T1(i1, j1, k1) * T2(i2, j2, k2)
    # where i = i1 * n2 + i2, j = j1 * m2 + j2, k = k1 * p2 + k2
    
    # For the factor matrices:
    # U[i, r] = U1[i1, r1] * U2[i2, r2] where i = i1 * n2 + i2, r = r1 * rank2 + r2
    # V[j, r] = V1[j1, r1] * V2[j2, r2] where j = j1 * m2 + j2, r = r1 * rank2 + r2
    # W[k, r] = W1[k1, r1] * W2[k2, r2] where k = k1 * p2 + k2, r = r1 * rank2 + r2
    
    # Let's construct U1, V1, W1 for T1 (1x2 * 2x4).
    # n1=1, m1=2, p1=4
    # U1 has shape (2, 8)
    # V1 has shape (8, 8)
    # W1 has shape (4, 8)
    
    # The multiplication is C[i, k] = sum_j A[i, j] B[j, k]
    # i is in 0..0, j is in 0..1, k is in 0..3
    # So C[0, k] = A[0, 0] B[0, k] + A[0, 1] B[1, k]
    
    # U1[i1*m1 + j1, r1]
    # i1=0, j1=0 -> index 0
    # i1=0, j1=1 -> index 1
    # U1[0, r1] = 1 if r1 corresponds to A[0,0]
    # U1[1, r1] = 1 if r1 corresponds to A[0,1]
    
    # V1[j1*p1 + k1, r1]
    # j1=0, k1=0..3 -> indices 0..3
    # j1=1, k1=0..3 -> indices 4..7
    # V1[0, r1] = 1 if r1 corresponds to B[0,0]
    # V1[1, r1] = 1 if r1 corresponds to B[0,1]
    # V1[2, r1] = 1 if r1 corresponds to B[0,2]
    # V1[3, r1] = 1 if r1 corresponds to B[0,3]
    # V1[4, r1] = 1 if r1 corresponds to B[1,0]
    # V1[5, r1] = 1 if r1 corresponds to B[1,1]
    # V1[6, r1] = 1 if r1 corresponds to B[1,2]
    # V1[7, r1] = 1 if r1 corresponds to B[1,3]
    
    # W1[k1*n1 + i1, r1]
    # k1=0..3, i1=0 -> indices 0..3
    # W1[0, r1] = 1 if r1 corresponds to C[0,0]
    # W1[1, r1] = 1 if r1 corresponds to C[0,1]
    # W1[2, r1] = 1 if r1 corresponds to C[0,2]
    # W1[3, r1] = 1 if r1 corresponds to C[0,3]
    
    # The terms are:
    # r1=0: A[0,0] * B[0,0] -> C[0,0]
    # r1=1: A[0,0] * B[0,1] -> C[0,1]
    # r1=2: A[0,0] * B[0,2] -> C[0,2]
    # r1=3: A[0,0] * B[0,3] -> C[0,3]
    # r1=4: A[0,1] * B[1,0] -> C[0,0]
    # r1=5: A[0,1] * B[1,1] -> C[0,1]
    # r1=6: A[0,1] * B[1,2] -> C[0,2]
    # r1=7: A[0,1] * B[1,3] -> C[0,3]
    
    U1 = [[0]*8 for _ in range(2)]
    V1 = [[0]*8 for _ in range(8)]
    W1 = [[0]*8 for _ in range(4)]
    
    U1[0][0] = 1; U1[0][1] = 1; U1[0][2] = 1; U1[0][3] = 1
    U1[1][4] = 1; U1[1][5] = 1; U1[1][6] = 1; U1[1][7] = 1
    
    V1[0][0] = 1; V1[1][1] = 1; V1[2][2] = 1; V1[3][3] = 1
    V1[4][4] = 1; V1[5][5] = 1; V1[6][6] = 1; V1[7][7] = 1
    
    W1[0][0] = 1; W1[0][4] = 1
    W1[1][1] = 1; W1[1][5] = 1
    W1[2][2] = 1; W1[2][6] = 1
    W1[3][3] = 1; W1[3][7] = 1
    
    # Now U2, V2, W2 for T2 (2x2 * 2x2) using Strassen.
    # n2=2, m2=2, p2=2
    # U2 has shape (4, 7)
    # V2 has shape (4, 7)
    # W2 has shape (4, 7)
    
    # Strassen coefficients:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A10)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # U2 indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # V2 indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # W2 indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    # M1: A00+A11, B00+B11
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    
    # M2: A10+A11, B00
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    
    # M3: A00, B10-B11
    U2[0][2] = 1
    V2[2][2] = 1; V2[3][2] = -1
    
    # M4: A11, B01-B00
    U2[3][3] = 1
    V2[1][3] = 1; V2[0][3] = -1
    
    # M5: A00+A10, B11
    U2[0][4] = 1; U2[2][4] = 1
    V2[3][4] = 1
    
    # M6: A10-A00, B00+B01
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    
    # M7: A01-A11, B10+B11
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    
    # Now combine using Kronecker product.
    # rank1 = 8, rank2 = 7
    # rank = 56
    
    rank1 = 8
    rank2 = 7
    rank = rank1 * rank2
    
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # n1=1, m1=2, p1=4
    # n2=2, m2=2, p2=2
    
    n1, m1, p1 = 1, 2, 4
    n2, m2, p2 = 2, 2, 2
    
    # U[i, r] = U1[i1, r1] * U2[i2, r2]
    # i = i1 * n2 + i2
    # r = r1 * rank2 + r2
    
    for i1 in range(n1):
        for i2 in range(n2):
            i = i1 * n2 + i2
            for r1 in range(rank1):
                for r2 in range(rank2):
                    r = r1 * rank2 + r2
                    U[i][r] = U1[i1 * m1 + i2][r1] * U2[i2][r2]
    
    # Wait, the index for U1 is i1*m1 + j1? No, U1 is for A, so index is i1*m1 + j1.
    # But in the Kronecker product, the indices are combined.
    # Let's re-examine the index mapping.
    
    # T1 has indices (i1, j1, k1) with i1 in 0..n1-1, j1 in 0..m1-1, k1 in 0..p1-1
    # T2 has indices (i2, j2, k2) with i2 in 0..n2-1, j2 in 0..m2-1, k2 in 0..p2-1
    # T has indices (i, j, k) with i = i1*n2 + i2, j = j1*m2 + j2, k = k1*p2 + k2
    
    # U[i, r] = U1[i1, j1, r1] * U2[i2, j2, r2] ?
    # No, U is for the A matrix, which has indices (i, j).
    # So U[i, j, r] = U1[i1, j1, r1] * U2[i2, j2, r2]
    # where i = i1*n2 + i2, j = j1*m2 + j2, r = r1*rank2 + r2
    
    # But in our representation, U is a 2D array with shape (n*m, rank).
    # So U[i*m + j, r] = U1[i1*m1 + j1, r1] * U2[i2*m2 + j2, r2]
    
    # Let's re-construct U, V, W with this correct mapping.
    
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for i1 in range(n1):
        for j1 in range(m1):
            for i2 in range(n2):
                for j2 in range(m2):
                    i = i1 * n2 + i2
                    j = j1 * m2 + j2
                    idx_u = i * m + j
                    for r1 in range(rank1):
                        for r2 in range(rank2):
                            r = r1 * rank2 + r2
                            U[idx_u][r] = U1[i1 * m1 + j1][r1] * U2[i2 * m2 + j2][r2]
    
    for j1 in range(m1):
        for k1 in range(p1):
            for j2 in range(m2):
                for k2 in range(p2):
                    j = j1 * m2 + j2
                    k = k1 * p2 + k2
                    idx_v = j * p + k
                    for r1 in range(rank1):
                        for r2 in range(rank2):
                            r = r1 * rank2 + r2
                            V[idx_v][r] = V1[j1 * p1 + k1][r1] * V2[j2 * p2 + k2][r2]
    
    for k1 in range(p1):
        for i1 in range(n1):
            for k2 in range(p2):
                for i2 in range(n2):
                    k = k1 * p2 + k2
                    i = i1 * n2 + i2
                    idx_w = k * n + i
                    for r1 in range(rank1):
                        for r2 in range(rank2):
                            r = r1 * rank2 + r2
                            W[idx_w][r] = W1[k1 * n1 + i1][r1] * W2[k2 * n2 + i2][r2]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
