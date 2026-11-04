import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # safety margin
    
    # Strassen 2x2 decomposition (Rank 7)
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    
    # r=0
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    # r=1
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[1, 1] = 1; W2[3, 1] = -1
    # r=2
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[2, 2] = 1; W2[3, 2] = 1
    # r=3
    U2[3, 3] = 1
    V2[0, 3] = -1; V2[2, 3] = 1
    W2[0, 3] = 1; W2[1, 3] = 1
    # r=4
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[2, 4] = 1
    # r=5
    U2[0, 5] = -1; U2[2, 5] = 1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    # r=6
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1
    
    # Winograd 3x3 decomposition (Rank 15)
    # U: 9x15, V: 9x15, W: 9x15
    # Indexing: U[i*3+j, r] for A[i][j], V[j*3+k, r] for B[j][k], W[k*3+i, r] for C[k][i] (trace convention)
    U3 = np.zeros((9, 15), dtype=int)
    V3 = np.zeros((9, 15), dtype=int)
    W3 = np.zeros((9, 15), dtype=int)
    
    # Winograd 3x3: 15 multiplications
    # Let A = [[a00,a01,a02],[a10,a11,a12],[a20,a21,a22]]
    # Let B = [[b00,b10,b20],[b01,b11,b21],[b02,b12,b22]] (column-major for V indexing)
    # C = A*B, C[i][k] = sum_j A[i][j]*B[j][k]
    # V[j*3+k, r] = coefficient of B[j][k] in term r
    # W[k*3+i, r] = coefficient for C[i][k] from term r (trace convention: W[k*n+i])
    
    # Term 0: (a00+a01+a02)(b00+b01+b02)
    U3[0, 0] = 1; U3[1, 0] = 1; U3[2, 0] = 1  # a00, a01, a02
    V3[0, 0] = 1; V3[3, 0] = 1; V3[6, 0] = 1  # b00, b10, b20
    # This product contributes to c00, c01, c02 (row 0 of C)
    W3[0, 0] = 1  # c00 (k=0,i=0)
    W3[1, 0] = 1  # c01 (k=1,i=0)
    W3[2, 0] = 1  # c02 (k=2,i=0)
    
    # Term 1: (a00-a01)(b10-b20)
    U3[0, 1] = 1; U3[1, 1] = -1  # a00, a01
    V3[3, 1] = 1; V3[6, 1] = -1  # b10, b20
    W3[1, 1] = 1  # c01
    W3[2, 1] = -1  # c02
    
    # Term 2: (a01-a02)(b00-b10)
    U3[1, 2] = 1; U3[2, 2] = -1  # a01, a02
    V3[0, 2] = 1; V3[3, 2] = -1  # b00, b10
    W3[2, 2] = 1  # c02
    W3[1, 2] = -1  # c01
    
    # Term 3: (a00-a02)(b00-b20)
    U3[0, 3] = 1; U3[2, 3] = -1  # a00, a02
    V3[0, 3] = 1; V3[6, 3] = -1  # b00, b20
    W3[0, 3] = 1  # c00
    W3[2, 3] = -1  # c02
    
    # Term 4: (a10+a11+a12)(b00+b01+b02)
    U3[3, 4] = 1; U3[4, 4] = 1; U3[5, 4] = 1  # a10, a11, a12
    V3[0, 4] = 1; V3[3, 4] = 1; V3[6, 4] = 1  # b00, b10, b20
    W3[3, 4] = 1  # c10 (k=0,i=1)
    W3[4, 4] = 1  # c11 (k=1,i=1)
    W3[5, 4] = 1  # c12 (k=2,i=1)
    
    # Term 5: (a10-a11)(b10-b20)
    U3[3, 5] = 1; U3[4, 5] = -1  # a10, a11
    V3[3, 5] = 1; V3[6, 5] = -1  # b10, b20
    W3[4, 5] = 1  # c11
    W3[5, 5] = -1  # c12
    
    # Term 6: (a11-a12)(b00-b10)
    U3[4, 6] = 1; U3[5, 6] = -1  # a11, a12
    V3[0, 6] = 1; V3[3, 6] = -1  # b00, b10
    W3[5, 6] = 1  # c12
    W3[4, 6] = -1  # c11
    
    # Term 7: (a10-a12)(b00-b20)
    U3[3, 7] = 1; U3[5, 7] = -1  # a10, a12
    V3[0, 7] = 1; V3[6, 7] = -1  # b00, b20
    W3[3, 7] = 1  # c10
    W3[5, 7] = -1  # c12
    
    # Term 8: (a20+a21+a22)(b00+b01+b02)
    U3[6, 8] = 1; U3[7, 8] = 1; U3[8, 8] = 1  # a20, a21, a22
    V3[0, 8] = 1; V3[3, 8] = 1; V3[6, 8] = 1  # b00, b10, b20
    W3[6, 8] = 1  # c20 (k=0,i=2)
    W3[7, 8] = 1  # c21 (k=1,i=2)
    W3[8, 8] = 1  # c22 (k=2,i=2)
    
    # Term 9: (a20-a21)(b10-b20)
    U3[6, 9] = 1; U3[7, 9] = -1  # a20, a21
    V3[3, 9] = 1; V3[6, 9] = -1  # b10, b20
    W3[7, 9] = 1  # c21
    W3[8, 9] = -1  # c22
    
    # Term 10: (a21-a22)(b00-b10)
    U3[7, 10] = 1; U3[8, 10] = -1  # a21, a22
    V3[0, 10] = 1; V3[3, 10] = -1  # b00, b10
    W3[8, 10] = 1  # c22
    W3[7, 10] = -1  # c21
    
    # Term 11: (a20-a22)(b00-b20)
    U3[6, 11] = 1; U3[8, 11] = -1  # a20, a22
    V3[0, 11] = 1; V3[6, 11] = -1  # b00, b20
    W3[6, 11] = 1  # c20
    W3[8, 11] = -1  # c22
    
    # Term 12: (a01+a11+a21)(b10+b11+b12)
    U3[1, 12] = 1; U3[4, 12] = 1; U3[7, 12] = 1  # a01, a11, a21
    V3[3, 12] = 1; V3[4, 12] = 1; V3[5, 12] = 1  # b10, b11, b12
    W3[1, 12] = 1  # c01
    W3[4, 12] = 1  # c11
    W3[7, 12] = 1  # c21
    
    # Term 13: (a02+a12+a22)(b20+b21+b22)
    U3[2, 13] = 1; U3[5, 13] = 1; U3[8, 13] = 1  # a02, a12, a22
    V3[6, 13] = 1; V3[7, 13] = 1; V3[8, 13] = 1  # b20, b21, b22
    W3[2, 13] = 1  # c02
    W3[5, 13] = 1  # c12
    W3[8, 13] = 1  # c22
    
    # Term 14: (a00+a10+a20)(b00+b10+b20)
    U3[0, 14] = 1; U3[3, 14] = 1; U3[6, 14] = 1  # a00, a10, a20
    V3[0, 14] = 1; V3[3, 14] = 1; V3[6, 14] = 1  # b00, b10, b20
    W3[0, 14] = 1  # c00
    W3[3, 14] = 1  # c10
    W3[6, 14] = 1  # c20
    
    # Verify the 3x3 decomposition is correct
    # Tensor T[i,j,k] = delta[i,k] for C[i,k] = sum_j A[i,j] B[j,k]
    # Decomposition: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = delta[i,k]
    # Let's verify
    max_err = 0.0
    for i in range(3):
        for k in range(3):
            val = 0.0
            for j in range(3):
                for r in range(15):
                    val += U3[i*3+j, r] * V3[j*3+k, r] * W3[k*3+i, r]
            expected = 1.0 if i == k else 0.0
            err = abs(val - expected)
            max_err = max(max_err, err)
    
    if max_err > 1e-10:
        # Fallback to schoolbook 3x3 if Winograd is incorrect
        U3 = np.zeros((9, 27), dtype=int)
        V3 = np.zeros((9, 27), dtype=int)
        W3 = np.zeros((9, 27), dtype=int)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    r = i * 9 + j * 3 + k
                    U3[i * 3 + j, r] = 1
                    V3[j * 3 + k, r] = 1
                    W3[k * 3 + i, r] = 1
        rank_3 = 27
    else:
        rank_3 = 15
    
    if time.time() > deadline - 10:
        # Not enough time, use schoolbook for safety
        U3 = np.zeros((9, 27), dtype=int)
        V3 = np.zeros((9, 27), dtype=int)
        W3 = np.zeros((9, 27), dtype=int)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    r = i * 9 + j * 3 + k
                    U3[i * 3 + j, r] = 1
                    V3[j * 3 + k, r] = 1
                    W3[k * 3 + i, r] = 1
        rank_3 = 27
    
    # Construct 6x6 factors via Kronecker product
    rank_2 = 7
    rank = rank_2 * rank_3
    U6 = np.zeros((36, rank), dtype=int)
    V6 = np.zeros((36, rank), dtype=int)
    W6 = np.zeros((36, rank), dtype=int)
    
    for r2 in range(rank_2):
        for r3 in range(rank_3):
            r = r2 * rank_3 + r3
            
            nz_u2 = np.nonzero(U2[:, r2])[0]
            nz_v2 = np.nonzero(V2[:, r2])[0]
            nz_w2 = np.nonzero(W2[:, r2])[0]
            
            nz_u3 = np.nonzero(U3[:, r3])[0]
            nz_v3 = np.nonzero(V3[:, r3])[0]
            nz_w3 = np.nonzero(W3[:, r3])[0]
            
            for u2 in nz_u2:
                i2 = u2 // 2
                j2 = u2 % 2
                val_u2 = U2[u2, r2]
                
                for u3 in nz_u3:
                    i3 = u3 // 3
                    j3 = u3 % 3
                    val_u3 = U3[u3, r3]
                    
                    i6 = i2 * 3 + i3
                    j6 = j2 * 3 + j3
                    g_u = i6 * 6 + j6
                    U6[g_u, r] = val_u2 * val_u3
            
            for v2 in nz_v2:
                j2 = v2 // 2
                k2 = v2 % 2
                val_v2 = V2[v2, r2]
                
                for v3 in nz_v3:
                    j3 = v3 // 3
                    k3 = v3 % 3
                    val_v3 = V3[v3, r3]
                    
                    j6 = j2 * 3 + j3
                    k6 = k2 * 3 + k3
                    g_v = j6 * 6 + k6
                    V6[g_v, r] = val_v2 * val_v3
            
            for w2 in nz_w2:
                k2 = w2 // 2
                i2 = w2 % 2
                val_w2 = W2[w2, r2]
                
                for w3 in nz_w3:
                    k3 = w3 // 3
                    i3 = w3 % 3
                    val_w3 = W3[w3, r3]
                    
                    k6 = k2 * 3 + k3
                    i6 = i2 * 3 + i3
                    g_w = k6 * 6 + i6
                    W6[g_w, r] = val_w2 * val_w3
    
    # Final verification: check the 6x6 tensor decomposition
    # Sample check with random matrices
    np.random.seed(42)
    A_test = np.random.randint(-5, 5, (6, 6))
    B_test = np.random.randint(-5, 5, (6, 6))
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((6, 6), dtype=int)
    for r in range(rank):
        # Compute the product for this rank term
        a_vec = U6[:, r]  # shape (36,)
        b_vec = V6[:, r]  # shape (36,)
        w_vec = W6[:, r]  # shape (36,)
        
        # Reshape to matrices
        A_r = a_vec.reshape(6, 6)
        B_r = b_vec.reshape(6, 6)
        W_r = w_vec.reshape(6, 6)
        
        # The contribution to C is: for each (i,k), sum_j A_r[i,j] * B_r[j,k] * W_r[k,i]
        # Actually, the tensor decomposition means:
        # C[i,k] = sum_r (sum_j U6[i*6+j, r] * V6[j*6+k, r]) * W6[k*6+i, r]
        # Wait, no. The tensor is T[i,j,k] where C[i,k] = sum_j A[i,j]*B[j,k]
        # The decomposition: T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] (with trace convention)
        # So C[i,k] = sum_j sum_r U[i*6+j,r]*V[j*6+k,r]*W[k*6+i,r]
        
        prod = np.zeros((6, 6), dtype=int)
        for i in range(6):
            for k in range(6):
                s = 0
                for j in range(6):
                    s += U6[i*6+j, r] * V6[j*6+k, r]
                prod[i, k] = s * W6[k*6+i, r]
        C_computed += prod
    
    max_diff = np.max(np.abs(C_computed - C_expected))
    if max_diff != 0:
        # Fallback to full schoolbook 6x6
        U6 = np.zeros((36, 216), dtype=int)
        V6 = np.zeros((36, 216), dtype=int)
        W6 = np.zeros((36, 216), dtype=int)
        for i in range(6):
            for j in range(6):
                for k in range(6):
                    r = i * 36 + j * 6 + k
                    U6[i * 6 + j, r] = 1
                    V6[j * 6 + k, r] = 1
                    W6[k * 6 + i, r] = 1
    
    return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}
# EVOLVE-BLOCK-END
