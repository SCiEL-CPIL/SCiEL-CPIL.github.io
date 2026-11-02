import time

def construct():
    deadline = time.time() + 120
    
    # Dimensions
    n, m, p = 6, 6, 6
    
    # Strassen 2x2 decomposition
    # U2, V2, W2 of shape (4, 7)
    # Local indices: 0:00, 1:01, 2:10, 3:11
    U2 = [
        [1, 1, 0, 0, 1, 0, 1],
        [0, 1, 1, 0, 0, 1, -1],
        [0, 0, 1, 1, -1, 0, 1],
        [1, 0, 0, 1, 0, -1, -1]
    ]
    V2 = [
        [1, 0, 1, 0, 1, 1, 1],
        [0, 1, 0, 1, 1, 0, -1],
        [0, 0, 1, 0, 0, 1, 1],
        [1, 0, 0, 1, 0, -1, -1]
    ]
    W2 = [
        [1, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [1, 1, 0, -1, 0, 0, 1]
    ]
    
    # Verify Strassen 2x2
    # C[i,j] = sum_k A[i,k] B[k,j]
    # sum_s U2[i*2+k, s] V2[k*2+j, s] W2[j*2+i, s] = 1 for all i,k,j?
    # No, the tensor condition is:
    # sum_s U2[i*2+k, s] V2[k*2+j, s] W2[j*2+i, s] should be 1?
    # Let's check the indices.
    # U index: i*2+k (row i, col k of A)
    # V index: k*2+j (row k, col j of B)
    # W index: j*2+i (col j of C? No, W is p*n. Index j*n+i? No, W is 2*2. Index j*2+i)
    # C[i,j] = sum_k A[i,k] B[k,j].
    # The coefficient of A[i,k] B[k,j] in C[i,j] is 1.
    # So sum_s U2[i*2+k, s] V2[k*2+j, s] W2[j*2+i, s] must be 1.
    
    def verify_strassen():
        for i in range(2):
            for k in range(2):
                for j in range(2):
                    s = 0
                    for s_idx in range(7):
                        s += U2[i*2+k][s_idx] * V2[k*2+j][s_idx] * W2[j*2+i][s_idx]
                    if s != 1:
                        return False
        return True
    
    if not verify_strassen():
        # Fallback to schoolbook if Strassen is wrong (it shouldn't be)
        return _build_schoolbook()

    # Schoolbook 3x3 decomposition
    # U3, V3, W3 of shape (9, 27)
    U3 = [[0]*27 for _ in range(9)]
    V3 = [[0]*27 for _ in range(9)]
    W3 = [[0]*27 for _ in range(9)]
    
    r = 0
    for i in range(3):
        for k in range(3):
            for j in range(3):
                U3[i*3+j][r] = 1
                V3[j*3+k][r] = 1
                W3[k*3+i][r] = 1
                r += 1
    
    # Construct 6x6 decomposition
    rank = 27 * 7
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for i in range(3):
        for k in range(3):
            for j in range(3):
                # Block multiplication term: A[i,j] * B[j,k] -> C[i,k]
                # U3 index: i*3+j
                # V3 index: j*3+k
                # W3 index: k*3+i
                u3_val = U3[i*3+j][r]
                v3_val = V3[j*3+k][r]
                w3_val = W3[k*3+i][r]
                
                for s in range(7):
                    # Strassen term s
                    for u1 in range(2):
                        for v1 in range(2):
                            # A global index: (i*2+u1, j*2+v1)
                            U[(i*2+u1)*m + (j*2+v1)][col] += u3_val * U2[u1*2+v1][s]
                            
                            # B global index: (j*2+u1, k*2+v1) ?
                            # Strassen U2 index is u1*2+v1.
                            # Strassen V2 index is u1*2+v1.
                            # This corresponds to the same "position" in the block?
                            # Yes, Strassen operates on the 2x2 block.
                            # The input A block is A[i,j], input B block is B[j,k].
                            # The Strassen decomposition factors the product of these two blocks.
                            # So we use U2 and V2 with the same local indices.
                            
                            V[(j*2+u1)*p + (k*2+v1)][col] += v3_val * V2[u1*2+v1][s]
                            
                            # C global index: (i*2+u1, k*2+v1) ?
                            # Strassen W2 index is u1*2+v1.
                            # W2 maps to output C block.
                            # The output block is C[i,k].
                            # The local output indices are u1, v1.
                            # So global C index is (i*2+u1, k*2+v1).
                            # W index in tensor: k_global * n + i_global.
                            # k_global = k*2+v1, i_global = i*2+u1.
                            W[(k*2+v1)*n + (i*2+u1)][col] += w3_val * W2[u1*2+v1][s]
                            
                    col += 1
                r += 1
                
    assert col == rank, col
    
    return {"U": U, "V": V, "W": W}

def _build_schoolbook():
    n, m, p = 6, 6, 6
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
