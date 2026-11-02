"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _strassen_2x2():
    """
    Return rank-7 Strassen decomposition for 2x2 matrix multiplication.
    Returns U (4,7), V (4,7), W (4,7) where:
    U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    # Strassen 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # U[i*2+j, r] = coefficient of A[i,j] in X_r
    # V[j*2+k, r] = coefficient of B[j,k] in Y_r
    # W[k*2+i, r] = coefficient of M_r in C[i,j]
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1: (A00+A11)(B00+B11)
    # X1 = A00+A11, Y1 = B00+B11
    U[0, 0] = 1  # A00
    U[3, 0] = 1  # A11
    V[0, 0] = 1  # B00
    V[3, 0] = 1  # B11
    # C00 += M1, C11 += M1
    W[0, 0] = 1  # C00
    W[3, 0] = 1  # C11
    
    # M2: (A10+A11)B00
    # X2 = A10+A11, Y2 = B00
    U[2, 1] = 1  # A10
    U[3, 1] = 1  # A11
    V[0, 1] = 1  # B00
    # C10 += M2, C11 -= M2
    W[2, 1] = 1  # C10
    W[3, 1] = -1 # C11
    
    # M3: A00(B01-B11)
    # X3 = A00, Y3 = B01-B11
    U[0, 2] = 1  # A00
    V[1, 2] = 1  # B01
    V[3, 2] = -1 # B11
    # C01 += M3, C11 += M3
    W[1, 2] = 1  # C01
    W[3, 2] = 1  # C11
    
    # M4: A11(B10-B00)
    # X4 = A11, Y4 = B10-B00
    U[3, 3] = 1  # A11
    V[2, 3] = 1  # B10
    V[0, 3] = -1 # B00
    # C00 += M4, C10 += M4
    W[0, 3] = 1  # C00
    W[2, 3] = 1  # C10
    
    # M5: (A00+A01)B11
    # X5 = A00+A01, Y5 = B11
    U[0, 4] = 1  # A00
    U[1, 4] = 1  # A01
    V[3, 4] = 1  # B11
    # C00 -= M5, C01 += M5
    W[0, 4] = -1 # C00
    W[1, 4] = 1  # C01
    
    # M6: (A10-A00)(B00+B01)
    # X6 = A10-A00, Y6 = B00+B01
    U[2, 5] = 1  # A10
    U[0, 5] = -1 # A00
    V[0, 5] = 1  # B00
    V[1, 5] = 1  # B01
    # C11 += M6
    W[3, 5] = 1  # C11
    
    # M7: (A01-A11)(B10+B11)
    # X7 = A01-A11, Y7 = B10+B11
    U[1, 6] = 1  # A01
    U[3, 6] = -1 # A11
    V[2, 6] = 1  # B10
    V[3, 6] = 1  # B11
    # C00 += M7
    W[0, 6] = 1  # C00
    
    return U, V, W

def _kron_decompose(U1, V1, W1, U2, V2, W2):
    """
    Combine two decompositions via Kronecker product.
    If T1 = sum_r u1_r v1_r w1_r and T2 = sum_s u2_s v2_s w2_s,
    then T1 ⊗ T2 = sum_{r,s} (u1_r ⊗ u2_s)(v1_r ⊗ v2_s)(w1_r ⊗ w2_s)
    """
    r1 = U1.shape[1]
    r2 = U2.shape[1]
    total_rank = r1 * r2
    
    n1, m1, p1 = U1.shape[0] // 2, 2, 2  # Assume 2x2x2
    n2, m2, p2 = U2.shape[0] // 2, 2, 2
    
    n = n1 * n2
    m = m1 * m2
    p = p1 * p2
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    for r in range(r1):
        for s in range(r2):
            # U: (i1*m1+j1, i2*m2+j2) -> ((i1*m2+i2)*(m1*m2) + (j1*m2+j2))
            # Actually, U shape is (n*m, rank) where n = n1*n2, m = m1*m2
            # Index in U: i*n*m... wait, let me think about the indexing.
            # U[i*m+j, r] where i in [0,n), j in [0,m)
            # For Kronecker: i = i1*n2 + i2, j = j1*m2 + j2
            # So U index = i*m + j = (i1*n2+i2)*m + (j1*m2+j2)
            
            # V[j*p+k, r] where j in [0,m), k in [0,p)
            # j = j1*m2 + j2, k = k1*p2 + k2
            # V index = j*p + k = (j1*m2+j2)*p + (k1*p2+k2)
            
            # W[k*n+i, r] where k in [0,p), i in [0,n)
            # k = k1*p2 + k2, i = i1*n2 + i2
            # W index = k*n + i = (k1*p2+k2)*n + (i1*n2+i2)
            
            for i1 in range(n1):
                for j1 in range(m1):
                    for i2 in range(n2):
                        for j2 in range(m2):
                            i = i1 * n2 + i2
                            j = j1 * m2 + j2
                            U_idx = i * m + j
                            U[U_idx, col] = U1[i1 * m1 + j1, r] * U2[i2 * m2 + j2, s]
            
            for j1 in range(m1):
                for k1 in range(p1):
                    for j2 in range(m2):
                        for k2 in range(p2):
                            j = j1 * m2 + j2
                            k = k1 * p2 + k2
                            V_idx = j * p + k
                            V[V_idx, col] = V1[j1 * p1 + k1, r] * V2[j2 * p2 + k2, s]
            
            for k1 in range(p1):
                for i1 in range(n1):
                    for k2 in range(p2):
                        for i2 in range(n2):
                            k = k1 * p2 + k2
                            i = i1 * n2 + i2
                            W_idx = k * n + i
                            W[W_idx, col] = W1[k1 * n1 + i1, r] * W2[k2 * n2 + i2, s]
            
            col += 1
    
    return U, V, W

def _construct_4x4x4_strassen():
    """
    Construct rank-49 decomposition for 4x4x4 matrix multiplication
    using two levels of Strassen.
    """
    # Level 1: 2x2x2 Strassen (rank 7)
    U1, V1, W1 = _strassen_2x2()
    
    # Level 2: Apply Strassen again to get 4x4x4
    U2, V2, W2 = _strassen_2x2()
    
    # Kronecker product gives 4x4x4 with rank 7*7 = 49
    U, V, W = _kron_decompose(U1, V1, W1, U2, V2, W2)
    
    return U, V, W

def _schoolbook_part(n, m, p, start_col):
    """
    Schoolbook decomposition for n x m x p, starting at column start_col.
    Returns U, V, W of shapes (n*m, p), (m*p, p), (p*n, p)
    """
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify the decomposition is correct."""
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    # Compute T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    # For matrix multiplication, T[i,j,k] should be 1 if this is the matmul tensor
    # Actually, the matmul tensor T[i,j,k] = 1 for all i,j,k (it's the all-ones tensor for matmul)
    # Wait, no. The matrix multiplication tensor is defined such that:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor T[i,j,k] = 1 for all valid i,j,k
    return np.allclose(T, 1.0, atol=1e-10)

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Strategy: 4x4x7 = 4x4x4 + 4x4x3
    # 4x4x4: rank 49 (Strassen)
    # 4x4x3: rank 48 (schoolbook)
    # Total: rank 97
    
    # Part 1: 4x4x4 (columns 0-3 of B and C)
    U_444, V_444, W_444 = _construct_4x4x4_strassen()
    rank_444 = U_444.shape[1]  # Should be 49
    
    # Part 2: 4x4x3 (columns 4-6 of B and C)
    # Schoolbook for 4x4x3: rank = 4*4*3 = 48
    U_443, V_443, W_443 = _schoolbook_part(4, 4, 3, 0)
    rank_443 = U_443.shape[1]  # Should be 48
    
    total_rank = rank_444 + rank_443
    
    # Combine
    U = np.zeros((n * m, total_rank))  # 16 x 97
    V = np.zeros((m * p, total_rank))  # 28 x 97
    W = np.zeros((p * n, total_rank))  # 28 x 97
    
    # Place 4x4x4 part (B cols 0-3, C cols 0-3)
    U[:, 0:rank_444] = U_444
    V[:, 0:rank_444] = V_444[:, 0:4*4]  # V_444 is 16x49, we need first 16 cols (B cols 0-3)
    W[:, 0:rank_444] = W_444[:, 0:4*4]  # W_444 is 16x49, we need first 16 cols (C cols 0-3)
    
    # Place 4x4x3 part (B cols 4-6, C cols 4-6)
    # V_443 is 12x48 (m*p = 4*3), we need to place it in V[:, rank_444:rank_444+48]
    # V index: j*p + k where j in [0,4), k in [0,7)
    # For the 4x4x3 part, k is offset by 4, so V[j*7 + (4+k3), col] = V_443[j*3 + k3, col]
    for j in range(4):
        for k3 in range(3):
            v_idx = j * 7 + (4 + k3)
            v_idx_443 = j * 3 + k3
            V[v_idx, rank_444:rank_444+rank_443] = V_443[v_idx_443, :]
    
    # W index: k*n + i where k in [0,7), i in [0,4)
    # For the 4x4x3 part, k is offset by 4
    for k3 in range(3):
        for i in range(4):
            w_idx = (4 + k3) * 4 + i
            w_idx_443 = k3 * 4 + i
            W[w_idx, rank_444:rank_444+rank_443] = W_443[w_idx_443, :]
    
    # Verify
    if time.time() < deadline:
        if not _verify_decomposition(U, V, W, n, m, p):
            # Fallback to schoolbook if verification fails
            rank = n * m * p
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j, col] = 1
                        V[j * p + k, col] = 1
                        W[k * n + i, col] = 1
                        col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
