import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Winograd 4x4 matrices
    # M_A, M_B, M_C such that C = M_C * ( (A * M_A) o (M_B * B) )
    # Actually, the standard form is C = M_L * (hat_A o hat_B) * M_R
    # hat_A = A * M_A, hat_B = M_B * B
    
    # For 4x4, the matrices are:
    # M_A = [[1, 1, 1, 1],
    #        [1, -1, 1, -1],
    #        [1, 1, -1, -1],
    #        [1, -1, -1, 1]]
    # This is the Hadamard matrix H4.
    # If we use H4, the inverse is H4/4.
    # So C = (H4/4) * ( (A * H4) o (H4 * B) ) * (H4/4)^T ?
    # Let's check.
    # A_hat = A * H4
    # B_hat = H4 * B
    # C_hat = A_hat o B_hat
    # C = (H4/4) * C_hat * (H4/4)^T
    # This works because H4 is symmetric and orthogonal (up to scale).
    # Rank is 16.
    
    H4 = np.array([
        [1, 1, 1, 1],
        [1, -1, 1, -1],
        [1, 1, -1, -1],
        [1, -1, -1, 1]
    ])
    
    # The decomposition for 4x4x4:
    # C[i,k] = sum_{r,s} (H4/4)[i,r] * (A*H4)[r,s] * (H4*B)[r,s] * (H4/4)[k,s]
    # Wait, (H4/4)^T is (H4/4).
    # So C[i,k] = sum_{r,s} (H4[i,r]/4) * (A*H4)[r,s] * (H4*B)[r,s] * (H4[k,s]/4)
    
    # We need to map this to U, V, W.
    # U is (n*m, rank). Here n=4, m=4. U is 16x16.
    # V is (m*p, rank). Here m=4, p=4. V is 16x16.
    # W is (p*n, rank). Here p=4, n=4. W is 16x16.
    
    # The rank indices are (r, s) for r in 0..3, s in 0..3.
    # U[i*m + j, r*4 + s] = (A*H4)[r,s] coefficient for A[i,j]
    # A*H4 = sum_j A[i,j] H4[j,s] ... wait.
    # (A*H4)[r,s] = sum_j A[i,j] H4[j,s]? No, A is 4x4.
    # (A*H4)[i,s] = sum_j A[i,j] H4[j,s].
    # So U[i*4 + j, r*4 + s] should be 0 unless r==i?
    # No, the tensor is C[i,k] = sum_{j} A[i,j] B[j,k].
    # In the Winograd scheme:
    # C[i,k] = sum_{r,s} M_L[i,r] * hat_C[r,s] * M_R[s,k]
    # hat_C[r,s] = hat_A[r,s] * hat_B[r,s]
    # hat_A[r,s] = sum_j A[r,j] M_A[j,s]  (if hat_A = A * M_A)
    # hat_B[r,s] = sum_j M_B[r,j] B[j,s]  (if hat_B = M_B * B)
    
    # So U[r*4 + j, r*4 + s] = M_A[j,s] ? No.
    # U is indexed by (i, j) for A, and (r, s) for rank.
    # U[i*4 + j, r*4 + s] = 1 if r==i, else 0? No.
    # hat_A[r,s] depends on A[r,j].
    # So U[r*4 + j, r*4 + s] = M_A[j,s].
    # But our U is for the specific problem.
    
    # Let's build the 4x4x4 decomposition first.
    # U_444[i*4 + j, r*4 + s] = M_A[j,s] if r==i else 0?
    # No, hat_A[r,s] = sum_j A[r,j] M_A[j,s].
    # So U[r*4 + j, r*4 + s] = M_A[j,s].
    # V_444[j*4 + k, r*4 + s] = M_B[r,j] if k==s else 0?
    # hat_B[r,s] = sum_j M_B[r,j] B[j,s].
    # So V[j*4 + s, r*4 + s] = M_B[r,j].
    # W_444[k*4 + i, r*4 + s] = M_L[i,r] * M_R[s,k] / 16?
    # C[i,k] = sum_{r,s} M_L[i,r] hat_C[r,s] M_R[s,k].
    # So W[k*4 + i, r*4 + s] = M_L[i,r] * M_R[s,k].
    
    # M_L = H4/4, M_R = H4/4, M_A = H4, M_B = H4.
    
    M_A = H4
    M_B = H4
    M_L = H4 / 4.0
    M_R = H4 / 4.0
    
    # Build U, V, W for 4x4x4
    U_444 = np.zeros((16, 16))
    V_444 = np.zeros((16, 16))
    W_444 = np.zeros((16, 16))
    
    for i in range(4):
        for j in range(4):
            for r in range(4):
                for s in range(4):
                    # U[i*4 + j, r*4 + s]
                    if r == i:
                        U_444[i*4 + j, r*4 + s] = M_A[j, s]
                    
                    # V[j*4 + s, r*4 + s] -> V is (m*p, rank) = (4*4, 16)
                    # V index is (j, s) for B[j,s]? No, B is 4x4.
                    # V[j*4 + k, r*4 + s] where k is col of B.
                    # hat_B[r,s] = sum_j M_B[r,j] B[j,s].
                    # So V[j*4 + s, r*4 + s] = M_B[r, j].
                    # Here k=s.
                    V_444[j*4 + s, r*4 + s] = M_B[r, j]
                    
                    # W[k*4 + i, r*4 + s]
                    # W index is (k, i) for C[i,k].
                    # W[k*4 + i, r*4 + s] = M_L[i, r] * M_R[s, k].
                    W_444[k*4 + i, r*4 + s] = M_L[i, r] * M_R[s, k]
    
    # Now for 2x4x7
    # We split into 2x4x4 and 2x4x3.
    # For 2x4x4:
    # A is 2x4, pad to 4x4.
    # B1 is 4x4.
    # Use U_444, V_444, W_444.
    # U is (2*4, 16) = (8, 16).
    # V is (4*4, 16) = (16, 16).
    # W is (4*2, 16) = (8, 16).
    
    U1 = np.zeros((8, 16))
    V1 = np.zeros((16, 16))
    W1 = np.zeros((8, 16))
    
    for i in range(2):
        for j in range(4):
            for r in range(4):
                for s in range(4):
                    U1[i*4 + j, r*4 + s] = U_444[i*4 + j, r*4 + s]
                    V1[j*4 + s, r*4 + s] = V_444[j*4 + s, r*4 + s]
                    W1[s*2 + i, r*4 + s] = W_444[s*2 + i, r*4 + s] # k=s, n=2
    
    # For 2x4x3:
    # B2 is 4x3, pad to 4x4 (col 3 is 0).
    # Use U_444, V_444, W_444.
    # U is (8, 16).
    # V is (16, 16).
    # W is (3*2, 16) = (6, 16).
    
    U2 = np.zeros((8, 16))
    V2 = np.zeros((16, 16))
    W2 = np.zeros((6, 16))
    
    for i in range(2):
        for j in range(4):
            for r in range(4):
                for s in range(4):
                    U2[i*4 + j, r*4 + s] = U_444[i*4 + j, r*4 + s]
                    # B2[j, k] for k=0,1,2.
                    # V[j*4 + k, r*4 + s] = V_444[j*4 + k, r*4 + s] for k=0,1,2.
                    # For k=3, B2[j,3]=0, so we don't need it.
                    for k in range(3):
                        V2[j*4 + k, r*4 + s] = V_444[j*4 + k, r*4 + s]
                    # W[k*2 + i, r*4 + s] for k=0,1,2.
                    for k in range(3):
                        W2[k*2 + i, r*4 + s] = W_444[k*2 + i, r*4 + s]
    
    # Combine
    U = np.concatenate([U1, U2], axis=1)
    V = np.concatenate([V1, V2], axis=1)
    W = np.concatenate([W1, W2], axis=1)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
