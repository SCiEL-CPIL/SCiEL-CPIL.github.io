import numpy as np

DIMENSIONS = [2, 2, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank 7 decomposition for 2x2
    # u: 4x7, v: 4x7
    u_222 = np.array([
        [1, 0, 0, 0, 1, 0, 0], # a
        [0, 1, 0, 0, 1, 0, 1], # b
        [0, 0, 1, 0, 0, 1, 0], # c
        [0, 0, 0, 1, 0, 1, 1]  # d
    ])
    v_222 = np.array([
        [1, 0, 0, 0, 1, 1, 0], # e
        [0, 1, 0, 0, 0, 1, 0], # f
        [0, 0, 1, 0, 1, 0, 1], # g
        [0, 0, 0, 1, 0, 0, 1]  # h
    ])
    
    # Solve for w_222: 4x7
    # C = AB
    # C_flat = S W^T
    # S_r = (u_r^T A) (v_r^T B)
    
    # Use random A, B to solve for W
    np.random.seed(42)
    N_samples = 100
    A_samples = np.random.rand(N_samples, 2, 2)
    B_samples = np.random.rand(N_samples, 2, 2)
    C_samples = np.einsum('nij,njk->nik', A_samples, B_samples)
    C_flat = C_samples.reshape(N_samples, 4)
    
    # Compute S: N x 7
    # u_r^T A: sum u_r[i,j] A[i,j]
    # A_flat: N x 4
    A_flat = A_samples.reshape(N_samples, 4)
    B_flat = B_samples.reshape(N_samples, 4)
    
    # u_222 is 4x7. u_222.T is 7x4.
    # S_A = A_flat @ u_222.T -> N x 7
    S_A = A_flat @ u_222.T
    S_B = B_flat @ v_222.T
    S = S_A * S_B # N x 7
    
    # Solve C_flat = S @ W_222.T
    # W_222.T is 7x4
    W_222_T, _, _, _ = np.linalg.lstsq(S, C_flat.T, rcond=None)
    W_222 = W_222_T.T # 4x7
    
    # Verify
    # C_recon = S @ W_222.T
    # assert np.allclose(C_flat, C_recon)
    
    # Construct full U, V, W
    rank = 21
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for k in range(3):
        # U is same for all blocks
        U[:, k*7:(k+1)*7] = u_222
        
        # V
        # V_222 is 4x7. Rows correspond to B indices j*2+l
        # V full rows correspond to B indices j*6+k_col
        # For block k, B cols are k*2, k*2+1
        for j in range(2):
            for l in range(2):
                v_idx = j * 6 + k * 2 + l
                v_222_idx = j * 2 + l
                V[v_idx, k*7:(k+1)*7] = v_222[v_222_idx, :]
                
        # W
        # W_222 is 4x7. Rows correspond to C indices k'*2+i
        # W full rows correspond to C indices k_col*2+i
        # For block k, C cols are k*2, k*2+1
        for i in range(2):
            for l in range(2):
                w_idx = (k * 2 + l) * 2 + i # k_col * n + i
                w_222_idx = l * 2 + i
                W[w_idx, k*7:(k+1)*7] = W_222[w_222_idx, :]
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
