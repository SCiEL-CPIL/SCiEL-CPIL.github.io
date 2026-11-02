"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix by a 4x4 matrix.
    
    The tensor for matrix multiplication A(2x4) * B(4x4) = C(2x4):
    T = sum_{i,j,k} e_{i,j} ⊗ f_{j,k} ⊗ g_{i,k}
    
    Using the trace convention: W index is k*n + i.
    
    U_dim = n*m = 8, V_dim = m*p = 16, W_dim = p*n = 8.
    
    The schoolbook rank is 32. We use a greedy rank-1 decomposition to find
    an exact decomposition, hoping to find one with rank < 32.
    """
    import numpy as np
    
    n, m, p = DIMENSIONS
    U_dim = n * m  # 8
    V_dim = m * p  # 16
    W_dim = p * n  # 8
    
    T = np.zeros((U_dim, V_dim, W_dim))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
    
    # Greedy rank-1 decomposition: pick the largest-magnitude entry,
    # create a rank-1 term matching it, subtract from residual.
    residual = T.copy()
    U_cols = []
    V_cols = []
    W_cols = []
    
    max_rank = U_dim * V_dim * W_dim
    for r in range(max_rank):
        flat_idx = np.argmax(np.abs(residual))
        u_idx, v_idx, w_idx = np.unravel_index(flat_idx, residual.shape)
        val = residual[u_idx, v_idx, w_idx]
        
        if abs(val) < 1e-10:
            break
        
        u_vec = np.zeros(U_dim)
        v_vec = np.zeros(V_dim)
        w_vec = np.zeros(W_dim)
        u_vec[u_idx] = 1.0
        v_vec[v_idx] = 1.0
        w_vec[w_idx] = 1.0
        
        residual[u_idx, v_idx, w_idx] -= val
        
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
    
    rank = len(U_cols)
    
    U_mat = np.array(U_cols).T if U_cols else np.zeros((U_dim, 0))
    V_mat = np.array(V_cols).T if V_cols else np.zeros((V_dim, 0))
    W_mat = np.array(W_cols).T if W_cols else np.zeros((W_dim, 0))
    
    # Verify correctness
    if rank > 0:
        recon = np.einsum('ur,vr,wr->uvw', U_mat, V_mat, W_mat)
        if not np.allclose(recon, T, atol=1e-10):
            # Fallback to schoolbook
            rank = n * m * p
            U_mat = np.zeros((U_dim, rank))
            V_mat = np.zeros((V_dim, rank))
            W_mat = np.zeros((W_dim, rank))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_mat[i * m + j, col] = 1
                        V_mat[j * p + k, col] = 1
                        W_mat[k * n + i, col] = 1
                        col += 1
    
    U_list = [[float(U_mat[i, r]) for r in range(rank)] for i in range(U_dim)]
    V_list = [[float(V_mat[i, r]) for r in range(rank)] for i in range(V_dim)]
    W_list = [[float(W_mat[i, r]) for r in range(rank)] for i in range(W_dim)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
