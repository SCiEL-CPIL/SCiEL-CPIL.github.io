import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    n, m, p = 4, 4, 4

    # Build rank-49 Strassen^2 decomposition
    # 2x2 Strassen factors
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)

    # M0 = (A00 + A11) * (B00 + B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1

    # M1 = (A10 + A11) * B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = -1

    # M2 = A00 * (B01 - B11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1

    # M3 = A11 * (B10 - B00)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1

    # M4 = (A00 + A01) * B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1

    # M5 = (A10 - A00) * (B00 + B01)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1

    # M6 = (A01 - A11) * (B10 + B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1

    # Build 49-rank decomposition
    U_arr = np.zeros((16, 49), dtype=int)
    V_arr = np.zeros((16, 49), dtype=int)
    W_arr = np.zeros((16, 49), dtype=int)
    
    I_map = [i // 2 for i in range(4)]
    J_map = [j // 2 for j in range(4)]
    ii_map = [i % 2 for i in range(4)]
    jj_map = [j % 2 for j in range(4)]
    
    for s in range(7):
        for t in range(7):
            r = s * 7 + t
            for i in range(4):
                for j in range(4):
                    coeff = U2[I_map[i]*2 + J_map[j], s] * U2[ii_map[i]*2 + jj_map[j], t]
                    if coeff != 0:
                        U_arr[i*4 + j, r] = coeff
            for k in range(4):
                for l in range(4):
                    coeff = V2[I_map[k]*2 + J_map[l], s] * V2[ii_map[k]*2 + jj_map[l], t]
                    if coeff != 0:
                        V_arr[k*4 + l, r] = coeff
            for j in range(4):
                for i in range(4):
                    coeff = W2[I_map[i]*2 + J_map[j], s] * W2[ii_map[i]*2 + jj_map[j], t]
                    if coeff != 0:
                        W_arr[j*4 + i, r] = coeff
    
    # Verify the rank-49 solution (vectorized)
    def verify_decomp(U_mat, V_mat, W_mat, n, m, p):
        """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
        U_mat = np.array(U_mat)
        V_mat = np.array(V_mat)
        W_mat = np.array(W_mat)
        rank = U_mat.shape[1]
        
        # Test with multiple random matrices for robustness
        np.random.seed(42)
        for trial in range(3):
            A = np.random.randint(-5, 5, (n, m)).astype(float)
            B = np.random.randint(-5, 5, (m, p)).astype(float)
            C_expected = A @ B
            
            # U: (n*m, rank) -> reshape to (rank, n, m)
            # V: (m*p, rank) -> reshape to (rank, m, p)
            # W: (p*n, rank) -> W[k*n+i, r] where k=j, i=row -> reshape to (rank, p, n)
            U_block = U_mat.T.reshape(rank, n, m)  # (rank, n, m)
            V_block = V_mat.T.reshape(rank, m, p)  # (rank, m, p)
            W_block = W_mat.T.reshape(rank, p, n)  # (rank, p, n)
            
            # For each rank element r: C[i,j] += W[r,j,i] * sum_a U[r,i,a] * V[r,a,j]
            # Use einsum: U[r,i,a] * V[r,a,j] -> sum over a -> (rank, n, p)
            UV = np.einsum('ria,raj->rij', U_block, V_block)  # (rank, n, p)
            # C[i,j] = sum_r W[r,j,i] * UV[r,i,j]
            C_computed = np.einsum('rji,rij->ij', W_block, UV)
            
            if not np.allclose(C_computed, C_expected, atol=1e-10):
                return False
        
        return True
    
    if not verify_decomp(U_arr, V_arr, W_arr, n, m, p):
        # Error in verification, return as-is (shouldn't happen)
        return {"U": U_arr.tolist(), "V": V_arr.tolist(), "W": W_arr.tolist()}
    
    # Now try to find rank 48 by checking if any single column is redundant
    found_rank_48 = False
    
    for remove_col in range(49):
        if time.time() > deadline:
            break
        
        U_test = np.delete(U_arr, remove_col, axis=1)
        V_test = np.delete(V_arr, remove_col, axis=1)
        W_test = np.delete(W_arr, remove_col, axis=1)
        
        if verify_decomp(U_test, V_test, W_test, n, m, p):
            found_rank_48 = True
            rank = 48
            U_arr = U_test
            V_arr = V_test
            W_arr = W_test
            break
    
    if not found_rank_48:
        # Try to find a linear dependence among the 49 rank-1 tensors
        # Build the 16*16*16 tensor and check rank of the 49 vectors
        tensor_dim = (n * m) * (m * p) * (p * n)  # 16*16*16 = 4096
        
        # Build matrix of 49 rank-1 tensor vectors
        M = np.zeros((49, tensor_dim))
        for r in range(49):
            u = U_arr[:, r]  # (16,)
            v = V_arr[:, r]  # (16,)
            w = W_arr[:, r]  # (16,)
            # Outer product reshaped
            outer = (u[:, None, None] * v[None, :, None] * w[None, None, :]).flatten()
            M[r, :] = outer
        
        # Check numerical rank
        try:
            sv = np.linalg.svd(M, compute_uv=False)
            # Count significant singular values
            threshold = sv[0] * 1e-10
            num_indep = np.sum(sv > threshold)
        except:
            num_indep = 49
        
        if num_indep < 49:
            # We can potentially reduce to num_indep rank
            # Find a basis for the column space
            Q, R = np.linalg.qr(M.T)  # M.T is (4096, 49)
            # Find pivot columns
            diag = np.abs(np.diag(R))
            pivots = np.where(diag > threshold * 49)[0]
            
            if len(pivots) < 49:
                # Keep only pivot columns
                U_arr = U_arr[:, pivots]
                V_arr = V_arr[:, pivots]
                W_arr = W_arr[:, pivots]
                rank = len(pivots)
                
                if verify_decomp(U_arr, V_arr, W_arr, n, m, p):
                    pass  # Good, reduced rank works
                else:
                    # Revert to rank 49
                    U_arr = np.zeros((16, 49), dtype=int)
                    V_arr = np.zeros((16, 49), dtype=int)
                    W_arr = np.zeros((16, 49), dtype=int)
                    for s in range(7):
                        for t in range(7):
                            r = s * 7 + t
                            for i in range(4):
                                for j in range(4):
                                    coeff = U2[I_map[i]*2 + J_map[j], s] * U2[ii_map[i]*2 + jj_map[j], t]
                                    if coeff != 0:
                                        U_arr[i*4 + j, r] = coeff
                            for k in range(4):
                                for l in range(4):
                                    coeff = V2[I_map[k]*2 + J_map[l], s] * V2[ii_map[k]*2 + jj_map[l], t]
                                    if coeff != 0:
                                        V_arr[k*4 + l, r] = coeff
                            for j in range(4):
                                for i in range(4):
                                    coeff = W2[I_map[i]*2 + J_map[j], s] * W2[ii_map[i]*2 + jj_map[j], t]
                                    if coeff != 0:
                                        W_arr[j*4 + i, r] = coeff
                    rank = 49
    
    return {"U": U_arr.tolist(), "V": V_arr.tolist(), "W": W_arr.tolist()}
