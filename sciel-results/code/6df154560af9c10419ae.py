<<<<<<< SEARCH
    # Fast Verification
    def verify_decomp(U_mat, V_mat, W_mat):
        rank = U_mat.shape[1]
        # We need to check if sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        # for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U_mat[i * m + j, r] * V_mat[j * p + k, r] * W_mat[k * n + i, r]
                    if s != 1:
                        return False
        return True
=======
    # Fast Verification using vectorized tensor contraction
    def verify_decomp(U_mat, V_mat, W_mat):
        rank = U_mat.shape[1]
        if rank == 0:
            return False
        # Compute tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # U shape: (nm, rank), V shape: (mp, rank), W shape: (pn, rank)
        # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        U_r = U_mat.reshape(n, m, rank)
        V_r = V_mat.reshape(m, p, rank)
        W_r = W_mat.reshape(p, n, rank)
        # For each (i,j,k): sum_r U_r[i,j,r] * V_r[j,k,r] * W_r[k,i,r]
        # Use einsum: T[i,j,k] = U_r[i,j,r] * V_r[j,k,r] * W_r[k,i,r]
        # First compute UV[i,j,k,r] = U_r[i,j,r] * V_r[j,k,r]
        UV = U_r[:, :, None, :] * V_r[None, :, :, :]  # shape (n, m, p, rank)
        # Then T[i,j,k] = sum_r UV[i,j,k,r] * W_r[k,i,r]
        # W_r has shape (p, n, rank), we need W_r[k,i,r]
        # Transpose W_r to (n, p, rank) -> W_r.T[0] gives (n, p, rank)
        W_transposed = W_r.transpose(1, 0, 2)  # shape (n, p, rank)
        T = np.einsum('ijkk->ijk', UV, W_transposed[:, :, :, None])  # This isn't right
        
        # Let me be more careful. W_r has shape (p, n, rank)
        # We need W_r[k, i, r] for each i, j, k
        # So for fixed i, we need W_r[:, i, :] which has shape (p, rank)
        # T[i, j, k] = sum_r UV[i, j, k, r] * W_r[k, i, r]
        # UV shape: (n, m, p, rank)
        # W_r shape: (p, n, rank)
        # We want to contract over r: UV[i,j,k,r] * W_r[k,i,r]
        # Let's use: T[i,j,k] = sum_r UV[i,j,k,r] * W_r[k,i,r]
        # This is like: for each i, take UV[i,:,:,:] (shape m,p,rank) and multiply element-wise with W_r[:,i,:] broadcast
        # W_r[:,i,:] has shape (p, rank), broadcast to (m, p, rank)
        T = np.zeros((n, m, p), dtype=int)
        for i in range(n):
            # UV[i,:,:,:] shape (m, p, rank)
            # W_r[:,i,:] shape (p, rank)
            W_part = W_r[:, i, :]  # (p, rank)
            # Broadcast: W_part[None, :, :] shape (1, p, rank)
            T[i, :, :] = np.einsum('mpr,pr->mp', UV[i, :, :, :], W_part)
        
        return np.all(T == 1)
>>>>>>> REPLACE

