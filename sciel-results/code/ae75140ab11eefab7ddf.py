<<<<<<< SEARCH
    # Verify a decomposition: check sum_r U[i*m+j][r]*V[j*p+k][r]*W[k*n+i][r] == 1 for all i,j,k
    def verify(U, V, W, rank):
        # U: (n*m, rank)
        # V: (m*p, rank)
        # W: (p*n, rank)
        # We need T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        # We can compute this efficiently.
        # For each r, we have u_r (n*m), v_r (m*p), w_r (p*n).
        # The contribution to T is outer product?
        # T_{ijk} += u_r[i,j] v_r[j,k] w_r[k,i]
        
        # Let's reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        U_reshaped = U.reshape(n, m, rank)
        V_reshaped = V.reshape(m, p, rank)
        W_reshaped = W.reshape(p, n, rank)
        
        # Compute T
        # T[i,j,k] = sum_r U_reshaped[i,j,r] * V_reshaped[j,k,r] * W_reshaped[k,i,r]
        # This is a bit tricky with simple einsum because indices are mixed.
        # Let's use a loop over r for verification, it's fast enough for small n,m,p.
        T = np.zeros((n, m, p))
        for r in range(rank):
            u_r = U_reshaped[:, :, r] # (n, m)
            v_r = V_reshaped[:, :, r] # (m, p)
            w_r = W_reshaped[:, :, r] # (p, n)
            
            # We need to add u_r[i,j] * v_r[j,k] * w_r[k,i] to T[i,j,k]
            # This is not a standard matmul.
            # For each i, j: T[i,j,:] += u_r[i,j] * (v_r[j,:] * w_r[:,i])
            # v_r[j,:] is (p,). w_r[:,i] is (p,).
            # So for fixed i, j, the vector over k is v_r[j,:] * w_r[:,i].
            
            # We can do this with broadcasting.
            # u_r: (n, m, 1)
            # v_r: (1, m, p) -> need to align j
            # w_r: (p, n) -> need to align k and i.
            
            # Let's just loop i, j for clarity and safety, p is small (7).
            for i in range(n):
                for j in range(m):
                    if u_r[i, j] != 0:
                        # v_r[j, :] is (p,)
                        # w_r[:, i] is (p,)
                        T[i, j, :] += u_r[i, j] * (v_r[j, :] * w_r[:, i])
                        
        if not np.allclose(T, 1.0):
            return False
        return True
=======
    # Verify a decomposition: check sum_r U[i*m+j][r]*V[j*p+k][r]*W[k*n+i][r] == 1 for all i,j,k
    def verify(U, V, W, rank):
        # U: (n*m, rank)
        # V: (m*p, rank)
        # W: (p*n, rank)
        # We need T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        
        # Reshape to 3D tensors for easy einsum
        U_reshaped = U.reshape(n, m, rank)
        V_reshaped = V.reshape(m, p, rank)
        W_reshaped = W.reshape(p, n, rank)
        
        # Compute T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # Indices: i, j, k are output. r is summed.
        # U: i, j, r
        # V: j, k, r
        # W: k, i, r
        T = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        
        # Check if T is all 1s
        # Use exact integer check if possible, but float is fine with tolerance
        # Since entries are half-integers, products are quarters, sums are quarters.
        # 1.0 is exactly representable.
        return np.allclose(T, 1.0, atol=1e-9)
>>>>>>> REPLACE

