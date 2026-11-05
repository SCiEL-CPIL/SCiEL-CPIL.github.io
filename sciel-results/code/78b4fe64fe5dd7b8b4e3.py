<<<<<<< SEARCH
def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify if U, V, W form a valid decomposition of the matrix multiplication tensor.
    """
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    rank = U_np.shape[1]
    
    # Reshape
    U_r = U_np.reshape((n, m, rank))
    V_r = V_np.reshape((m, p, rank))
    W_r = W_np.reshape((p, n, rank))
    
    # Check all indices
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for kp in range(p):
                    for ip in range(n):
                        if kp == k and ip == i:
                            expected = 1.0
                        else:
                            expected = 0.0
                        
                        s = 0.0
                        for c in range(rank):
                            s += U_r[i, j, c] * V_r[j, k, c] * W_r[kp, ip, c]
                        
                        if not np.isclose(s, expected, atol=1e-9):
                            return False
    return True
=======
def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify if U, V, W form a valid decomposition of the matrix multiplication tensor.
    Uses vectorized numpy for speed.
    """
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    rank = U_np.shape[1]
    
    # Reshape: U[i,j,c], V[j,k,c], W[k,i,c]
    U_r = U_np.reshape((n, m, rank))
    V_r = V_np.reshape((m, p, rank))
    W_r = W_np.reshape((p, n, rank))
    
    # For each (i,j,k), compute sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] should be 1
    # and sum_c U[i,j,c]*V[j,k,c]*W[kp,ip,c] should be 0 for (kp,ip) != (k,i)
    
    # Efficient check: for each (i,j), compute the outer product of U[i,j,:] with 
    # the appropriate V and W slices
    
    # Check all (i,j,k) -> should give 1 at W[k,i]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Compute sum over c of U[i,j,c] * V[j,k,c] * W[:, :, c]
                # This gives a p x n matrix
                # Use einsum: U[i,j,:] * V[j,k,:] * W[:,:,c] -> W[k,i] should be 1
                uv = U_r[i, j, :] * V_r[j, k, :]  # shape (rank,)
                # Now sum over c: uv[c] * W[kp, ip, c] for all kp, ip
                result = np.einsum('c,cpi->kp i', uv, W_r)  # Wait, W_r is (p, n, rank)
                # W_r[kp, ip, c], so we need np.einsum('c,kpi->kp i', uv, W_r)
                result = np.einsum('c,kpi->kp i', uv, W_r)
                
                # Check that result[k, i] == 1 and all other entries == 0
                if not np.isclose(result[k, i], 1.0, atol=1e-9):
                    return False
                # Check other entries
                mask = np.ones((p, n), dtype=bool)
                mask[k, i] = False
                if not np.allclose(result[mask], 0.0, atol=1e-9):
                    return False
    return True
>>>>>>> REPLACE

