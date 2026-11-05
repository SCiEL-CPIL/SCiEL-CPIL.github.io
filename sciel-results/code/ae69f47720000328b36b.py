<<<<<<< SEARCH
    # Build the reduced matrices
    keep_indices = [r for r in range(rank) if not remove[r]]
    
    if len(keep_indices) < rank:
        U = U[:, keep_indices]
        V = V[:, keep_indices]
        W = W[:, keep_indices]
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
=======
    # Build the reduced matrices
    keep_indices = [r for r in range(rank) if not remove[r]]
    
    if len(keep_indices) < rank:
        U = U[:, keep_indices]
        V = V[:, keep_indices]
        W = W[:, keep_indices]
    
    # Verify correctness with a random test
    if time.time() < _deadline:
        np.random.seed(42)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test
        
        # Reconstruct C from decomposition
        C_recon = np.zeros((n, p))
        for r in range(U.shape[1]):
            # U[:,r] is (n*m,), V[:,r] is (m*p,), W[:,r] is (p*n,)
            A_r = U[:, r].reshape(n, m)
            B_r = V[:, r].reshape(m, p)
            C_r = np.zeros((n, p))
            W_r = W[:, r].reshape(p, n)  # W[k*n+i] -> C[i,k]
            for i in range(n):
                for k in range(p):
                    C_r[i, k] = A_r @ B_r * W_r[k, i]
            C_recon += C_r
        
        if not np.allclose(C_recon, C_expected, atol=1e-8):
            # Fallback to full decomposition
            U = np.array(U_parts).T
            V = np.array(V_parts).T
            W = np.array(W_parts).T
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
