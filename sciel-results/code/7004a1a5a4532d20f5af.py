    def verify_full(U, V, W):
        """Check if U, V, W form a valid decomposition of the multiplication tensor."""
        n, m, p = DIMENSIONS
        r = U.shape[1]
        
        # Check half-integer constraint
        if not np.allclose(U * 2, np.round(U * 2), atol=1e-9): return False
        if not np.allclose(V * 2, np.round(V * 2), atol=1e-9): return False
        if not np.allclose(W * 2, np.round(W * 2), atol=1e-9): return False
        
        # Snap to exact half-integers
        U_int = np.round(U * 2) / 2.0
        V_int = np.round(V * 2) / 2.0
        W_int = np.round(W * 2) / 2.0
        
        # Check all tensor components
        # T[i, j, j', k, i', k'] should be 1 if i==i', j==j', k==k' else 0
        for i in range(n):
            for j in range(m):
                for j_prime in range(m):
                    for k in range(p):
                        for i_prime in range(n):
                            for k_prime in range(p):
                                u_idx = i * m + j
                                v_idx = j_prime * p + k
                                # W index convention: k * n + i
                                # The tensor component corresponds to C[i', k']
                                # So W index should be k_prime * n + i_prime
                                w_idx = k_prime * n + i_prime
                                
                                s = np.dot(U_int[u_idx, :], V_int[v_idx, :] * W_int[w_idx, :])
                                
                                target = 1.0 if (i == i_prime and j == j_prime and k == k_prime) else 0.0
                                if abs(s - target) > 1e-9:
                                    return False
        return True
