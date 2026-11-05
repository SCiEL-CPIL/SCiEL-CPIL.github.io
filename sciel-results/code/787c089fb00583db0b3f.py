<<<<<<< SEARCH
    def check_exact_fast(U, V, W, tol=1e-10):
        """Vectorized check."""
        r = U.shape[1]
        # Build the full tensor and compare
        # T_expected[(i,j),(j',k),(i',k')] = delta(i,i')*delta(j,j')*delta(k,k')
        # T_actual[(i,j),(j',k),(i',k')] = sum_r U[i*m+j,r]*V[j'*p+k,r]*W[i'*n+k',r]
        
        # For efficiency, check only the nonzero entries of T_expected
        # T_expected is nonzero when i=i', j=j', k=k'
        # So we need: sum_r U[i*m+j,r]*V[j*p+k,r]*W[i*n+k,r] = 1 for all i,j,k
        # And: sum_r U[i*m+j,r]*V[j'*p+k,r]*W[i'*n+k',r] = 0 for all other combos
        
        # Check nonzero entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for rr in range(r):
                        s += U[i*m+j, rr] * V[j*p+k, rr] * W[i*n+k, rr]
                    if abs(s - 1.0) > tol:
                        return False
        
        # Check a sample of zero entries
        for i in range(n):
            for j in range(m):
                for jp in range(m*p):
                    j_prime = jp // p
                    k = jp % p
                    if j_prime == j:
                        continue  # Already checked above
                    for ip in range(n*p):
                        i_prime = ip // p
                        k_prime = ip % p
                        s = 0.0
                        for rr in range(r):
                            s += U[i*m+j, rr] * V[j_prime*p+k, rr] * W[i_prime*n+k_prime, rr]
                        if abs(s) > tol:
                            return False
        return True
=======
    def check_exact_fast(U, V, W, tol=1e-10):
        """Vectorized check."""
        # Reshape to 3D for easy einsum
        # U: (n, m, r)
        # V: (m, p, r)
        # W: (n, p, r)
        U_3d = U.reshape(n, m, -1)
        V_3d = V.reshape(m, p, -1)
        W_3d = W.reshape(n, p, -1)
        
        # Compute T_actual[i, j, j_prime, k, i_prime, k_prime]
        # U[i, j, r] * V[j_prime, k, r] * W[i_prime, k_prime, r]
        # Sum over r
        # Result shape: (n, m, m, p, n, p)
        T_actual = np.einsum('ijr,jkr,ikr->ijkjik', U_3d, V_3d, W_3d)
        
        # Create T_expected
        # T_expected[i, j, j_prime, k, i_prime, k_prime] = 1 if i==i_prime, j==j_prime, k==k_prime
        # We can construct this by taking the diagonal of the 6D tensor?
        # Or simply construct it.
        # T_expected is sparse.
        # Let's just check the max absolute difference.
        
        # Construct expected tensor efficiently
        # It is 1 when indices match.
        # We can use np.eye for each dimension?
        # E_i[i, i_prime] = delta(i, i_prime)
        # E_j[j, j_prime] = delta(j, j_prime)
        # E_k[k, k_prime] = delta(k, k_prime)
        
        E_i = np.eye(n)
        E_j = np.eye(m)
        E_k = np.eye(p)
        
        # T_expected[i, j, j_prime, k, i_prime, k_prime] = E_i[i, i_prime] * E_j[j, j_prime] * E_k[k, k_prime]
        # Note the index order in T_actual: i, j, j_prime, k, i_prime, k_prime
        # We need to broadcast E_i, E_j, E_k to match this shape.
        
        # E_i shape (n, 1, 1, 1, n, 1) -> (n, 1, 1, 1, n, 1)
        # E_j shape (1, m, m, 1, 1, 1) -> (1, m, m, 1, 1, 1)
        # E_k shape (1, 1, 1, p, 1, p) -> (1, 1, 1, p, 1, p)
        
        T_expected = E_i[:, None, None, None, :, None] * \
                     E_j[None, :, :, None, None, None] * \
                     E_k[None, None, None, :, None, :]
                     
        T_expected = T_expected.astype(float)
        
        diff = np.abs(T_actual - T_expected)
        return np.max(diff) < tol
>>>>>>> REPLACE

Now, the ALS solver is still likely to be slow if it builds huge matrices in loops. I will replace `als_solve` with a faster version that uses vectorized operations for the least squares steps, or simply reduce the iterations and ranks to guarantee termination. Given the directive to "keep this design", I will keep the ALS structure but make it much faster by pre-computing parts or using `np.einsum` for the matrix construction in the least squares.

However, constructing the full $A$ matrix for each row of $U$ (size $m \cdot p \cdot n \cdot p \times r$) is $14112 \times r$. Doing this for $n \cdot m = 12$ rows is $12 \times 14112 \times r$ operations. This is manageable if vectorized.

Let's rewrite `als_solve` to be faster.

