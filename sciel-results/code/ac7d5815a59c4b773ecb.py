<<<<<<< SEARCH
    def get_errors(U, V, W):
        """
        Returns a vector of errors for each term in the tensor identity.
        Identity:
        sum_r U[i'*m+j', r] V[j''*p+k', r] W[k*n+i, r] = delta_{i', i} delta_{k', k} delta_{j', j''}
        
        We compute the error for every valid (i, k, i', k', j', j'') tuple.
        There are n*p*n*p*m*m = 1024 such tuples.
        """
        U_arr = np.asarray(U)
        V_arr = np.asarray(V)
        W_arr = np.asarray(W)
        R = U_arr.shape[1]
        
        errors = np.zeros(1024)
        idx = 0
        for i in range(n):
            for k in range(p):
                w_idx = k * n + i
                for i_prime in range(n):
                    for k_prime in range(p):
                        for j_prime in range(m):
                            u_idx = i_prime * m + j_prime
                            for j_double_prime in range(m):
                                v_idx = j_double_prime * p + k_prime
                                target = 1.0 if (i == i_prime and k == k_prime and j_prime == j_double_prime) else 0.0
                                val = np.dot(U_arr[u_idx] * V_arr[v_idx], W_arr[w_idx])
                                errors[idx] = val - target
                                idx += 1
        return errors
=======
    def get_errors(U, V, W):
        """
        Returns a vector of errors for each term in the tensor identity.
        Uses vectorized computation for speed.
        """
        U_arr = np.asarray(U, dtype=np.float64)
        V_arr = np.asarray(V, dtype=np.float64)
        W_arr = np.asarray(W, dtype=np.float64)
        R = U_arr.shape[1]
        
        # Precompute all products U[i',j',r] * V[j'',k',r]
        # Shape: (n, m, m, p, R)
        UV = np.einsum('ijr,jkr->ijk r', U_arr, V_arr)
        
        # Now compute sum_r UV[i',j',j'',k',r] * W[k,i,r] for all (i,k,i',k',j',j'')
        # W has shape (p, n, R) -> W[k,i,r]
        # We need: for each (i,k,i',k',j',j''): sum_r UV[i',j',j'',k',r] * W[k,i,r]
        
        # Reshape W to (p, n, R)
        W_reshaped = W_arr.reshape(p, n, R)
        
        # UV has shape (n, m, m, p, R) = (i', j', j'', k', r)
        # We want to contract over r with W[k, i, r]
        
        # Let's build the full tensor step by step
        # For each (i, k, i', k', j', j''):
        #   val = sum_r U[i',j',r] * V[j'',k',r] * W[k,i,r]
        
        # Use einsum to compute all values at once
        # U: (i', j', r), V: (j'', k', r), W: (k, i, r)
        # Result: (i, k, i', k', j', j'')
        
        # This might be too large: 2*4*2*4*4*4 = 1024, which is fine
        # But the intermediate might be large. Let's use a different approach.
        
        # Alternative: loop over the smaller dimensions
        errors = np.zeros((n, p, n, p, m, m))
        
        for i in range(n):
            for k in range(p):
                # W[k, i, :] has shape (R,)
                w_vec = W_reshaped[k, i, :]
                # For each (i', k', j', j''):
                #   sum_r U[i',j',r] * V[j'',k',r] * w_vec[r]
                # = sum_r (U[i',j',r] * w_vec[r]) * V[j'',k',r]
                
                # Compute U_weighted[i', j', r] = U[i', j', r] * w_vec[r]
                U_weighted = U_arr * w_vec[np.newaxis, np.newaxis, :]  # (n, m, R)
                
                # Now for each (k', j', j''):
                #   sum_r U_weighted[i', j', r] * V[j'', k', r]
                # This is a batched dot product
                
                # U_weighted: (n, m, R), V: (m, p, R)
                # We want: for each i', k', j', j'': sum_r U_weighted[i', j', r] * V[j'', k', r]
                
                # Reshape: U_weighted -> (n, m, R), V -> (p, m, R) [transpose]
                V_t = V_arr.transpose(1, 0, 2)  # (p, m, R)
                
                # Use einsum: U_weighted[i', j', r] * V_t[k', j'', r] -> (i', k', j', j'')
                vals = np.einsum('ijr,kjr->ikjj', U_weighted, V_t)
                
                errors[i, k, :, :, :, :] = vals
        
        # Reshape to flat and subtract targets
        errors_flat = errors.reshape(-1)
        
        # Build target vector
        targets = np.zeros(1024)
        idx = 0
        for i in range(n):
            for k in range(p):
                for i_prime in range(n):
                    for k_prime in range(p):
                        for j_prime in range(m):
                            for j_double_prime in range(m):
                                target = 1.0 if (i == i_prime and k == k_prime and j_prime == j_double_prime) else 0.0
                                targets[idx] = target
                                idx += 1
        
        return errors_flat - targets
>>>>>>> REPLACE

