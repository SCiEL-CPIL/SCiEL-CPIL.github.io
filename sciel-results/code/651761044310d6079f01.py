<<<<<<< SEARCH
    # Search for lower-rank decomposition of [2,3,3] sub-tensor
    # If we find rank r for [2,3,3], we can use 3 blocks for 9 columns
    # Total rank would be 3*r + 18 (for the 2x1x9 part)
    # Wait, that's not quite right. Let me think about this more carefully.
    
    # Actually, the 2x3x9 tensor can be decomposed as:
    # C[i,j] = sum_{k=0}^{2} A[i,k] B[k,j]
    # = sum_{k=0}^{1} A[i,k] B[k,j] + A[i,2] B[2,j]
    # The first part is 2x2x9, the second is 2x1x9.
    
    # For 2x2x9, we can split into three 2x2x3 blocks.
    # If we find rank r for 2x2x3, total for 2x2x9 is 3r.
    # Plus 18 for 2x1x9.
    # Total: 3r + 18.
    # Current: 32 + 18 = 50.
    # If r=10, total=48. If r=9, total=45.
    
    # Let's search for rank of [2,2,3] tensor.
    # T[i,k,j] = 1 if this is a valid multiplication term, else 0.
    # For 2x2x3: i in {0,1}, k in {0,1,2}, j in {0,1,2}
    # The tensor is sum over all (i,k,j) of e_i ⊗ e_k ⊗ e_j
    
    def search_2x2x3_rank(max_rank, time_budget=30):
        """Search for low-rank decomposition of [2,2,3] tensor."""
        if time.time() > deadline - 20:
            return None
        
        n2, m2, p2 = 2, 2, 3
        # Create the tensor
        tensor = np.zeros((n2, m2, p2))
        for i in range(n2):
            for k in range(m2):
                for j in range(p2):
                    tensor[i, k, j] = 1.0
        
        for r in range(8, max_rank + 1):
            if time.time() > deadline - 20:
                break
            # Try to find rank-r decomposition
            # U: (n2*m2, r), V: (m2*p2, r), W: (p2*n2, r)
            np.random.seed(r * 1000)
            U = np.random.randn(n2 * m2, r)
            V = np.random.randn(m2 * p2, r)
            W = np.random.randn(p2 * n2, r)
            
            # Alternating minimization
            for iteration in range(200):
                if time.time() > deadline - 20:
                    break
                
                # Update U: for each i,k, U[i*m2+k, :] = argmin ||tensor[i,k,:] - V[:,r] outer W[:,r] contracted||
                # This is complex. Let's use a simpler approach.
                # Actually, let's use a direct least squares approach.
                
                # The tensor equation: T[i,k,j] = sum_r U[i*m2+k, r] * V[k*p2+j, r] * W[j*n2+i, r]
                # This is a 3-way tensor decomposition, which is non-convex.
                # Let's use a simple gradient-free approach: try random initializations and check.
                
                # Actually, let me just check if the current factors work
                # and do a few alternating updates.
                
                # For each r, compute the residual
                # This is getting too complex for the time budget.
                pass
            
            # Simple check: can we verify with a few random initializations?
            for trial in range(50):
                if time.time() > deadline - 20:
                    break
                np.random.seed(trial * 7919 + r)
                U = np.random.randn(n2 * m2, r)
                V = np.random.randn(m2 * p2, r)
                W = np.random.randn(p2 * n2, r)
                
                # Check if these factors can represent the tensor
                # T[i,k,j] should equal sum_r U[i*m2+k, r] * V[k*p2+j, r] * W[j*n2+i, r]
                # Let's compute the actual tensor from these factors and compare
                T_computed = np.zeros((n2, m2, p2))
                for rr in range(r):
                    for i in range(n2):
                        for k in range(m2):
                            for j in range(p2):
                                T_computed[i, k, j] += U[i*m2+k, rr] * V[k*p2+j, rr] * W[j*n2+i, rr]
                
                if np.allclose(T_computed, tensor, atol=1e-10):
                    # Found a valid decomposition!
                    # Convert to list format
                    U_list = [[U[i, rr] for rr in range(r)] for i in range(n2*m2)]
                    V_list = [[V[i, rr] for rr in range(r)] for i in range(m2*p2)]
                    W_list = [[W[i, rr] for rr in range(r)] for i in range(p2*n2)]
                    return (r, U_list, V_list, W_list)
        
        return None
=======
    # The 2x3x9 tensor decomposes as:
    # C[i,j] = sum_{k=0}^{2} A[i,k] B[k,j]
    # = sum_{k=0}^{1} A[i,k] B[k,j] + A[i,2] B[2,j]
    # First part: 2x2x9, split into three 2x2x3 blocks
    # Second part: 2x1x9, rank 18
    # Total: 3*r + 18 where r is rank of 2x2x3
    
    def search_2x2x3_rank(max_rank, time_budget=60):
        """Search for low-rank decomposition of [2,2,3] tensor using ALS."""
        if time.time() > deadline - 20:
            return None
        
        n2, m2, p2 = 2, 2, 3
        # Create the tensor: T[i,k,j] = 1 for all valid indices
        tensor = np.ones((n2, m2, p2))
        
        best_result = None
        
        for r in range(7, max_rank + 1):
            if time.time() > deadline - 20:
                break
            
            # Try multiple random initializations
            for trial in range(30):
                if time.time() > deadline - 20:
                    break
                
                np.random.seed(trial * 7919 + r * 100)
                U = np.random.randn(n2 * m2, r) * 0.5
                V = np.random.randn(m2 * p2, r) * 0.5
                W = np.random.randn(p2 * n2, r) * 0.5
                
                # Alternating least squares
                for iteration in range(300):
                    if time.time() > deadline - 20:
                        break
                    
                    # Update U: for each (i,k), solve for U[i*m2+k, :]
                    # T[i,k,j] = sum_r U[i*m2+k, r] * V[k*p2+j, r] * W[j*n2+i, r]
                    # For fixed V,W: T[i,k,:] = U[i*m2+k,:] @ (V[k,:] ⊙ W[:,i])
                    # where ⊙ denotes element-wise product after proper reshaping
                    
                    for i in range(n2):
                        for k in range(m2):
                            # Compute the effective matrix: M[j, r] = V[k*p2+j, r] * W[j*n2+i, r]
                            M = np.zeros((p2, r))
                            for j in range(p2):
                                M[j, :] = V[k*p2+j, :] * W[j*n2+i, :]
                            # Solve: T[i,k,:] = U[i*m2+k,:] @ M
                            # U[i*m2+k,:] = T[i,k,:] @ pinv(M)
                            try:
                                U[i*m2+k, :] = np.linalg.lstsq(M, tensor[i,k,:], rcond=None)[0]
                            except:
                                pass
                    
                    # Update V: for each (k,j), solve for V[k*p2+j, :]
                    for k in range(m2):
                        for j in range(p2):
                            # T[i,k,j] = sum_r U[i*m2+k, r] * V[k*p2+j, r] * W[j*n2+i, r]
                            # For fixed U,W: T[:,k,j] = V[k*p2+j,:] @ (U[:,k] ⊙ W[j,:])
                            M = np.zeros((n2, r))
                            for i in range(n2):
                                M[i, :] = U[i*m2+k, :] * W[j*n2+i, :]
                            try:
                                V[k*p2+j, :] = np.linalg.lstsq(M, tensor[:,k,j], rcond=None)[0]
                            except:
                                pass
                    
                    # Update W: for each (j,i), solve for W[j*n2+i, :]
                    for j in range(p2):
                        for i in range(n2):
                            # T[i,k,j] = sum_r U[i*m2+k, r] * V[k*p2+j, r] * W[j*n2+i, r]
                            # For fixed U,V: T[i,:,j] = W[j*n2+i,:] @ (V[:,j] ⊙ U[i,:])
                            M = np.zeros((m2, r))
                            for k in range(m2):
                                M[k, :] = V[k*p2+j, :] * U[i*m2+k, :]
                            try:
                                W[j*n2+i, :] = np.linalg.lstsq(M, tensor[i,:,j], rcond=None)[0]
                            except:
                                pass
                    
                    # Check convergence
                    T_computed = np.zeros((n2, m2, p2))
                    for rr in range(r):
                        for i in range(n2):
                            for k in range(m2):
                                for j in range(p2):
                                    T_computed[i, k, j] += U[i*m2+k, rr] * V[k*p2+j, rr] * W[j*n2+i, rr]
                    
                    error = np.linalg.norm(T_computed - tensor)
                    if error < 1e-10:
                        # Found a valid decomposition!
                        U_list = [[U[i, rr] for rr in range(r)] for i in range(n2*m2)]
                        V_list = [[V[i, rr] for rr in range(r)] for i in range(m2*p2)]
                        W_list = [[W[i, rr] for rr in range(r)] for i in range(p2*n2)]
                        if best_result is None or r < best_result[0]:
                            best_result = (r, U_list, V_list, W_list)
                        return best_result
        
        return best_result
>>>>>>> REPLACE

