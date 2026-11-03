<<<<<<< SEARCH
def find_low_rank_decomposition(target_rank, n=2, m=4, p=4, max_restarts=100):
    """Use scipy.optimize.least_squares to find a decomposition of target_rank."""
    n, m, p = 2, 4, 4
    size_U = n * m * target_rank
    size_V = m * p * target_rank
    size_W = p * n * target_rank
    total_size = size_U + size_V + size_W
    
    best_result = None
    best_error = float('inf')
    
    # Use a set of possible values for sparse initialization
    sparse_vals = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
    
    for restart in range(max_restarts):
        if time_up():
            break
            
        # Random initialization with sparse structure
        x0 = np.zeros(total_size)
        
        # Randomly select a subset of entries to be non-zero
        # We want a structured sparsity, so we pick random indices
        num_nonzero = int(total_size * 0.2) # 20% non-zero
        indices = np.random.choice(total_size, num_nonzero, replace=False)
        
        for idx in indices:
            x0[idx] = np.random.choice(sparse_vals)
        
        # Also try a dense random initialization occasionally
        if restart % 5 == 0:
            x0 = np.random.randn(total_size) * 0.5
        
        try:
            # Use 'trf' method which is more robust
            result = opt.least_squares(residuals, x0, args=(n, m, p, target_rank), method='trf', max_nfev=5000)
            err = np.linalg.norm(result.fun)
            
            if err < best_error:
                best_error = err
                best_result = result.x
                
            if err < 1e-8:
                # Found a close solution, try to round to half-integers
                U_rec = best_result[:size_U].reshape(n*m, target_rank)
                V_rec = best_result[size_U : size_U + size_V].reshape(m*p, target_rank)
                W_rec = best_result[size_U + size_V : ].reshape(p*n, target_rank)
                
                # Round to nearest half-integer
                U_rounded = np.round(U_rec * 2) / 2
                V_rounded = np.round(V_rec * 2) / 2
                W_rounded = np.round(W_rec * 2) / 2
                
                # Verify rounded solution
                valid, _ = verify_decomposition(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p)
                if valid:
                    return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
                    
                # If rounding failed, check if the unrounded solution is exact enough
                # and return it (though the task requires integer/half-integer)
                # The task says "integer/half-integer entries".
                # So we must return a valid half-integer solution.
                
        except Exception as e:
            pass
            
    return None
=======
def find_low_rank_decomposition(target_rank, n=2, m=4, p=4, max_restarts=200):
    """Use scipy.optimize.least_squares to find a decomposition of target_rank."""
    n, m, p = 2, 4, 4
    size_U = n * m * target_rank
    size_V = m * p * target_rank
    size_W = p * n * target_rank
    total_size = size_U + size_V + size_W
    
    best_result = None
    best_error = float('inf')
    
    # Use a set of possible values for sparse initialization
    sparse_vals = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
    
    for restart in range(max_restarts):
        if time_up():
            break
            
        # Random initialization with sparse structure
        x0 = np.zeros(total_size)
        
        # Randomly select a subset of entries to be non-zero
        # We want a structured sparsity, so we pick random indices
        num_nonzero = int(total_size * 0.2) # 20% non-zero
        indices = np.random.choice(total_size, num_nonzero, replace=False)
        
        for idx in indices:
            x0[idx] = np.random.choice(sparse_vals)
        
        # Also try a dense random initialization occasionally
        if restart % 5 == 0:
            x0 = np.random.randn(total_size) * 0.5
        
        try:
            # Use 'trf' method which is more robust
            result = opt.least_squares(residuals, x0, args=(n, m, p, target_rank), method='trf', max_nfev=8000)
            err = np.linalg.norm(result.fun)
            
            if err < best_error:
                best_error = err
                best_result = result.x.copy()
                
            if err < 1e-8:
                # Found a close solution, try to round to half-integers
                U_rec = best_result[:size_U].reshape(n*m, target_rank)
                V_rec = best_result[size_U : size_U + size_V].reshape(m*p, target_rank)
                W_rec = best_result[size_U + size_V : ].reshape(p*n, target_rank)
                
                # Round to nearest half-integer
                U_rounded = np.round(U_rec * 2) / 2
                V_rounded = np.round(V_rec * 2) / 2
                W_rounded = np.round(W_rec * 2) / 2
                
                # Verify rounded solution
                valid, _ = verify_decomposition(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p)
                if valid:
                    return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
                    
        except Exception as e:
            pass
            
    return None
>>>>>>> REPLACE

