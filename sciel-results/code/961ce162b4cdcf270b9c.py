import numpy as np
import time
import itertools

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # 1. Construct Schoolbook Decomposition (Rank 84)
    def schoolbook(n, m, p):
        rank = n*m*p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W

    U_school, V_school, W_school = schoolbook(n, m, p)
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = n*m*p
    
    # 2. Support Pruning Search
    # Strategy: Start with a random ensemble of rank R_start (e.g., 40).
    # Project onto the "all-ones" target? No, we need to solve the system.
    # The system is: sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1.
    # This is a bilinear/trilinear system.
    # We can try to minimize the error E = || T_hat - T_target ||_F^2.
    # T_target is the all-ones tensor of shape (n, m, p).
    
    # Let's try to find a low rank solution using random restarts and local optimization (gradient descent).
    # Since the dimension is small (3x4x7=84 entries), we can afford some computation.
    
    def compute_residual(U, V, W):
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        # T_hat[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We want T_hat to be all 1s.
        # Return the residual tensor (n, m, p) and its norm.
        
        # Vectorized approach:
        # Let's reshape U, V, W to 3D arrays for easier indexing?
        # U_3d[i, j, r] = U[i*m+j, r]
        # V_3d[j, k, r] = V[j*p+k, r]
        # W_3d[k, i, r] = W[k*n+i, r]
        
        U_3d = U.reshape(n, m, -1)
        V_3d = V.reshape(m, p, -1)
        W_3d = W.reshape(p, n, -1) # Note: W index is k*n+i, so dims are (p, n, R)
        
        # T_hat[i, j, k] = sum_r U_3d[i, j, r] * V_3d[j, k, r] * W_3d[k, i, r]
        
        # We can compute this using einsum or loops.
        # Given small size, loops are fine, but let's try to be efficient.
        # T_hat = np.einsum('ijr,jkr,ki r->ijk', U_3d, V_3d, W_3d) # Syntax error in thought, need spaces.
        # Actually, standard einsum:
        # U_3d shape (n, m, R)
        # V_3d shape (m, p, R)
        # W_3d shape (p, n, R)
        # We want output (n, m, p)
        # Sum over r.
        # T_hat[i,j,k] = sum_r U_3d[i,j,r] V_3d[j,k,r] W_3d[k,i,r]
        
        # Let's use np.einsum
        # 'ijr,jkr,kip->ijk' ? No, W is (p,n,R) so indices are k, i, r.
        # So W_3d[k,i,r].
        # T_hat = np.einsum('ijr,jkr,ki r->ijk', U_3d, V_3d, W_3d) -> 'ijr,jkr,kir->ijk'
        
        T_hat = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        residual = T_hat - 1.0
        return residual, np.linalg.norm(residual)

    def try_optimize(R, time_budget=10.0, seed=0):
        np.random.seed(seed)
        start_time = time.time()
        
        # Initialize random matrices
        U = np.random.randn(n*m, R) * 0.1
        V = np.random.randn(m*p, R) * 0.1
        W = np.random.randn(p*n, R) * 0.1
        
        # Gradient Descent
        # Loss = 0.5 * || T_hat - 1 ||^2
        # dL/dU[i,j,r] = sum_{k} residual[i,j,k] * V[j,k,r] * W[k,i,r]
        # dL/dV[j,k,r] = sum_{i} residual[i,j,k] * U[i,j,r] * W[k,i,r]
        # dL/dW[k,i,r] = sum_{j} residual[i,j,k] * U[i,j,r] * V[j,k,r]
        
        lr = 0.01
        for step in range(1000):
            if time.time() - start_time > time_budget:
                break
            
            residual, norm = compute_residual(U, V, W)
            if norm < 1e-5:
                # Found a solution!
                return U, V, W, norm
            
            # Compute Gradients
            U_3d = U.reshape(n, m, R)
            V_3d = V.reshape(m, p, R)
            W_3d = W.reshape(p, n, R)
            
            # dU[i,j,r] = sum_k res[i,j,k] * V[j,k,r] * W[k,i,r]
            # dV[j,k,r] = sum_i res[i,j,k] * U[i,j,r] * W[k,i,r]
            # dW[k,i,r] = sum_j res[i,j,k] * U[i,j,r] * V[j,k,r]
            
            # Using einsum
            # dU: res(i,j,k), V(j,k,r), W(k,i,r) -> sum over k -> (i,j,r)
            dU = np.einsum('ijk,jkr,kir->ijr', residual, V_3d, W_3d)
            # dV: res(i,j,k), U(i,j,r), W(k,i,r) -> sum over i -> (j,k,r)
            dV = np.einsum('ijk,ijr,kir->jkr', residual, U_3d, W_3d)
            # dW: res(i,j,k), U(i,j,r), V(j,k,r) -> sum over j -> (k,i,r)
            dW = np.einsum('ijk,ijr,jkr->kir', residual, U_3d, V_3d)
            
            # Update
            U -= lr * dU.reshape(n*m, R)
            V -= lr * dV.reshape(m*p, R)
            W -= lr * dW.reshape(p*n, R)
            
            # Adaptive learning rate? Maybe reduce if norm increases.
            # For now, simple decay.
            lr *= 0.999
            
            # Pruning heuristic:
            # If a term r has very small contribution to the residual reduction or is "useless", remove it.
            # This is tricky. Instead, we can try to zero out columns that are close to 0.
            # But that might break the gradient flow.
            # Let's stick to pure GD for now to see if we can reach low rank.
            
        return None

    # Try different ranks
    # We know rank 84 is possible.
    # Let's try to find a rank < 84.
    # Start with R=40 and go up? Or start high and prune?
    # The directive suggests: "initializing a random rank-40 factor matrix ensemble and iteratively removing the term..."
    # My GD approach effectively does this if we allow rank reduction, but standard GD doesn't reduce rank.
    # I can implement a "hard thresholding" step or try to solve for R=1, 2, ...
    
    # Let's try R from 20 to 83.
    # If we find a valid solution for R, we return it.
    
    # To respect the time limit (150s total), we allocate ~120s for search.
    search_start = time.time()
    max_search_time = 120.0
    
    # We will try a few random seeds for each rank to increase chances.
    # But GD on a non-convex problem is hard.
    # Maybe use a smarter initialization?
    # The schoolbook solution is sparse.
    # Maybe initialize U, V, W with the schoolbook solution and then try to merge terms?
    
    # Alternative: Use the known Laderman decomposition for 3x3x3 (Rank 23) and extend it.
    # But hardcoding Laderman is complex.
    
    # Let's stick to the GD search but with a "restart" strategy.
    # If we don't find a solution for R=40 in 10s, try R=45, etc.
    
    # Actually, finding an *exact* integer/half-integer solution via floating point GD is very unlikely to converge to the exact rational values required.
    # The task requires "integer/half-integer entries".
    # Floating point GD will produce floating point numbers.
    # We need to snap to the nearest half-integer and check if it's exact.
    
    def snap_and_check(U, V, W, tol=1e-6):
        # Snap to nearest 0.5
        U_s = np.round(U * 2) / 2
        V_s = np.round(V * 2) / 2
        W_s = np.round(W * 2) / 2
        
        # Check if snapped solution is exact
        _, norm = compute_residual(U_s, V_s, W_s)
        if norm < tol:
            return U_s, V_s, W_s, norm
        return None

    # Modify try_optimize to include snapping
    def try_optimize_snap(R, time_budget=5.0, seed=0):
        np.random.seed(seed)
        start_time = time.time()
        
        U = np.random.randn(n*m, R) * 0.5
        V = np.random.randn(m*p, R) * 0.5
        W = np.random.randn(p*n, R) * 0.5
        
        lr = 0.05
        for step in range(2000):
            if time.time() - start_time > time_budget:
                break
            
            residual, norm = compute_residual(U, V, W)
            
            # Check for convergence to a snapped solution
            if norm < 0.1: # Close enough to try snapping
                snapped = snap_and_check(U, V, W)
                if snapped:
                    return snapped
            
            # Compute Gradients
            U_3d = U.reshape(n, m, R)
            V_3d = V.reshape(m, p, R)
            W_3d = W.reshape(p, n, R)
            
            dU = np.einsum('ijk,jkr,kir->ijr', residual, V_3d, W_3d)
            dV = np.einsum('ijk,ijr,kir->jkr', residual, U_3d, W_3d)
            dW = np.einsum('ijk,ijr,jkr->kir', residual, U_3d, V_3d)
            
            U -= lr * dU.reshape(n*m, R)
            V -= lr * dV.reshape(m*p, R)
            W -= lr * dW.reshape(p*n, R)
            
            lr *= 0.995
            
        # Final check
        snapped = snap_and_check(U, V, W)
        if snapped:
            return snapped
            
        return None

    # Search loop
    # We know 84 works.
    # Let's try to find R < 84.
    # We'll try R in range(20, 84, 5) with a few seeds.
    
    # To maximize chance, let's try lower ranks first?
    # No, lower ranks are harder to find.
    # Let's try R=83, 82, ... down to 20?
    # Or just a few specific ranks.
    
    # Given the difficulty, let's just try to improve on the schoolbook if possible.
    # If we don't find a better one, we return schoolbook.
    
    # Let's try a few ranks with limited time each.
    ranks_to_try = [80, 75, 70, 60, 50, 40]
    
    for R in ranks_to_try:
        if time.time() - search_start > max_search_time - 10: # Leave 10s margin
            break
            
        # Try a few seeds for this rank
        for seed in range(5):
            if time.time() - search_start > max_search_time - 5:
                break
                
            res = try_optimize_snap(R, time_budget=2.0, seed=seed)
            if res:
                U_sol, V_sol, W_sol, norm = res
                # Verify strictly
                _, final_norm = compute_residual(U_sol, V_sol, W_sol)
                if final_norm < 1e-5:
                    best_U, best_V, best_W = U_sol, V_sol, W_sol
                    best_rank = R
                    # We found a better rank! We can stop or try to go lower.
                    # Let's try to go lower.
                    # But we need to update the list of ranks to try.
                    # For simplicity, we break and return this.
                    # Or we can continue searching for even lower rank.
                    # Let's continue searching for lower ranks.
                    # We need to adjust the loop.
                    pass
        
    # The above loop is a bit messy with "continue searching for lower ranks".
    # Let's restructure:
    # We will iterate R from 20 to 83.
    # If we find a solution, we update best.
    # We stop if time is up.
    
    # Re-do search properly
    search_start = time.time()
    max_search_time = 120.0
    
    best_found = False
    best_rank_found = 84
    
    # Try ranks from 20 to 83
    for R in range(20, 84):
        if time.time() - search_start > max_search_time - 5:
            break
            
        # Try a few seeds
        for seed in range(3):
            if time.time() - search_start > max_search_time - 2:
                break
                
            res = try_optimize_snap(R, time_budget=1.0, seed=seed)
            if res:
                U_sol, V_sol, W_sol, norm = res
                # Verify
                _, final_norm = compute_residual(U_sol, V_sol, W_sol)
                if final_norm < 1e-5:
                    best_U, best_V, best_W = U_sol, V_sol, W_sol
                    best_rank = R
                    best_found = True
                    # We found a solution for R.
                    # We can break out of seed loop and try lower R?
                    # No, we are iterating R upwards.
                    # If we find R=30, we should try R=29, 28...
                    # So we should iterate R downwards?
                    # But lower ranks are harder.
                    # Maybe iterate R downwards from 83 to 20?
                    # If we find R=50, we still might find R=40.
                    # So iterate R downwards.
                    
                    # Let's restart the loop logic?
                    # It's better to just iterate R downwards in the main loop.
                    
                    break # Break seed loop, go to next R (which is higher in current loop, so we miss lower ones if we go up)
                    
    # Wait, if I iterate R from 20 to 83, and I find R=30, I will continue to 31, 32...
    # I won't find R=25.
    # So I should iterate R from 83 down to 20.
    # If I find R=50, I record it. Then I try 49, 48...
    # This is better.
    
    # Let's re-run the search with R descending.
    search_start = time.time()
    max_search_time = 120.0
    
    # Reset best to schoolbook
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = 84
    
    for R in range(83, 19, -1):
        if time.time() - search_start > max_search_time - 5:
            break
            
        for seed in range(5):
            if time.time() - search_start > max_search_time - 2:
                break
                
            res = try_optimize_snap(R, time_budget=0.5, seed=seed)
            if res:
                U_sol, V_sol, W_sol, norm = res
                _, final_norm = compute_residual(U_sol, V_sol, W_sol)
                if final_norm < 1e-5:
                    best_U, best_V, best_W = U_sol, V_sol, W_sol
                    best_rank = R
                    # Don't break R loop, keep looking for lower R.
                    # But break seed loop to save time.
                    break
                    
    # Return the best found
    # Convert to list of lists as required by interface?
    # The interface says: factor matrices of shapes ... with integer/half-integer entries.
    # The seed program returned lists of lists.
    # Let's return lists of lists to be safe, or numpy arrays?
    # "factor matrices of shapes (n*m, rank) ... with integer/half-integer entries"
    # Usually numpy arrays are fine. The seed returned lists.
    # Let's check the seed return: `return {"U": U, "V": V, "W": W}` where U, V, W were lists.
    # My schoolbook function returns numpy arrays.
    # Let's convert to lists to match seed behavior exactly?
    # Or just return numpy arrays. The evaluator likely handles both.
    # The prompt says "Keep the function names and signatures ... return interface EXACTLY as in the current program".
    # Current program returns `{"U": U, "V": V, "W": W}` where U is a list of lists.
    # So I should return lists of lists.
    
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
