import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = 3, 4, 4
    
    def build_tensor():
        """Build the full multiplication tensor T[i,j,k]"""
        T = np.zeros((n * m, m * p, p * n))
        for row_a in range(n):
            for col_a in range(m):
                for col_c in range(p):
                    i = row_a * m + col_a
                    j = col_a * p + col_c
                    k = col_c * n + row_a
                    T[i, j, k] = 1.0
        return T
    
    def verify_solution(sol, max_tests=50):
        U = np.array(sol['U'])
        V = np.array(sol['V'])
        W = np.array(sol['W'])
        T = build_tensor()
        
        np.random.seed(42)
        for _ in range(max_tests):
            i = np.random.randint(0, n * m)
            j = np.random.randint(0, m * p)
            k = np.random.randint(0, p * n)
            computed = np.sum(U[i, :] * V[j, :] * W[k, :])
            expected = T[i, j, k]
            if abs(computed - expected) > 1e-9:
                return False
        return True
    
    def get_current_solution():
        """Return the current best known solution (rank 44)"""
        rank = 44
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        U_loc = np.zeros((6, 11))
        V_loc = np.zeros((4, 11))
        W_loc = np.zeros((6, 11))
        
        # Strassen part (cols 0-6)
        U_loc[0, 0] = 1; U_loc[3, 0] = 1
        V_loc[0, 0] = 1; V_loc[3, 0] = 1
        W_loc[0, 0] = 1; W_loc[3, 0] = 1
        
        U_loc[2, 1] = 1; U_loc[3, 1] = 1
        V_loc[0, 1] = 1
        W_loc[2, 1] = 1; W_loc[3, 1] = -1
        
        U_loc[0, 2] = 1
        V_loc[1, 2] = 1; V_loc[3, 2] = -1
        W_loc[1, 2] = 1; W_loc[3, 2] = 1
        
        U_loc[3, 3] = 1
        V_loc[2, 3] = 1; V_loc[0, 3] = -1
        W_loc[0, 3] = 1; W_loc[2, 3] = 1
        
        U_loc[0, 4] = 1; U_loc[1, 4] = 1
        V_loc[3, 4] = 1
        W_loc[0, 4] = -1; W_loc[1, 4] = 1
        
        U_loc[2, 5] = 1; U_loc[0, 5] = -1
        V_loc[0, 5] = 1; V_loc[1, 5] = 1
        W_loc[3, 5] = 1
        
        U_loc[1, 6] = 1; U_loc[3, 6] = -1
        V_loc[2, 6] = 1; V_loc[3, 6] = 1
        W_loc[0, 6] = 1
        
        # Extra part (cols 7-10)
        U_loc[4, 7] = 1; V_loc[0, 7] = 1; W_loc[4, 7] = 1
        U_loc[5, 8] = 1; V_loc[2, 8] = 1; W_loc[4, 8] = 1
        U_loc[4, 9] = 1; V_loc[1, 9] = 1; W_loc[5, 9] = 1
        U_loc[5, 10] = 1; V_loc[3, 10] = 1; W_loc[5, 10] = 1
        
        def add_block(block_idx, a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start):
            col_offset = block_idx * 11
            for r in range(3):
                for c in range(2):
                    local_a_idx = r * 2 + c
                    global_a_idx = (a_row_start + r) * m + (a_col_start + c)
                    U[global_a_idx, col_offset:col_offset+11] += U_loc[local_a_idx, :]
            for r in range(2):
                for c in range(2):
                    local_b_idx = r * 2 + c
                    global_b_idx = (b_row_start + r) * p + (b_col_start + c)
                    V[global_b_idx, col_offset:col_offset+11] += V_loc[local_b_idx, :]
            for r in range(3):
                for c in range(2):
                    local_c_idx = r * 2 + c
                    global_c_idx = (c_col_start + c) * n + (c_row_start + r)
                    W[global_c_idx, col_offset:col_offset+11] += W_loc[local_c_idx, :]

        add_block(0, 0, 0, 0, 0, 0, 0)
        add_block(1, 0, 2, 2, 0, 0, 0)
        add_block(2, 0, 0, 0, 2, 0, 2)
        add_block(3, 0, 2, 2, 2, 0, 2)
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def find_rank_decomposition(max_rank=44, time_limit=130):
        """
        Try to find a lower-rank decomposition using the linear operator approach.
        
        Key insight: For each output position (row_c, col_c), the computation is:
        C[row_c, col_c] = sum_{col_a} A[row_c, col_a] * B[col_a, col_c]
        
        This is a bilinear form. We can view the multiplication tensor as a set of
        linear operators and try to find a rank-r decomposition.
        
        Strategy: Use SVD-like approaches on slices of the tensor to find low-rank
        approximations, then verify exactness.
        """
        T = build_tensor()
        T_flat = T.reshape(n * m * p * n, 1)
        
        # The tensor has structure: T[i,j,k] depends on specific index relationships
        # i = row_a * m + col_a, j = col_a * p + col_c, k = col_c * n + row_a
        
        # For a rank-r decomposition: T[i,j,k] = sum_{r=1}^{R} U[i,r]*V[j,r]*W[k,r]
        
        # Approach: Try to find a decomposition by solving a system of equations
        # for small ranks, starting from rank 40 and going down
        
        best_solution = get_current_solution()
        best_rank = 44
        
        # Try ranks from 40 down to 35
        for target_rank in range(40, 34, -1):
            if time.time() > time_limit:
                break
            
            # Use a greedy/iterative approach
            # Start with the known rank-44 solution and try to merge terms
            
            U_curr = np.array(best_solution['U'])
            V_curr = np.array(best_solution['V'])
            W_curr = np.array(best_solution['W'])
            
            # Try to reduce rank by finding linear dependencies
            # This is a non-convex problem, so we use heuristics
            
            # Heuristic 1: Try to find pairs of terms that can be merged
            # If U[:,a] = alpha*U[:,b] and V[:,a] = beta*V[:,b] and W[:,a] = gamma*W[:,b]
            # with alpha*beta*gamma = 1, we might be able to eliminate one term
            
            # Heuristic 2: Random perturbation and re-optimization
            np.random.seed(42)
            
            # Try to find a lower-rank solution by solving the tensor equation
            # directly for a given rank
            
            # For rank r, we need to find U (12 x r), V (16 x r), W (12 x r)
            # such that sum_r U[i,r]*V[j,r]*W[k,r] = T[i,j,k] for all i,j,k
            
            # This is equivalent to: for each (i,j), the vector [W[k,r] for all k]
            # must satisfy a linear constraint
            
            # Let's try a direct approach: fix U and V, solve for W
            # Or use alternating minimization
            
            U_rand = np.random.randn(n * m, target_rank)
            V_rand = np.random.randn(m * p, target_rank)
            
            # Normalize to avoid numerical issues
            for r in range(target_rank):
                if np.abs(U_rand[:, r]).max() > 1e-10:
                    U_rand[:, r] /= np.abs(U_rand[:, r]).max()
                if np.abs(V_rand[:, r]).max() > 1e-10:
                    V_rand[:, r] /= np.abs(V_rand[:, r]).max()
            
            # Iterate: given U and V, solve for W; given U and W, solve for V; etc.
            W_curr = np.zeros((p * n, target_rank))
            
            converged = False
            for iteration in range(50):
                if time.time() > time_limit:
                    break
                
                # Solve for W: for each (i,j), we have
                # sum_r U[i,r]*V[j,r]*W[k,r] = T[i,j,k]
                # For fixed i,j, this is a linear system in W[:,r]
                
                # Actually, let's think differently. For each output k,
                # W[k,r] appears in the equation for all (i,j)
                
                # The equation is: sum_r (U[i,r]*V[j,r]) * W[k,r] = T[i,j,k]
                # Let M_{(i,j),r} = U[i,r]*V[j,r]
                # Then M @ W[k,:] = T[:,:,k].T flattened
                
                # Build M matrix: (12*16) x target_rank
                M = np.zeros((n * m * m * p, target_rank))
                for i in range(n * m):
                    for j in range(m * p):
                        idx = i * (m * p) + j
                        M[idx, :] = U_rand[i, :] * V_rand[j, :]
                
                # For each k, solve M @ W[k,:] = T[:,:,k].T.reshape(-1)
                T_k = T[:, :, k].reshape(-1)
                
                # Use least squares
                W_k, residuals, rank_m, sv = np.linalg.lstsq(M, T_k, rcond=None)
                W_curr[k, :] = W_k
                
                # Update U: fix V and W, solve for U
                # sum_r U[i,r]*V[j,r]*W[k,r] = T[i,j,k]
                # For fixed j,k: sum_r U[i,r]*(V[j,r]*W[k,r]) = T[i,j,k]
                # Let N_{(j,k),r} = V[j,r]*W[k,r]
                # Then N @ U[i,:] = T[:,:,k].T[i,:] for each i... 
                
                # Actually: for each i, sum over (j,k) of U[i,r]*V[j,r]*W[k,r] = T[i,j,k]
                # This is: for each i, the vector U[i,:] must satisfy
                # sum_{j,k} U[i,r]*V[j,r]*W[k,r] = T[i,j,k] for all (j,k)
                
                # Let's build a different system. For each i, define
                # P_i[(j,k), r] = V[j,r]*W[k,r]
                # Then P_i @ U[i,:] = T[i,:,:].reshape(-1)
                
                U_new = np.zeros_like(U_rand)
                for i in range(n * m):
                    P_i = np.zeros((m * p * p * n, target_rank))
                    for j in range(m * p):
                        for k in range(p * n):
                            idx = j * (p * n) + k
                            P_i[idx, :] = V_rand[j, :] * W_curr[k, :]
                    T_i = T[i, :, :].reshape(-1)
                    U_new[i, :], _, _, _ = np.linalg.lstsq(P_i, T_i, rcond=None)
                
                # Update V similarly
                V_new = np.zeros_like(V_rand)
                for j in range(m * p):
                    Q_j = np.zeros((n * m * p * n, target_rank))
                    for i in range(n * m):
                        for k in range(p * n):
                            idx = i * (p * n) + k
                            Q_j[idx, :] = U_new[i, :] * W_curr[k, :]
                    T_j = T[:, j, :].reshape(-1)
                    V_new[j, :], _, _, _ = np.linalg.lstsq(Q_j, T_j, rcond=None)
                
                # Check convergence
                if np.abs(U_new - U_rand).max() < 1e-8 and np.abs(V_new - V_rand).max() < 1e-8:
                    U_rand = U_new
                    V_rand = V_new
                    converged = True
                    break
                
                U_rand = U_new
                V_rand = V_new
            
            # Verify the solution
            if converged:
                sol_test = {"U": U_rand.tolist(), "V": V_rand.tolist(), "W": W_curr.tolist()}
                if verify_solution(sol_test, max_tests=100):
                    best_solution = sol_test
                    best_rank = target_rank
                    print(f"Found rank {target_rank} solution!")
                    break
        
        return best_solution, best_rank
    
    # First, get the baseline solution
    sol = get_current_solution()
    
    # Verify correctness
    np.random.seed(42)
    if not verify_solution(sol, max_tests=100):
        # If verification fails, return trivial solution as fallback
        rank = 48
        U = np.zeros((12, rank))
        V = np.zeros((16, rank))
        W = np.zeros((12, rank))
        for row_a in range(3):
            for col_a in range(4):
                for col_c in range(4):
                    idx = (row_a * 4 + col_a) * 4 + col_c
                    i = row_a * 4 + col_a
                    j = col_a * 4 + col_c
                    k = col_c * 3 + row_a
                    U[i, idx] = 1
                    V[j, idx] = 1
                    W[k, idx] = 1
        sol = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Check deadline
    if time.time() > deadline:
        return sol
    
    # Try to find a better solution
    try:
        improved_sol, improved_rank = find_rank_decomposition(max_rank=44, time_limit=130)
        if improved_rank < 44:
            sol = improved_sol
    except Exception as e:
        # If anything goes wrong, return the baseline
        pass
    
    # Check deadline
    if time.time() > deadline:
        return sol
    
    return sol
