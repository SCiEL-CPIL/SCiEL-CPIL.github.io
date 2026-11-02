import time
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s limit
    
    n, m, p = 2, 3, 4
    
    # Schoolbook decomposition (Rank 24) - always valid fallback
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook), dtype=np.float64)
    V_sb = np.zeros((m * p, rank_schoolbook), dtype=np.float64)
    W_sb = np.zeros((p * n, rank_schoolbook), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1
                
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_schoolbook
    
    # Try to find a lower rank decomposition using greedy set cover approach
    # The tensor T[i,j,k] = 1 for all i,j,k
    # We want to decompose T = sum_r u_r ⊗ v_r ⊗ w_r
    
    # Pre-compute candidate rank-1 tensors with small integer/half-integer entries
    # For efficiency, we'll use a bounded search space
    
    # Define the target tensor
    T = np.ones((n, m, p), dtype=np.float64)
    
    # Generate candidate vectors with entries in {-1, -0.5, 0, 0.5, 1}
    # But to keep it manageable, let's focus on {0, 1} first, then expand
    
    # Strategy: Use greedy algorithm to select rank-1 terms that cover the most unsatisfied entries
    # Start with residual = T
    
    def verify_decomposition(U, V, W, tol=1e-9):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        rank = U.shape[1]
        # Compute the tensor from the decomposition
        T_recon = np.zeros((n, m, p), dtype=np.float64)
        for r in range(rank):
            u = U[:, r]  # shape (n*m,)
            v = V[:, r]  # shape (m*p,)
            w = W[:, r]  # shape (p*n,)
            
            # Reshape
            u_mat = u.reshape(n, m)
            v_mat = v.reshape(m, p)
            w_mat = w.reshape(p, n)
            
            # The contribution to T[i,j,k] is u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_recon[i, j, k] += u_mat[i, j] * v_mat[j, k] * w_mat[k, i]
        
        # Check if T_recon matches T
        return np.allclose(T_recon, T, atol=tol, rtol=0)
    
    # Try rank 20 using a more systematic approach
    # We'll use a local search / simulated annealing style approach
    
    def try_find_decomposition(rank, max_iterations=10000, time_limit=30):
        """Try to find a rank-rank decomposition using iterative improvement."""
        start = time.time()
        
        # Initialize with random vectors
        np.random.seed(42)
        
        # Candidate values for entries
        candidates = [0.0, 0.5, 1.0, -0.5, -1.0]
        
        # Initialize U, V, W with random candidate values
        U = np.random.choice(candidates, size=(n*m, rank)).astype(np.float64)
        V = np.random.choice(candidates, size=(m*p, rank)).astype(np.float64)
        W = np.random.choice(candidates, size=(p*n, rank)).astype(np.float64)
        
        # Compute initial error
        def compute_error(U, V, W):
            T_recon = np.zeros((n, m, p), dtype=np.float64)
            for r in range(rank):
                u = U[:, r].reshape(n, m)
                v = V[:, r].reshape(m, p)
                w = W[:, r].reshape(p, n)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_recon[i, j, k] += u[i, j] * v[j, k] * w[k, i]
            return np.sum((T_recon - T) ** 2)
        
        current_error = compute_error(U, V, W)
        best_error = current_error
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iterations):
            if time.time() - start > time_limit:
                break
            
            # Pick a random vector to modify
            mat_choice = np.random.randint(3)  # 0=U, 1=V, 2=W
            row = np.random.randint(n*m if mat_choice == 0 else (m*p if mat_choice == 1 else p*n))
            col = np.random.randint(rank)
            
            # Try changing this entry
            old_val = (U if mat_choice == 0 else (V if mat_choice == 1 else W))[row, col]
            new_val = np.random.choice(candidates)
            
            if new_val == old_val:
                continue
            
            # Apply change
            if mat_choice == 0:
                U[row, col] = new_val
            elif mat_choice == 1:
                V[row, col] = new_val
            else:
                W[row, col] = new_val
            
            new_error = compute_error(U, V, W)
            
            # Accept if better or with some probability (simulated annealing)
            if new_error < current_error:
                current_error = new_error
                if new_error < best_error:
                    best_error = new_error
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    if best_error < 1e-9:
                        return best_U, best_V, best_W
            else:
                # Revert
                if mat_choice == 0:
                    U[row, col] = old_val
                elif mat_choice == 1:
                    V[row, col] = old_val
                else:
                    W[row, col] = old_val
        
        # Check if best solution is valid
        if best_error < 1e-9:
            return best_U, best_V, best_W
        
        return None
    
    # Try rank 20
    if time.time() < deadline - 5:
        result_20 = try_find_decomposition(20, max_iterations=5000, time_limit=25)
        if result_20 is not None:
            U_20, V_20, W_20 = result_20
            if verify_decomposition(U_20, V_20, W_20):
                best_result = {"U": U_20, "V": V_20, "W": W_20}
                best_rank = 20
    
    # If we found rank 20, try rank 19
    if best_rank == 20 and time.time() < deadline - 5:
        result_19 = try_find_decomposition(19, max_iterations=5000, time_limit=25)
        if result_19 is not None:
            U_19, V_19, W_19 = result_19
            if verify_decomposition(U_19, V_19, W_19):
                best_result = {"U": U_19, "V": V_19, "W": W_19}
                best_rank = 19
    
    # If we found rank 19, try rank 18
    if best_rank == 19 and time.time() < deadline - 5:
        result_18 = try_find_decomposition(18, max_iterations=5000, time_limit=25)
        if result_18 is not None:
            U_18, V_18, W_18 = result_18
            if verify_decomposition(U_18, V_18, W_18):
                best_result = {"U": U_18, "V": V_18, "W": W_18}
                best_rank = 18
    
    # Final safety check
    if time.time() > deadline - 2:
        # Return the best we have
        pass
    
    # Ensure we return valid numpy arrays
    return {
        "U": np.array(best_result["U"], dtype=np.float64),
        "V": np.array(best_result["V"], dtype=np.float64),
        "W": np.array(best_result["W"], dtype=np.float64)
    }
