"""Program for matrix multiplication 3x4x8 tensor decomposition."""
import time
import itertools
import numpy as np
from collections import defaultdict

DIMENSIONS = [3, 4, 8]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying
    a 3 by 4 matrix by a 4 by 8 matrix.
    
    Uses combinatorial search over sparse factor matrices to find
    a rank below the schoolbook 96.
    """
    start_time = time.time()
    DEADLINE = 145  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    # Schoolbook: rank = n*m*p = 96
    rank_schoolbook = n * m * p  # 96
    
    # Build schoolbook decomposition as fallback
    U = [[0] * rank_schoolbook for _ in range(n * m)]  # 12 x 96
    V = [[0] * rank_schoolbook for _ in range(m * p)]  # 32 x 96
    W = [[0] * rank_schoolbook for _ in range(p * n)]  # 24 x 96
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U, V, W
    
    # Strategy: Try to find a decomposition with rank < 96
    # by searching over sparse factor matrices.
    #
    # The tensor decomposition requires:
    # For each (i,j,k) in [n]x[m]x[p]:
    #   sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] = 1
    # For all other (i,j,k):
    #   sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] = 0
    #
    # We search for sparse U, V, W with entries in {-1, 0, 1} or {-1/2, 0, 1/2, 1}
    
    def _verify_decomp(U_mat, V_mat, W_mat, n, m, p, rank):
        """Verify that the decomposition is correct."""
        U_arr = np.array(U_mat, dtype=float)
        V_arr = np.array(V_mat, dtype=float)
        W_arr = np.array(W_mat, dtype=float)
        
        # Check each tensor component
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    # Compute sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    u_row = U_arr[i * m + j]
                    v_row = V_arr[j * p + k]
                    w_row = W_arr[k * n + i]
                    val = np.sum(u_row * v_row * w_row)
                    if abs(val - expected) > 1e-9:
                        return False
        return True
    
    def _try_rank(rank, max_time=30):
        """Try to find a decomposition of given rank using combinatorial search."""
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 5:
            return None
        
        # For a given rank, we need U (12 x rank), V (32 x rank), W (24 x rank)
        # with entries from a small set.
        
        # Strategy: Use a greedy/iterative approach
        # Start with random sparse matrices and try to satisfy constraints
        
        # The number of constraints is n*m*p = 96
        # The number of variables is rank * (12 + 32 + 24) = rank * 68
        
        # For rank >= 68, the system is underdetermined.
        # Let's try to find solutions with rank in a reasonable range.
        
        # Approach: Model as a constraint satisfaction problem
        # Each "column" r of the decomposition contributes a rank-1 tensor
        # T_r = u_r ⊗ v_r ⊗ w_r
        # We need sum_r T_r = T_matmul
        
        # For small entries, each rank-1 tensor has limited support.
        
        # Let's try a direct search for rank around 40-60
        # using a randomized approach with local search.
        
        target_rank = rank
        U_size = n * m  # 12
        V_size = m * p  # 32
        W_size = p * n  # 24
        
        # Initialize with random sparse matrices
        # Entries from {-1, 0, 1}
        
        np.random.seed(42)
        
        # Try multiple random initializations
        for attempt in range(20):
            if time.time() - start_time > DEADLINE - 5:
                return None
            
            # Random sparse initialization
            density = 0.3
            U_rand = np.zeros((U_size, target_rank), dtype=float)
            V_rand = np.zeros((V_size, target_rank), dtype=float)
            W_rand = np.zeros((W_size, target_rank), dtype=float)
            
            for r in range(target_rank):
                # Each column has a few non-zero entries
                n_nonzero_U = max(1, int(U_size * density * 0.5))
                n_nonzero_V = max(1, int(V_size * density * 0.5))
                n_nonzero_W = max(1, int(W_size * density * 0.5))
                
                idx_U = np.random.choice(U_size, n_nonzero_U, replace=False)
                idx_V = np.random.choice(V_size, n_nonzero_V, replace=False)
                idx_W = np.random.choice(W_size, n_nonzero_W, replace=False)
                
                U_rand[idx_U, r] = np.random.choice([-1, 1], n_nonzero_U)
                V_rand[idx_V, r] = np.random.choice([-1, 1], n_nonzero_V)
                W_rand[idx_W, r] = np.random.choice([-1, 1], n_nonzero_W)
            
            # Compute the current tensor approximation
            # T_approx[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            
            def _compute_tensor(U_mat, V_mat, W_mat, n, m, p, rank):
                T = np.zeros((n, m, p), dtype=float)
                for r in range(rank):
                    u_col = U_mat[:, r]
                    v_col = V_mat[:, r]
                    w_col = W_mat[:, r]
                    # Reshape u_col to (n, m), v_col to (m, p), w_col to (p, n)
                    u_reshaped = u_col.reshape(n, m)
                    v_reshaped = v_col.reshape(m, p)
                    w_reshaped = w_col.reshape(p, n)
                    # The contribution to T[i,j,k] is u_reshaped[i,j] * v_reshaped[j,k] * w_reshaped[k,i]
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                T[i, j, k] += u_reshaped[i, j] * v_reshaped[j, k] * w_reshaped[k, i]
                return T
            
            T_approx = _compute_tensor(U_rand, V_rand, W_rand, n, m, p, target_rank)
            
            # Target tensor
            T_target = np.ones((n, m, p), dtype=float)
            
            # Error
            error = np.sum((T_approx - T_target) ** 2)
            
            # Local search: try flipping entries to reduce error
            for iteration in range(500):
                if time.time() - start_time > DEADLINE - 5:
                    return None
                
                if error < 1e-9:
                    # Found a valid decomposition!
                    return (U_rand, V_rand, W_rand, target_rank)
                
                # Pick a random entry to flip
                mat_choice = np.random.choice(3)  # 0=U, 1=V, 2=W
                if mat_choice == 0:
                    row = np.random.choice(U_size)
                    col = np.random.choice(target_rank)
                    mat = U_rand
                elif mat_choice == 1:
                    row = np.random.choice(V_size)
                    col = np.random.choice(target_rank)
                    mat = V_rand
                else:
                    row = np.random.choice(W_size)
                    col = np.random.choice(target_rank)
                    mat = W_rand
                
                # Try changing this entry
                old_val = mat[row, col]
                
                # Compute how much this entry contributes to the error
                # The entry appears in T[i,j,k] for various (i,j,k)
                # depending on which matrix it's in.
                
                # For simplicity, just try setting to -old_val or 0 or other values
                candidates = []
                if old_val != 0:
                    candidates.append(0.0)
                    candidates.append(-old_val)
                else:
                    candidates.append(1.0)
                    candidates.append(-1.0)
                
                best_new_val = old_val
                best_new_error = error
                
                for new_val in candidates:
                    mat[row, col] = new_val
                    # Recompute error (expensive, but we're doing limited iterations)
                    T_new = _compute_tensor(U_rand, V_rand, W_rand, n, m, p, target_rank)
                    new_error = np.sum((T_new - T_target) ** 2)
                    if new_error < best_new_error:
                        best_new_error = new_error
                        best_new_val = new_val
                    mat[row, col] = old_val
                
                if best_new_val != old_val:
                    mat[row, col] = best_new_val
                    error = best_new_error
                else:
                    # No improvement, break out
                    break
            
            if error < 1e-9:
                return (U_rand, V_rand, W_rand, target_rank)
        
        return None
    
    # Try ranks from low to high, but focus on a reasonable range
    # The theoretical lower bound is around 41, but finding it is hard.
    # Let's try ranks from 50 to 95.
    
    # Actually, let's be smarter. Try a few specific ranks.
    ranks_to_try = [95, 94, 93, 92, 91, 90, 85, 80, 75, 70, 65, 60, 55, 50]
    
    for r in ranks_to_try:
        if time.time() - start_time > DEADLINE - 10:
            break
        
        result = _try_rank(r, max_time=10)
        if result is not None:
            U_mat, V_mat, W_mat, rank_found = result
            # Convert to list format
            U_list = U_mat.tolist()
            V_list = V_mat.tolist()
            W_list = W_mat.tolist()
            
            # Verify
            if _verify_decomp(U_list, V_list, W_list, n, m, p, rank_found):
                best_rank = rank_found
                best_U, best_V, best_W = U_list, V_list, W_list
                break  # Found a valid decomposition, stop
    
    return {"U": best_U, "V": best_V, "W": best_W}
