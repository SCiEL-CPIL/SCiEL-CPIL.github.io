import time
import sys
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Build the matrix multiplication tensor T[i,j,k] = 1 if the (i,k) entry of A*B
    # depends on A[i,j]*B[j,k]. For standard matrix multiplication, T[i,j,k]=1 for all valid i,j,k.
    # Actually, the tensor is T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # because C[i,k] = sum_j A[i,j]*B[j,k], so every (i,j,k) triple contributes.
    
    # Start with rank-56 Strassen decomposition
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_U = [
        {(0,0): 1, (1,1): 1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1},
        {(1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(1,0): 1, (0,0): -1},
        {(0,1): 1, (1,1): -1},
    ]
    strassen_V = [
        {(0,0): 1, (1,1): 1},
        {(0,0): 1},
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (0,0): -1},
        {(1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(1,0): 1, (1,1): 1},
    ]
    strassen_W = [
        {(0,0): 1, (1,1): 1},
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(0,0): -1, (1,0): 1},
        {(1,1): 1},
        {(0,0): 1},
    ]
    
    def add_block(base_rank, i_base, j_base, k_base):
        for r in range(7):
            cr = base_rank + r
            for (li, lj), val in strassen_U[r].items():
                gi = i_base + li
                gj = j_base + lj
                U[gi * 4 + gj][cr] += val
            for (lj, lk), val in strassen_V[r].items():
                gj = j_base + lj
                gk = k_base + lk
                V[gj * 8 + gk][cr] += val
            for (lk, li), val in strassen_W[r].items():
                gk = k_base + lk
                gi = i_base + li
                W[gk * 2 + gi][cr] += val
    
    add_block(0, 0, 0, 0)
    add_block(7, 0, 0, 2)
    add_block(14, 0, 0, 4)
    add_block(21, 0, 0, 6)
    add_block(28, 0, 2, 0)
    add_block(35, 0, 2, 2)
    add_block(42, 0, 2, 4)
    add_block(49, 0, 2, 6)
    
    # Convert to numpy for efficient tensor computation
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # The tensor T[i,j,k] should equal sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
    # For matrix multiplication, T[i,j,k] = 1 for all i,j,k
    # Verify the decomposition is correct
    def compute_tensor(U_arr, V_arr, W_arr, rank_val):
        """Compute the decomposed tensor from factor matrices."""
        tensor = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Sum over rank dimension
                    u_vals = U_arr[i*m+j, :rank_val]
                    v_vals = V_arr[j*p+k, :rank_val]
                    w_vals = W_arr[k*n+i, :rank_val]
                    tensor[i,j,k] = np.dot(u_vals, v_vals * w_vals)
        return tensor
    
    # Check if decomposition is valid
    def is_valid(U_arr, V_arr, W_arr, rank_val):
        """Check if the decomposition exactly computes matrix multiplication."""
        tensor = compute_tensor(U_arr, V_arr, W_arr, rank_val)
        # For matrix multiplication, all entries should be 1
        expected = np.ones((n, m, p))
        return np.allclose(tensor, expected, atol=1e-10)
    
    # Verify initial decomposition
    if not is_valid(U_np, V_np, W_np, rank):
        # Fallback: return the original Strassen decomposition
        return {"U": U, "V": V, "W": W}
    
    # Now attempt rank reduction by trying to zero out columns
    # Strategy: for each column r, check if setting it to zero still gives a valid decomposition
    # If yes, remove that column and continue with rank-1
    # This is a greedy approach - not guaranteed optimal but may find reductions
    
    best_rank = rank
    best_U = U_np.copy()
    best_V = V_np.copy()
    best_W = W_np.copy()
    
    # Try to reduce rank greedily
    current_rank = rank
    current_U = U_np.copy()
    current_V = V_np.copy()
    current_W = W_np.copy()
    
    # Repeatedly try to zero out a column
    improved = True
    while improved:
        improved = False
        for r in range(current_rank):
            if time.time() > deadline - 5:
                break
            
            # Try zeroing out column r
            test_U = current_U.copy()
            test_V = current_V.copy()
            test_W = current_W.copy()
            test_U[:, r] = 0
            test_V[:, r] = 0
            test_W[:, r] = 0
            
            if is_valid(test_U, test_V, test_W, current_rank):
                # Remove column r
                mask = [x for x in range(current_rank) if x != r]
                current_U = current_U[:, mask]
                current_V = current_V[:, mask]
                current_W = current_W[:, mask]
                current_rank -= 1
                improved = True
                if current_rank < best_rank:
                    best_rank = current_rank
                    best_U = current_U.copy()
                    best_V = current_V.copy()
                    best_W = current_W.copy()
                break  # Restart the loop since indices changed
        
        if time.time() > deadline - 5:
            break
    
    # Now try gradient-based optimization to find lower-rank decompositions
    # We'll try to find a rank-r decomposition for r < best_rank
    if time.time() < deadline - 20:
        # Try decreasing ranks from best_rank - 1 down to some minimum
        for target_rank in range(best_rank - 1, max(15, best_rank - 10), -1):
            if time.time() > deadline - 10:
                break
            
            # Try multiple random initializations
            for init_seed in range(3):
                if time.time() > deadline - 10:
                    break
                
                np.random.seed(init_seed + 1000)
                
                # Initialize with small random perturbations around the Strassen structure
                # Or use a scaled version of the known decomposition
                U_init = np.random.randn(n*m, target_rank) * 0.1
                V_init = np.random.randn(m*p, target_rank) * 0.1
                W_init = np.random.randn(p*n, target_rank) * 0.1
                
                # Add some structure from the known decomposition
                for r in range(min(target_rank, rank)):
                    U_init[:, r] += U_np[:, r] * 0.5
                    V_init[:, r] += V_np[:, r] * 0.5
                    W_init[:, r] += W_np[:, r] * 0.5
                
                # Define the optimization problem
                def objective(params):
                    if time.time() > deadline - 5:
                        return float('inf')
                    
                    # Unpack parameters
                    U_flat = params[:n*m*target_rank]
                    V_flat = params[n*m*target_rank : n*m*target_rank + m*p*target_rank]
                    W_flat = params[n*m*target_rank + m*p*target_rank :]
                    
                    U_opt = U_flat.reshape(n*m, target_rank)
                    V_opt = V_flat.reshape(m*p, target_rank)
                    W_opt = W_flat.reshape(p*n, target_rank)
                    
                    # Compute tensor
                    tensor = np.zeros((n, m, p))
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                u_vals = U_opt[i*m+j, :]
                                v_vals = V_opt[j*p+k, :]
                                w_vals = W_opt[k*n+i, :]
                                tensor[i,j,k] = np.dot(u_vals, v_vals * w_vals)
                    
                    # Objective: sum of squared errors
                    error = tensor - 1.0
                    return np.sum(error**2)
                
                # Initial parameters
                init_params = np.concatenate([
                    U_init.flatten(),
                    V_init.flatten(),
                    W_init.flatten()
                ])
                
                # Run optimization
                result = minimize(
                    objective,
                    init_params,
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                if time.time() > deadline - 5:
                    break
                
                # Check if we found a valid solution
                if result.fun < 1e-10:
                    # Extract the solution
                    U_sol = result.x[:n*m*target_rank].reshape(n*m, target_rank)
                    V_sol = result.x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                    W_sol = result.x[n*m*target_rank + m*p*target_rank :].reshape(p*n, target_rank)
                    
                    # Verify
                    if is_valid(U_sol, V_sol, W_sol, target_rank):
                        best_rank = target_rank
                        best_U = U_sol.copy()
                        best_V = V_sol.copy()
                        best_W = W_sol.copy()
                        break  # Found valid solution for this rank, try lower rank
            
            if time.time() > deadline - 10:
                break
    
    # Convert back to lists
    if best_rank < rank:
        U_final = best_U.tolist()
        V_final = best_V.tolist()
        W_final = best_W.tolist()
        
        # Trim to best_rank columns
        U_final = [row[:best_rank] for row in U_final]
        V_final = [row[:best_rank] for row in V_final]
        W_final = [row[:best_rank] for row in W_final]
        
        # Round to nearest half-integer if close
        for mat in [U_final, V_final, W_final]:
            for i in range(len(mat)):
                for j in range(len(mat[i])):
                    val = mat[i][j]
                    rounded = round(val * 2) / 2
                    if abs(val - rounded) < 1e-10:
                        mat[i][j] = rounded
        
        return {"U": U_final, "V": V_final, "W": W_final}
    else:
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
