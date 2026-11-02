import numpy as np
import time

def construct():
    n, m, p = 2, 3, 3
    # Dimensions of U, V, W
    dim_u = n * m # 6
    dim_v = m * p # 9
    dim_w = p * n # 6
    
    # Target tensor T
    # T is a 3D array of shape (n, m, p) but we flatten it to a vector for comparison
    # The decomposition is sum_r U[:, r] @ V[:, r] @ W[:, r] = C
    # Flattened C is W_flattened.
    # Let's define the target tensor in flattened form corresponding to W[k*n+i]
    # T_flat[index] = 1 if the corresponding (i, j, k) is in the multiplication
    # Wait, the tensor T_{i,j,k} = 1 for all i,j,k.
    # The decomposition is:
    # C_{ik} = sum_j A_{ij} B_{jk}
    # This is linear in A and B.
    # The tensor is T = sum_{i,j,k} e_i \otimes e_j \otimes e_k
    # No, T is the structure tensor.
    # T_{i,j,k} = 1.
    # We want sum_r u_r[i] v_r[j] w_r[k] = 1 if (i,j,k) is a valid term?
    # No, the tensor for matrix multiplication is:
    # T_{i,j,k} = 1 for all i,j,k.
    # And we want C_{ik} = sum_j A_{ij} B_{jk}.
    # This corresponds to the tensor T where T_{i,j,k} = 1.
    # And the decomposition is T = sum_r u_r \otimes v_r \otimes w_r.
    # So we need sum_r u_r[i] v_r[j] w_r[k] = 1 for all i,j,k.
    
    # Create target tensor
    T = np.ones((n, m, p))
    
    # Search for rank 17
    rank = 17
    # Initialize with random integers
    # Use a fixed seed for reproducibility in search, but random restarts
    rng = np.random.default_rng(42)
    
    best_score = np.inf
    best_solution = None
    
    start_time = time.time()
    timeout = 10.0 # 10 seconds for search
    
    # Heuristic: Start with schoolbook (18 terms) and try to merge/remove
    # Or just random search with local optimization
    
    # Let's try a local search starting from random initialization
    for restart in range(100): # 100 restarts
        if time.time() - start_time > timeout:
            break
            
        # Initialize U, V, W
        U = rng.integers(-1, 2, size=(dim_u, rank))
        V = rng.integers(-1, 2, size=(dim_v, rank))
        W = rng.integers(-1, 2, size=(dim_w, rank))
        
        # Local search
        for step in range(1000):
            if time.time() - start_time > timeout:
                break
            
            # Compute current tensor
            # Tensor = np.einsum('ir,jr,kr->ijk', U, V, W)
            # This is slow. Let's compute error directly.
            # Error = || T - Tensor ||^2
            
            # Pick a random term r and a random coordinate
            r = rng.integers(0, rank)
            tensor_choice = rng.integers(0, 3)
            
            if tensor_choice == 0:
                idx = rng.integers(0, dim_u)
                old_val = U[idx, r]
                # Try changing by +1, -1, or 0
                for delta in [-1, 1]:
                    new_val = old_val + delta
                    if -2 <= new_val <= 2: # Keep small
                        U[idx, r] = new_val
                        # Compute error
                        # Efficiently compute change in error?
                        # Or just recompute. 17 * 21 * 18 * 6 * 6 * 6 is small.
                        # 17 terms, each 6x9x6.
                        # Total ops ~ 17 * 324 = 5508.
                        # Very fast.
                        tensor = np.einsum('ir,jr,kr->ijk', U, V, W)
                        error = np.sum((T - tensor) ** 2)
                        if error < best_score:
                            best_score = error
                            if error == 0:
                                # Found exact solution
                                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                            # Store best
                            best_solution = (U.copy(), V.copy(), W.copy())
                        else:
                            U[idx, r] = old_val # Revert
            elif tensor_choice == 1:
                idx = rng.integers(0, dim_v)
                old_val = V[idx, r]
                for delta in [-1, 1]:
                    new_val = old_val + delta
                    if -2 <= new_val <= 2:
                        V[idx, r] = new_val
                        tensor = np.einsum('ir,jr,kr->ijk', U, V, W)
                        error = np.sum((T - tensor) ** 2)
                        if error < best_score:
                            best_score = error
                            if error == 0:
                                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                            best_solution = (U.copy(), V.copy(), W.copy())
                        else:
                            V[idx, r] = old_val
            else:
                idx = rng.integers(0, dim_w)
                old_val = W[idx, r]
                for delta in [-1, 1]:
                    new_val = old_val + delta
                    if -2 <= new_val <= 2:
                        W[idx, r] = new_val
                        tensor = np.einsum('ir,jr,kr->ijk', U, V, W)
                        error = np.sum((T - tensor) ** 2)
                        if error < best_score:
                            best_score = error
                            if error == 0:
                                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                            best_solution = (U.copy(), V.copy(), W.copy())
                        else:
                            W[idx, r] = old_val
                            
    # If no exact solution found, return schoolbook
    rank_school = n * m * p
    U_school = [[0.0] * rank_school for _ in range(dim_u)]
    V_school = [[0.0] * rank_school for _ in range(dim_v)]
    W_school = [[0.0] * rank_school for _ in range(dim_w)]
    
    col_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                
                U_school[idx_u][col_idx] = 1.0
                V_school[idx_v][col_idx] = 1.0
                W_school[idx_w][col_idx] = 1.0
                col_idx += 1
                
    return {"U": U_school, "V": V_school, "W": W_school}
