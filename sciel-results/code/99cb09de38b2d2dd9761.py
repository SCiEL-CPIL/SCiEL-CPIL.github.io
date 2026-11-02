import numpy as np
import time

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0

    # Dimensions
    dim_U = n * m  # 10
    dim_V = m * p  # 25
    dim_W = p * n  # 10

    # Target Tensor T[i, j, k] = 1 if valid, 0 otherwise
    # Valid indices: i in [0, n), j in [0, m), k in [0, p)
    # In the flattened output index gamma = k*n + i
    # The tensor T is of shape (n, m, p)
    T_target = np.zeros((n, m, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i, j, k] = 1.0

    # Schoolbook Solution (Rank 50)
    R_school = n * m * p
    U_school = np.zeros((dim_U, R_school), dtype=np.int8)
    V_school = np.zeros((dim_V, R_school), dtype=np.int8)
    W_school = np.zeros((dim_W, R_school), dtype=np.int8)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    best_rank = R_school
    best_U = U_school.astype(np.float64)
    best_V = V_school.astype(np.float64)
    best_W = W_school.astype(np.float64)

    def compute_error(T_current):
        return np.sum((T_current - T_target) ** 2)

    def try_rank(R):
        nonlocal best_rank, best_U, best_V, best_W
        if R >= best_rank:
            return False
            
        # Try to find a solution of rank R
        # Use Simulated Annealing / Local Search with Fast Updates
        
        num_restarts = 200
        steps_per_restart = 50000
        
        for restart in range(num_restarts):
            if time.time() > deadline - 2.0:
                return best_rank < R_school

            # Initialize U, V, W with random integers in {-1, 0, 1}
            # Bias towards sparse? Random uniform is fine.
            U = np.random.randint(-1, 2, (dim_U, R), dtype=np.int8)
            V = np.random.randint(-1, 2, (dim_V, R), dtype=np.int8)
            W = np.random.randint(-1, 2, (dim_W, R), dtype=np.int8)
            
            # Compute initial T_current and Error
            T_current = np.einsum('ir,jr,kr->ijk', U, V, W, optimize=True)
            current_error = compute_error(T_current)
            
            if current_error < 1e-9:
                best_rank = R
                best_U = U.astype(np.float64)
                best_V = V.astype(np.float64)
                best_W = W.astype(np.float64)
                return True

            best_error_in_run = current_error
            best_U_run = U.copy()
            best_V_run = V.copy()
            best_W_run = W.copy()
            
            # Temperature for Simulated Annealing
            T_temp = 1.0
            cooling = 0.999
            
            for step in range(steps_per_restart):
                if time.time() > deadline - 2.0:
                    break
                
                # Pick a random matrix and index
                mat_type = np.random.randint(0, 3)
                if mat_type == 0:
                    i_idx = np.random.randint(0, dim_U)
                    r_idx = np.random.randint(0, R)
                    old_val = U[i_idx, r_idx]
                    new_val = np.random.randint(-1, 2)
                    if new_val == old_val: continue
                    
                    # Calculate Delta T
                    # T[i, j, k] += (new - old) * V[j, r] * W[k, r]
                    delta = (new_val - old_val) * (V[:, r_idx] * W[:, r_idx]) # shape (m*p) -> (5*5)
                    # Reshape to (m, p)
                    delta_T = delta.reshape((m, p))
                    
                    # Old Error contribution for fixed i
                    old_contrib = np.sum((T_current[i_idx] - T_target[i_idx]) ** 2)
                    
                    T_current[i_idx] += delta_T
                    new_contrib = np.sum((T_current[i_idx] - T_target[i_idx]) ** 2)
                    
                    delta_error = new_contrib - old_contrib
                    
                    if delta_error < 0:
                        U[i_idx, r_idx] = new_val
                        current_error += delta_error
                        if current_error < 1e-9:
                            best_rank = R
                            best_U = U.astype(np.float64)
                            best_V = V.astype(np.float64)
                            best_W = W.astype(np.float64)
                            return True
                        if current_error < best_error_in_run:
                            best_error_in_run = current_error
                            best_U_run = U.copy()
                            best_V_run = V.copy()
                            best_W_run = W.copy()
                    else:
                        # Simulated Annealing acceptance
                        if np.random.rand() < np.exp(-delta_error / T_temp):
                            U[i_idx, r_idx] = new_val
                            current_error += delta_error
                elif mat_type == 1:
                    j_idx = np.random.randint(0, dim_V)
                    r_idx = np.random.randint(0, R)
                    old_val = V[j_idx, r_idx]
                    new_val = np.random.randint(-1, 2)
                    if new_val == old_val: continue
                    
                    # V is flattened (m, p). j_idx is k + j*m? No, V index is j*p+k.
                    # V[j_idx, r_idx] affects T[i, j, k] where j*p+k == j_idx.
                    # So j = j_idx // p, k = j_idx % p.
                    j_val = j_idx // p
                    k_val = j_idx % p
                    
                    delta = (new_val - old_val) * (U[:, r_idx] * W[:, r_idx]) # shape (n*n) -> (2*2)
                    # U is flattened (n, m). Wait.
                    # T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
                    # U index is i*m + j? No.
                    # Standard matrix mult: A (n x m) * B (m x p) = C (n x p).
                    # C[i, k] = sum_j A[i, j] B[j, k].
                    # Tensor: T[i, j, k] = A[i, j] B[j, k].
                    # U corresponds to A. Shape (n*m, R). Index a = i*m + j.
                    # V corresponds to B. Shape (m*p, R). Index b = j*p + k.
                    # W corresponds to C. Shape (p*n, R). Index c = k*n + i.
                    
                    # So if we change V[j_idx, r], j_idx = j*p + k.
                    # It affects T[i, j, k] for all i.
                    # Delta T[i, j, k] = (new - old) * U[i, r] * W[k, r] ??
                    # No. T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r].
                    # If V changes, delta T[i, j, k] = U[i, r] * (V_new - V_old) * W[k, r].
                    # U is indexed by i? No, U is indexed by (i, j_A).
                    # Wait, the tensor definition in the prompt:
                    # "output factor follows the official trace convention W[k*n+i,r]"
                    # This implies the third index of the tensor is related to W.
                    # Standard: T[i, j, k] = A[i, j] B[j, k].
                    # U is A. V is B. W is C.
                    # U shape (n*m, R). V shape (m*p, R). W shape (p*n, R).
                    
                    # If we change V[j_idx, r], where j_idx = j_B * p + k.
                    # The term is U[i_U, r] * V[j_idx, r] * W[k_W, r].
                    # For the product to be non-zero in the sum for T[i, j, k],
                    # we need the indices to match the tensor coordinates.
                    # The tensor coordinates are (i, j, k).
                    # U index a = i*m + j.
                    # V index b = j*p + k.
                    # W index c = k*n + i.
                    
                    # So, if we change V[b, r], it affects T[i, j, k] only if b = j*p + k.
                    # And for that specific (i, j, k), the contribution is U[a, r] * V[b, r] * W[c, r].
                    # So Delta T[i, j, k] = U[i*m+j, r] * (V_new - V_old) * W[k*n+i, r].
                    
                    # We need to update T[i, j, k] for all i (since j, k are fixed by b).
                    # There are n=2 values of i.
                    j_B = j_idx // p
                    k_B = j_idx % p
                    
                    # Loop over i
                    old_contrib = 0.0
                    new_contrib = 0.0
                    # We need to update T_current[i, j_B, k_B]
                    # Note: T_current is stored as (n, m, p) i.e., (i, j, k)
                    
                    for i in range(n):
                        a_idx = i * m + j_B
                        c_idx = k_B * n + i
                        
                        old_val_T = T_current[i, j_B, k_B]
                        target_val = T_target[i, j_B, k_B]
                        old_contrib += (old_val_T - target_val) ** 2
                        
                        delta_val = U[a_idx, r_idx] * (new_val - old_val) * W[c_idx, r_idx]
                        new_val_T = old_val_T + delta_val
                        T_current[i, j_B, k_B] = new_val_T
                        new_contrib += (new_val_T - target_val) ** 2
                    
                    delta_error = new_contrib - old_contrib
                    
                    if delta_error < 0:
                        V[j_idx, r_idx] = new_val
                        current_error += delta_error
                        if current_error < 1e-9:
                            best_rank = R
                            best_U = U.astype(np.float64)
                            best_V = V.astype(np.float64)
                            best_W = W.astype(np.float64)
                            return True
                        if current_error < best_error_in_run:
                            best_error_in_run = current_error
                            best_U_run = U.copy()
                            best_V_run = V.copy()
                            best_W_run = W.copy()
                    else:
                        if np.random.rand() < np.exp(-delta_error / T_temp):
                            V[j_idx, r_idx] = new_val
                            current_error += delta_error

                else: # mat_type == 2
                    k_idx = np.random.randint(0, dim_W)
                    r_idx = np.random.randint(0, R)
                    old_val = W[k_idx, r_idx]
                    new_val = np.random.randint(-1, 2)
                    if new_val == old_val: continue
                    
                    # W index c = k*n + i.
                    # So k = c // n, i = c % n.
                    # It affects T[i, j, k] for all j.
                    k_W = k_idx // n
                    i_W = k_idx % n
                    
                    old_contrib = 0.0
                    new_contrib = 0.0
                    for j in range(m):
                        a_idx = i_W * m + j
                        b_idx = j * p + k_W
                        
                        old_val_T = T_current[i_W, j, k_W]
                        target_val = T_target[i_W, j, k_W]
                        old_contrib += (old_val_T - target_val) ** 2
                        
                        delta_val = U[a_idx, r_idx] * V[b_idx, r_idx] * (new_val - old_val)
                        new_val_T = old_val_T + delta_val
                        T_current[i_W, j, k_W] = new_val_T
                        new_contrib += (new_val_T - target_val) ** 2
                        
                    delta_error = new_contrib - old_contrib
                    
                    if delta_error < 0:
                        W[k_idx, r_idx] = new_val
                        current_error += delta_error
                        if current_error < 1e-9:
                            best_rank = R
                            best_U = U.astype(np.float64)
                            best_V = V.astype(np.float64)
                            best_W = W.astype(np.float64)
                            return True
                        if current_error < best_error_in_run:
                            best_error_in_run = current_error
                            best_U_run = U.copy()
                            best_V_run = V.copy()
                            best_W_run = W.copy()
                    else:
                        if np.random.rand() < np.exp(-delta_error / T_temp):
                            W[k_idx, r_idx] = new_val
                            current_error += delta_error

                T_temp *= cooling

            # End of local search for this restart
            # If we found a better solution than previous best for this R, keep it?
            # Actually, if we didn't find error ~ 0, we just try next restart.
            # But if we want to "warm start" next restart, we could use best_U_run.
            # Let's stick to pure random restarts for simplicity and diversity.

        return best_rank < R_school

    # Search for lower ranks
    for R in range(R_school - 1, 24, -1):
        if time.time() > deadline - 5.0:
            break
        try_rank(R)
        if best_rank == R:
            break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
