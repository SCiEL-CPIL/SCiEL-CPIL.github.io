import numpy as np
import time

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # 120s limit, keep margin
    
    # Schoolbook solution as fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    best_result = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
    best_rank = rank_school

    # Define the target tensor vector b
    # The tensor has dimensions (n*m, m*p, p*n)
    # We flatten it to a vector of size (n*m) * (m*p) * (p*n)
    # b[idx] = 1 if the indices correspond to a valid multiplication term, 0 otherwise
    
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    total_size = dim_u * dim_v * dim_w
    
    b = np.zeros(total_size)
    
    # Set b to 1 for valid terms
    # Index in b: ((i*m + j) * dim_v + (j*p + k)) * dim_w + (k*n + i)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                flat_idx = (idx_u * dim_v + idx_v) * dim_w + idx_w
                b[flat_idx] = 1.0

    # Search for lower ranks
    # Try ranks from 21 up to 41
    for r in range(21, best_rank):
        if time.time() > deadline - 10: # Stop if close to deadline
            break
            
        # Try a few random initializations for U and V
        for attempt in range(100): # Limit attempts per rank
            if time.time() > deadline - 10:
                break
                
            # Random U and V with 0/1 entries
            # Using a sparse structure might help, but random dense is simpler
            U = np.random.randint(0, 2, size=(dim_u, r))
            V = np.random.randint(0, 2, size=(dim_v, r))
            
            # Construct the linear system M * W = b
            # M has shape (total_size, dim_w * r)
            # W is flattened to size dim_w * r
            # The coefficient for W[idx_w, t] in equation (idx_u, idx_v) is U[idx_u, t] * V[idx_v, t]
            
            # We can build M efficiently by iterating over t
            # M is huge: 1764 * (14 * r). For r=21, 1764 * 294 ~ 500k elements.
            # Building it explicitly might be slow.
            # Let's use a sparse representation or just check the residual via projection.
            
            # Alternative: Solve M W = b using least squares on the fly?
            # M is defined by M[flat_idx, idx_w * r + t] = U[idx_u, t] * V[idx_v, t]
            
            # Let's try to solve it.
            # Since M is sparse (each row has at most r non-zeros? No, each row (idx_u, idx_v) has r non-zeros in W's columns?
            # Wait, for a fixed (idx_u, idx_v), the equation is sum_t U[idx_u, t] V[idx_v, t] W[idx_w, t] = b[flat_idx]
            # This equation involves W[idx_w, t] for all t?
            # No, idx_w is fixed for a given (idx_u, idx_v) ONLY IF the term is valid?
            # No, the tensor is defined for ALL triples (idx_u, idx_v, idx_w).
            # For a fixed (idx_u, idx_v), there are dim_w possible idx_w values.
            # So we have dim_w equations for each (idx_u, idx_v).
            # Each equation involves the same set of W variables?
            # W has indices (idx_w, t).
            # So for a fixed (idx_u, idx_v), we have a system of dim_w equations for the variables W[:, t] (all t)?
            # No, the variable is W[idx_w, t].
            # So for fixed (idx_u, idx_v), we have dim_w equations, each involving r variables (W[idx_w, 0...r-1] for the specific idx_w? No).
            
            # Let's re-read the tensor definition.
            # T[idx_u, idx_v, idx_w] = sum_t U[idx_u, t] V[idx_v, t] W[idx_w, t]
            # We want T[idx_u, idx_v, idx_w] = b[flat_idx]
            
            # So for each (idx_u, idx_v, idx_w), we have an equation.
            # The variables are W[idx_w, t] for t=1..r.
            # Note that W[idx_w, t] appears in ALL equations where the third index is idx_w.
            # So the system is coupled.
            
            # We can flatten W into a vector w of size dim_w * r.
            # The coefficient of w[idx_w * r + t] in equation (idx_u, idx_v, idx_w) is U[idx_u, t] * V[idx_v, t].
            
            # So M[flat_idx, idx_w * r + t] = U[idx_u, t] * V[idx_v, t] if the indices match.
            # Wait, idx_u, idx_v, idx_w are determined by flat_idx.
            
            # Building M explicitly:
            M = np.zeros((total_size, dim_w * r))
            
            for idx_u in range(dim_u):
                for idx_v in range(dim_v):
                    # For each idx_w
                    for idx_w in range(dim_w):
                        flat_idx = (idx_u * dim_v + idx_v) * dim_w + idx_w
                        # Coefficients for this equation
                        # w[idx_w * r + t] is multiplied by U[idx_u, t] * V[idx_v, t]
                        coeff = U[idx_u, :] * V[idx_v, :]
                        M[flat_idx, idx_w * r : (idx_w + 1) * r] = coeff
            
            # Solve M W = b
            try:
                W_flat, residuals, rank, s = np.linalg.lstsq(M, b, rcond=None)
                W = W_flat.reshape(dim_w, r)
                
                # Check if solution is exact
                # Reconstruct T
                # T[idx_u, idx_v, idx_w] = sum_t U[idx_u, t] V[idx_v, t] W[idx_w, t]
                # We can check the residual
                T_flat = M @ W_flat
                if np.allclose(T_flat, b, atol=1e-9):
                    # Found a valid decomposition
                    best_rank = r
                    best_result = {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
                    break # Stop searching for this rank, try next lower? No, we want minimal rank.
                    # Since we iterate r from 21 upwards, the first found is the minimal in this range.
                    # We can break out of the rank loop too.
                    
            except np.linalg.LinAlgError:
                pass

        if best_rank < rank_school:
            break

    return best_result
