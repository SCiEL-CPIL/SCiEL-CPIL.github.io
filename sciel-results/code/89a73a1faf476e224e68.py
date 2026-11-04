import numpy as np
import time

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    start_time = time.time()
    time_limit = 110  # Leave 10s margin
    
    # Tensor T[i, j, k] = 1 for all i, j, k
    # We need U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] sum to 1
    
    # Precompute indices for vectorized error calculation
    # This is tricky because indices are coupled.
    # We will use a loop for error calculation for simplicity, as n*m*p is small (72).
    
    def compute_error(U, V, W):
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    prod_sum = np.dot(U[u_idx, :], V[v_idx, :])
                    # Wait, the decomposition is sum_r U[i,j,r] V[j,k,r] W[k,i,r]
                    # This is not a simple dot product of U and V because W depends on i and k.
                    # We must sum over r.
                    # err_term = sum_r U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                    # Vectorized: sum_r (U[u_idx, :] * V[v_idx, :]) * W[w_idx, :]
                    term = np.sum((U[u_idx, :] * V[v_idx, :]) * W[w_idx, :])
                    err += (1.0 - term) ** 2
        return err

    # Try to find a lower rank
    # We will try ranks from 71 down to 40
    best_result = None
    best_rank = n * m * p
    
    for rank in range(best_rank - 1, 39, -1):
        if time.time() - start_time > time_limit:
            break
            
        # Try a few random restarts for this rank
        for restart in range(5):
            if time.time() - start_time > time_limit:
                break
                
            # Random initialization with small integers
            U = np.random.randint(-2, 3, (n * m, rank))
            V = np.random.randint(-2, 3, (m * p, rank))
            W = np.random.randint(-2, 3, (p * n, rank))
            
            # Simple local search: try flipping a single entry
            # This is very slow. Better to use gradient descent on reals.
            
            # Convert to float for optimization
            U_f = U.astype(float)
            V_f = V.astype(float)
            W_f = W.astype(float)
            
            # Gradient descent
            lr = 0.01
            for step in range(50):
                if time.time() - start_time > time_limit:
                    break
                err = compute_error(U_f, V_f, W_f)
                if err < 1e-6:
                    break
                
                # Calculate gradients numerically? Too slow.
                # Just do random perturbations (Simulated Annealing style)
                # Pick a random matrix and entry, perturb it
                mat_idx = np.random.randint(3)
                if mat_idx == 0:
                    r, c = np.random.randint(n * m), np.random.randint(rank)
                    delta = np.random.normal(0, 0.1)
                    old_val = U_f[r, c]
                    U_f[r, c] += delta
                    new_err = compute_error(U_f, V_f, W_f)
                    if new_err < err:
                        err = new_err
                    else:
                        U_f[r, c] = old_val
                elif mat_idx == 1:
                    r, c = np.random.randint(m * p), np.random.randint(rank)
                    delta = np.random.normal(0, 0.1)
                    old_val = V_f[r, c]
                    V_f[r, c] += delta
                    new_err = compute_error(U_f, V_f, W_f)
                    if new_err < err:
                        err = new_err
                    else:
                        V_f[r, c] = old_val
                else:
                    r, c = np.random.randint(p * n), np.random.randint(rank)
                    delta = np.random.normal(0, 0.1)
                    old_val = W_f[r, c]
                    W_f[r, c] += delta
                    new_err = compute_error(U_f, V_f, W_f)
                    if new_err < err:
                        err = new_err
                    else:
                        W_f[r, c] = old_val
            
            # Check if we found a solution
            if err < 1e-6:
                # Snap to nearest half-integer
                U_int = np.round(U_f * 2) / 2
                V_int = np.round(V_f * 2) / 2
                W_int = np.round(W_f * 2) / 2
                
                # Verify exactness
                if compute_error(U_int, V_int, W_int) < 1e-9:
                    best_result = {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
                    best_rank = rank
                    break # Found a solution for this rank, try lower? No, return best found so far if time is tight.
                    # Actually, we want minimal rank, so we continue to lower ranks.
                    # But if we found one, we can break the restart loop for this rank.
        
        # If we found a solution at this rank, we can continue to lower ranks
        # But we must be careful with time.

    if best_result is not None:
        return best_result
    else:
        return schoolbook()
# EVOLVE-BLOCK-END
