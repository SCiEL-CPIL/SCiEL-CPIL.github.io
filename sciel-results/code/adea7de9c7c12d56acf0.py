"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115.0
    
    # Schoolbook initialization
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    def verify_half_integer(U, V, W):
        """Check if U, V, W are half-integers and form a valid decomposition."""
        # Check half-integer constraint
        if not np.allclose(U * 2, np.round(U * 2), atol=1e-9): return False
        if not np.allclose(V * 2, np.round(V * 2), atol=1e-9): return False
        if not np.allclose(W * 2, np.round(W * 2), atol=1e-9): return False
        
        # Snap to exact half-integers for calculation
        U_int = np.round(U * 2) / 2.0
        V_int = np.round(V * 2) / 2.0
        W_int = np.round(W * 2) / 2.0
        
        # Verify decomposition: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
        # We can vectorize this check partially
        # For each c, compute the tensor contribution? No, just check the 45 points.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    s = np.dot(U_int[u_idx, :], V_int[v_idx, :] * W_int[w_idx, :])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def solve_for_W(U, V):
        """Given U and V, find W that satisfies the equations in least squares sense."""
        r = U.shape[1]
        W_sol = np.zeros((p * n, r), dtype=np.float64)
        
        for k in range(p):
            for i in range(n):
                # Construct matrix A of size m x r
                # A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                A = np.zeros((m, r), dtype=np.float64)
                for j in range(m):
                    A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                
                # Solve A * x = 1 (vector of ones)
                # x is W[k*n+i, :]
                try:
                    x, residuals, rank_A, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                    W_sol[k * n + i, :] = x
                except np.linalg.LinAlgError:
                    # Fallback to zero if singular
                    W_sol[k * n + i, :] = 0.0
                    
        return W_sol

    def snap_to_half_integer(mat):
        return np.round(mat * 2.0) / 2.0

    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Local search to reduce rank
    current_rank = rank
    current_U, current_V, current_W = U.copy(), V.copy(), W.copy()
    
    attempts = 0
    max_attempts = 2000 # Limit iterations to ensure time safety
    
    while time.time() < deadline - 5.0 and attempts < max_attempts:
        attempts += 1
        
        # Try to reduce rank by 1
        if current_rank <= 1:
            break
            
        # Choose a column to drop randomly
        # Or choose the column with smallest norm to minimize impact
        norms_U = np.linalg.norm(current_U, axis=0)
        norms_V = np.linalg.norm(current_V, axis=0)
        norms_W = np.linalg.norm(current_W, axis=0)
        
        # Simple heuristic: drop random column
        drop_idx = np.random.randint(current_rank)
        
        # Create new U, V by dropping the column
        mask = [i for i in range(current_rank) if i != drop_idx]
        new_U = current_U[:, mask]
        new_V = current_V[:, mask]
        
        # Add small noise to escape local minima
        noise_scale = 0.1
        new_U = new_U + noise_scale * np.random.randn(*new_U.shape)
        new_V = new_V + noise_scale * np.random.randn(*new_V.shape)
        
        # Solve for W
        new_W = solve_for_W(new_U, new_V)
        
        # Snap to half-integers
        snap_U = snap_to_half_integer(new_U)
        snap_V = snap_to_half_integer(new_V)
        snap_W = snap_to_half_integer(new_W)
        
        # Verify
        if verify_half_integer(snap_U, snap_V, snap_W):
            # Success!
            current_rank -= 1
            current_U, current_V, current_W = snap_U, snap_V, snap_W
            best_rank = current_rank
            best_U, best_V, best_W = current_U.copy(), current_V.copy(), current_W.copy()
            
            # If we hit a very low rank, we might stop or continue
            # Theoretical lower bound is around 36.
            if current_rank <= 36:
                break
        else:
            # Failure, keep current state
            pass

    # Final check and return
    if verify_half_integer(best_U, best_V, best_W):
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    else:
        # Fallback to schoolbook if search failed
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }

# EVOLVE-BLOCK-END
