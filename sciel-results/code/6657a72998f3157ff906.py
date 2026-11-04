import numpy as np
import time

DIMENSIONS = [3, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 120s limit, leave 5s margin
    
    n, m, p = DIMENSIONS
    dim_U = n * m  # 12
    dim_V = m * p  # 28
    dim_W = p * n  # 21
    
    num_eqs = n * m * p  # 84
    
    # Pre-compute index mappings
    idx_U = np.zeros(num_eqs, dtype=int)
    idx_V = np.zeros(num_eqs, dtype=int)
    idx_W_row = np.zeros(num_eqs, dtype=int)
    
    e = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_U[e] = i * m + j
                idx_V[e] = j * p + k
                idx_W_row[e] = k * n + i
                e += 1
    
    def get_schoolbook():
        rank = num_eqs
        U = np.zeros((dim_U, rank))
        V = np.zeros((dim_V, rank))
        W = np.zeros((dim_W, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def verify(U, V, W):
        """Verify that the decomposition is exact."""
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        r = U.shape[1]
        
        # Compute sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for all (i,j,k)
        U_exp = U[idx_U, :]  # (84, r)
        V_exp = V[idx_V, :]  # (84, r)
        W_exp = W[idx_W_row, :]  # (84, r)
        
        vals = np.sum(U_exp * V_exp * W_exp, axis=1)
        return np.all(np.abs(vals - 1.0) < 1e-8)
    
    def solve_for_W(U, V):
        """Given U and V, solve for W such that the decomposition holds."""
        r = U.shape[1]
        
        U_exp = U[idx_U, :]  # (84, r)
        V_exp = V[idx_V, :]  # (84, r)
        coeffs = U_exp * V_exp  # (84, r)
        
        # Build A matrix: A[e, idx_W_row[e]*r + c] = coeffs[e, c]
        A = np.zeros((num_eqs, dim_W * r))
        for c in range(r):
            A[np.arange(num_eqs), idx_W_row * r + c] = coeffs[:, c]
        
        b = np.ones(num_eqs)
        
        try:
            W_flat, residuals, rank_sol, s = np.linalg.lstsq(A, b, rcond=None)
        except Exception:
            return None, None
        
        if residuals.size > 0 and residuals[0] > 1e-6:
            return None, None
        
        W_sol = W_flat.reshape(dim_W, r)
        return W_sol, residuals[0] if residuals.size > 0 else 0.0
    
    # Try ranks from lower bound to schoolbook
    best_result = None
    
    for r in range(28, 85):
        if time.time() > deadline - 3:
            break
        
        # Number of tries: allocate more to lower ranks
        if r <= 35:
            tries = 50
        elif r <= 50:
            tries = 30
        elif r <= 70:
            tries = 15
        else:
            tries = 5
        
        for trial in range(tries):
            if time.time() > deadline - 2:
                break
            
            # Generate random U and V with integer entries
            # Use a mix of strategies
            if trial % 3 == 0:
                # Strategy 1: Small integers [-2, 2]
                U = np.random.randint(-2, 3, size=(dim_U, r))
                V = np.random.randint(-2, 3, size=(dim_V, r))
            elif trial % 3 == 1:
                # Strategy 2: Slightly larger integers [-3, 3]
                U = np.random.randint(-3, 4, size=(dim_U, r))
                V = np.random.randint(-3, 4, size=(dim_V, r))
            else:
                # Strategy 3: Sparse - mostly 0, some ±1, ±2
                U = np.zeros((dim_U, r))
                V = np.zeros((dim_V, r))
                for _ in range(dim_U * r // 3):
                    U[np.random.randint(dim_U), np.random.randint(r)] = np.random.choice([-2, -1, 1, 2])
                for _ in range(dim_V * r // 3):
                    V[np.random.randint(dim_V), np.random.randint(r)] = np.random.choice([-2, -1, 1, 2])
            
            # Ensure no row is all zeros (would make that equation unsatisfiable)
            U_rows_zero = np.all(U == 0, axis=1)
            V_rows_zero = np.all(V == 0, axis=1)
            if np.any(U_rows_zero) or np.any(V_rows_zero):
                continue
            
            W_sol, residual = solve_for_W(U, V)
            if W_sol is None:
                continue
            
            # Check if W has half-integer entries
            W_scaled = W_sol * 2
            W_rounded = np.round(W_scaled)
            
            if np.max(np.abs(W_scaled - W_rounded)) < 1e-4:
                W_int = W_rounded / 2.0
                
                # Verify
                if verify(U, V, W_int):
                    result = {"U": U.tolist(), "V": V.tolist(), "W": W_int.tolist()}
                    if best_result is None or r < best_result[0]:
                        best_result = (r, result)
                        # Found a valid solution, keep searching for better
                        break  # Move to next rank
        
        if best_result is not None and r < best_result[0]:
            # We found something better at a lower rank, but we're at a higher rank now
            # Continue to see if we can find even better
            pass
    
    if best_result is not None:
        return best_result[1]
    
    return get_schoolbook()
