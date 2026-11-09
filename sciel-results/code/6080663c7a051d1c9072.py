import numpy as np
import time

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Target dimensions for U, V, W
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Construct the list of non-zero indices of T
    # T[i*m+j, j*p+k, k*n+i] = 1
    nz_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                nz_indices.append((u_idx, v_idx, w_idx))
    
    start_time = time.time()
    deadline = 140.0 # Leave 10s margin for 150s limit
    
    # Lower bound is max(dim_u, dim_v, dim_w)
    lower_bound = max(dim_u, dim_v, dim_w)
    
    # We iterate through possible ranks
    for r in range(lower_bound, rank_schoolbook):
        if time.time() > deadline:
            break
            
        # Try to find a decomposition of rank r
        # We use a greedy heuristic: iteratively find the best rank-1 term
        # with multiple random restarts
        
        found = False
        num_restarts = 50 # Increased restarts
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
                
            residual = np.zeros((dim_u, dim_v, dim_w))
            # Initialize residual to T
            for u_idx, v_idx, w_idx in nz_indices:
                residual[u_idx, v_idx, w_idx] = 1.0
            
            U_factors = []
            V_factors = []
            W_factors = []
            Coeffs = []
            
            success = False
            for _ in range(r):
                if time.time() > deadline:
                    break
                    
                # Initialize with random half-integers to bias towards the solution space
                # Values: -1, -0.5, 0, 0.5, 1
                vals = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
                
                u = np.random.choice(vals, size=dim_u)
                v = np.random.choice(vals, size=dim_v)
                w = np.random.choice(vals, size=dim_w)
                
                # Ensure non-zero norm
                if np.linalg.norm(u) < 1e-10: u = np.random.randn(dim_u)
                if np.linalg.norm(v) < 1e-10: v = np.random.randn(dim_v)
                if np.linalg.norm(w) < 1e-10: w = np.random.randn(dim_w)
                
                u = u / np.linalg.norm(u)
                v = v / np.linalg.norm(v)
                w = w / np.linalg.norm(w)
                
                prev_coeff = 0
                converged = False
                for _iter in range(200): # Max iterations
                    # Update u
                    u_new = np.einsum('ijk,j,k->i', residual, v, w)
                    norm_u = np.linalg.norm(u_new)
                    if norm_u < 1e-10:
                        break
                    u_new = u_new / norm_u
                    
                    # Update v
                    v_new = np.einsum('ijk,i,k->j', residual, u_new, w)
                    norm_v = np.linalg.norm(v_new)
                    if norm_v < 1e-10:
                        break
                    v_new = v_new / norm_v
                    
                    # Update w
                    w_new = np.einsum('ijk,i,j->k', residual, u_new, v_new)
                    norm_w = np.linalg.norm(w_new)
                    if norm_w < 1e-10:
                        break
                    w_new = w_new / norm_w
                    
                    # Check convergence
                    coeff = np.einsum('ijk,i,j,k->', residual, u_new, v_new, w_new)
                    if abs(coeff - prev_coeff) < 1e-8:
                        u, v, w = u_new, v_new, w_new
                        converged = True
                        break
                    u, v, w = u_new, v_new, w_new
                    prev_coeff = coeff
                
                if not converged:
                    # If it didn't converge, try to proceed with the current best guess
                    # or break. Breaking is safer to avoid garbage.
                    continue # Restart this rank attempt
                
                # Calculate coefficient
                coeff = np.einsum('ijk,i,j,k->', residual, u, v, w)
                
                # Update residual
                residual -= coeff * np.einsum('i,j,k->ijk', u, v, w)
                
                # Store factors
                U_factors.append(u)
                V_factors.append(v)
                W_factors.append(w)
                Coeffs.append(coeff)
                
                # Check if residual is negligible
                if np.linalg.norm(residual) < 1e-8:
                    success = True
                    break
            
            if not success:
                continue
                
            # We have a real-valued decomposition.
            # Construct U, V, W matrices
            U_mat = np.zeros((dim_u, r))
            V_mat = np.zeros((dim_v, r))
            W_mat = np.zeros((dim_w, r))
            
            for col in range(r):
                U_mat[:, col] = Coeffs[col] * U_factors[col]
                V_mat[:, col] = V_factors[col]
                W_mat[:, col] = W_factors[col]
            
            # Check for Integer/Half-Integer representation
            # Condition: 2 * mat entries are integers
            U_check = 2.0 * U_mat
            V_check = 2.0 * V_mat
            W_check = 2.0 * W_mat
            
            # Use a slightly more relaxed tolerance for snapping to handle numerical drift
            # from the power iteration
            atol = 1e-4
            rtol = 1e-4
            
            if np.allclose(U_check, np.round(U_check), atol=atol, rtol=rtol) and \
               np.allclose(V_check, np.round(V_check), atol=atol, rtol=rtol) and \
               np.allclose(W_check, np.round(W_check), atol=atol, rtol=rtol):
                
                # The factors are half-integers.
                U_final = np.round(U_check) / 2.0
                V_final = np.round(V_check) / 2.0
                W_final = np.round(W_check) / 2.0
                
                # Verify exactness
                T_final = np.zeros((dim_u, dim_v, dim_w))
                for col in range(r):
                    T_final += np.einsum('i,j,k->ijk', U_final[:, col], V_final[:, col], W_final[:, col])
                
                valid = True
                for u_idx, v_idx, w_idx in nz_indices:
                    if not np.isclose(T_final[u_idx, v_idx, w_idx], 1.0, atol=1e-5):
                        valid = False
                        break
                
                if valid:
                    return {
                        "U": U_final.tolist(),
                        "V": V_final.tolist(),
                        "W": W_final.tolist()
                    }
            
            # If snapping failed, try next restart
            
    # Fallback: Schoolbook
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
