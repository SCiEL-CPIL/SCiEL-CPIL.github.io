import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Target tensor T is defined by T[i, j, j', k', k'', i''] = 1 if i==i'', j==j', k'==k'' else 0
    # We represent the linear system for U given V, W
    # Equation: sum_c U[im+j, c] * V[j'p+k', c] * W[k''n+i'', c] = RHS
    
    # Precompute indices
    # U indices: i, j -> im+j
    # V indices: j', k' -> j'p+k'
    # W indices: k'', i'' -> k''n+i''
    
    # We will try ranks from 18 to 24
    # Time limit 150s. Start time now.
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    best_rank = n * m * p
    best_solution = None
    
    # Generate the system matrix structure
    # For each equation (i, j, j', k', k'', i''):
    # Coeff for U[im+j, c] is V[j'p+k', c] * W[k''n+i'', c]
    
    # To speed up, we can precompute the list of (eq_idx, var_idx) pairs?
    # Or just build the matrix each time.
    # Matrix size: 576 x (6*r).
    # Building 576x108 matrix is fast.
    
    ranks_to_try = [18, 19, 20, 21, 22, 23, 24]
    
    for r in ranks_to_try:
        if time.time() > deadline:
            break
            
        # Try multiple random restarts for this rank
        max_restarts = 50
        if r < 20:
            max_restarts = 100
            
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Initialize U, V, W with random integers -1, 0, 1
            U = np.random.randint(-1, 2, size=(n*m, r)).astype(float)
            V = np.random.randint(-1, 2, size=(m*p, r)).astype(float)
            W = np.random.randint(-1, 2, size=(p*n, r)).astype(float)
            
            # Ensure not all zero
            if np.all(U == 0) or np.all(V == 0) or np.all(W == 0):
                continue
                
            success = False
            for iter in range(50):
                # Solve for U
                # Build A_U and b_U
                A_U = np.zeros((n*m*p*p*m*n, n*m*r)) # This is too big?
                # Wait, the number of equations is n*m * m*p * p*n?
                # No, the indices are (i, j, j', k', k'', i'').
                # i in [0,n), j in [0,m), j' in [0,m), k' in [0,p), k'' in [0,p), i'' in [0,n).
                # Total equations: n*m*m*p*p*n = 576.
                
                # We can build A_U more efficiently.
                # A_U[eq_idx, var_idx]
                # var_idx = (im+j)*r + c
                # eq_idx corresponds to (i, j, j', k', k'', i'')
                
                # Let's just loop and fill
                A_U = np.zeros((576, n*m*r))
                b_U = np.zeros(576)
                
                eq_idx = 0
                for i in range(n):
                    for j in range(m):
                        for jp in range(m): # j'
                            for kp in range(p): # k'
                                for kpp in range(p): # k''
                                    for ip in range(n): # i''
                                        # RHS
                                        if i == ip and j == jp and kp == kpp:
                                            b_U[eq_idx] = 1
                                        else:
                                            b_U[eq_idx] = 0
                                        
                                        # Coeffs for U[im+j, c]
                                        # V[j'p+k', c] * W[k''n+i'', c]
                                        v_idx = jp * p + kp
                                        w_idx = kpp * n + ip
                                        u_idx = i * m + j
                                        
                                        for c in range(r):
                                            A_U[eq_idx, u_idx * r + c] = V[v_idx, c] * W[w_idx, c]
                                        
                                        eq_idx += 1
                
                # Solve A_U u = b_U
                # u is flattened U
                try:
                    u_sol, residuals, rank, s = np.linalg.lstsq(A_U, b_U, rcond=None)
                    U = u_sol.reshape(n*m, r)
                except:
                    break
                
                # Similarly for V
                A_V = np.zeros((576, m*p*r))
                b_V = np.zeros(576)
                eq_idx = 0
                for i in range(n):
                    for j in range(m):
                        for jp in range(m):
                            for kp in range(p):
                                for kpp in range(p):
                                    for ip in range(n):
                                        if i == ip and j == jp and kp == kpp:
                                            b_V[eq_idx] = 1
                                        else:
                                            b_V[eq_idx] = 0
                                        
                                        v_idx = jp * p + kp
                                        w_idx = kpp * n + ip
                                        u_idx = i * m + j
                                        
                                        for c in range(r):
                                            A_V[eq_idx, v_idx * r + c] = U[u_idx, c] * W[w_idx, c]
                                        
                                        eq_idx += 1
                try:
                    v_sol, residuals, rank, s = np.linalg.lstsq(A_V, b_V, rcond=None)
                    V = v_sol.reshape(m*p, r)
                except:
                    break
                
                # Similarly for W
                A_W = np.zeros((576, p*n*r))
                b_W = np.zeros(576)
                eq_idx = 0
                for i in range(n):
                    for j in range(m):
                        for jp in range(m):
                            for kp in range(p):
                                for kpp in range(p):
                                    for ip in range(n):
                                        if i == ip and j == jp and kp == kpp:
                                            b_W[eq_idx] = 1
                                        else:
                                            b_W[eq_idx] = 0
                                        
                                        v_idx = jp * p + kp
                                        w_idx = kpp * n + ip
                                        u_idx = i * m + j
                                        
                                        for c in range(r):
                                            A_W[eq_idx, w_idx * r + c] = U[u_idx, c] * V[v_idx, c]
                                        
                                        eq_idx += 1
                try:
                    w_sol, residuals, rank, s = np.linalg.lstsq(A_W, b_W, rcond=None)
                    W = w_sol.reshape(p*n, r)
                except:
                    break
                
                # Check if close to half-integers
                # Round to nearest half-integer
                U_h = np.round(U * 2) / 2
                V_h = np.round(V * 2) / 2
                W_h = np.round(W * 2) / 2
                
                # Verify exactness with half-integers
                # Check residual
                # We can compute the tensor product and compare to T
                # To save time, we can just check the norm of (U_h V_h W_h - T)
                # But computing T is expensive.
                # Instead, check if the residual of the linear system is small with the snapped values?
                # Or just check a few random equations?
                
                # Better: Check the maximum error
                # We can reconstruct the tensor for a subset of indices?
                # Or, since we solved least squares, the residual should be small.
                # If the snapped values are close to the solution, they should be a valid solution.
                
                # Let's check if the error is < 1e-9
                # We can compute the error for all 576 equations using the snapped values
                # This is fast: 576 * r operations.
                
                err = 0
                eq_idx = 0
                for i in range(n):
                    for j in range(m):
                        for jp in range(m):
                            for kp in range(p):
                                for kpp in range(p):
                                    for ip in range(n):
                                        target = 1.0 if (i == ip and j == jp and kp == kpp) else 0.0
                                        
                                        val = 0
                                        v_idx = jp * p + kp
                                        w_idx = kpp * n + ip
                                        u_idx = i * m + j
                                        
                                        for c in range(r):
                                            val += U_h[u_idx, c] * V_h[v_idx, c] * W_h[w_idx, c]
                                        
                                        err = max(err, abs(val - target))
                                        
                                        eq_idx += 1
                
                if err < 1e-9:
                    # Found a solution
                    # Convert to list of lists for return
                    U_out = U_h.tolist()
                    V_out = V_h.tolist()
                    W_out = W_h.tolist()
                    return {"U": U_out, "V": V_out, "W": W_out}
                    
            # End restart loop
        # End rank loop
        
    # If no solution found, return schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
