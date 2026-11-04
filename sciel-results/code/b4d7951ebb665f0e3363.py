import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s kill
    
    n, m, p = 2, 4, 8
    
    # Try to find a low-rank decomposition
    # The constraint: for all i in [n], j in [m], k in [p]:
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Start with schoolbook and try to reduce rank
    # Try various ranks from low to high
    
    best_result = None
    best_rank = n * m * p  # 64
    
    # First, create the schoolbook as fallback
    U_school = [[0] * best_rank for _ in range(n * m)]
    V_school = [[0] * best_rank for _ in range(m * p)]
    W_school = [[0] * best_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    
    # Try to find lower rank solutions
    # Use the structure: fix U and V, solve for W, or vice versa
    
    # Approach: For a given rank r, we have 56r variables and 64 constraints.
    # Try to find a solution with integer/half-integer entries.
    
    # Let's try a specific constructive approach:
    # Decompose using the structure of the indices.
    
    # For i in {0,1}, j in {0,1,2,3}, k in {0,...,7}
    # Write j = 2*j2 + j1, k = 4*k2 + k1
    
    # C[i, 4*k2+k1] = sum_{j2,j1} A[i, 2*j2+j1] * B[2*j2+j1, 4*k2+k1]
    
    # For each (i, k2, k1), we have 4 terms.
    # Total: 2 * 2 * 4 * 4 = 64 terms (same as schoolbook)
    
    # To reduce rank, we need to find linear dependencies.
    
    # Let me try: for fixed i and k2, consider the 4x4 matrix M[j1, k1] = B[2*j2+j1, 4*k2+k1]
    # This doesn't directly help.
    
    # Alternative: use the fact that we can share computation across k values.
    
    # Let me try a numerical approach: set up the problem as a system of equations
    # and use least squares or SVD to find a low-rank solution.
    
    # For rank r, we can write:
    # Let U be 8xr, V be 32xr, W be 16xr
    # Constraint: for each (i,j,k): sum_c U[i*4+j,c] V[j*8+k,c] W[k*2+i,c] = 1
    
    # This is a tensor equation. Let me flatten it.
    
    # Actually, let me try a different approach: use the fact that for each (i,j),
    # the vector v_{i,j} in R^p defined by v_{i,j}[k] = 1 must be expressed as
    # sum_c U[i*4+j,c] * (V[j*8:,:c] * W[:,k*2+i,c] element-wise)
    
    # This is getting complicated. Let me just try to find a solution via optimization.
    
    # Try rank 56:
    for r in [56, 52, 48, 44, 40, 36, 32, 28, 24, 20, 16, 12, 8, 64]:
        if time.time() > deadline - 10:
            break
            
        # Initialize with random values and try to solve
        # Use the schoolbook as a starting point and zero out some columns
        
        if r >= 64:
            # Use schoolbook
            U = [[0]*r for _ in range(n*m)]
            V = [[0]*r for _ in range(m*p)]
            W = [[0]*r for _ in range(p*n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if col < r:
                            U[i*m+j][col] = 1
                            V[j*p+k][col] = 1
                            W[k*n+i][col] = 1
                            col += 1
            # Verify
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for c in range(r):
                            s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                        if abs(s - 1) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if valid:
                if best_result is None or r < best_rank:
                    best_rank = r
                    best_result = {"U": U, "V": V, "W": W}
            continue
        
        # For r < 64, try to find a solution
        # Use a greedy/optimization approach
        
        # Initialize with schoolbook entries in first r columns, then try to optimize
        U = [[0]*r for _ in range(n*m)]
        V = [[0]*r for _ in range(m*p)]
        W = [[0]*r for _ in range(p*n)]
        
        # Put schoolbook terms in
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if col < r:
                        U[i*m+j][col] = 1
                        V[j*p+k][col] = 1
                        W[k*n+i][col] = 1
                        col += 1
                    else:
                        break
                if col >= r:
                    break
            if col >= r:
                break
        
        # Now we have r of the 64 terms. Need to adjust to satisfy all 64 constraints.
        # This is a hard problem. Let me try a different initialization.
        
        # Use random initialization and optimize
        np.random.seed(42)
        U_np = np.random.randn(n*m, r) * 0.1
        V_np = np.random.randn(m*p, r) * 0.1
        W_np = np.random.randn(p*n, r) * 0.1
        
        # Define the objective: minimize sum of (constraint - 1)^2
        # Constraint for (i,j,k): sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1
        
        def compute_residuals(U_np, V_np, W_np):
            residuals = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(r):
                            s += U_np[i*m+j, c] * V_np[j*p+k, c] * W_np[k*n+i, c]
                        residuals[i,j,k] = s - 1.0
            return residuals
        
        # This is a non-convex optimization problem. Let me try gradient descent.
        # But with 56r variables and only 64 constraints, we might be able to find a solution.
        
        # Actually, let me try a smarter approach: for each (i,j,k), the constraint is
        # a trilinear form. Let me fix two of the matrices and solve for the third.
        
        # Fix U and V, solve for W:
        # For each (i,j,k): sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1
        # For fixed (i,j): sum_c (U[i*m+j,c] V[j*p+k,c]) W[k*n+i,c] = 1 for all k
        # This is a linear system in W[k*n+i, :] for each (i,j)
        
        # Actually, for fixed i, the equations for all j,k give:
        # sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1
        
        # Let me group by i. For i=0:
        # For j=0: sum_c U[0,c] V[k,c] W[k*2,c] = 1 for k=0..7  (8 equations)
        # For j=1: sum_c U[1,c] V[8+k,c] W[k*2,c] = 1 for k=0..7  (8 equations)
        # For j=2: sum_c U[2,c] V[16+k,c] W[k*2,c] = 1 for k=0..7  (8 equations)
        # For j=3: sum_c U[3,c] V[24+k,c] W[k*2,c] = 1 for k=0..7  (8 equations)
        # Total 32 equations for W[0:16,:] (16*r variables)
        
        # Similarly for i=1: 32 equations for W[16:32,:] (16*r variables)
        
        # So if I fix U and V, I can solve for W via linear systems!
        
        # Let me try: fix U and V to be the schoolbook values (first r columns),
        # then solve for W.
        
        U_fixed = [[0]*r for _ in range(n*m)]
        V_fixed = [[0]*r for _ in range(m*p)]
        
        # Use schoolbook for U and V in first r columns
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if col < r:
                        U_fixed[i*m+j][col] = 1
                        V_fixed[j*p+k][col] = 1
                        col += 1
                    else:
                        break
                if col >= r:
                    break
            if col >= r:
                break
        
        # Now solve for W
        # For each i in {0,1}, we have a linear system:
        # For each j in {0,1,2,3}, k in {0,...,7}:
        #   sum_c U_fixed[i*m+j][c] * V_fixed[j*p+k][c] * W[k*n+i][c] = 1
        
        W_sol = [[0]*r for _ in range(p*n)]
        solvable = True
        
        for i in range(n):
            # Build the linear system for W[k*n+i, :] for k=0..p-1
            # Unknowns: W[0*n+i, :], W[1*n+i, :], ..., W[(p-1)*n+i, :]
            # That's p vectors of length r, so p*r unknowns
            # Equations: m*p = 32 equations
            
            # A is (m*p) x (p*r), b is (m*p,)
            A = np.zeros((m*p, p*r))
            b = np.ones(m*p)
            
            eq_idx = 0
            for j in range(m):
                for k in range(p):
                    for c in range(r):
                        A[eq_idx, k*r + c] = U_fixed[i*m+j][c] * V_fixed[j*p+k][c]
                    eq_idx += 1
            
            # Solve A x = b where x is the flattened W for this i
            try:
                x, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                # Check if solution is good
                res = np.max(np.abs(A @ x - b))
                if res < 1e-6:
                    for k in range(p):
                        for c in range(r):
                            W_sol[k*n+i][c] = x[k*r + c]
                else:
                    solvable = False
            except:
                solvable = False
        
        if solvable:
            # Verify the solution
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for c in range(r):
                            s += U_fixed[i*m+j][c] * V_fixed[j*p+k][c] * W_sol[k*n+i][c]
                        if abs(s - 1) > 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                # Round to nearest half-integer if possible
                U_out = [[round(U_fixed[i][c] * 2) / 2 for c in range(r)] for i in range(n*m)]
                V_out = [[round(V_fixed[i][c] * 2) / 2 for c in range(r)] for i in range(m*p)]
                W_out = [[round(W_sol[i][c] * 2) / 2 for c in range(r)] for i in range(p*n)]
                
                # Verify rounded solution
                valid_rounded = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0
                            for c in range(r):
                                s += U_out[i*m+j][c] * V_out[j*p+k][c] * W_out[k*n+i][c]
                            if abs(s - 1) > 1e-6:
                                valid_rounded = False
                                break
                        if not valid_rounded:
                            break
                    if not valid_rounded:
                        break
                
                if valid_rounded:
                    if best_result is None or r < best_rank:
                        best_rank = r
                        best_result = {"U": U_out, "V": V_out, "W": W_out}
                else:
                    # Use unrounded if it's exact enough
                    if best_result is None or r < best_rank:
                        best_rank = r
                        best_result = {"U": U_fixed, "V": V_fixed, "W": W_sol}
        
        if time.time() > deadline - 5:
            break
    
    if best_result is not None:
        return best_result
    else:
        return {"U": U_school, "V": V_school, "W": W_school}
