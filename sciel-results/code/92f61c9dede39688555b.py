import numpy as np

def construct():
    """
    5x5 matrix multiplication - randomized algebraic rank-reduction heuristic.
    Attempts to find a rank-r decomposition (r < 125) with integer/half-integer entries
    by iteratively projecting onto the null space of the tensor flattening matrix.
    Falls back to schoolbook if no improvement found within time budget.
    """
    import random
    import time
    
    n, m, p = 5, 5, 5
    t_start = time.time()
    time_budget = 100  # seconds, leave margin for 120s limit
    
    def time_remaining():
        return time_budget - (time.time() - t_start)
    
    # Build the tensor flattening matrix M of shape (n*p, n*m*m*p)
    # M[k*n + i, (i2*m + j2)*m*p + (j*p + k2)] = 1 if i==i2, j==j2, k==k2
    # Actually, let's use the more direct Kronecker structure.
    # The condition is: sum_j A[i,j] B[j,k] = sum_c W[k*n+i,c] * (U^T a)_c * (V^T b)_c
    # This must hold for ALL A, B. So we need:
    # For each (i,k), the bilinear form L_{i,k}(a,b) = sum_j A[i,j]B[j,k] must equal
    # sum_c W[k*n+i,c] * L_c(a) * M_c(b)
    
    # The matrix M flattening: M[output_idx, input_idx] where
    # output_idx = k*n + i
    # input_idx encodes the product of U and V column vectors.
    
    # Actually, the standard approach: the tensor T has entries T[i,j,k] = 1.
    # A rank-r decomposition exists iff the tensor T can be written as sum_c U_c ⊗ V_c ⊗ W_c
    # where U_c is in R^{n*m}, V_c in R^{m*p}, W_c in R^{p*n}.
    
    # The flattening matrix M has shape (n*p, n*m * m*p) where
    # M[k*n+i, (im+j)*(m*p) + (jp+k')] = 1 if i==im, j==jp, k==k'... 
    # No, that's not quite right. Let me think more carefully.
    
    # The bilinear map: (a, b) -> c where c[i,k] = sum_j a[i,j] b[j,k]
    # This is a linear map from R^{n*m} ⊗ R^{m*p} to R^{n*p}
    # The matrix M: R^{n*m * m*p} -> R^{n*p}
    # M[k*n+i, (i*m+j)*(m*p) + (j*p+k')] = 1 if i==i, j==j, k==k'... 
    # Actually: M[k*n+i, idx] where idx = (i*m+j)*(m*p) + (j*p+k')
    # The entry is 1 iff the (i,j) index of a and (j',k') index of b match j==j' and k==k'
    # Wait, no. c[i,k] = sum_j a[i,j] b[j,k]. So for a fixed output (i,k),
    # the input indices that contribute are (i,j) from a and (j,k) from b for each j.
    # So M[k*n+i, (i*m+j)*(m*p) + (j*p+k)] = 1 for each j.
    
    # Let me build M properly.
    # M shape: (n*p, n*m * m*p)
    # Row index: k*n + i
    # Column index: (i_a * m + j_a) * (m*p) + (j_b * p + k_b)
    # M[row, col] = 1 if i_a == i and j_a == j_b and k_b == k, i.e., the a-index is (i,j) and b-index is (j,k)
    
    M = np.zeros((n * p, n * m * m * p), dtype=float)
    for i in range(n):
        for k in range(p):
            for j in range(m):
                row = k * n + i
                col = (i * m + j) * (m * p) + (j * p + k)
                M[row, col] = 1.0
    
    # Now, a rank-r decomposition corresponds to choosing r vectors u_c in R^{n*m}, v_c in R^{m*p}, w_c in R^{p*n}
    # such that for all (i,k): sum_j a[i,j]b[j,k] = sum_c w_c[k*n+i] * (u_c^T a) * (v_c^T b)
    
    # Equivalently, the matrix M can be "factored" in a specific way. The condition is that
    # M = sum_c (w_c ⊗ (u_c ⊗ v_c^T)) ... no, that's not the right structure.
    
    # Let me re-derive. We need:
    # c[i,k] = sum_j a[i,j] b[j,k]
    # = sum_c w[k*n+i, c] * (sum_{a,b} U[a*m+b, c] a[a,b]) * (sum_{b',d} V[b'*p+d, c] b[b',d])
    
    # For this to hold for all a, b, we need the coefficient of each a[x,y] b[z,w] to match.
    # LHS coefficient of a[x,y] b[z,w]: 1 if x==i, y==z, w==k, else 0 (for output c[i,k])
    # RHS: sum_c w[k*n+i,c] * U[x*m+y, c] * V[z*p+w, c]
    
    # So the condition is: for all (i,k,x,y,z,w):
    # sum_c W[k*n+i,c] U[x*m+y,c] V[z*p+w,c] = [x==i and y==z and w==k]
    
    # This means the tensor T[i,k,x,y,z,w] = [x==i and y==z and w==k] must be decomposable.
    
    # The flattening approach: consider M as a matrix. A rank-r decomposition of the
    # matrix multiplication tensor corresponds to M being expressible as a sum of r rank-1
    # "structured" terms. But actually, the null space approach is:
    
    # If we fix U and V, then W is determined (up to null space) by the linear equations.
    # Or if we fix two of them, the third is constrained.
    
    # Strategy: randomize U and V, then for each row of W, solve the linear system.
    # The constraint for W[k*n+i, :] is:
    # For all (x,y,z,w): sum_c U[x*m+y,c] V[z*p+w,c] W[k*n+i,c] = [x==i and y==z and w==k]
    
    # This is a linear system in W[k*n+i, c] for c=1..r.
    # Number of unknowns: r
    # Number of equations: n*m*m*p (all combinations of x,y,z,w)
    # But many are redundant. The effective number of independent equations is n*p*m (for each i,k, the j values).
    
    # Actually, for a fixed (i,k), the equation becomes:
    # sum_j (sum_c U[i*m+j,c] W[k*n+i,c]) * (sum_c V[j*p+k,c] ... ) 
    # No, that's not separable that way.
    
    # Let me try a different approach. For each (i,k), define:
    # For each j, we need: the coefficient of a[i,j] b[j,k] in the RHS to be 1, and all other coefficients to be 0.
    
    # The RHS expanded:
    # sum_c W[k*n+i,c] * (sum_x,y U[x*m+y,c] a[x,y]) * (sum_z,w V[z*p+w,c] b[z,w])
    # = sum_x,y,z,w (sum_c W[k*n+i,c] U[x*m+y,c] V[z*p+w,c]) a[x,y] b[z,w]
    
    # So for output (i,k), we need:
    # sum_c W[k*n+i,c] U[x*m+y,c] V[z*p+w,c] = 1 if (x,y,z,w) = (i,j,j,k) for some j, else 0
    
    # For a fixed (i,k), this gives a system of equations. Let's define the matrix A_{i,k} of shape (n*m*m*p, r)
    # where A_{i,k}[idx, c] = U[x*m+y,c] * V[z*p+w,c] where idx encodes (x,y,z,w)
    # And the target vector b_{i,k}[idx] = 1 if (x,y,z,w) corresponds to (i,j,j,k) for some j, else 0
    
    # Then we need A_{i,k} W[k*n+i, :] = b_{i,k}
    
    # This is an overdetermined system (n*m*m*p equations, r unknowns).
    # We need a W row that satisfies this. If r is large enough, a solution exists.
    
    # For the algorithm:
    # 1. Randomly initialize U, V with small integers.
    # 2. For each (i,k), solve A_{i,k} w = b_{i,k} using least squares.
    # 3. Check if the solution is exact (residual ~ 0).
    # 4. If yes, round to nearest integer/half-integer and verify.
    # 5. If no exact solution, perturb and retry.
    
    # But this is expensive. Let's try a simpler approach:
    # Use the fact that for schoolbook, U, V, W are all 0/1. 
    # Try to find a lower rank by looking for linear dependencies.
    
    # Given the complexity and time constraints, let's try a direct approach:
    # For a given rank r, randomly generate U, V, and try to solve for W.
    # Start with r close to 125 and try to reduce.
    
    # Actually, let's try a more direct randomized approach:
    # For rank r, generate random integer U, V of appropriate shapes.
    # For each (i,k), form the linear system and solve.
    # Check if solution exists and is integral/half-integral.
    
    best_rank = n * m * p  # start with schoolbook
    best_U = best_V = best_W = None
    
    # Build schoolbook as fallback
    U_s = np.zeros((n*m, best_rank), dtype=float)
    V_s = np.zeros((m*p, best_rank), dtype=float)
    W_s = np.zeros((p*n, best_rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i*m+j, col] = 1.0
                V_s[j*p+k, col] = 1.0
                W_s[k*n+i, col] = 1.0
                col += 1
    best_U, best_V, best_W = U_s.copy(), V_s.copy(), W_s.copy()
    
    # Try to find lower rank
    for r in range(best_rank - 1, max(20, best_rank - 50), -1):
        if time_remaining() < 5:
            break
        
        for attempt in range(50):
            if time_remaining() < 2:
                break
            
            random.seed(attempt * 1000 + r)
            np.random.seed(attempt * 1000 + r)
            
            # Random integer initialization
            U_r = np.random.randint(-2, 3, size=(n*m, r))
            V_r = np.random.randint(-2, 3, size=(m*p, r))
            
            success = True
            W_r = np.zeros((p*n, r), dtype=float)
            
            for i in range(n):
                for k in range(p):
                    if time_remaining() < 1:
                        success = False
                        break
                    
                    # Build the linear system for W[k*n+i, :]
                    # For each (x,y,z,w), equation: sum_c U[x*m+y,c]*V[z*p+w,c]*W[k*n+i,c] = target
                    # Target = 1 if x==i and y==z and w==k, else 0
                    # But we only need equations where target is 1, and check others are 0.
                    
                    # Number of equations: n*m*m*p, but we can use just the ones with target=1
                    # and verify the rest.
                    
                    eq_indices = []
                    for j in range(m):
                        eq_indices.append((i, j, j, k))
                    
                    # Form the system: for each (x,y,z,w) in eq_indices, 
                    # sum_c U[x*m+y,c]*V[z*p+w,c]*W[c] = 1
                    # This gives m equations in r unknowns.
                    # If r > m, there are many solutions.
                    
                    A_sys = np.zeros((m, r))
                    for jj in range(m):
                        x, y, z, w = i, jj, jj, k
                        for c in range(r):
                            A_sys[jj, c] = U_r[x*m+y, c] * V_r[z*p+w, c]
                    
                    # Solve A_sys @ W_row = ones(m)
                    try:
                        sol, residuals, rank_sys, sv = np.linalg.lstsq(A_sys, np.ones(m), rcond=None)
                        if residuals is not None and len(residuals) > 0:
                            if residuals[0] > 1e-6:
                                success = False
                                break
                        W_r[k*n+i, :] = sol
                    except:
                        success = False
                        break
                
                if not success:
                    break
            
            if not success:
                continue
            
            # Verify the full constraint
            # Check all (i,k,x,y,z,w)
            max_err = 0
            for i in range(n):
                for k in range(p):
                    for x in range(n):
                        for y in range(m):
                            for z in range(m):
                                for w in range(p):
                                    expected = 1.0 if (x == i and y == z and w == k) else 0.0
                                    actual = 0.0
                                    for c in range(r):
                                        actual += W_r[k*n+i, c] * U_r[x*m+y, c] * V_r[z*p+w, c]
                                    max_err = max(max_err, abs(actual - expected))
            
            if max_err < 1e-6:
                # Round to nearest half-integer
                U_r = np.round(U_r * 2) / 2
                V_r = np.round(V_r * 2) / 2
                W_r = np.round(W_r * 2) / 2
                
                # Re-verify after rounding
                max_err2 = 0
                for i in range(n):
                    for k in range(p):
                        for x in range(n):
                            for y in range(m):
                                for z in range(m):
                                    for w in range(p):
                                        expected = 1.0 if (x == i and y == z and w == k) else 0.0
                                        actual = 0.0
                                        for c in range(r):
                                            actual += W_r[k*n+i, c] * U_r[x*m+y, c] * V_r[z*p+w, c]
                                        max_err2 = max(max_err2, abs(actual - expected))
                
                if max_err2 < 1e-9:
                    best_rank = r
                    best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
                    break  # break out of attempt loop, try lower r
        
        if time_remaining() < 5:
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
