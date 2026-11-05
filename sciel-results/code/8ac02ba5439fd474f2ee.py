import numpy as np
from itertools import product
import time

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank as fallback
    rank_school = n * m * p
    
    def verify(U, V, W):
        """Verify that U, V, W give an exact decomposition."""
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        r = U.shape[1]
        # Build the tensor from the factors
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        # For each c, the contribution is outer product structure
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        # Check a few random entries and also all entries for exactness
        # Since we need exact, let's check all entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = 0.0
                    for c in range(r):
                        actual += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(actual - expected) > 1e-10:
                        return False
        return True
    
    # Try to find minimal rank
    # Lower bound: max(n*m, m*p, p*n) = 30
    lower_bound = max(n*m, m*p, p*n)
    
    best_rank = rank_school
    best_U = None
    best_V = None
    best_W = None
    
    # Build schoolbook as initial best
    U_sb = np.zeros((n*m, rank_school))
    V_sb = np.zeros((m*p, rank_school))
    W_sb = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_school
    
    # Try to find lower rank decompositions
    # Strategy: for a given rank r, randomly initialize and try to solve
    # The equations are: for all (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    # We can use the approach: fix W randomly, then for each (i,k), 
    # we need sum_c W[k*n+i,c] * U[i*m+j,c] * V[j*p+k,c] = 1 for all j
    # This is a bilinear system in U and V for fixed W.
    
    # Alternative: use random linear combinations to reduce to a matrix problem
    # For fixed random vectors a (length n*m), b (length m*p), c (length p*n):
    # sum_{i,j,k} a[i*m+j]*b[j*p+k]*c[k*n+i]*T[i,j,k] = sum_c (sum_{i,j} a[i*m+j]*U[i*m+j,c]) * (sum_{j,k} b[j*p+k]*V[j*p+k,c]) * (sum_{i,k} c[k*n+i]*W[k*n+i,c])
    # This must equal sum_c (a^T U_c)(b^T V_c)(c^T W_c) for the decomposition to work
    # And it should equal sum_{i,j,k} a[i*m+j]*b[j*p+k]*c[k*n+i] = (sum_i a[i*m+j] for appropriate) ... 
    
    # Actually, the contraction of T with a,b,c gives:
    # sum_{i,j,k} a[i*m+j]*b[j*p+k]*c[k*n+i]*T[i,j,k]
    # = sum_{i,j,k} a[i*m+j]*b[j*p+k]*c[k*n+i]  (since T[i,j,k]=1 for all valid i,j,k)
    # = (sum_i sum_j a[i*m+j]) * ... no, it's not that simple because the indices are coupled
    
    # Let me just do a direct search with random initialization and Newton's method or similar
    
    deadline = time.time() + 100  # 100 second deadline
    
    for r in range(lower_bound, best_rank):
        if time.time() > deadline:
            break
        # Try multiple random initializations
        max_attempts = 200
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Random initialization
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            W = np.random.randn(p*n, r)
            
            # Try to optimize: minimize sum of squared errors
            # Error = sum_{i,j,k} (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
            
            # Use gradient descent or L-BFGS
            def objective(params):
                U_p = params[:n*m*r].reshape(n*m, r)
                V_p = params[n*m*r:n*m*r+m*p*r].reshape(m*p, r)
                W_p = params[n*m*r+m*p*r:].reshape(p*n, r)
                
                # Compute all entries
                errors = []
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for c in range(r):
                                val += U_p[i*m+j, c] * V_p[j*p+k, c] * W_p[k*n+i, c]
                            errors.append(val - 1.0)
                return sum(e*e for e in errors)
            
            # This is too slow with pure Python. Let's use numpy vectorization.
            # Actually, let me reformulate.
            
            # For each c, define u_c = U[:,c] (length n*m), v_c = V[:,c] (length m*p), w_c = W[:,c] (length p*n)
            # The (i,j,k) entry is sum_c u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i]
            
            # Let's use a different approach: for fixed V and W, the problem is linear in U
            # For each (i,j,k): sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1
            # So for fixed V,W, we have a linear system in U.
            
            # Similarly, for fixed U,W, linear in V. For fixed U,V, linear in W.
            # Use alternating least squares!
            
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            W = np.random.randn(p*n, r)
            
            converged = False
            for iteration in range(200):
                if time.time() > deadline:
                    break
                
                # Update U: for each (i,j), solve linear system
                # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k
                # For fixed (i,j), this is: U[i*m+j,:] * diag(V[j*p+k,:]*W[k*n+i,:]) = 1 for all k
                # i.e., U[i*m+j,:] * M_{ij} = ones(p) where M_{ij}[k,c] = V[j*p+k,c]*W[k*n+i,c]
                
                new_U = np.zeros((n*m, r))
                for i in range(n):
                    for j in range(m):
                        # M_{ij} is p x r: M_{ij}[k,c] = V[j*p+k,c] * W[k*n+i,c]
                        M_ij = np.zeros((p, r))
                        for k in range(p):
                            for c in range(r):
                                M_ij[k, c] = V[j*p+k, c] * W[k*n+i, c]
                        # Solve: new_U[i*m+j, :] * M_ij = ones(p)
                        # i.e., M_ij.T @ new_U[i*m+j, :].T = ones(p)
                        # Use least squares
                        try:
                            sol, residuals, rank_m, sv = np.linalg.lstsq(M_ij.T, np.ones(p), rcond=None)
                            new_U[i*m+j, :] = sol
                        except:
                            new_U[i*m+j, :] = U[i*m+j, :]
                U = new_U
                
                # Update V: for each (j,k), solve linear system
                # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i
                # For fixed (j,k): V[j*p+k,:] * M_{jk} = ones(n) where M_{jk}[i,c] = U[i*m+j,c]*W[k*n+i,c]
                
                new_V = np.zeros((m*p, r))
                for j in range(m):
                    for k in range(p):
                        M_jk = np.zeros((n, r))
                        for i in range(n):
                            for c in range(r):
                                M_jk[i, c] = U[i*m+j, c] * W[k*n+i, c]
                        try:
                            sol, residuals, rank_m, sv = np.linalg.lstsq(M_jk.T, np.ones(n), rcond=None)
                            new_V[j*p+k, :] = sol
                        except:
                            new_V[j*p+k, :] = V[j*p+k, :]
                V = new_V
                
                # Update W: for each (i,k), solve linear system
                # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all j
                # For fixed (i,k): W[k*n+i,:] * M_{ik} = ones(m) where M_{ik}[j,c] = U[i*m+j,c]*V[j*p+k,c]
                
                new_W = np.zeros((p*n, r))
                for i in range(n):
                    for k in range(p):
                        M_ik = np.zeros((m, r))
                        for j in range(m):
                            for c in range(r):
                                M_ik[j, c] = U[i*m+j, c] * V[j*p+k, c]
                        try:
                            sol, residuals, rank_m, sv = np.linalg.lstsq(M_ik.T, np.ones(m), rcond=None)
                            new_W[k*n+i, :] = sol
                        except:
                            new_W[k*n+i, :] = W[k*n+i, :]
                W = new_W
                
                # Check convergence
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for c in range(r):
                                val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            max_err = max(max_err, abs(val - 1.0))
                
                if max_err < 1e-10:
                    converged = True
                    break
                
                # Damping to help convergence
                U = 0.5 * U + 0.5 * new_U if False else new_U
                V = 0.5 * V + 0.5 * new_V if False else new_V
                W = 0.5 * W + 0.5 * new_W if False else new_W
            
            if converged:
                # Verify exactly
                if verify(U, V, W):
                    best_rank = r
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    break  # Found minimal rank for this r, try next lower
    
    # Convert to lists for output
    U_out = [[float(U_out_val) for U_out_val in row] for row in best_U.tolist()]
    V_out = [[float(V_out_val) for V_out_val in row] for row in best_V.tolist()]
    W_out = [[float(W_out_val) for W_out_val in row] for row in best_W.tolist()]
    
    return {"U": U_out, "V": V_out, "W": W_out}
