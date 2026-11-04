"""Tensor decomposition for 4x5 x 5x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def construct():
    """
    Construct a rank-minimal (or near-minimal) exact tensor decomposition
    for multiplying a 4x5 matrix by a 5x6 matrix.
    
    Returns dict with U (20 x rank), V (30 x rank), W (24 x rank).
    The decomposition satisfies: for all i in [0,n), k in [0,p),
    sum_c U[i*m + i2, c] * V[i2*p + k, c] * W[k*n + i, c] = delta(i2 exists)
    Actually the convention is: (AB)_{i,k} = sum_j A_{i,j} B_{j,k}
    Tensor T[i,j,k] = 1 if all indices valid, and decomposition is:
    T = sum_c U[:,c] V[:,c] W[:,c] where:
    U has shape (n*m, rank) indexed by (i, j)
    V has shape (m*p, rank) indexed by (j, k)
    W has shape (p*n, rank) indexed by (k, i)
    """
    start_time = time.time()
    DEADLINE = 135  # seconds, with safety margin
    
    n, m, p = DIMENSIONS  # 4, 5, 6
    
    # First, get the schoolbook decomposition as a baseline
    rank_schoolbook = n * m * p  # 120
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    # Convert to lists for return format
    def mat_to_list(M):
        return [[float(M[i, j]) for j in range(M.shape[1])] for i in range(M.shape[0])]
    
    # Try to find a lower rank decomposition using numerical optimization
    # Strategy: Use iterative tensor decomposition (CP decomposition) with
    # exact integer/rational recovery
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_schoolbook
    
    # Try various rank values from low to high
    # Theoretical lower bound: rank >= max(n*m, m*p, p*n) = max(20, 30, 24) = 30
    # Known results suggest rank for 4x5x6 is likely around 40-60
    
    # Use ALS (Alternating Least Squares) for CP decomposition, then try to
    # snap to rational values
    
    def try_rank(R, max_iters=2000, tol=1e-10, n_restarts=3):
        """Try to find a rank-R CP decomposition of the matrix multiplication tensor."""
        if time.time() > DEADLINE:
            return None
        
        # Create the tensor in flattened form
        # T[i,j,k] = 1 for all valid i,j,k
        # Flatten: T_flat[i*m + j, k] gives a 2D slice
        # Actually, let's work with the tensor directly
        
        # The tensor T has dimensions (n, m, p) with T[i,j,k] = 1
        # We want T = sum_c U_c V_c W_c where:
        # U_c is a vector of length n*m (indexed by (i,j))
        # V_c is a vector of length m*p (indexed by (j,k))  
        # W_c is a vector of length p*n (indexed by (k,i))
        
        # Build the tensor as a 3D array
        T = np.ones((n, m, p))
        
        # Flatten dimensions for CP: T[i,j,k] = sum_c U[(i,j),c] * V[(j,k),c] * W[(k,i),c]
        # This is NOT a standard CP decomposition because the index pairings are different
        # U pairs (i,j), V pairs (j,k), W pairs (k,i)
        
        # Let's set up the problem properly
        # For each (i,j,k), we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k] = 1
        
        # This is a constrained tensor decomposition. Let's use a different approach.
        
        # Build constraint matrix: for each (i,j,k), the equation is linear in the entries
        # if we fix two of the three matrices. Use ALS-style approach.
        
        # Initialize
        best_cost = np.inf
        best_solution = None
        
        for restart in range(n_restarts):
            np.random.seed(42 + restart)
            
            U = np.random.randn(n * m, R)
            V = np.random.randn(m * p, R)
            W = np.random.randn(p * n, R)
            
            # Normalize to avoid scale issues
            for c in range(R):
                norm = np.sqrt(np.sum(U[:, c]**2) + np.sum(V[:, c]**2) + np.sum(W[:, c]**2))
                if norm > 1e-10:
                    U[:, c] /= norm
                    V[:, c] /= norm
                    W[:, c] /= norm
            
            for it in range(max_iters):
                if time.time() > DEADLINE:
                    return None
                
                # Update U: for each (i,j) row, solve least squares
                # For fixed i,j: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all k
                # This is: U[i*m+j, :] @ (V * W) for appropriate indices = ones(p)
                
                U_new = np.zeros_like(U)
                for i in range(n):
                    for j in range(m):
                        # Build matrix M where M[k, c] = V[j*p+k, c] * W[k*n+i, c]
                        M = np.zeros((p, R))
                        for k in range(p):
                            M[k, :] = V[j * p + k, :] * W[k * n + i, :]
                        # Solve: U[i*m+j, :] @ M = ones(p)
                        try:
                            sol, residuals, rank_m, sv = np.linalg.lstsq(M.T, np.ones(p), rcond=None)
                            U_new[i * m + j, :] = sol
                        except:
                            U_new[i * m + j, :] = U[i * m + j, :]
                
                # Update V
                V_new = np.zeros_like(V)
                for j in range(m):
                    for k in range(p):
                        # Build matrix M where M[i, c] = U[i*m+j, c] * W[k*n+i, c]
                        M = np.zeros((n, R))
                        for i in range(n):
                            M[i, :] = U[i * m + j, :] * W[k * n + i, :]
                        try:
                            sol, residuals, rank_m, sv = np.linalg.lstsq(M.T, np.ones(n), rcond=None)
                            V_new[j * p + k, :] = sol
                        except:
                            V_new[j * p + k, :] = V[j * p + k, :]
                
                # Update W
                W_new = np.zeros_like(W)
                for k in range(p):
                    for i in range(n):
                        # Build matrix M where M[j, c] = U[i*m+j, c] * V[j*p+k, c]
                        M = np.zeros((m, R))
                        for j in range(m):
                            M[j, :] = U[i * m + j, :] * V[j * p + k, :]
                        try:
                            sol, residuals, rank_m, sv = np.linalg.lstsq(M.T, np.ones(m), rcond=None)
                            W_new[k * n + i, :] = sol
                        except:
                            W_new[k * n + i, :] = W[k * n + i, :]
                
                # Check convergence
                cost = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                            cost += (val - 1.0) ** 2
                
                if cost < best_cost:
                    best_cost = cost
                    best_solution = (U_new.copy(), V_new.copy(), W_new.copy())
                
                U, V, W = U_new, V_new, W_new
                
                if cost < tol:
                    break
            
            if best_cost < tol:
                break
        
        if best_cost < 1e-6:
            # Try to snap to rational values
            U_s, V_s, W_s = best_solution
            U_rat, V_rat, W_rat = _snap_to_rational(U_s), _snap_to_rational(V_s), _snap_to_rational(W_s)
            
            # Verify the rational solution
            if _verify(U_rat, V_rat, W_rat, n, m, p):
                return (U_rat, V_rat, W_rat)
            else:
                # Return the numerical solution if it's close enough
                # But the problem requires exact decomposition...
                # Let's try harder to snap
                for den in [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 30, 40, 48, 60]:
                    U_rat2 = _snap_to_rational_den(U_s, den)
                    V_rat2 = _snap_to_rational_den(V_s, den)
                    W_rat2 = _snap_to_rational_den(W_s, den)
                    if _verify(U_rat2, V_rat2, W_rat2, n, m, p):
                        return (U_rat2, V_rat2, W_rat2)
        
        return None
    
    def _snap_to_rational(M, max_den=10):
        """Snap matrix entries to nearby rational numbers."""
        M_s = M.copy()
        for i in range(M_s.shape[0]):
            for j in range(M_s.shape[1]):
                x = M_s[i, j]
                best_val = x
                best_err = abs(x - round(x))
                for den in range(1, max_den + 1):
                    num = round(x * den)
                    val = num / den
                    err = abs(x - val)
                    if err < best_err:
                        best_err = err
                        best_val = val
                M_s[i, j] = best_val
        return M_s
    
    def _snap_to_rational_den(M, den):
        """Snap matrix entries to multiples of 1/den."""
        M_s = M.copy()
        for i in range(M_s.shape[0]):
            for j in range(M_s.shape[1]):
                M_s[i, j] = round(M_s[i, j] * den) / den
        return M_s
    
    def _verify(U, V, W, n, m, p, tol=1e-8):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(U.shape[1]):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Try ranks from low to high
    # Start with a reasonable guess; the schoolbook is 120
    # Try to find something better
    for R in [35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 110, 120]:
        if time.time() > DEADLINE - 10:
            break
        
        result = try_rank(R)
        if result is not None:
            U_found, V_found, W_found = result
            # Convert to lists
            U_list = mat_to_list(U_found)
            V_list = mat_to_list(V_found)
            W_list = mat_to_list(W_found)
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # If we couldn't find a lower rank, return the schoolbook decomposition
    return {"U": mat_to_list(U_sb), "V": mat_to_list(V_sb), "W": mat_to_list(W_sb)}

# EVOLVE-BLOCK-END
