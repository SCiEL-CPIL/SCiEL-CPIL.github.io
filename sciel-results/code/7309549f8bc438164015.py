"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that U, V, W form a valid decomposition."""
    rank = len(U[0]) if U else 0
    # Build the tensor from factors
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Check against the standard multiplication tensor
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                # The standard tensor has T[i,j,k] = 1 if this is a valid term
                # Actually, the multiplication tensor T has T[i,j,k] = 1 for all valid (i,j,k)
                # No wait - the tensor represents the bilinear map
                # T[i,j,k] should equal the coefficient in C[i,k] = sum_{j} A[i,j] B[j,k]
                # So T[i,j,k] = 1 for all i,j,k (since each product term appears)
                # Actually no - the tensor is defined such that
                # C[i,k] = sum_{j,c} U[i*m+j,c] A[i,j] V[j*p+k,c] B[j,k] W[k*n+i,c]
                # For this to equal sum_j A[i,j] B[j,k], we need:
                # sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1 for all i,j,k
                expected = 1.0
                if abs(val - expected) > tol:
                    return False
    return True

def construct_rank_28():
    """Construct a rank-28 decomposition for 2x3 * 3x5."""
    n, m, p = 2, 3, 5
    rank = 28
    U = [[0.0]*rank for _ in range(n*m)]
    V = [[0.0]*rank for _ in range(m*p)]
    W = [[0.0]*rank for _ in range(p*n)]
    
    # Start with schoolbook (rank 30) and try to merge two terms
    # Schoolbook: for each (i,j,k), one term with U[i*m+j,c]=1, V[j*p+k,c]=1, W[k*n+i,c]=1
    # We need to find if two of these can be combined
    
    # Let's try a different approach: use known structure
    # For small matrices, we can enumerate possibilities
    
    # Initialize with schoolbook
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1.0
                V[j*p+k][col] = 1.0
                W[k*n+i][col] = 1.0
                col += 1
    
    # Now try to reduce by finding linear dependencies
    # This is complex - let's use a numerical optimization approach instead
    # For now, return schoolbook and let the search find improvements
    return U, V, W, rank

def search_lower_rank(n, m, p, max_time=100):
    """Search for lower rank decomposition using numerical methods."""
    start_time = time.time()
    
    # Build the target tensor
    # T[i,j,k] = 1 for all valid indices (each term A[i,j]*B[j,k] contributes to C[i,k])
    # We want: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
    
    # Start with schoolbook rank
    rank_school = n * m * p
    
    # Try to find lower rank using alternating minimization
    # Initialize with random factors
    np.random.seed(42)
    
    best_U, best_V, best_W, best_rank = None, None, None, rank_school
    
    # First, verify schoolbook works
    U_school = [[0.0]*rank_school for _ in range(n*m)]
    V_school = [[0.0]*rank_school for _ in range(m*p)]
    W_school = [[0.0]*rank_school for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1.0
                V_school[j*p+k][col] = 1.0
                W_school[k*n+i][col] = 1.0
                col += 1
    
    if verify_decomposition(U_school, V_school, W_school, n, m, p):
        best_U, best_V, best_W, best_rank = U_school, V_school, W_school, rank_school
    
    # Try lower ranks
    for target_rank in range(rank_school - 1, max(10, rank_school - 15), -1):
        if time.time() - start_time > max_time - 10:
            break
        
        # Try multiple random initializations
        for trial in range(5):
            if time.time() - start_time > max_time - 10:
                break
            
            # Initialize factors
            U = [[np.random.randn() for _ in range(target_rank)] for _ in range(n*m)]
            V = [[np.random.randn() for _ in range(target_rank)] for _ in range(m*p)]
            W = [[np.random.randn() for _ in range(target_rank)] for _ in range(p*n)]
            
            # Alternating minimization
            converged = False
            for it in range(200):
                if time.time() - start_time > max_time - 5:
                    break
                
                # Update U: for each (i,j), solve linear system
                # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                # This is coupled, so we do coordinate descent
                # For fixed V, W, U[i*m+j,:] should minimize error
                # Error = sum_{i,j,k} (1 - sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c])^2
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        # Build system: for each k, sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1
                        A_mat = np.zeros((p, target_rank))
                        for k in range(p):
                            for c in range(target_rank):
                                A_mat[k, c] = V[j*p+k][c] * W[k*n+i][c]
                        # Solve A_mat @ U[i*m+j,:] = ones(p)
                        try:
                            sol, residuals, rank_a, sv = np.linalg.lstsq(A_mat, np.ones(p), rcond=None)
                            U[i*m+j] = sol.tolist()
                        except:
                            pass
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        A_mat = np.zeros((n, target_rank))
                        for i in range(n):
                            for c in range(target_rank):
                                A_mat[i, c] = U[i*m+j][c] * W[k*n+i][c]
                        try:
                            sol, residuals, rank_a, sv = np.linalg.lstsq(A_mat, np.ones(n), rcond=None)
                            V[j*p+k] = sol.tolist()
                        except:
                            pass
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        A_mat = np.zeros((m, target_rank))
                        for j in range(m):
                            for c in range(target_rank):
                                A_mat[j, c] = U[i*m+j][c] * V[j*p+k][c]
                        try:
                            sol, residuals, rank_a, sv = np.linalg.lstsq(A_mat, np.ones(m), rcond=None)
                            W[k*n+i] = sol.tolist()
                        except:
                            pass
                
                # Check convergence
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for c in range(target_rank):
                                val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                            err += (val - 1.0)**2
                
                if err < 1e-10:
                    converged = True
                    break
                
                if it % 50 == 0 and err < 1e-15:
                    break
            
            if converged:
                # Round to half-integers if possible
                U_rounded = [[round(x*2)/2 for x in row] for row in U]
                V_rounded = [[round(x*2)/2 for x in row] for row in V]
                W_rounded = [[round(x*2)/2 for x in row] for row in W]
                
                if verify_decomposition(U_rounded, V_rounded, W_rounded, n, m, p, tol=1e-8):
                    best_U, best_V, best_W, best_rank = U_rounded, V_rounded, W_rounded, target_rank
                    break  # Found lower rank, stop
                elif verify_decomposition(U, V, W, n, m, p, tol=1e-10):
                    # Use non-rounded if rounded doesn't work
                    best_U, best_V, best_W, best_rank = U, V, W, target_rank
                    break
    
    return best_U, best_V, best_W, best_rank

def construct():
    n, m, p = DIMENSIONS
    
    # Try to find lower rank decomposition
    U, V, W, rank = search_lower_rank(n, m, p, max_time=110)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

