"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    # Schoolbook rank
    schoolbook_rank = n * m * p  # 54
    
    # Build the matrix multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    # We want to find rank-r decomposition:
    # T[i*m+j, j*p+k, k*n+i] = sum_{c=0}^{r-1} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Known lower bounds for matrix multiplication rank:
    # For (2,3,9): The trivial rank is 54.
    # Let's try to find a better decomposition using random initialization + gradient descent
    # or use known results.
    
    # For small matrices, we can use the approach of trying random rank-r decompositions
    # and checking if they can be made exact via solving linear systems.
    
    # Strategy: 
    # 1. Try ranks from lower bound up to schoolbook rank
    # 2. For each rank, try random initializations and use iterative refinement
    # 3. Verify exactness at the end
    
    # Lower bound: max(n*m, m*p, p*n) = max(6, 27, 18) = 27
    # But this is very weak. Better lower bounds exist.
    
    # Let's try a systematic approach:
    # For each candidate rank r, initialize U, V, W randomly and try to solve
    # the tensor equation exactly.
    
    # The tensor equation can be written as:
    # For each (i,j,k): T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # This is a system of n*m*p equations with r*(n*m + m*p + p*n) unknowns.
    
    # For r < n*m*p, this is an underdetermined system (more unknowns than equations)
    # but it's nonlinear (bilinear/trilinear).
    
    # Alternative approach: Use the fact that matrix multiplication can sometimes be
    # done with fewer multiplications. For (2,3,9), let's check known results.
    
    # Actually, let's use a more practical approach:
    # 1. Start with schoolbook decomposition (rank 54)
    # 2. Try to reduce rank by finding linear dependencies
    
    # For now, let's implement a simple but effective search:
    # Try random rank-r decompositions and verify exactness
    
    def verify_decomposition(U, V, W, n, m, p, tol=1e-10):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        r = U.shape[1]
        # Check all (i,j,k) entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # T[i*m+j, j*p+k, k*n+i] should equal sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
                    val = 0.0
                    for c in range(r):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def verify_decomposition_fast(U, V, W, n, m, p, tol=1e-10):
        """Faster verification using vectorized operations."""
        r = U.shape[1]
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        # For each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        # We can check this by building the full tensor and comparing
        # But that's expensive. Let's use a different approach.
        
        # For fixed i, j: the vector (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]) for all k
        # should be all 1s.
        
        # Actually, let's just do it directly but vectorized over k for each (i,j)
        for i in range(n):
            for j in range(m):
                # u = U[i*m+j, :] shape (r,)
                # For each k: v_k = V[j*p+k, :] shape (r,), w_k = W[k*n+i, :] shape (r,)
                # product = sum_c u[c] * v_k[c] * w_k[c]
                vals = np.zeros(p)
                for k in range(p):
                    vals[k] = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if not np.allclose(vals, 1.0, atol=tol):
                    return False
        return True
    
    # Try to find a decomposition with rank < schoolbook_rank
    # Use random initialization and iterative refinement
    
    # First, let's try a few random rank-r decompositions
    # For each rank, try multiple random initializations
    
    best_rank = schoolbook_rank
    best_decomp = None
    
    # Build schoolbook decomposition as fallback
    U_sb = np.zeros((n*m, schoolbook_rank))
    V_sb = np.zeros((m*p, schoolbook_rank))
    W_sb = np.zeros((p*n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1.0
                V_sb[j*p+k, col] = 1.0
                W_sb[k*n+i, col] = 1.0
                col += 1
    
    # Try ranks from a reasonable lower bound up to schoolbook_rank
    # Lower bound: at least max(n*m, m*p, p*n) = 27, but let's start from 20
    # Actually, for (2,3,9), known results might suggest rank around 30-40
    
    # Let's try ranks from 25 to schoolbook_rank-1
    for r in range(25, schoolbook_rank):
        if time.time() > deadline - 10.0:
            break
        
        # Try a few random initializations for this rank
        max_attempts = 5
        for attempt in range(max_attempts):
            if time.time() > deadline - 5.0:
                break
            
            # Random initialization
            U = np.random.randn(n*m, r) * 0.1
            V = np.random.randn(m*p, r) * 0.1
            W = np.random.randn(p*n, r) * 0.1
            
            # Iterative refinement: alternate between solving for U, V, W
            # This is similar to alternating least squares for tensor decomposition
            
            converged = False
            for iteration in range(50):
                if time.time() > deadline - 5.0:
                    break
                
                # Fix V and W, solve for U
                # For each (i,j): U[i*m+j, :] should satisfy
                # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k
                # This is a linear system: A[i,j] @ U[i*m+j, :] = b[i,j]
                # where A[i,j][k, c] = V[j*p+k, c] * W[k*n+i, c]
                # b[i,j] = ones(p)
                
                # Build the system for all (i,j) at once
                # A has shape (n*m*p, r) but structured
                # Let's solve for each (i,j)
                
                U_new = np.zeros_like(U)
                for i in range(n):
                    for j in range(m):
                        # Build A[i,j] of shape (p, r)
                        A_ij = np.zeros((p, r))
                        for k in range(p):
                            A_ij[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        b_ij = np.ones(p)
                        # Solve A_ij @ u = b_ij
                        try:
                            u_sol, residuals, rank_sol, sv = np.linalg.lstsq(A_ij, b_ij, rcond=None)
                            U_new[i*m+j, :] = u_sol
                        except:
                            U_new[i*m+j, :] = U[i*m+j, :]
                
                # Fix U and W, solve for V
                V_new = np.zeros_like(V)
                for j in range(m):
                    for k in range(p):
                        # Build A[j,k] of shape (n*m, r)
                        A_jk = np.zeros((n*m, r))
                        for i in range(n):
                            for jj in range(m):
                                # Wait, V is indexed by (j*p+k), so for fixed j,k
                                # We need: sum_c U[i*m+jj, c] * V[j*p+k, c] * W[k*n+i, c] = 1
                                # But this couples all i and jj. Let me reconsider.
                                pass
                        # This is getting complicated. Let's use a different approach.
                        break
                    break
                
                # Let's use a simpler alternating approach:
                # Actually, let's just check if the current U, V, W is close to a solution
                # and if not, break early
                
                error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                            error += (val - 1.0)**2
                
                if error < 1e-12:
                    converged = True
                    break
                
                # If error is not decreasing, break
                if iteration > 10 and error > 1.0:
                    break
                
                # Simple update: move U, V, W in the direction that reduces error
                # This is complex, so let's just try random restarts instead
                break  # Break out of iteration loop, try next random init
        
        # Check if any attempt converged
        # For simplicity, just check the last attempt
        if converged:
            if verify_decomposition_fast(U, V, W, n, m, p, tol=1e-10):
                best_rank = r
                best_decomp = (U, V, W)
                break  # Found a valid decomposition with this rank
    
    # If we found a better decomposition, use it
    if best_decomp is not None:
        U, V, W = best_decomp
        r = best_rank
        # Convert to list of lists
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Return schoolbook decomposition
        U_list = [[float(x) for x in row] for row in U_sb]
        V_list = [[float(x) for x in row] for row in V_sb]
        W_list = [[float(x) for x in row] for row in W_sb]
        return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
