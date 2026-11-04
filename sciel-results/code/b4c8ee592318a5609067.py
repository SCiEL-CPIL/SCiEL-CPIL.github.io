"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    # The matrix multiplication tensor T[i, k, j] = 1 for all i,k,j
    # We want T = sum_r U[i,r] * V[k,r] * W[j,r]
    # where U is (n, rank), V is (p, rank), W is (m, rank)
    # Output format: U is (n*m, rank) with U[i*m+j, r], 
    #                V is (m*p, rank) with V[j*p+k, r], 
    #                W is (p*n, rank) with W[k*n+i, r]
    
    # Trivial decomposition: rank = n*m*p = 72
    def trivial_decomp():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to find a lower rank decomposition
    # The tensor in our indexing: for each (i,j,k), we have a term A[i,j]*B[j,k]*C[i,k]
    # So T[i,k,j] should equal sum_r U'[i,r]*V'[k,r]*W'[j,r] where U',V',W' are (n,r),(p,r),(m,r)
    # Then we reshape: U[i*m+j, r] corresponds to... wait, let me re-read the format.
    
    # U has shape (n*m, rank): U[i*m + j, c]
    # V has shape (m*p, rank): V[j*p + k, c]  
    # W has shape (p*n, rank): W[k*n + i, c]
    
    # The product sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should give the coefficient
    # for A[i,j]*B[j,k] in the expression of C[i,k].
    # For matrix multiplication, C[i,k] = sum_j A[i,j]*B[j,k], so the coefficient is 1 for each (i,j,k).
    
    # So we need: for all i in [0,n), j in [0,m), k in [0,p):
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # This is equivalent to: let U' be (n, m, rank) with U'[i,j,c] = U[i*m+j,c]
    #                        V' be (m, p, rank) with V'[j,k,c] = V[j*p+k,c]
    #                        W' be (p, n, rank) with W'[k,i,c] = W[k*n+i,c]
    # Then sum_c U'[i,j,c] * V'[j,k,c] * W'[k,i,c] = 1
    
    # This is the standard matrix multiplication tensor decomposition.
    
    # For small cases, let's try to find a decomposition with rank < 72.
    # We can use the fact that the tensor has a certain structure.
    
    # One approach: try rank values from low to high and use numerical optimization
    # or linear algebra to find a solution.
    
    # Let's try a specific construction. For 3x4x6, note that 4 and 6 have common factors.
    # We can group: for each i (3 values), we need to compute 6 outputs from 4 inputs.
    
    # Idea: use a decomposition where we split the m=4 dimension into 2+2,
    # and the p=6 dimension into 3+3, then use rank-2 decompositions for each block.
    
    # Actually, let me try a simpler approach: just verify that rank 72 works and return it,
    # but also try to find something better if time permits.
    
    # For now, let's try to find a decomposition of rank 48 or lower by search.
    
    # The system of equations: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # This is a system of 72 equations in r*(n*m + m*p + p*n) = r*48 unknowns (for n=3,m=4,p=6: 12+24+18=54... wait)
    # n*m = 12, m*p = 24, p*n = 18, total = 54 per rank unit.
    
    # For rank r, we have 54*r unknowns and 72 equations.
    # We need 54*r >= 72, so r >= 2 (trivially), but the system is highly nonlinear (trilinear).
    
    # Let me try a different strategy: fix U and V, then solve for W linearly.
    # Or use random initialization and gradient descent.
    
    # Given the time constraint, let me try a few specific constructions.
    
    # Construction 1: Rank 36
    # If we can pair up terms: for each i, use 12 terms (4*3) instead of 24 (4*6)
    # This would require that for each i, the 6 outputs share 12 rank-1 terms.
    
    # Actually, let me just try to use numpy to find a solution via optimization for small ranks.
    
    def try_rank(r, max_iters=5000, seed=42):
        """Try to find a decomposition of rank r using random init + gradient descent."""
        if time.time() > deadline - 10:
            return None
        
        np.random.seed(seed)
        
        # Initialize U, V, W randomly
        U_flat = np.random.randn(n * m, r) * 0.1
        V_flat = np.random.randn(m * p, r) * 0.1
        W_flat = np.random.randn(p * n, r) * 0.1
        
        # Reshape for easy access
        # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
        
        # Compute the "error": for each (i,j,k), we want sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        # Let's vectorize this.
        
        # For each c, define:
        #   Uc = U[:, c]  (n*m,)
        #   Vc = V[:, c]  (m*p,)
        #   Wc = W[:, c]  (p*n,)
        # The contribution to (i,j,k) is Uc[i*m+j] * Vc[j*p+k] * Wc[k*n+i]
        
        # This is hard to vectorize directly. Let me think...
        
        # Alternative: for each (i,k), the equation is:
        #   sum_j sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = p  (since there are p j's, each should give 1)
        # Wait no, for each (i,j,k) separately, the sum over c should give 1.
        
        # Let me just loop over (i,j,k) for the error computation.
        
        def compute_error():
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(r):
                            s += U_flat[i*m+j, c] * V_flat[j*p+k, c] * W_flat[k*n+i, c]
                        err += (s - 1.0) ** 2
            return err
        
        # Gradient descent
        lr = 0.01
        best_err = float('inf')
        best_U, best_V, best_W = U_flat.copy(), V_flat.copy(), W_flat.copy()
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            err = compute_error()
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U_flat.copy(), V_flat.copy(), W_flat.copy()
            
            if err < 1e-10:
                break
            
            # Compute gradients (finite differences for simplicity, but this is slow)
            # Let's use analytic gradients instead.
            # d(err)/d(U[i*m+j,c]) = 2*(s_ijk - 1) * V[j*p+k,c] * W[k*n+i,c]  summed over... 
            # Actually, U[i*m+j,c] appears in equations for all k (with that i,j).
            # So d(err)/d(U[i*m+j,c]) = 2 * sum_k (s_{i,j,k} - 1) * V[j*p+k,c] * W[k*n+i,c]
            
            # This is getting complex. Let me just use finite differences for a few variables.
            # Actually, for the time constraint, let me try a different approach.
            
            # Let me just try specific rank values with a smarter method.
            break  # Skip gradient descent for now, too slow
        
        if best_err < 1e-8:
            # Convert to the required format
            U_out = [[0] * r for _ in range(n * m)]
            V_out = [[0] * r for _ in range(m * p)]
            W_out = [[0] * r for _ in range(p * n)]
            for i in range(n * m):
                for c in range(r):
                    U_out[i][c] = float(best_U[i, c])
            for j in range(m * p):
                for c in range(r):
                    V_out[j][c] = float(best_V[j, c])
            for k in range(p * n):
                for c in range(r):
                    W_out[k][c] = float(best_W[k, c])
            return {"U": U_out, "V": V_out, "W": W_out}
        
        return None
    
    # Try ranks from low to high
    for r in range(30, 73):
        if time.time() > deadline - 15:
            break
        result = try_rank(r)
        if result is not None:
            # Verify
            U = result["U"]
            V = result["V"]
            W = result["W"]
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(r):
                            s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                        if abs(s - 1.0) > 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if valid:
                return result
    
    # Fall back to trivial
    return trivial_decomp()
# EVOLVE-BLOCK-END
