"""Program for finding rank-minimal exact tensor decomposition of 2x5 * 5x6 matrix multiplication."""
import numpy as np
import random

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # The matrix multiplication tensor T has entries T[i,j,k] = 1 for all valid (i,j,k)
    # where i in [0,n), j in [0,m), k in [0,p)
    # We seek rank-r decomposition: T = sum_t u_t ⊗ v_t ⊗ w_t
    # where u_t is in R^(n*m), v_t in R^(m*p), w_t in R^(p*n)
    # such that for all (i,j,k): sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] = 1
    
    # Try structured low-rank constructions
    
    def schoolbook():
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
    
    def is_exact(U, V, W, n, m, p, tol=1e-9):
        """Check if the given factors exactly decompose the matrix multiplication tensor."""
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        r = U.shape[1]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for t in range(r):
                        s += U[i * m + j, t] * V[j * p + k, t] * W[k * n + i, t]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    # Try a structured approach based on exploiting the small dimensions
    # For 2x5*5x6, we can try to find decompositions with rank < 60
    
    # One idea: use the fact that we have only 2 rows in A and 6 columns in B
    # Let's try to construct a decomposition where we group terms
    
    # Try rank values from a lower bound up to schoolbook
    # Lower bound: max(n*m, m*p, p*n) = max(10, 30, 12) = 30
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Try to find better decompositions using random search with structured initialization
    # We'll try various ranks and use random restarts
    
    for target_rank in range(30, best_rank):
        found = False
        for trial in range(50):
            # Initialize with random values
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            # Try to project onto a solution
            # This is a hard non-convex problem, so we'll use a simple local search
            
            # Compute the residual
            # For each (i,j,k), we want sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] = 1
            
            # Use gradient descent to minimize the error
            lr = 0.01
            for step in range(100):
                # Compute gradients
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # Current value
                            val = 0.0
                            for t in range(target_rank):
                                val += U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                            err = val - 1.0
                            for t in range(target_rank):
                                grad_U[i*m+j, t] += err * V[j*p+k, t] * W[k*n+i, t]
                                grad_V[j*p+k, t] += err * U[i*m+j, t] * W[k*n+i, t]
                                grad_W[k*n+i, t] += err * U[i*m+j, t] * V[j*p+k, t]
                
                # Update
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Check convergence
                if is_exact(U, V, W, n, m, p, tol=1e-6):
                    found = True
                    break
            
            if found:
                # Round to nearest half-integer if close
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                if is_exact(U_rounded, V_rounded, W_rounded, n, m, p, tol=1e-6):
                    best_result = {
                        "U": U_rounded.tolist(),
                        "V": V_rounded.tolist(),
                        "W": W_rounded.tolist()
                    }
                    best_rank = target_rank
                    break
                else:
                    # Use the unrounded version if it's exact
                    if is_exact(U, V, W, n, m, p, tol=1e-9):
                        best_result = {
                            "U": U.tolist(),
                            "V": V.tolist(),
                            "W": W.tolist()
                        }
                        best_rank = target_rank
                        break
        
        if found:
            break
    
    return best_result
# EVOLVE-BLOCK-END
