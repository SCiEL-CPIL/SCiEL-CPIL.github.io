"""Matrix multiplication 3x4x5: rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 3x4 by 4x5 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Schoolbook as fallback
    def schoolbook():
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
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Try to find lower rank using iterative refinement
    # The constraint: for all (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    def compute_error(U, V, W, rank):
        """Compute max absolute error."""
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    err = abs(s - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err
    
    # Try ranks from 55 down
    for rank in range(55, 35, -1):
        if time.time() > deadline - 15:
            break
        
        found = False
        for trial in range(20):
            if time.time() > deadline - 15:
                break
            if found:
                break
            
            np.random.seed(rank * 1000 + trial)
            U = np.random.randn(n*m, rank) * 0.3
            V = np.random.randn(m*p, rank) * 0.3
            W = np.random.randn(p*n, rank) * 0.3
            
            # Iterative refinement: fix two, optimize one
            for iteration in range(200):
                if time.time() > deadline - 15:
                    break
                
                # Optimize U given V, W
                # For each row of U, we have linear constraints
                # U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :] summed over c should give 1
                # This is nonlinear, so use gradient descent
                
                # Compute gradient of error w.r.t. U
                err = compute_error(U.tolist(), V.tolist(), W.tolist(), rank)
                if err < 1e-8:
                    break
                
                # Simple gradient step
                lr = 0.01
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for c in range(rank):
                                s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                            # Gradient of (s-1)^2 w.r.t. U[i*m+j][c]
                            for c in range(rank):
                                U[i*m+j][c] -= lr * 2 * (s - 1) * V[j*p+k][c] * W[k*n+i][c]
                
                # Similarly for V and W
                for j in range(m):
                    for k in range(p):
                        for i in range(n):
                            s = 0.0
                            for c in range(rank):
                                s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                            for c in range(rank):
                                V[j*p+k][c] -= lr * 2 * (s - 1) * U[i*m+j][c] * W[k*n+i][c]
                
                for i in range(n):
                    for k in range(p):
                        for j in range(m):
                            s = 0.0
                            for c in range(rank):
                                s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                            for c in range(rank):
                                W[k*n+i][c] -= lr * 0.01 * 2 * (s - 1) * U[i*m+j][c] * V[j*p+k][c]
            
            # Check if converged
            err = compute_error(U.tolist(), V.tolist(), W.tolist(), rank)
            if err < 1e-6:
                # Convert to fractions
                U_frac = [[float(U[i, c]) for c in range(rank)] for i in range(n*m)]
                V_frac = [[float(V[i, c]) for c in range(rank)] for i in range(m*p)]
                W_frac = [[float(W[i, c]) for c in range(rank)] for i in range(p*n)]
                
                # Round to nearest half-integer
                U_frac = [[round(x * 2) / 2 for x in row] for row in U_frac]
                V_frac = [[round(x * 2) / 2 for x in row] for row in V_frac]
                W_frac = [[round(x * 2) / 2 for x in row] for row in W_frac]
                
                # Verify with rounded values
                err_rounded = compute_error(U_frac, V_frac, W_frac, rank)
                if err_rounded < 1e-6:
                    best_result = {"U": U_frac, "V": V_frac, "W": W_frac}
                    best_rank = rank
                    found = True
                    break
    
    return best_result
# EVOLVE-BLOCK-END
