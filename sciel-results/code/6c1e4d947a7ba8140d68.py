"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 4
    deadline = time.time() + 110.0  # Safety margin
    
    # Schoolbook decomposition as fallback (rank 36)
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

    def verify(U, V, W):
        """Verify that the decomposition exactly represents matrix multiplication."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        
        # Check if sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > 1e-6:
                        return False
        return True

    def check_half_integer(mat):
        """Check if all entries are close to half-integers (multiples of 0.5)."""
        doubled = mat * 2
        rounded = np.round(doubled)
        return np.allclose(doubled, rounded, atol=1e-4)

    def verify_and_round(U, V, W):
        """Try to round to half-integers and verify. Returns (U, V, W) if valid, else None."""
        # Round to nearest half-integer
        U_rounded = np.round(U * 2) / 2.0
        V_rounded = np.round(V * 2) / 2.0
        W_rounded = np.round(W * 2) / 2.0
        
        # Verify the rounded solution
        if verify(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()):
            return U_rounded, V_rounded, W_rounded
        return None

    # Try to find a lower rank decomposition
    # We search for ranks from 24 to 35
    for target_rank in range(24, 36):
        if time.time() > deadline:
            break
            
        # Number of variables: 9*r + 12*r + 12*r = 33*r
        num_vars = (n*m + m*p + p*n) * target_rank
        
        # Objective function - vectorized for speed
        def objective(params):
            U_flat = params[:n*m*target_rank]
            V_flat = params[n*m*target_rank : n*m*target_rank + m*p*target_rank]
            W_flat = params[n*m*target_rank + m*p*target_rank:]
            
            U = U_flat.reshape((n*m, target_rank))
            V = V_flat.reshape((m*p, target_rank))
            W = W_flat.reshape((p*n, target_rank))
            
            # Compute all products: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
            # Vectorized: create arrays of indices
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Sum over c of U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                        val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                        error += (val - 1.0)**2
                        
            return error

        # Initial guess: random small values
        # We use multiple restarts for each rank
        best_result = None
        best_score = float('inf')
        
        # Limit restarts to stay within time
        max_restarts = 5
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break
                
            # Initialize with random values in {-1, 0, 1} or similar
            # Try different initialization strategies
            if restart == 0:
                x0 = np.random.randn(num_vars)
            elif restart == 1:
                # Sparse initialization
                x0 = np.random.choice([-1, 0, 1], size=num_vars)
            elif restart == 2:
                # Larger scale
                x0 = np.random.randn(num_vars) * 2.0
            else:
                # Mixed
                x0 = np.random.randn(num_vars) * 0.5
                x0[np.random.rand(num_vars) < 0.3] = 0
            
            try:
                # Use L-BFGS-B with more iterations
                res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8})
                
                # Check if the raw error is low enough to potentially round
                if res.fun < 1e-6:
                    U_flat = res.x[:n*m*target_rank]
                    V_flat = res.x[n*m*target_rank : n*m*target_rank + m*p*target_rank]
                    W_flat = res.x[n*m*target_rank + m*p*target_rank:]
                    
                    U = U_flat.reshape((n*m, target_rank))
                    V = V_flat.reshape((m*p, target_rank))
                    W = W_flat.reshape((p*n, target_rank))
                    
                    # Try to round and verify
                    result = verify_and_round(U, V, W)
                    if result is not None:
                        U_rounded, V_rounded, W_rounded = result
                        return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
                        
            except Exception as e:
                pass
        
    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
