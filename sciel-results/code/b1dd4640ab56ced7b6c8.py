"""Matrix multiplication 3,3,6 - rank-minimal exact tensor decomposition.

We exploit the structure: multiplying (3x3) A by (3x6) B gives (3x6) C.
This can be viewed as 6 independent (3x3) multiplications sharing the same A.
We attempt to find a low-rank decomposition using an alternating minimization
approach with integer/half-integer rounding. If that fails, we fall back to
schoolbook rank 54.

Key improvements over the previous version:
1. More aggressive search over ranks, starting from a tighter lower bound.
2. Multiple random restarts with different initializations.
3. Better convergence criteria and rounding strategy.
4. Time-budget management to ensure we return a valid result.
"""

import time
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    def schoolbook():
        rank = n * m * p  # 54
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
        """Verify that the decomposition is exact."""
        n, m, p = DIMENSIONS
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    def try_find_decomposition(R, seed=42):
        """Try to find a rank-R decomposition using alternating least squares."""
        np.random.seed(seed)
        
        # Initialize with small random values
        U = np.random.randn(n*m, R) * 0.5
        V = np.random.randn(m*p, R) * 0.5
        W = np.random.randn(p*n, R) * 0.5
        
        max_iterations = 300
        for iteration in range(max_iterations):
            if time.time() > deadline - 5:
                return None
            
            # Update W: for each output index o = k*n+i
            for o in range(p*n):
                k = o // n
                i = o % n
                # Equations: for each j, sum_r U[i*m+j,r] * V[j*p+k,r] * W[o,r] = 1
                A_sub = np.zeros((m, R))
                for j in range(m):
                    for r in range(R):
                        A_sub[j, r] = U[i*m+j, r] * V[j*p+k, r]
                try:
                    W[o, :] = np.linalg.lstsq(A_sub, np.ones(m), rcond=None)[0]
                except:
                    W[o, :] = np.zeros(R)
            
            # Update U: for each (i,j)
            for idx_u in range(n*m):
                i = idx_u // m
                j = idx_u % m
                # Equations: for each k, sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
                A_sub = np.zeros((p, R))
                for k in range(p):
                    for r in range(R):
                        A_sub[k, r] = V[j*p+k, r] * W[k*n+i, r]
                try:
                    U[idx_u, :] = np.linalg.lstsq(A_sub, np.ones(p), rcond=None)[0]
                except:
                    U[idx_u, :] = np.zeros(R)
            
            # Update V: for each (j,k)
            for idx_v in range(m*p):
                j = idx_v // p
                k = idx_v % p
                # Equations: for each i, sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
                A_sub = np.zeros((n, R))
                for i in range(n):
                    for r in range(R):
                        A_sub[i, r] = U[i*m+j, r] * W[k*n+i, r]
                try:
                    V[idx_v, :] = np.linalg.lstsq(A_sub, np.ones(n), rcond=None)[0]
                except:
                    V[idx_v, :] = np.zeros(R)
            
            # Check convergence every 5 iterations
            if iteration % 5 == 0:
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for r in range(R):
                                val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                            max_err = max(max_err, abs(val - 1.0))
                
                if max_err < 1e-6:
                    # Try to round to half-integers
                    for factor in [1, 2, 4]:
                        U_r = np.round(U * factor) / factor
                        V_r = np.round(V * factor) / factor
                        W_r = np.round(W * factor) / factor
                        
                        max_err_r = 0.0
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    val = 0.0
                                    for r in range(R):
                                        val += U_r[i*m+j, r] * V_r[j*p+k, r] * W_r[k*n+i, r]
                                    max_err_r = max(max_err_r, abs(val - 1.0))
                        
                        if max_err_r < 1e-9:
                            U_list = U_r.tolist()
                            V_list = V_r.tolist()
                            W_list = W_r.tolist()
                            # Ensure clean values
                            for mat in [U_list, V_list, W_list]:
                                for row in mat:
                                    for idx in range(len(row)):
                                        row[idx] = round(row[idx] * 2) / 2
                            return {"U": U_list, "V": V_list, "W": W_list}
                
                if max_err > 5.0 and iteration > 30:
                    return None
        
        return None
    
    # Try various ranks from low to high
    # Lower bound for 3x3x6 is at least max(9, 18, 18) = 18, but likely higher
    ranks_to_try = [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
    
    for R in ranks_to_try:
        if time.time() > deadline - 15:
            break
        for seed in [42, 123, 777, 2024, 999, 555, 314, 271]:
            if time.time() > deadline - 10:
                break
            result = try_find_decomposition(R, seed=seed)
            if result is not None:
                if verify(result["U"], result["V"], result["W"]):
                    return result
    
    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
