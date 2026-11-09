import numpy as np
import time

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Target ranks to try
    # We know schoolbook is 40.
    # Let's try to find a lower rank.
    # Based on the linear system logic, any rank R where 38*R >= 160 (R>=5) *might* work.
    # But constraints on U, V being fixed might make it hard.
    # We will iterate R from 10 to 39.
    
    for R in range(10, 40):
        if time.time() > deadline:
            break
            
        # Try multiple random initializations for U, V
        for trial in range(100):
            if time.time() > deadline:
                break
                
            # Random U, V
            # U shape (nm, R), V shape (mp, R)
            U = np.random.randn(n * m, R)
            V = np.random.randn(m * p, R)
            
            # Construct the linear system for W
            # Equations: sum_r (U[im+j1, r] * V[j2*p+k, r]) * W[k*n+i, r] = delta(j1, j2)
            # Number of equations: n * p * m * m
            # Number of variables: R * p * n
            
            num_eq = n * p * m * m
            num_var = R * p * n
            
            # Build matrix A (num_eq x num_var) and vector b (num_eq)
            # Indexing:
            # Eq idx: ((i * p + k) * m + j1) * m + j2
            # Var idx: r * (p * n) + (k * n + i)
            
            A = np.zeros((num_eq, num_var))
            b = np.zeros(num_eq)
            
            for i in range(n):
                for k in range(p):
                    for j1 in range(m):
                        for j2 in range(m):
                            eq_idx = ((i * p + k) * m + j1) * m + j2
                            b[eq_idx] = 1.0 if j1 == j2 else 0.0
                            
                            for r in range(R):
                                var_idx = r * (p * n) + (k * n + i)
                                coeff = U[i * m + j1, r] * V[j2 * p + k, r]
                                A[eq_idx, var_idx] = coeff
            
            # Solve A W_flat = b
            # Use least squares
            try:
                W_flat, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
                # Check if solution is valid (residuals should be close to 0)
                if residuals[0] < 1e-9:
                    W = W_flat.reshape(R, p * n)
                    # Convert to lists
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = W.T.tolist() # W shape is (R, pn), need (pn, R)
                    
                    return {"U": U_list, "V": V_list, "W": W_list}
            except np.linalg.LinAlgError:
                pass

    # Fallback to schoolbook
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
