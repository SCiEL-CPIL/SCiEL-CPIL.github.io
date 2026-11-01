import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 2 by 3 matrix by a 3 by 4 matrix.
    
    Uses Alternating Least Squares (ALS) to find a rank 20 decomposition.
    Falls back to rank 24 if not found within time limit.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n, m, p = 2, 3, 4
    R = 20  # Target rank
    
    # Initialize U, V, W
    np.random.seed(42)
    U = np.random.randn(n * m, R)
    V = np.random.randn(m * p, R)
    W = np.random.randn(p * n, R)
    
    # ALS Loop
    for it in range(2000):
        if time.time() > deadline - 5.0:
            break
            
        # Solve for U
        for i in range(n):
            for j1 in range(m):
                A = np.zeros((m * p, R))
                b = np.zeros(m * p)
                for k in range(p):
                    for j2 in range(m):
                        idx = j2 * p + k
                        b[idx] = 1.0 if j1 == j2 else 0.0
                        for r in range(R):
                            A[idx, r] = W[k * n + i, r] * V[j2 * p + k, r]
                
                try:
                    U[i * m + j1, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except Exception:
                    pass
                    
        # Solve for V
        for j2 in range(m):
            for k in range(p):
                A = np.zeros((n * m, R))
                b = np.zeros(n * m)
                for i in range(n):
                    for j1 in range(m):
                        idx = i * m + j1
                        b[idx] = 1.0 if j1 == j2 else 0.0
                        for r in range(R):
                            A[idx, r] = W[k * n + i, r] * U[i * m + j1, r]
                            
                try:
                    V[j2 * p + k, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except Exception:
                    pass
                    
        # Solve for W
        for i in range(n):
            for k in range(p):
                A = np.zeros((m * m, R))
                b = np.zeros(m * m)
                for j1 in range(m):
                    for j2 in range(m):
                        idx = j1 * m + j2
                        b[idx] = 1.0 if j1 == j2 else 0.0
                        for r in range(R):
                            A[idx, r] = U[i * m + j1, r] * V[j2 * p + k, r]
                            
                try:
                    W[k * n + i, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except Exception:
                    pass
                    
        # Check error
        err = 0.0
        for i in range(n):
            for k in range(p):
                for j1 in range(m):
                    for j2 in range(m):
                        val = 0.0
                        for r in range(R):
                            val += W[k * n + i, r] * U[i * m + j1, r] * V[j2 * p + k, r]
                        target = 1.0 if j1 == j2 else 0.0
                        err += abs(val - target)
                        
        if err < 1e-8:
            # Round to nearest 0.5
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            # Verify rounded solution
            err_r = 0.0
            for i in range(n):
                for k in range(p):
                    for j1 in range(m):
                        for j2 in range(m):
                            val = 0.0
                            for r in range(R):
                                val += W_r[k * n + i, r] * U_r[i * m + j1, r] * V_r[j2 * p + k, r]
                            target = 1.0 if j1 == j2 else 0.0
                            err_r += abs(val - target)
            if err_r < 1e-8:
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                
    # Fallback to schoolbook rank 24
    R_fb = 24
    U_fb = [[0] * R_fb for _ in range(n * m)]
    V_fb = [[0] * R_fb for _ in range(m * p)]
    W_fb = [[0] * R_fb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
                
    return {"U": U_fb, "V": V_fb, "W": W_fb}
# EVOLVE-BLOCK-END
