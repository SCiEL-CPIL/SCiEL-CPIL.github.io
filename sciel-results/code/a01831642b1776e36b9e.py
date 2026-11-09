"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 115.0  # seconds - safety margin before 120s kill
    n, m, p = DIMENSIONS  # 3, 4, 5
    
    def verify_solution(U, V, W, r, tol=1e-8):
        """Verify that the decomposition is exact."""
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for rr in range(r):
                        val += U[i*m+j, rr] * V[j*p+k, rr] * W[k*n+i, rr]
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err < tol
    
    def compute_error(U, V, W, r):
        """Compute max error efficiently using vectorized operations."""
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :r], V[j*p+k, :r] * W[k*n+i, :r])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err
    
    def polish(U, V, W, r, max_iters=500, tol=1e-13):
        """High-precision polish using alternating least squares with tighter tolerance."""
        for it in range(max_iters):
            if time.time() > deadline - 8:
                break
            err = compute_error(U, V, W, r)
            if err < tol:
                break
            
            # Update U
            for i in range(n):
                for j in range(m):
                    M = np.zeros((p, r))
                    for k in range(p):
                        M[k, :] = V[j*p+k, :r] * W[k*n+i, :r]
                    rhs = np.ones(p)
                    try:
                        u, _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)
                        U[i*m+j, :r] = u
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    M = np.zeros((n, r))
                    for i in range(n):
                        M[i, :] = U[i*m+j, :r] * W[k*n+i, :r]
                    rhs = np.ones(n)
                    try:
                        v, _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)
                        V[j*p+k, :r] = v
                    except:
                        pass
            
            # Update W
            for k in range(p):
                for i in range(n):
                    M = np.zeros((m, r))
                    for j in range(m):
                        M[j, :] = U[i*m+j, :r] * V[j*p+k, :r]
                    rhs = np.ones(m)
                    try:
                        w, _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)
                        W[k*n+i, :r] = w
                    except:
                        pass
        
        return U, V, W
    
    def try_rank(r, seed=42, max_iters=3000, init_type="random"):
        """Try to find a rank-r decomposition."""
        rng = np.random.RandomState(seed)
        
        if init_type == "random":
            U = rng.randn(n * m, r) * 0.5
            V = rng.randn(m * p, r) * 0.5
            W = rng.randn(p * n, r) * 0.5
        elif init_type == "structured":
            # Initialize with schoolbook pattern plus small random perturbation
            U = np.zeros((n * m, r))
            V = np.zeros((m * p, r))
            W = np.zeros((p * n, r))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if col < r:
                            U[i*m+j, col] = 1.0
                            V[j*p+k, col] = 1.0
                            W[k*n+i, col] = 1.0
                            col += 1
            U += rng.randn(n * m, r) * 0.1
            V += rng.randn(m * p, r) * 0.1
            W += rng.randn(p * n, r) * 0.1
        elif init_type == "sparse":
            U = rng.randn(n * m, r)
            V = rng.randn(m * p, r)
            W = rng.randn(p * n, r)
            mask_u = rng.rand(n * m, r) < 0.2
            mask_v = rng.rand(m * p, r) < 0.2
            mask_w = rng.rand(p * n, r) < 0.2
            U *= mask_u
            V *= mask_v
            W *= mask_w
        elif init_type == "mixed":
            # Combine structured and random
            U = np.zeros((n * m, r))
            V = np.zeros((m * p, r))
            W = np.zeros((p * n, r))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if col < r:
                            U[i*m+j, col] = 1.0
                            V[j*p+k, col] = 1.0
                            W[k*n+i, col] = 1.0
                            col += 1
            U += rng.randn(n * m, r) * 0.3
            V += rng.randn(m * p, r) * 0.3
            W += rng.randn(p * n, r) * 0.3
        
        # Alternating least squares with adaptive iteration checking
        for it in range(max_iters):
            if time.time() > deadline - 12:
                break
            
            # Update U
            for i in range(n):
                for j in range(m):
                    M = np.zeros((p, r))
                    for k in range(p):
                        M[k, :] = V[j*p+k, :r] * W[k*n+i, :r]
                    try:
                        u, _, _, _ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                        U[i*m+j, :r] = u
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    M = np.zeros((n, r))
                    for i in range(n):
                        M[i, :] = U[i*m+j, :r] * W[k*n+i, :r]
                    try:
                        v, _, _, _ = np.linalg.lstsq(M, np.ones(n), rcond=None)
                        V[j*p+k, :r] = v
                    except:
                        pass
            
            # Update W
            for k in range(p):
                for i in range(n):
                    M = np.zeros((m, r))
                    for j in range(m):
                        M[j, :] = U[i*m+j, :r] * V[j*p+k, :r]
                    try:
                        w, _, _, _ = np.linalg.lstsq(M, np.ones(m), rcond=None)
                        W[k*n+i, :r] = w
                    except:
                        pass
            
            # Check convergence more frequently in later stages
            check_interval = 20 if it < 500 else 10
            if it % check_interval == check_interval - 1:
                err = compute_error(U, V, W, r)
                if err < 1e-11:
                    break
        
        # Polish with tighter tolerance
        U, V, W = polish(U, V, W, r, max_iters=300, tol=1e-13)
        
        # Check final error
        err = compute_error(U, V, W, r)
        if err < 1e-10:
            # Try to round to half-integers
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            if verify_solution(U_rounded, V_rounded, W_rounded, r):
                return U_rounded, V_rounded, W_rounded, True
            else:
                return U, V, W, True
        
        return None, None, None, False
    
    # Try various ranks and initialization strategies
    init_types = ["random", "structured", "sparse", "mixed"]
    
    for rank in [16, 17, 18, 19, 20]:
        if time.time() > deadline - 25:
            break
        
        for init_type in init_types:
            for seed in range(15):
                if time.time() > deadline - 12:
                    break
                
                U, V, W, success = try_rank(rank, seed=seed, max_iters=2000, init_type=init_type)
                if success:
                    U_list = [[float(U[i][j]) for j in range(rank)] for i in range(n*m)]
                    V_list = [[float(V[i][j]) for j in range(rank)] for i in range(m*p)]
                    W_list = [[float(W[i][j]) for j in range(rank)] for i in range(p*n)]
                    return {"U": U_list, "V": V_list, "W": W_list}
    
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
# EVOLVE-BLOCK-END
