"""Program for matrix multiplication tensor decomposition 2x4x7.
Uses alternating minimization to find low-rank decompositions.
"""
import time
import itertools
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin: 145s < 150s limit
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=7
    
    def make_schoolbook():
        """Return the standard rank n*m*p decomposition."""
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
    
    best_result = make_schoolbook()
    best_rank = n * m * p  # 56
    
    def compute_residual(U, V, W):
        """Compute max absolute residual of the decomposition."""
        max_res = 0.0
        for i in range(n):
            for j in range(m):
                V_jk = V[j*p:(j+1)*p, :]
                W_ki = W[i::n, :]
                Prod = V_jk * W_ki
                Dots = Prod @ U[i*m+j, :]
                max_res = max(max_res, np.max(np.abs(Dots - 1.0)))
        return max_res
    
    def polish(U, V, W, max_iters=200, tol=1e-12):
        """Final high-precision polish pass with tighter tolerance."""
        ones_p = np.ones(p)
        ones_n = np.ones(n)
        ones_m = np.ones(m)
        
        for it in range(max_iters):
            if time.time() > deadline - 2:
                break
            
            # Update U
            for i in range(n):
                for j in range(m):
                    A = V[j*p:(j+1)*p, :] * W[i::n, :]
                    try:
                        U[i*m+j, :] = np.linalg.lstsq(A.T, ones_p, rcond=None)[0]
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    B = U[j::m, :] * W[k*n:(k+1)*n, :]
                    try:
                        V[j*p+k, :] = np.linalg.lstsq(B.T, ones_n, rcond=None)[0]
                    except:
                        pass
            
            # Update W
            for k in range(p):
                for i in range(n):
                    C = U[i*m:(i+1)*m, :] * V[k::p, :]
                    try:
                        W[k*n+i, :] = np.linalg.lstsq(C.T, ones_m, rcond=None)[0]
                    except:
                        pass
            
            if it % 10 == 9 or it == max_iters - 1:
                res = compute_residual(U, V, W)
                if res < tol:
                    return True, U, V, W
        
        res = compute_residual(U, V, W)
        return res < tol, U, V, W
    
    def try_alternating_minimization(rank, max_iters=600, tol=1e-8, seed=0, init_type=0):
        """Try alternating minimization to find a rank-r decomposition."""
        if time.time() > deadline:
            return None
        
        np.random.seed(seed)
        
        if init_type == 0:
            # Standard random init
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
        elif init_type == 1:
            # Sparse random init
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            # Zero out ~70% of entries
            mask_u = np.random.rand(n*m, rank) > 0.7
            mask_v = np.random.rand(m*p, rank) > 0.7
            mask_w = np.random.rand(p*n, rank) > 0.7
            U[mask_u] = 0
            V[mask_v] = 0
            W[mask_w] = 0
        else:
            # Structured init: try to match schoolbook pattern partially
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            # Fill first min(rank, n*m*p) columns with schoolbook-like structure
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if col >= rank:
                            break
                        U[i*m+j, col] = np.random.choice([-1, 1])
                        V[j*p+k, col] = np.random.choice([-1, 1])
                        W[k*n+i, col] = np.random.choice([-1, 1])
                        col += 1
            # Add some random noise to remaining columns
            if col < rank:
                U[:, col:] = np.random.randn(n*m, rank - col) * 0.1
                V[:, col:] = np.random.randn(m*p, rank - col) * 0.1
                W[:, col:] = np.random.randn(p*n, rank - col) * 0.1
        
        ones_p = np.ones(p)
        ones_n = np.ones(n)
        ones_m = np.ones(m)
        
        best_res = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Update U
            for i in range(n):
                for j in range(m):
                    A = V[j*p:(j+1)*p, :] * W[i::n, :]
                    try:
                        U[i*m+j, :] = np.linalg.lstsq(A.T, ones_p, rcond=None)[0]
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    B = U[j::m, :] * W[k*n:(k+1)*n, :]
                    try:
                        V[j*p+k, :] = np.linalg.lstsq(B.T, ones_n, rcond=None)[0]
                    except:
                        pass
            
            # Update W
            for k in range(p):
                for i in range(n):
                    C = U[i*m:(i+1)*m, :] * V[k::p, :]
                    try:
                        W[k*n+i, :] = np.linalg.lstsq(C.T, ones_m, rcond=None)[0]
                    except:
                        pass
            
            # Check residual every 5 iterations or at end
            if it % 5 == 4 or it == max_iters - 1:
                max_res = compute_residual(U, V, W)
                
                if max_res < best_res:
                    best_res = max_res
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    if max_res < tol:
                        # Try polish
                        success, U_p, V_p, W_p = polish(U.copy(), V.copy(), W.copy())
                        if success:
                            return {
                                "U": U_p.tolist(),
                                "V": V_p.tolist(),
                                "W": W_p.tolist()
                            }
                        elif max_res < 1e-6:
                            return {
                                "U": U.tolist(),
                                "V": V.tolist(),
                                "W": W.tolist()
                            }
        
        # If we got close but not exact, try polish on best found
        if best_res < 1e-4:
            success, U_p, V_p, W_p = polish(best_U.copy(), best_V.copy(), best_W.copy())
            if success:
                return {
                    "U": U_p.tolist(),
                    "V": V_p.tolist(),
                    "W": W_p.tolist()
                }
        
        return None
    
    # Try alternating minimization for various ranks
    # Focus on lower ranks first with more seeds and diverse inits
    ranks_to_try = [28, 29, 30, 31, 32, 33, 34, 36, 38, 40, 42, 44, 48, 52, 54]
    
    for r in ranks_to_try:
        if time.time() > deadline - 15:
            break
        
        # More seeds and diverse inits for smaller ranks
        if r <= 30:
            seeds = 40
            init_types = [0, 1, 2]
        elif r <= 34:
            seeds = 25
            init_types = [0, 1]
        else:
            seeds = 12
            init_types = [0]
        
        found = None
        for seed in range(seeds):
            if time.time() > deadline - 5:
                break
            for init_type in init_types:
                if time.time() > deadline - 5:
                    break
                found = try_alternating_minimization(r, max_iters=500, seed=seed, init_type=init_type)
                if found is not None:
                    best_rank = r
                    best_result = found
                    return best_result
    
    return best_result
# EVOLVE-BLOCK-END
