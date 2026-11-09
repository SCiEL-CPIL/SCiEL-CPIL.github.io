"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 115.0  # seconds (safety margin)
    n, m, p = DIMENSIONS  # 3, 4, 5
    
    def compute_max_error(U, V, W, r):
        """Compute maximum error in tensor reconstruction."""
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
        return max_err
    
    def try_rank_als(r, seed=42, max_iters=2000):
        """Try to find a rank-r decomposition using ALS with multiple restarts."""
        rng = np.random.RandomState(seed)
        
        best_U, best_V, best_W = None, None, None
        best_err = float('inf')
        
        # Multiple random initializations
        for init in range(5):
            if time.time() > deadline - 5:
                break
                
            # Initialize
            U = rng.randn(n * m, r) * 0.3
            V = rng.randn(m * p, r) * 0.3
            W = rng.randn(p * n, r) * 0.3
            
            prev_err = float('inf')
            stall_count = 0
            
            for it in range(max_iters):
                if time.time() > deadline - 5:
                    break
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        M = np.zeros((p, r))
                        for k in range(p):
                            M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        try:
                            u, _, _, _ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                            U[i*m+j, :] = u
                        except:
                            pass
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        M = np.zeros((n, r))
                        for i in range(n):
                            M[i, :] = U[i*m+j, :] * W[k*n+i, :]
                        try:
                            v, _, _, _ = np.linalg.lstsq(M, np.ones(n), rcond=None)
                            V[j*p+k, :] = v
                        except:
                            pass
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        M = np.zeros((m, r))
                        for j in range(m):
                            M[j, :] = U[i*m+j, :] * V[j*p+k, :]
                        try:
                            w, _, _, _ = np.linalg.lstsq(M, np.ones(m), rcond=None)
                            W[k*n+i, :] = w
                        except:
                            pass
                
                # Check convergence
                curr_err = compute_max_error(U, V, W, r)
                
                if curr_err < best_err:
                    best_err = curr_err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if curr_err < 1e-12:
                    break
                
                if abs(prev_err - curr_err) < 1e-14:
                    stall_count += 1
                    if stall_count > 10:
                        break
                else:
                    stall_count = 0
                
                prev_err = curr_err
        
        if best_U is not None and best_err < 1e-10:
            # Try to round to half-integers
            U_rounded = np.round(best_U * 2) / 2
            V_rounded = np.round(best_V * 2) / 2
            W_rounded = np.round(best_W * 2) / 2
            
            max_err_round = compute_max_error(U_rounded, V_rounded, W_rounded, r)
            
            if max_err_round < 1e-10:
                return U_rounded, V_rounded, W_rounded, True
            else:
                return best_U, best_V, best_W, True
        
        return None, None, None, False
    
    # Try rank 16 with multiple seeds
    for seed in range(15):
        if time.time() > deadline - 10:
            break
        U, V, W, success = try_rank_als(16, seed=seed, max_iters=1500)
        if success:
            U_list = [[float(U[i][j]) for j in range(16)] for i in range(n*m)]
            V_list = [[float(V[i][j]) for j in range(16)] for i in range(m*p)]
            W_list = [[float(W[i][j]) for j in range(16)] for i in range(p*n)]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Try rank 17
    for seed in range(10):
        if time.time() > deadline - 10:
            break
        U, V, W, success = try_rank_als(17, seed=seed, max_iters=1500)
        if success:
            U_list = [[float(U[i][j]) for j in range(17)] for i in range(n*m)]
            V_list = [[float(V[i][j]) for j in range(17)] for i in range(m*p)]
            W_list = [[float(W[i][j]) for j in range(17)] for i in range(p*n)]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Try rank 18
    for seed in range(8):
        if time.time() > deadline - 10:
            break
        U, V, W, success = try_rank_als(18, seed=seed, max_iters=1500)
        if success:
            U_list = [[float(U[i][j]) for j in range(18)] for i in range(n*m)]
            V_list = [[float(V[i][j]) for j in range(18)] for i in range(m*p)]
            W_list = [[float(W[i][j]) for j in range(18)] for i in range(p*n)]
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
