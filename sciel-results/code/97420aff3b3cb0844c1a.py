import time
import numpy as np
import scipy.optimize as opt

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def _construct_rank_55():
    """
    Hardcoded rank-55 decomposition for 2x3x10 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for t in range(5):
        k0 = 2 * t
        k1 = 2 * t + 1
        
        U[0 * 3 + 0][col] += 1
        U[1 * 3 + 1][col] += 1
        V[0 * 10 + k0][col] += 1
        V[1 * 10 + k1][col] += 1
        W[k0 * 2 + 0][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1
        
        U[1 * 3 + 0][col] += 1
        U[1 * 3 + 1][col] += 1
        V[0 * 10 + k0][col] += 1
        W[k0 * 2 + 1][col] += 1
        W[k1 * 2 + 1][col] -= 1
        col += 1
        
        U[0 * 3 + 0][col] += 1
        V[0 * 10 + k1][col] += 1
        V[1 * 10 + k1][col] -= 1
        W[k1 * 2 + 0][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1
        
        U[1 * 3 + 1][col] += 1
        V[1 * 10 + k0][col] += 1
        V[0 * 10 + k0][col] -= 1
        W[k0 * 2 + 0][col] += 1
        W[k0 * 2 + 1][col] += 1
        col += 1
        
        U[0 * 3 + 0][col] += 1
        U[0 * 3 + 1][col] += 1
        V[1 * 10 + k1][col] += 1
        W[k0 * 2 + 0][col] -= 1
        W[k1 * 2 + 0][col] += 1
        col += 1
        
        U[1 * 3 + 0][col] += 1
        U[0 * 3 + 0][col] -= 1
        V[0 * 10 + k0][col] += 1
        V[0 * 10 + k1][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1
        
        U[0 * 3 + 1][col] += 1
        U[1 * 3 + 1][col] -= 1
        V[1 * 10 + k0][col] += 1
        V[1 * 10 + k1][col] += 1
        W[k0 * 2 + 0][col] += 1
        col += 1
        
        U[0 * 3 + 2][col] += 1
        V[2 * 10 + k0][col] += 1
        W[k0 * 2 + 0][col] += 1
        col += 1
        
        U[0 * 3 + 2][col] += 1
        V[2 * 10 + k1][col] += 1
        W[k1 * 2 + 0][col] += 1
        col += 1
        
        U[1 * 3 + 2][col] += 1
        V[2 * 10 + k0][col] += 1
        W[k0 * 2 + 1][col] += 1
        col += 1
        
        U[1 * 3 + 2][col] += 1
        V[2 * 10 + k1][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1
        
    return np.array(U, dtype=float), np.array(V, dtype=float), np.array(W, dtype=float)

def _build_target_tensor():
    n, m, p = DIMENSIONS
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    return T

def _verify_decomposition(U, V, W, T, tol=1e-7):
    """Check if U, V, W exactly decompose T."""
    T_recon = np.einsum('ur,vr,wr->uvw', U, V, W)
    return np.allclose(T_recon, T, atol=tol, rtol=tol)

def _try_find_rank(T, rank, n, m, p, time_budget, init_U=None, init_V=None, init_W=None):
    """
    Attempt to find a decomposition of rank `rank` for tensor T.
    Returns (U, V, W) if found, else None.
    """
    start = time.time()
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    num_vars = rank * (dim_u + dim_v + dim_w)
    
    def objective(params):
        U = params[:rank * dim_u].reshape(dim_u, rank)
        V = params[rank * dim_u : rank * dim_u + rank * dim_v].reshape(dim_v, rank)
        W = params[rank * dim_u + rank * dim_v :].reshape(dim_w, rank)
        T_recon = np.einsum('ur,vr,wr->uvw', U, V, W)
        error = T_recon - T
        return np.sum(error ** 2)
    
    best_result = None
    best_score = float('inf')
    
    # Strategy 1: If initial matrices are provided, use them.
    # We need to project the initial higher-rank solution to this rank.
    # Simplest: take the first `rank` columns of the init matrices.
    if init_U is not None:
        # Assume init_U has rank >= rank
        U_init = init_U[:, :rank]
        V_init = init_V[:, :rank]
        W_init = init_W[:, :rank]
        
        init_params = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        # Optimize from this start
        try:
            res = opt.minimize(objective, init_params, method='L-BFGS-B', 
                               options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12})
            
            if res.fun < best_score:
                best_score = res.fun
                best_result = res.x
                
                if best_score < 1e-8:
                    U = best_result[:rank * dim_u].reshape(dim_u, rank)
                    V = best_result[rank * dim_u : rank * dim_u + rank * dim_v].reshape(dim_v, rank)
                    W = best_result[rank * dim_u + rank * dim_v :].reshape(dim_w, rank)
                    
                    # Try rounding to integer
                    U_int = np.rint(U)
                    V_int = np.rint(V)
                    W_int = np.rint(W)
                    
                    if _verify_decomposition(U_int, V_int, W_int, T, tol=1e-7):
                        return U_int, V_int, W_int
                    
                    # Try rounding to half-integer
                    U_half = np.rint(U * 2) / 2
                    V_half = np.rint(V * 2) / 2
                    W_half = np.rint(W * 2) / 2
                    
                    if _verify_decomposition(U_half, V_half, W_half, T, tol=1e-7):
                        return U_half, V_half, W_half
                        
        except Exception:
            pass
            
    # Strategy 2: Random restarts if init strategy failed or no init
    if time.time() - start < time_budget:
        max_restarts = 5
        for r in range(max_restarts):
            if time.time() - start > time_budget:
                break
            init_params = np.random.randn(num_vars) * 0.1
            try:
                res = opt.minimize(objective, init_params, method='L-BFGS-B', 
                                   options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12})
                if res.fun < best_score:
                    best_score = res.fun
                    best_result = res.x
                    if best_score < 1e-8:
                        U = best_result[:rank * dim_u].reshape(dim_u, rank)
                        V = best_result[rank * dim_u : rank * dim_u + rank * dim_v].reshape(dim_v, rank)
                        W = best_result[rank * dim_u + rank * dim_v :].reshape(dim_w, rank)
                        
                        U_int = np.rint(U)
                        V_int = np.rint(V)
                        W_int = np.rint(W)
                        if _verify_decomposition(U_int, V_int, W_int, T, tol=1e-7):
                            return U_int, V_int, W_int
                        
                        U_half = np.rint(U * 2) / 2
                        V_half = np.rint(V * 2) / 2
                        W_half = np.rint(W * 2) / 2
                        if _verify_decomposition(U_half, V_half, W_half, T, tol=1e-7):
                            return U_half, V_half, W_half
            except Exception:
                pass

    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin
    
    # Step 1: Get baseline rank 55
    U55, V55, W55 = _construct_rank_55()
    best_rank = 55
    best_U, best_V, best_W = U55, V55, W55
    
    # Step 2: Build target tensor
    T = _build_target_tensor()
    
    # Step 3: Try to find lower rank decompositions
    # We try ranks 54, 53...
    # Allocate time for each rank attempt
    
    # Try rank 54 first, using rank 55 as initialization
    if time.time() < DEADLINE - 20:
        time_per_attempt = min(20.0, (DEADLINE - time.time()) / 2)
        result = _try_find_rank(T, 54, n, m, p, time_per_attempt, init_U=U55, init_V=V55, init_W=W55)
        if result is not None:
            U_new, V_new, W_new = result
            best_rank = 54
            best_U, best_V, best_W = U_new, V_new, W_new
            
            # If found 54, try 53?
            if time.time() < DEADLINE - 10:
                time_per_attempt = min(10.0, (DEADLINE - time.time()) / 2)
                result = _try_find_rank(T, 53, n, m, p, time_per_attempt, init_U=U_new, init_V=V_new, init_W=W_new)
                if result is not None:
                    U_new, V_new, W_new = result
                    best_rank = 53
                    best_U, best_V, best_W = U_new, V_new, W_new

    # Convert to lists
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
