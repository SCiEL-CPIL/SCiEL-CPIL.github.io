"""Matrix multiplication 3x4x5 rank-minimal exact tensor decomposition search."""
import time
import random
import numpy as np

DIMENSIONS = [3, 4, 5]

def _build_tensor():
    n, m, p = DIMENSIONS
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1
    return T

def _validate(U, V, W, T, atol=1e-9):
    n, m, p = T.shape
    rank = U.shape[1]
    recon = np.zeros((n, m, p))
    for r in range(rank):
        u = U[:, r]
        v = V[:, r]
        w = W[:, r]
        u_mat = u.reshape(n, m)
        v_mat = v.reshape(m, p)
        w_mat = w.reshape(p, n)
        # contribution: sum over j of u_mat[i,j]*v_mat[j,k]*w_mat[k,i]
        inner = v_mat @ w_mat  # (m, p) @ (p, n) -> (m, n)
        # inner[j,i] = sum_k v[j,k]*w[k,i]
        # contrib[i,j] = sum_j u_mat[i,j] * inner[j,i]
        # Actually: contrib[i,j,k] is not separable this way.
        # Let's do it explicitly for correctness
        for i in range(n):
            for j in range(m):
                uj = u_mat[i, j]
                if uj == 0:
                    continue
                for k in range(p):
                    recon[i, j, k] += uj * v_mat[j, k] * w_mat[k, i]
    return np.allclose(recon, T, atol=atol)

def _is_half_integer(arr, max_val=1000):
    """Check if all entries are half-integers (x.0 or x.5) with |x| <= max_val."""
    if not np.all(np.isfinite(arr)):
        return False
    doubled = arr * 2
    if not np.all(np.isclose(doubled, np.round(doubled), atol=1e-6)):
        return False
    if not np.all(np.abs(arr) <= max_val):
        return False
    return True

def _snap_half(arr):
    """Snap to nearest half-integer."""
    return np.round(arr * 2) / 2

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W, rank

def _try_reduce_one(U, V, W, T, r_remove, max_val=1000):
    """Try to remove column r_remove by solving for adjusted U."""
    n, m, p = DIMENSIONS
    R = U.shape[1]
    if R <= 1:
        return None
    
    mask = [i for i in range(R) if i != r_remove]
    U_rem = U[:, mask]
    V_rem = V[:, mask]
    W_rem = W[:, mask]
    
    # Compute residual from removed column
    residual = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            uj = U[i * m + j, r_remove]
            if uj != 0:
                for k in range(p):
                    residual[i, j, k] = uj * V[j * p + k, r_remove] * W[k * n + i, r_remove]
    
    # Solve for delta_U
    delta_U = np.zeros((n * m, len(mask)))
    solvable = True
    for i in range(n):
        for j in range(m):
            M = np.zeros((p, len(mask)))
            for k in range(p):
                M[k, :] = V_rem[j * p + k, :] * W_rem[k * n + i, :]
            try:
                sol, res, rank_s, sv = np.linalg.lstsq(M.T, residual[i, j, :], rcond=None)
                delta_U[i * m + j, :] = sol
            except:
                solvable = False
                break
        if not solvable:
            break
    
    if not solvable:
        return None
    
    U_new = U_rem + delta_U
    
    # Check if U_new has half-integer entries
    if _is_half_integer(U_new, max_val=max_val):
        return U_new, V_rem, W_rem
    
    return None

def _try_reduce_rank(U, V, W, T, max_time=10, max_val=1000):
    """Try to reduce rank by removing one term at a time."""
    n, m, p = DIMENSIONS
    R = U.shape[1]
    start = time.time()
    
    while R > 1:
        if time.time() - start > max_time:
            break
        
        found = False
        for r_remove in range(R):
            if time.time() - start > max_time:
                break
            
            result = _try_reduce_one(U, V, W, T, r_remove, max_val=max_val)
            if result is not None:
                U, V, W = result
                R = U.shape[1]
                found = True
                break
        
        if not found:
            break
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    T = _build_tensor()
    
    # Start with schoolbook
    U, V, W, rank = _schoolbook()
    best = (U.copy(), V.copy(), W.copy(), rank)
    
    # Time budget
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for 150s limit
    
    rng = random.Random(42)
    np.random.seed(42)
    
    # Try random initializations with decreasing rank
    for R in range(35, 60):
        if time.time() > deadline - 20:
            break
        
        for attempt in range(5):
            if time.time() > deadline - 15:
                break
            
            # Random half-integer initialization
            def rand_half():
                return rng.choice([0, 0.5, 1, 1.5, 2, -0.5, -1, -1.5, -2])
            
            V_init = np.array([[rand_half() for _ in range(R)] for _ in range(m * p)])
            W_init = np.array([[rand_half() for _ in range(R)] for _ in range(p * n)])
            
            # Solve for U
            U_init = np.zeros((n * m, R))
            solvable = True
            for i in range(n):
                for j in range(m):
                    M = np.zeros((p, R))
                    for k in range(p):
                        M[k, :] = V_init[j * p + k, :] * W_init[k * n + i, :]
                    try:
                        sol, res, rank_s, sv = np.linalg.lstsq(M.T, T[i, j, :], rcond=None)
                        U_init[i * m + j, :] = sol
                    except:
                        solvable = False
                        break
                if not solvable:
                    break
            
            if not solvable:
                continue
            
            # Snap to half-integers
            U_init = _snap_half(U_init)
            V_init = _snap_half(V_init)
            W_init = _snap_half(W_init)
            
            # Validate
            if not _validate(U_init, V_init, W_init, T):
                continue
            
            # Try to reduce
            U_red, V_red, W_red = _try_reduce_rank(U_init, V_init, W_init, T, max_time=10)
            
            R_final = U_red.shape[1]
            if R_final < best[3]:
                best = (U_red.copy(), V_red.copy(), W_red.copy(), R_final)
    
    # Also try reducing from schoolbook
    if time.time() < deadline - 10:
        U_red, V_red, W_red = _try_reduce_rank(U.copy(), V.copy(), W.copy(), T, max_time=30)
        R_final = U_red.shape[1]
        if R_final < best[3]:
            best = (U_red.copy(), V_red.copy(), W_red.copy(), R_final)
    
    # Convert to lists
    U_best, V_best, W_best, R_best = best
    U_list = [[float(U_best[i, r]) for r in range(R_best)] for i in range(n * m)]
    V_list = [[float(V_best[i, r]) for r in range(R_best)] for i in range(m * p)]
    W_list = [[float(W_best[i, r]) for r in range(R_best)] for i in range(p * n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
