"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START

def _tensor_error(U, V, W, n, m, p):
    """Compute max error between decomposed tensor and delta_{jk}."""
    r = U.shape[1]
    T = np.zeros((n, m, p))
    for ri in range(r):
        u_col = U[:, ri].reshape(n, m)
        v_col = V[:, ri].reshape(m, p)
        w_col = W[:, ri].reshape(p, n)
        # T[i,j,k] += u[i,j] * v[j,k] * w[k,i]
        for i in range(n):
            for j in range(m):
                uj = u_col[i, j]
                if abs(uj) < 1e-14:
                    continue
                for k in range(p):
                    T[i, j, k] += uj * v_col[j, k] * w_col[k, i]
    T_expected = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            T_expected[i, j, j] = 1.0
    return np.max(np.abs(T - T_expected))


def _verify(U, V, W, tol=1e-6):
    n, m, p = DIMENSIONS
    r = len(U[0])
    Um = np.array(U, dtype=float)
    Vm = np.array(V, dtype=float)
    Wm = np.array(W, dtype=float)
    return _tensor_error(Um, Vm, Wm, n, m, p) < tol


def _snap_to_half(x, tol=0.4):
    """Snap to nearest integer or half-integer if close."""
    nearest_int = round(x)
    nearest_half = round(2 * x) / 2.0
    d_int = abs(x - nearest_int)
    d_half = abs(x - nearest_half)
    if d_int <= tol:
        return float(nearest_int)
    elif d_half <= tol:
        return float(nearest_half)
    return x


def _snap_matrix(M, tol=0.4):
    return [[_snap_to_half(x, tol) for x in row] for row in M]


def _als_optimize(n, m, p, rank, iters=200, lr=0.01, seed=0):
    """Gradient-based ALS optimization for tensor decomposition."""
    np.random.seed(seed)
    U = np.random.randn(n * m, rank) * 0.1
    V = np.random.randn(m * p, rank) * 0.1
    W = np.random.randn(p * n, rank) * 0.1
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(iters):
        # Compute current tensor
        T = np.zeros((n, m, p))
        for ri in range(rank):
            u_col = U[:, ri].reshape(n, m)
            v_col = V[:, ri].reshape(m, p)
            w_col = W[:, ri].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    uj = u_col[i, j]
                    if abs(uj) < 1e-14:
                        continue
                    for k in range(p):
                        T[i, j, k] += uj * v_col[j, k] * w_col[k, i]
        
        T_expected = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                T_expected[i, j, j] = 1.0
        
        err = np.max(np.abs(T - T_expected))
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        if err < 1e-8:
            break
        
        # Compute gradients using finite differences or analytical
        # dT[i,j,k] / dU[i*m+j, r] = V[j*p+k, r] * W[k*n+i, r]
        # dT[i,j,k] / dV[j*p+k, r] = U[i*m+j, r] * W[k*n+i, r]
        # dT[i,j,k] / dW[k*n+i, r] = U[i*m+j, r] * V[j*p+k, r]
        
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for ri in range(rank):
            u_col = U[:, ri].reshape(n, m)
            v_col = V[:, ri].reshape(m, p)
            w_col = W[:, ri].reshape(p, n)
            
            for i in range(n):
                for j in range(m):
                    uj = u_col[i, j]
                    for k in range(p):
                        residual = T[i, j, k] - T_expected[i, j, k]
                        # Gradient w.r.t. U[i*m+j, ri]
                        grad_U[i * m + j, ri] += residual * v_col[j, k] * w_col[k, i]
                        # Gradient w.r.t. V[j*p+k, ri]
                        grad_V[j * p + k, ri] += residual * u_col[i, j] * w_col[k, i]
                        # Gradient w.r.t. W[k*n+i, ri]
                        grad_W[k * n + i, ri] += residual * u_col[i, j] * v_col[j, k]
        
        # Update with learning rate
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
    
    return best_U, best_V, best_W, best_err


def _try_rank(n, m, p, rank, num_seeds=10, iters=150, deadline=None):
    """Try to find a valid decomposition with given rank."""
    best_result = None
    best_err = float('inf')
    
    for seed in range(num_seeds):
        if deadline is not None and time.time() > deadline:
            break
        
        U, V, W, err = _als_optimize(n, m, p, rank, iters=iters, seed=seed)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < 1e-6:
            # Found a solution, try to snap to half-integers
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            
            U_snapped = _snap_matrix(U_list)
            V_snapped = _snap_matrix(V_list)
            W_snapped = _snap_matrix(W_list)
            
            # Verify snapped version
            if _verify(U_snapped, V_snapped, W_snapped, tol=1e-6):
                return {"U": U_snapped, "V": V_snapped, "W": W_snapped}, err
            
            # If snapped doesn't work, use unsnapped (still valid if error small)
            if err < 1e-8:
                return {"U": U_list, "V": V_list, "W": W_list}, err
    
    if best_err < 1e-6:
        U_list = [[float(x) for x in row] for row in best_U]
        V_list = [[float(x) for x in row] for row in best_V]
        W_list = [[float(x) for x in row] for row in best_W]
        return {"U": U_list, "V": V_list, "W": W_list}, best_err
    
    return None, best_err


def _schoolbook():
    n, m, p = DIMENSIONS
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


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s
    
    best_result = _schoolbook()
    best_rank = n * m * p
    
    # Try ranks from low to high
    # Known: rank 6 should be achievable for 2x3 * 3x3
    for rank in range(6, best_rank + 1):
        if time.time() > deadline - 10:
            break
        
        result, err = _try_rank(n, m, p, rank, num_seeds=20, iters=100, deadline=deadline - 5)
        
        if result is not None and err < 1e-6:
            best_result = result
            best_rank = rank
            break
    
    return best_result

# EVOLVE-BLOCK-END
