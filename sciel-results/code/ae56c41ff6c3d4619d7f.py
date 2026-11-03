"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def _construct_tensor():
    n, m, p = DIMENSIONS
    # Tensor T[a, b, c] where a in [0, n*m), b in [0, m*p), c in [0, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _verify(U, V, W, T):
    n, m, p = DIMENSIONS
    rank = len(U[0]) if U else 0
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    # Reconstruct: sum_r U[:,r] * V[:,r] * W[:,r]
    recon = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(recon, T, atol=1e-10)

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _try_rank_decomposition(T, target_rank, max_iter=1000, tol=1e-8):
    """Try to find a rank-target_rank decomposition using ALS."""
    n, m, p = DIMENSIONS
    shape = T.shape
    U = np.random.randn(shape[0], target_rank)
    V = np.random.randn(shape[1], target_rank)
    W = np.random.randn(shape[2], target_rank)
    
    for it in range(max_iter):
        # Update U
        # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # Flatten b,c to k. T[a,k] = sum_r U[a,r] * (V[b,r] W[c,r])
        # Let X[k,r] = V[b,r] W[c,r]. Then T_mat = U @ X.T
        # T_mat.T = X @ U.T
        # Solve X @ Z = T_mat.T for Z (which is U.T)
        X = np.einsum('br,cr->bcr', V, W).reshape(-1, target_rank)
        T_matted = T.reshape(shape[0], -1)
        Z, _, _, _ = np.linalg.lstsq(X, T_matted.T, rcond=None)
        U = Z.T
        
        # Update V
        # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # Flatten a,c to k. T[k,b] = sum_r (U[a,r] W[c,r]) * V[b,r]
        # Let X[k,r] = U[a,r] W[c,r]. Then T_mat_V = V @ X.T
        # T_mat_V.T = X @ V.T
        X = np.einsum('ar,cr->acr', U, W).reshape(-1, target_rank)
        T_matted = T.transpose(1, 0, 2).reshape(shape[1], -1)
        Z, _, _, _ = np.linalg.lstsq(X, T_matted.T, rcond=None)
        V = Z.T
        
        # Update W
        # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # Flatten a,b to k. T[k,c] = sum_r (U[a,r] V[b,r]) * W[c,r]
        # Let X[k,r] = U[a,r] V[b,r]. Then T_mat_W = W @ X.T
        # T_mat_W.T = X @ W.T
        X = np.einsum('ar,br->abr', U, V).reshape(-1, target_rank)
        T_matted = T.transpose(2, 0, 1).reshape(shape[2], -1)
        Z, _, _, _ = np.linalg.lstsq(X, T_matted.T, rcond=None)
        W = Z.T
        
        # Check convergence
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.linalg.norm(recon - T) / np.linalg.norm(T)
        if err < tol:
            return U, V, W, err
    
    recon = np.einsum('ar,br,cr->abc', U, V, W)
    err = np.linalg.norm(recon - T) / np.linalg.norm(T)
    return U, V, W, err

def construct():
    start_time = time.time()
    deadline = start_time + 120
    
    T = _construct_tensor()
    
    # Start with schoolbook
    best = _schoolbook()
    best_rank = len(best["U"][0])
    
    # Try to reduce rank
    for target_rank in range(best_rank - 1, 20, -1):
        if time.time() > deadline - 30:
            break
        # Try multiple initializations
        found = False
        for trial in range(5):
            if time.time() > deadline - 30:
                break
            np.random.seed(trial * 1000 + target_rank)
            U, V, W, err = _try_rank_decomposition(T, target_rank, max_iter=500, tol=1e-9)
            if err < 1e-8:
                # Verify
                U_arr = np.array(U)
                V_arr = np.array(V)
                W_arr = np.array(W)
                recon = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
                if np.allclose(recon, T, atol=1e-8):
                    # Convert to list of lists
                    U_list = [[float(x) for x in row] for row in U_arr]
                    V_list = [[float(x) for x in row] for row in V_arr]
                    W_list = [[float(x) for x in row] for row in W_arr]
                    best = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = target_rank
                    found = True
                    break
        if found:
            continue
        else:
            # If we can't achieve this rank, try lower
            break
    
    return best
# EVOLVE-BLOCK-END
