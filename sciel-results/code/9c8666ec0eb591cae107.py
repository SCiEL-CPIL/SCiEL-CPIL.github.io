import time
import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    sub_n, sub_m, sub_p = 2, 2, 4
    
    best_rank = 14
    best_U, best_V, best_W = None, None, None
    
    for R in [10, 11, 12, 13]:
        if time.time() > deadline - 20:
            break
        found = _find_decomposition(sub_n, sub_m, sub_p, R)
        if found is not None:
            best_rank = R
            best_U, best_V, best_W = found
            break
    
    if best_U is None:
        best_rank = 14
        best_U, best_V, best_W = _strassen_2x2x4()
    
    num_tiles = p // sub_p
    total_rank = num_tiles * best_rank
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    for t in range(num_tiles):
        for r in range(best_rank):
            r_glob = t * best_rank + r
            for i in range(n):
                for j in range(m):
                    U[i * m + j][r_glob] = best_U[i * m + j][r]
            for j in range(m):
                for k in range(sub_p):
                    k_glob = t * sub_p + k
                    V[j * p + k_glob][r_glob] = best_V[j * sub_p + k][r]
            for k in range(sub_p):
                for i in range(n):
                    k_glob = t * sub_p + k
                    W[k_glob * n + i][r_glob] = best_W[k * sub_n + i][r]
    
    return {"U": U, "V": V, "W": W}


def _find_decomposition(n, m, p, R, max_iter=5000, tol=1e-10):
    """Find rank-R decomposition using alternating minimization with multiple restarts."""
    start = time.time()
    
    for seed in range(5):
        if time.time() > start + 15:
            break
        np.random.seed(seed)
        U = np.random.randn(n * m, R)
        V = np.random.randn(m * p, R)
        W = np.random.randn(p * n, R)
        
        # Scale to help convergence
        U /= np.sqrt(R)
        V /= np.sqrt(R)
        W /= np.sqrt(R)
        
        for iteration in range(max_iter):
            if time.time() > start + 15:
                break
            
            # Update U
            for i in range(n):
                for j in range(m):
                    A_mat = np.zeros((p, R))
                    for k in range(p):
                        for r in range(R):
                            A_mat[k, r] = V[j * p + k, r] * W[k * n + i, r]
                    b = np.ones(p)
                    try:
                        sol, _, _, _ = np.linalg.lstsq(A_mat, b, rcond=None)
                        U[i * m + j, :] = sol
                    except:
                        U = None
                        break
                if U is None:
                    break
            
            if U is None:
                break
            
            # Update V
            for j in range(m):
                for k in range(p):
                    A_mat = np.zeros((n, R))
                    for i in range(n):
                        for r in range(R):
                            A_mat[i, r] = U[i * m + j, r] * W[k * n + i, r]
                    b = np.ones(n)
                    try:
                        sol, _, _, _ = np.linalg.lstsq(A_mat, b, rcond=None)
                        V[j * p + k, :] = sol
                    except:
                        V = None
                        break
                if V is None:
                    break
            
            if V is None:
                break
            
            # Update W
            for k in range(p):
                for i in range(n):
                    A_mat = np.zeros((m, R))
                    for j in range(m):
                        for r in range(R):
                            A_mat[j, r] = U[i * m + j, r] * V[j * p + k, r]
                    b = np.ones(m)
                    try:
                        sol, _, _, _ = np.linalg.lstsq(A_mat, b, rcond=None)
                        W[k * n + i, :] = sol
                    except:
                        W = None
                        break
                if W is None:
                    break
            
            if W is None:
                break
            
            # Check convergence
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                        max_err = max(max_err, abs(val - 1.0))
            
            if max_err < tol:
                return (U.tolist(), V.tolist(), W.tolist())
    
    return None


def _strassen_2x2x4():
    """Two Strassen blocks for 2x2 x 2x4."""
    n, m, p = 2, 2, 4
    R = 14
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    terms = [
        {'U': {(0,0): 1, (1,1): 1}, 'V': {(0,0): 1, (1,1): 1}, 'W': {(0,0): 1, (1,1): 1}},
        {'U': {(1,0): 1, (1,1): 1}, 'V': {(0,0): 1}, 'W': {(0,1): 1, (1,1): -1}},
        {'U': {(0,0): 1}, 'V': {(0,1): 1, (1,1): -1}, 'W': {(1,0): 1, (1,1): 1}},
        {'U': {(1,1): 1}, 'V': {(1,0): 1, (0,0): -1}, 'W': {(0,0): 1, (0,1): 1}},
        {'U': {(0,0): 1, (0,1): 1}, 'V': {(1,1): 1}, 'W': {(0,0): -1, (1,0): 1}},
        {'U': {(1,0): 1, (0,0): -1}, 'V': {(0,0): 1, (0,1): 1}, 'W': {(1,1): 1}},
        {'U': {(0,1): 1, (1,1): -1}, 'V': {(1,0): 1, (1,1): 1}, 'W': {(0,0): 1}}
    ]
    
    for b in range(2):
        for t_idx in range(7):
            r = b * 7 + t_idx
            term = terms[t_idx]
            for (i, j), val in term['U'].items():
                U[i * m + j][r] = val
            for (j, k_loc), val in term['V'].items():
                k_glob = b * 2 + k_loc
                V[j * p + k_glob][r] = val
            for (k_loc, i), val in term['W'].items():
                k_glob = b * 2 + k_loc
                W[k_glob * n + i][r] = val
    
    return U, V, W
# EVOLVE-BLOCK-END
