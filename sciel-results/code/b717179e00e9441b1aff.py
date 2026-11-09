import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 6]
DEADLINE = 150.0
SAFETY_MARGIN = 5.0

def _verify(U, V, W, tol=1e-10):
    n, m, p = DIMENSIONS
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    # Compute tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Compare with expected: 1 if i == k else 0
    U_3d = U.reshape(n, m, rank)
    V_3d = V.reshape(m, p, rank)
    W_3d = W.reshape(p, n, rank)
    
    result = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
    
    expected = np.zeros((n, m, p))
    for i in range(n):
        for k in range(p):
            if i == k:
                expected[i, :, k] = 1.0
    
    if not np.allclose(result, expected, atol=tol):
        return False
    return True

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

def _try_rank_with_structure(n, m, p, rank, rng, max_iters=50):
    """
    Try to find a rank-rank decomposition using structured random search.
    Key insight: the tensor is multilinear, so we can exploit structure.
    """
    um, vm, wm = n*m, m*p, p*n
    val_set = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
    
    for it in range(max_iters):
        # Sparse initialization: each column has a few non-zeros
        U = np.zeros((um, rank))
        V = np.zeros((vm, rank))
        W = np.zeros((wm, rank))
        
        for c in range(rank):
            # 2-3 non-zeros per column
            nz_u = rng.randint(2, 3)
            nz_v = rng.randint(2, 3)
            nz_w = rng.randint(2, 3)
            
            idx_u = rng.sample(range(um), nz_u)
            idx_v = rng.sample(range(vm), nz_v)
            idx_w = rng.sample(range(wm), nz_w)
            
            for idx in idx_u:
                U[idx, c] = rng.choice(val_set)
            for idx in idx_v:
                V[idx, c] = rng.choice(val_set)
            for idx in idx_w:
                W[idx, c] = rng.choice(val_set)
        
        if _verify(U, V, W):
            return {"U": U, "V": V, "W": W}
    
    return None

def _try_dense_random(n, m, p, rank, rng, max_iters=20):
    """Dense random search with half-integer entries."""
    um, vm, wm = n*m, m*p, p*n
    val_set = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
    
    for it in range(max_iters):
        U = np.array([[rng.choice(val_set) for _ in range(rank)] for _ in range(um)])
        V = np.array([[rng.choice(val_set) for _ in range(rank)] for _ in range(vm)])
        W = np.array([[rng.choice(val_set) for _ in range(rank)] for _ in range(wm)])
        
        if _verify(U, V, W):
            return {"U": U, "V": V, "W": W}
    
    return None

def construct():
    start_time = time.time()
    
    # Baseline
    best = _schoolbook()
    best_rank = best["U"][0].__len__() if best["U"] else 54
    
    n, m, p = DIMENSIONS
    
    # Try ranks from 53 down to 30
    # Use both sparse and dense random search
    for target_rank in range(53, 29, -1):
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break
        
        # Multiple attempts with different seeds
        for attempt in range(8):
            if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
                break
            
            rng = random.Random(target_rank * 1000 + attempt * 7 + 42)
            
            # Try sparse structured search
            result = _try_rank_with_structure(n, m, p, target_rank, rng, max_iters=30)
            if result is not None:
                # Convert to lists
                return {
                    "U": result["U"].tolist(),
                    "V": result["V"].tolist(),
                    "W": result["W"].tolist()
                }
            
            # Try dense random search (fewer iterations as it's slower)
            result = _try_dense_random(n, m, p, target_rank, rng, max_iters=5)
            if result is not None:
                return {
                    "U": result["U"].tolist(),
                    "V": result["V"].tolist(),
                    "W": result["W"].tolist()
                }
    
    return best
