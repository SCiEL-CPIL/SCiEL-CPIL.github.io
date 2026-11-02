import time
import sys
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 6]

def _verify(U, V, W):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    n, m, p = 3, 3, 6
    rank = len(U[0])
    if len(U) != n * m:
        return False
    if len(V) != m * p:
        return False
    if len(W) != p * n:
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _verify_3x3x3(U, V, W):
    """Verify that U, V, W exactly decompose the 3x3x3 matrix multiplication tensor."""
    n, m, p = 3, 3, 3
    rank = len(U[0])
    if len(U) != n * m:
        return False
    if len(V) != m * p:
        return False
    if len(W) != p * n:
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _try_rank_3x3x3(rank, max_iters=500, time_limit=30.0, lr_init=0.01, seed=42):
    """Try to find a rank-`rank` decomposition for 3x3x3 via numerical optimization."""
    n, m, p = 3, 3, 3
    t0 = time.time()
    np.random.seed(seed)
    
    U = np.random.randn(n * m, rank)
    V = np.random.randn(m * p, rank)
    W = np.random.randn(p * n, rank)
    
    lr = lr_init
    best_loss = float('inf')
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    
    for it in range(max_iters):
        if time.time() - t0 > time_limit:
            break
        
        residual = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    residual[i, j, k] = s - 1.0
        
        loss = np.sum(residual ** 2)
        if loss < best_loss:
            best_loss = loss
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
            if loss < 1e-12:
                break
        
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r_ijk = residual[i, j, k]
                    for r in range(rank):
                        dU[i * m + j, r] += 2 * r_ijk * V[j * p + k, r] * W[k * n + i, r]
                        dV[j * p + k, r] += 2 * r_ijk * U[i * m + j, r] * W[k * n + i, r]
                        dW[k * n + i, r] += 2 * r_ijk * U[i * m + j, r] * V[j * p + k, r]
        
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        lr *= 0.99
        
    if best_loss < 1e-9:
        U_list = [[float(U[i, r]) for r in range(rank)] for i in range(n * m)]
        V_list = [[float(V[i, r]) for r in range(rank)] for i in range(m * p)]
        W_list = [[float(W[i, r]) for r in range(rank)] for i in range(p * n)]
        if _verify_3x3x3(U_list, V_list, W_list):
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def _construct_from_3x3x3(sol_3x3x3):
    """Construct a 3x3x6 decomposition from a 3x3x3 decomposition."""
    n, m, p = 3, 3, 6
    U1 = sol_3x3x3["U"]
    V1 = sol_3x3x3["V"]
    W1 = sol_3x3x3["W"]
    rank1 = len(U1[0])
    rank = 2 * rank1
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # First block: k=0,1,2
    for i in range(n):
        for j in range(m):
            for r in range(rank1):
                U[i * m + j][r] = U1[i * m + j][r]
                U[i * m + j][r + rank1] = U1[i * m + j][r]
        
        for j in range(m):
            for k in range(3):
                V[j * p + k][r] = V1[j * 3 + k][r]
                V[j * p + k][r + rank1] = 0.0
            
    for k in range(3):
        for i in range(n):
            for r in range(rank1):
                W[k * n + i][r] = W1[k * n + i][r]
                W[k * n + i][r + rank1] = 0.0

    # Second block: k=3,4,5
    for j in range(m):
        for k in range(3):
            for r in range(rank1):
                V[j * p + (k + 3)][r + rank1] = V1[j * 3 + k][r]
        
    for k in range(3):
        for i in range(n):
            for r in range(rank1):
                W[(k + 3) * n + i][r + rank1] = W1[k * n + i][r]
                
    return {"U": U, "V": V, "W": W}

def _schoolbook():
    n, m, p = 3, 3, 6
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

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 115.0
    
    # Try to find a rank 23 decomposition for 3x3x3
    # If found, construct rank 46 for 3x3x6
    if time.time() < deadline - 60:
        sol_3x3x3 = _try_rank_3x3x3(23, max_iters=1000, time_limit=50.0, seed=42)
        if sol_3x3x3 is not None:
            result = _construct_from_3x3x3(sol_3x3x3)
            if _verify(result["U"], result["V"], result["W"]):
                return result
        
        # Try rank 24
        if time.time() < deadline - 40:
            sol_3x3x3 = _try_rank_3x3x3(24, max_iters=1000, time_limit=30.0, seed=123)
            if sol_3x3x3 is not None:
                result = _construct_from_3x3x3(sol_3x3x3)
                if _verify(result["U"], result["V"], result["W"]):
                    return result

    # Fall back to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
