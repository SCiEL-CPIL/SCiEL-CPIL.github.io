"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
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
    return {"U": U, "V": V, "W": W, "rank": rank}

def _verify(decomp, n, m, p):
    """Verify that the tensor decomposition exactly multiplies the matrices."""
    U = decomp["U"]
    V = decomp["V"]
    W = decomp["W"]
    r = len(U[0])
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for idx in range(r):
                    val += U[i*m + j][idx] * V[j*p + k][idx] * W[k*n + i][idx]
                if not (abs(val - 1.0) < 1e-9):
                    return False
    return True

def _verify_exact(decomp, n, m, p):
    """Exact verification using fractions."""
    U = decomp["U"]
    V = decomp["V"]
    W = decomp["W"]
    r = len(U[0])
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = Fraction(0, 1)
                for idx in range(r):
                    val += Fraction(U[i*m + j][idx]) * Fraction(V[j*p + k][idx]) * Fraction(W[k*n + i][idx])
                if val != 1:
                    return False
    return True

def _try_rank(n, m, p, rank, max_iter=2000, seed=42):
    """Try to find a rank-r decomposition using randomized local search with adaptive learning rate."""
    rng = np.random.RandomState(seed)
    num_u = n * m
    num_v = m * p
    num_w = p * n
    
    # Initialize randomly with smaller scale
    U = rng.randn(num_u, rank) * 0.05
    V = rng.randn(num_v, rank) * 0.05
    W = rng.randn(num_w, rank) * 0.05
    
    lr = 0.01
    best_loss = float('inf')
    patience = 200
    no_improve = 0
    
    for it in range(max_iter):
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        loss = 0.0
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    
                    val = np.sum(U[u_idx] * V[v_idx] * W[w_idx])
                    diff = val - 1.0
                    loss += diff * diff
                    
                    factor = 2.0 * diff
                    grad_U[u_idx] += factor * V[v_idx] * W[w_idx]
                    grad_V[v_idx] += factor * U[u_idx] * W[w_idx]
                    grad_W[w_idx] += factor * U[u_idx] * V[v_idx]
        
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        if loss < best_loss:
            best_loss = loss
            no_improve = 0
        else:
            no_improve += 1
        
        # Adaptive learning rate
        if no_improve > patience:
            lr *= 0.5
            no_improve = 0
            if lr < 1e-8:
                break
        
        if loss < 1e-14:
            break
    
    if loss < 1e-12:
        # Try to snap to half-integers
        U_snapped = np.round(U * 2) / 2
        V_snapped = np.round(V * 2) / 2
        W_snapped = np.round(W * 2) / 2
        
        # Check if snapped version still works
        U_list = U_snapped.tolist()
        V_list = V_snapped.tolist()
        W_list = W_snapped.tolist()
        
        decomp = {"U": U_list, "V": V_list, "W": W_list, "rank": rank}
        if _verify(decomp, n, m, p):
            return decomp
        
        # Try original
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        decomp = {"U": U_list, "V": V_list, "W": W_list, "rank": rank}
        if _verify(decomp, n, m, p):
            return decomp
    
    return None

def construct():
    start_time = time.time()
    deadline = 145.0
    
    n, m, p = DIMENSIONS
    best_decomp = _schoolbook()
    best_rank = best_decomp["rank"]
    
    # Try ranks from lower to upper
    for rank in range(15, best_rank):
        if time.time() > deadline:
            break
        
        for seed in range(5):
            if time.time() > deadline:
                break
            
            result = _try_rank(n, m, p, rank, max_iter=1500, seed=seed + 50)
            if result is not None:
                if _verify(result, n, m, p):
                    best_decomp = result
                    best_rank = rank
                    break
        if best_rank == rank:
            break
    
    return {"U": best_decomp["U"], "V": best_decomp["V"], "W": best_decomp["W"]}
# EVOLVE-BLOCK-END
