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

def _snap_to_half_integers(U, V, W, n, m, p):
    """Try to snap to half-integer values and verify."""
    # Try multiple snapping strategies
    for scale in [2, 4, 8]:
        U_snap = np.round(U * scale) / scale
        V_snap = np.round(V * scale) / scale
        W_snap = np.round(W * scale) / scale
        
        U_list = U_snap.tolist()
        V_list = V_snap.tolist()
        W_list = W_snap.tolist()
        
        decomp = {"U": U_list, "V": V_list, "W": W_list}
        if _verify(decomp, n, m, p):
            return decomp
    
    # Also try original
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    decomp = {"U": U_list, "V": V_list, "W": W_list}
    if _verify(decomp, n, m, p):
        return decomp
    
    return None

def _try_rank_improved(n, m, p, rank, max_iter=3000, seed=42, time_limit=10.0):
    """Try to find a rank-r decomposition using improved optimization with momentum."""
    rng = np.random.RandomState(seed)
    num_u = n * m
    num_v = m * p
    num_w = p * n
    
    start_time = time.time()
    
    # Structured initialization: start from schoolbook and perturb
    U = np.zeros((num_u, rank))
    V = np.zeros((num_v, rank))
    W = np.zeros((num_w, rank))
    
    # Initialize first min(rank, n*m*p) columns with schoolbook pattern
    schoolbook_rank = min(rank, n * m * p)
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if column < rank:
                    U[i * m + j, column] = 1.0
                    V[j * p + k, column] = 1.0
                    W[k * n + i, column] = 1.0
                    column += 1
    
    # Add random perturbations to remaining columns or all columns
    for col in range(rank):
        if col >= schoolbook_rank:
            U[:, col] = rng.randn(num_u) * 0.1
            V[:, col] = rng.randn(num_v) * 0.1
            W[:, col] = rng.randn(num_w) * 0.1
        else:
            # Slight perturbation to existing
            U[:, col] += rng.randn(num_u) * 0.05
            V[:, col] += rng.randn(num_v) * 0.05
            W[:, col] += rng.randn(num_w) * 0.05
    
    # Adam-style optimization
    lr = 0.01
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    
    m_U = np.zeros_like(U)
    v_U = np.zeros_like(U)
    m_V = np.zeros_like(V)
    v_V = np.zeros_like(V)
    m_W = np.zeros_like(W)
    v_W = np.zeros_like(W)
    
    best_loss = float('inf')
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    patience = 500
    no_improve = 0
    t = 0
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        t += 1
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
        
        # Adam update
        m_U = beta1 * m_U + (1 - beta1) * grad_U
        v_U = beta2 * v_U + (1 - beta2) * (grad_U ** 2)
        m_hat_U = m_U / (1 - beta1 ** t)
        v_hat_U = v_U / (1 - beta2 ** t)
        U -= lr * m_hat_U / (np.sqrt(v_hat_U) + eps)
        
        m_V = beta1 * m_V + (1 - beta1) * grad_V
        v_V = beta2 * v_V + (1 - beta2) * (grad_V ** 2)
        m_hat_V = m_V / (1 - beta1 ** t)
        v_hat_V = v_V / (1 - beta2 ** t)
        V -= lr * m_hat_V / (np.sqrt(v_hat_V) + eps)
        
        m_W = beta1 * m_W + (1 - beta1) * grad_W
        v_W = beta2 * v_W + (1 - beta2) * (grad_W ** 2)
        m_hat_W = m_W / (1 - beta1 ** t)
        v_hat_W = v_W / (1 - beta2 ** t)
        W -= lr * m_hat_W / (np.sqrt(v_hat_W) + eps)
        
        if loss < best_loss:
            best_loss = loss
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
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
    
    # Use best found
    U, V, W = best_U, best_V, best_W
    
    if best_loss < 1e-10:
        # Try to snap to half-integers
        result = _snap_to_half_integers(U, V, W, n, m, p)
        if result is not None:
            return result
    
    return None

def construct():
    start_time = time.time()
    deadline = 145.0
    
    n, m, p = DIMENSIONS
    best_decomp = _schoolbook()
    best_rank = best_decomp["rank"]
    
    # Try ranks from lower to upper with more seeds
    for rank in range(15, best_rank):
        if time.time() > deadline:
            break
        
        found = False
        for seed in range(10):
            if time.time() > deadline:
                break
            
            remaining_time = deadline - time.time()
            if remaining_time < 15:
                break
            
            time_per_try = min(10.0, remaining_time / 3.0)
            result = _try_rank_improved(n, m, p, rank, max_iter=5000, seed=seed + 50, time_limit=time_per_try)
            if result is not None:
                if _verify(result, n, m, p):
                    best_decomp = result
                    best_rank = rank
                    found = True
                    break
        
        if found:
            break
    
    return {"U": best_decomp["U"], "V": best_decomp["V"], "W": best_decomp["W"]}
# EVOLVE-BLOCK-END
