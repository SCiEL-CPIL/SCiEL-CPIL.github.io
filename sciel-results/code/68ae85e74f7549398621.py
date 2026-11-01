"""Matrix multiplication 2x3x10 - rank-minimal tensor decomposition."""
import numpy as np
import time
from itertools import combinations

DIMENSIONS = [2, 3, 10]
n, m, p = DIMENSIONS

def _target_tensor():
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _residual(U, V, W, T):
    if U.shape[1] == 0:
        return np.linalg.norm(T)
    current = np.zeros((n*m, m*p, p*n))
    for r in range(U.shape[1]):
        current += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    return np.linalg.norm(current - T)

def _build_schoolbook():
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W, rank

def _local_search(U, V, W, T, max_iters=300):
    """Local search to reduce residual."""
    U = U.copy()
    V = V.copy()
    W = W.copy()
    lr = 0.01
    best_res = _residual(U, V, W, T)
    
    for it in range(max_iters):
        current = np.zeros((n*m, m*p, p*n))
        for r in range(U.shape[1]):
            current += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        diff = current - T
        res = np.sum(diff**2)
        
        if res < best_res:
            best_res = res
            if res < 1e-20:
                break
        
        # Compute gradients
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        for r in range(U.shape[1]):
            grad_U[:, r] = np.einsum('abc,b,c->a', diff, V[:, r], W[:, r])
            grad_V[:, r] = np.einsum('abc,a,c->b', diff, U[:, r], W[:, r])
            grad_W[:, r] = np.einsum('abc,a,b->c', diff, U[:, r], V[:, r])
        
        # Normalize gradients
        gnorm = np.linalg.norm(grad_U) + np.linalg.norm(grad_V) + np.linalg.norm(grad_W)
        if gnorm < 1e-15:
            break
        
        # Adaptive step
        step = lr / (1 + gnorm)
        U -= 2 * step * grad_U
        V -= 2 * step * grad_V
        W -= 2 * step * grad_W
        
        if it % 50 == 0:
            lr *= 0.95
    
    return U, V, W, best_res

_start_time = time.time()

def construct():
    global _start_time
    _start_time = time.time()
    
    T = _target_tensor()
    U, V, W, rank = _build_schoolbook()
    best_U, best_V, best_W, best_rank = U.copy(), V.copy(), W.copy(), rank
    
    # Try to reduce rank by removing terms
    for new_rank in range(rank - 1, 29, -1):
        if time.time() - _start_time > 110:
            break
        
        # Try removing different sets of terms
        # Start with removing the last (rank - new_rank) terms
        removed = rank - new_rank
        keep = list(range(new_rank))
        
        U_r = best_U[:, keep]
        V_r = best_V[:, keep]
        W_r = best_W[:, keep]
        
        U_opt, V_opt, W_opt, res = _local_search(U_r, V_r, W_r, T, max_iters=200)
        
        if res < 1e-10:
            best_U, best_V, best_W, best_rank = U_opt, V_opt, W_opt, new_rank
        else:
            # Try a different set: remove random terms
            for trial in range(5):
                if time.time() - _start_time > 110:
                    break
                keep = list(np.random.choice(rank, new_rank, replace=False))
                U_r = best_U[:, keep]
                V_r = best_V[:, keep]
                W_r = best_W[:, keep]
                U_opt, V_opt, W_opt, res = _local_search(U_r, V_r, W_r, T, max_iters=100)
                if res < 1e-10:
                    best_U, best_V, best_W, best_rank = U_opt, V_opt, W_opt, new_rank
                    break
            else:
                break
    
    def to_half_int(mat):
        return (np.round(mat * 2) / 2).tolist()
    
    return {"U": to_half_int(best_U), "V": to_half_int(best_V), "W": to_half_int(best_W)}
