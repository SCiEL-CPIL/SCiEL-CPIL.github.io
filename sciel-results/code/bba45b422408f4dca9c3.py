"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 4]

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
    return {"U": U, "V": V, "W": W}


def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Fast vectorized verification using einsum."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_3d = U_arr.reshape(n, m, rank)
    V_3d = V_arr.reshape(m, p, rank)
    W_3d = W_arr.reshape(p, n, rank)
    
    # T[i,j,k] = sum_c U_3d[i,j,c] * V_3d[j,k,c] * W_3d[k,i,c]
    # We need to compute this for all i, j, k
    # Use einsum: U[i,j,c] * V[j,k,c] * W[k,i,c] -> sum over c
    T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
    
    max_err = np.max(np.abs(T - 1.0))
    return max_err < tol, max_err


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that the decomposition exactly reconstructs the matrix multiplication tensor."""
    ok, _ = _verify_fast(U, V, W, n, m, p, tol)
    return ok


def _gradient_descent(n, m, p, r, max_time=10, seed=42):
    """Gradient descent to find a rank-r decomposition."""
    start_time = time.time()
    rng = np.random.RandomState(seed)
    
    # Initialize
    U = rng.randn(n * m, r) * 0.1
    V = rng.randn(m * p, r) * 0.1
    W = rng.randn(p * n, r) * 0.1
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    U_3d = U.reshape(n, m, r)
    V_3d = V.reshape(m, p, r)
    W_3d = W.reshape(p, n, r)
    
    lr = 0.05
    for iteration in range(5000):
        if time.time() - start_time > max_time:
            break
        
        # Compute T
        T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        err = np.sum((T - 1.0) ** 2)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < 1e-10:
            break
        
        # Compute gradients
        # dT/dU_3d[i,j,c] = V_3d[j,:,c] * W_3d[:,i,c] -> shape (n, m, r)
        # More precisely: dT[i,j,k]/dU[i,j,c] = V[j,k,c] * W[k,i,c]
        # So dL/dU[i,j,c] = sum_k 2*(T[i,j,k]-1) * V[j,k,c] * W[k,i,c]
        
        # grad_U_3d[i,j,c] = sum_k 2*(T[i,j,k]-1) * V_3d[j,k,c] * W_3d[k,i,c]
        # Using einsum: 2*(T-1)[i,j,k] * V[j,k,c] * W[k,i,c] -> sum over k
        dL_dU_3d = 2 * np.einsum('ijk,jkc,kic->ijc', T - 1.0, V_3d, W_3d)
        
        # grad_V_3d[j,k,c] = sum_i 2*(T[i,j,k]-1) * U[i,j,c] * W[k,i,c]
        dL_dV_3d = 2 * np.einsum('ijk,ijc,kic->jkc', T - 1.0, U_3d, W_3d)
        
        # grad_W_3d[k,i,c] = sum_j 2*(T[i,j,k]-1) * U[i,j,c] * V[j,k,c]
        dL_dW_3d = 2 * np.einsum('ijk,ijc,jkc->kic', T - 1.0, U_3d, V_3d)
        
        # Update
        U -= lr * dL_dU_3d.reshape(n * m, r)
        V -= lr * dL_dV_3d.reshape(m * p, r)
        W -= lr * dL_dW_3d.reshape(p * n, r)
        
        U_3d = U.reshape(n, m, r)
        V_3d = V.reshape(m, p, r)
        W_3d = W.reshape(p, n, r)
        
        # Adaptive learning rate
        lr *= 0.999
        
        if lr < 1e-6:
            break
    
    # Check if we found a good solution
    ok, final_err = _verify_fast(best_U.tolist(), best_V.tolist(), best_W.tolist(), n, m, p, tol=1e-8)
    if ok or final_err < 1e-6:
        return best_U, best_V, best_W, r
    
    return None


def _try_polish(U, V, W, n, m, p, max_time=5):
    """Try to polish a near-exact solution to exact fractions."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Try to snap to small rationals
    U_frac = [[Fraction(x).limit_denominator(100) for x in row] for row in U_arr.tolist()]
    V_frac = [[Fraction(x).limit_denominator(100) for x in row] for row in V_arr.tolist()]
    W_frac = [[Fraction(x).limit_denominator(100) for x in row] for row in W_arr.tolist()]
    
    ok, _ = _verify_fast(U_frac, V_frac, W_frac, n, m, p, tol=1e-9)
    if ok:
        return U_frac, V_frac, W_frac
    
    # Try with larger denominators
    U_frac = [[Fraction(x).limit_denominator(1000) for x in row] for row in U_arr.tolist()]
    V_frac = [[Fraction(x).limit_denominator(1000) for x in row] for row in V_arr.tolist()]
    W_frac = [[Fraction(x).limit_denominator(1000) for x in row] for row in W_arr.tolist()]
    
    ok, _ = _verify_fast(U_frac, V_frac, W_frac, n, m, p, tol=1e-9)
    if ok:
        return U_frac, V_frac, W_frac
    
    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Start with schoolbook
    result = _schoolbook()
    best_rank = n * m * p
    
    # Try gradient descent for various ranks
    for r in range(12, 25):
        if time.time() > deadline - 10:
            break
        
        max_time_for_this = min(20, deadline - time.time() - 10)
        if max_time_for_this < 5:
            break
        
        for seed in range(5):
            if time.time() > deadline - 10:
                break
            
            res = _gradient_descent(n, m, p, r, max_time=max_time_for_this / 5, seed=seed + 42)
            if res is not None:
                U, V, W, rank = res
                if rank < best_rank:
                    best_rank = rank
                    # Try to polish
                    polished = _try_polish(U, V, W, n, m, p, max_time=2)
                    if polished is not None:
                        U_f, V_f, W_f = polished
                        result = {"U": U_f, "V": V_f, "W": W_f}
                    else:
                        result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    break
        
        if best_rank <= 14:
            break
    
    return result

# EVOLVE-BLOCK-END
