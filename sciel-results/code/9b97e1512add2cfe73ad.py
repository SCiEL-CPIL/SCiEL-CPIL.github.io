"""Search for rank-minimal exact tensor decomposition of 2x3x3 matrix multiplication."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [2, 3, 3]
n, m, p = DIMENSIONS

def _build_tensor():
    """Build the matrix multiplication tensor T[i, j, k] = 1 for all valid indices.
    The tensor is defined such that:
      sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1  for all i, j, k
    """
    # T is indexed by (i, j, k) where i in [0,n), j in [0,m), k in [0,p)
    # Each entry should equal 1
    T = np.ones((n, m, p))
    return T

def _verify(U, V, W, tol=1e-10):
    """Verify that U, V, W form a valid decomposition.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    """
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _verify_numpy(U, V, W, tol=1e-10):
    """Fast verification using numpy einsum."""
    rank = U.shape[1]
    # Reshape: U -> (n, m, rank), V -> (m, p, rank), W -> (p, n, rank)
    Ur = U.reshape(n, m, rank)
    Vr = V.reshape(m, p, rank)
    Wr = W.reshape(p, n, rank)
    # Compute sum_c Ur[i,j,c] * Vr[j,k,c] * Wr[k,i,c]
    result = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
    return np.allclose(result, 1.0, atol=tol)

def _is_half_integer(val, tol=1e-9):
    """Check if val is a half-integer (multiple of 0.5)."""
    doubled = val * 2
    return abs(doubled - round(doubled)) < tol

def _check_half_integer_matrix(M, tol=1e-9):
    """Check if all entries are half-integers."""
    doubled = M * 2
    return np.allclose(doubled, np.round(doubled), atol=tol)

def _to_half_integer(M):
    """Round matrix to nearest half-integers."""
    return np.round(M * 2) / 2.0

def _random_init(rank, scale=1.0):
    """Random initialization of U, V, W."""
    U = np.random.randn(n * m, rank) * scale
    V = np.random.randn(m * p, rank) * scale
    W = np.random.randn(p * n, rank) * scale
    return U, V, W

def _objective(U, V, W):
    """Compute the error of the decomposition."""
    rank = U.shape[1]
    Ur = U.reshape(n, m, rank)
    Vr = V.reshape(m, p, rank)
    Wr = W.reshape(p, n, rank)
    result = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
    return np.sum((result - 1.0) ** 2)

def _gradient_step(U, V, W, lr=0.01):
    """One gradient descent step on the objective."""
    rank = U.shape[1]
    Ur = U.reshape(n, m, rank)
    Vr = V.reshape(m, p, rank)
    Wr = W.reshape(p, n, rank)
    
    # Compute residuals
    result = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
    residual = result - 1.0  # (n, m, p)
    
    # Gradients
    # d/dU[i*m+j, c] = sum_{k} residual[i,j,k] * V[j*p+k, c] * W[k*n+i, c]
    gU = np.zeros_like(U)
    gV = np.zeros_like(V)
    gW = np.zeros_like(W)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = residual[i, j, k]
                if abs(r) > 1e-15:
                    gU[i * m + j] += r * Vr[j, k] * Wr[k, i]
                    gV[j * p + k] += r * Ur[i, j] * Wr[k, i]
                    gW[k * n + i] += r * Ur[i, j] * Vr[j, k]
    
    U -= lr * gU
    V -= lr * gV
    W -= lr * gW
    return U, V, W

def _refine(U, V, W, max_iter=5000, lr_start=0.05, lr_end=0.001, tol=1e-12):
    """Refine with adaptive learning rate gradient descent."""
    for it in range(max_iter):
        obj = _objective(U, V, W)
        if obj < tol:
            break
        # Adaptive learning rate
        progress = it / max_iter
        lr = lr_start * (lr_end / lr_start) ** progress
        
        U_new, V_new, W_new = _gradient_step(U, V, W, lr)
        obj_new = _objective(U_new, V_new, W_new)
        
        if obj_new < obj:
            U, V, W = U_new, V_new, W_new
        else:
            # Try smaller step
            U2, V2, W2 = _gradient_step(U, V, W, lr * 0.1)
            obj2 = _objective(U2, V2, W2)
            if obj2 < obj:
                U, V, W = U2, V2, W2
            else:
                break
    return U, V, W

def _try_rank(rank, deadline, best_result):
    """Try to find a valid decomposition at given rank."""
    if time.time() > deadline:
        return best_result
    
    best_obj = float('inf')
    best_UVW = None
    
    n_attempts = 30
    for attempt in range(n_attempts):
        if time.time() > deadline:
            break
        
        # Different initialization strategies
        if attempt % 3 == 0:
            scale = 1.0
        elif attempt % 3 == 1:
            scale = 0.5
        else:
            scale = 2.0
        
        U, V, W = _random_init(rank, scale)
        U, V, W = _refine(U, V, W, max_iter=3000)
        
        obj = _objective(U, V, W)
        if obj < best_obj:
            best_obj = obj
            best_UVW = (U.copy(), V.copy(), W.copy())
        
        if best_obj < 1e-10:
            # Check half-integer constraint
            U_hi = _to_half_integer(best_UVW[0])
            V_hi = _to_half_integer(best_UVW[1])
            W_hi = _to_half_integer(best_UVW[2])
            if _check_half_integer_matrix(U_hi) and _check_half_integer_matrix(V_hi) and _check_half_integer_matrix(W_hi):
                if _verify_numpy(U_hi, V_hi, W_hi, tol=1e-9):
                    return {'U': U_hi.tolist(), 'V': V_hi.tolist(), 'W': W_hi.tolist()}
    
    # If we got close, try rounding and verifying
    if best_UVW is not None and best_obj < 1e-6:
        U_hi = _to_half_integer(best_UVW[0])
        V_hi = _to_half_integer(best_UVW[1])
        W_hi = _to_half_integer(best_UVW[2])
        if _check_half_integer_matrix(U_hi) and _check_half_integer_matrix(V_hi) and _check_half_integer_matrix(W_hi):
            if _verify_numpy(U_hi, V_hi, W_hi, tol=1e-9):
                return {'U': U_hi.tolist(), 'V': V_hi.tolist(), 'W': W_hi.tolist()}
    
    return best_result

def _schoolbook():
    """Fallback schoolbook decomposition of rank 18."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}

def construct():
    start_time = time.time()
    deadline = start_time + 120  # Safety margin
    
    # Start with schoolbook as fallback
    best_result = _schoolbook()
    
    # Try ranks from low to high
    # Lower bound: the tensor has 18 nonzero entries, but rank can be lower
    # Try ranks 10, 11, 12, 13, 14, 15, 16, 17
    for rank in range(10, 18):
        if time.time() > deadline - 5:
            break
        result = _try_rank(rank, deadline, best_result)
        if result is not best_result:
            best_result = result
            # Found a valid decomposition, return it
            return best_result
    
    return best_result
