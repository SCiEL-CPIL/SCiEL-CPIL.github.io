"""Matrix multiplication 3x3 x 3x5 — minimal-rank exact tensor decomposition."""
import time
import math
import itertools
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n=3, m=3, p=5):
    """Check that U * V^T * W^T reproduces the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(rank):
                    total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(total - 1) > 1e-9:
                    return False
    return True


def _check_half_integer(entries):
    """Check all entries are integers or half-integers (multiples of 0.5)."""
    for row in entries:
        for x in row:
            if abs(x * 2 - round(x * 2)) > 1e-9:
                return False
    return True


def _schoolbook(n=3, m=3, p=5):
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


def _build_target_tensor(n=3, m=3, p=5):
    """Build the target tensor T of shape (n*m, m*p, p*n)."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T


def _try_numerical_min_rank(n=3, m=3, p=5, deadline=None):
    """
    Use gradient descent to find a low-rank decomposition.
    Start from various random initializations and try to minimize the residual.
    """
    if deadline is None:
        deadline = time.time() + 80
    
    T = _build_target_tensor(n, m, p)
    best_result = None
    
    for r in [27, 25, 23, 21, 19, 18, 17]:
        if time.time() > deadline - 10:
            break
        
        found = False
        for trial in range(30):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(123 + trial * 7)
            U = np.random.randn(n*m, r) * 0.5
            V = np.random.randn(m*p, r) * 0.5
            W = np.random.randn(p*n, r) * 0.5
            
            lr = 0.05
            prev_err = np.inf
            stalled = 0
            
            for step in range(3000):
                if time.time() > deadline - 5:
                    break
                
                T_cur = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
                R = T - T_cur
                err = np.max(np.abs(R))
                
                if err < 1e-8:
                    break
                
                if err > prev_err * 0.99:
                    stalled += 1
                else:
                    stalled = 0
                prev_err = err
                
                if stalled > 200:
                    break
                
                grad_U = np.einsum('abc,br,cr->ar', R, V, W, optimize=True)
                grad_V = np.einsum('abc,ar,cr->br', R, U, W, optimize=True)
                grad_W = np.einsum('abc,ar,br->cr', R, U, V, optimize=True)
                
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
            
            T_cur = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            err = np.max(np.abs(T - T_cur))
            
            if err < 1e-6:
                # Try to snap to half-integers
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                
                err_snap = np.max(np.abs(T - np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap, optimize=True)))
                
                if err_snap < 1e-6:
                    U_list = [[float(x) for x in row] for row in U_snap]
                    V_list = [[float(x) for x in row] for row in V_snap]
                    W_list = [[float(x) for x in row] for row in W_snap]
                    
                    if _verify(U_list, V_list, W_list) and _check_half_integer(U_list) and _check_half_integer(V_list) and _check_half_integer(W_list):
                        best_result = {"U": U_list, "V": V_list, "W": W_list}
                        found = True
                        break
                else:
                    U_list = [[float(x) for x in row] for row in U]
                    V_list = [[float(x) for x in row] for row in V]
                    W_list = [[float(x) for x in row] for row in W]
                    
                    if _verify(U_list, V_list, W_list) and _check_half_integer(U_list) and _check_half_integer(V_list) and _check_half_integer(W_list):
                        best_result = {"U": U_list, "V": V_list, "W": W_list}
                        found = True
                        break
        
        if found:
            break
    
    return best_result


def construct():
    t0 = time.time()
    deadline = t0 + 100  # 100s hard limit, 20s safety margin for 120s budget
    
    # Try numerical rank minimization
    result = _try_numerical_min_rank(deadline=deadline)
    
    if result is not None:
        return result
    
    # Fallback to schoolbook
    return _schoolbook()

# EVOLVE-BLOCK-END
