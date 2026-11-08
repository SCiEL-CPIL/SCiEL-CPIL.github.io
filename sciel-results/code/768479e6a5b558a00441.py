"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def _build_target_tensor():
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _verify(U, V, W):
    T = _build_target_tensor()
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T_decomp, T, atol=1e-9)

def _to_half_int(x):
    return np.round(x * 2) / 2

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}

def _gradient_descent(U, V, W, T, max_iters=200, lr=0.005):
    for it in range(max_iters):
        T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
        err = T_decomp - T
        loss = np.sum(err**2)
        if loss < 1e-12:
            break
        grad_U = 2 * np.einsum('abc,br,cr->ar', err, V, W)
        grad_V = 2 * np.einsum('abc,ar,cr->br', err, U, W)
        grad_W = 2 * np.einsum('abc,ar,br->cr', err, U, V)
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
    return U, V, W, loss

def _try_rank(rank, num_trials=3, max_iters=80):
    n, m, p = DIMENSIONS
    T = _build_target_tensor()
    for trial in range(num_trials):
        np.random.seed(42 + trial * 137)
        U = np.random.randn(n*m, rank) * 0.5
        V = np.random.randn(m*p, rank) * 0.5
        W = np.random.randn(p*n, rank) * 0.5
        U, V, W, loss = _gradient_descent(U, V, W, T, max_iters=max_iters)
        if loss < 1e-8:
            U_h = _to_half_int(U)
            V_h = _to_half_int(V)
            W_h = _to_half_int(W)
            if _verify(U_h, V_h, W_h):
                return {'U': U_h.tolist(), 'V': V_h.tolist(), 'W': W_h.tolist()}
    return None

def construct():
    start = time.time()
    deadline = start + 110
    best = _schoolbook()
    for rank in range(25, 50):
        if time.time() > deadline - 3:
            break
        result = _try_rank(rank, num_trials=2, max_iters=40)
        if result is not None:
            best = result
            break
    return best
# EVOLVE-BLOCK-END
