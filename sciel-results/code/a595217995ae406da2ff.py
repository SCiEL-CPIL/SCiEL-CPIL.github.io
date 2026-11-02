"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 130.0  # safety margin before 150s kill

    # Schoolbook decomposition of rank n*m*p
    def schoolbook():
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

    best = schoolbook()
    best_rank = n * m * p

    # Nonzero entries of the tensor
    nz = set((i*m+j, j*p+k, k*n+i) for i in range(n) for j in range(m) for k in range(p))

    # Verify a decomposition: check that sum_t U[a,t]*V[b,t]*W[c,t] = 1 for (a,b,c) in nz, 0 otherwise
    def verify(U, V, W):
        r = len(U[0])
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    s = sum(U[a][t] * V[b][t] * W[c][t] for t in range(r))
                    expected = 1 if (a, b, c) in nz else 0
                    if s != expected:
                        return False
        return True

    # Tensor values
    T = np.zeros((n*m, m*p, p*n))
    for a, b, c in nz:
        T[a, b, c] = 1.0

    def try_rank(r, max_iter=100, lr=0.01):
        if time.time() > deadline - 5:
            return None
        
        # Try a few random initializations
        for _ in range(5):
            if time.time() > deadline - 5:
                return None
                
            # Initialize U, V, W with small random values
            U = np.random.randn(n*m, r) * 0.1
            V = np.random.randn(m*p, r) * 0.1
            W = np.random.randn(p*n, r) * 0.1
            
            # Gradient Descent on E = 0.5 * || U \cdot V \cdot W - T ||_F^2
            for it in range(max_iter):
                if time.time() > deadline - 5:
                    return None
                
                # Compute current tensor approximation
                X = np.zeros_like(T)
                for t in range(r):
                    X += np.einsum('a,b,c->abc', U[:, t], V[:, t], W[:, t])
                
                error = X - T
                E = np.sum(error**2)
                
                # Check for convergence
                if E < 1e-6:
                    # Try to snap to integers/half-integers
                    U_snap = np.round(U * 2) / 2
                    V_snap = np.round(V * 2) / 2
                    W_snap = np.round(W * 2) / 2
                    
                    # Verify exactness
                    if verify(U_snap.tolist(), V_snap.tolist(), W_snap.tolist()):
                        return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
                
                # Compute gradients
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for t in range(r):
                    dU[:, t] = np.einsum('abc,b,c->a', error, V[:, t], W[:, t])
                    dV[:, t] = np.einsum('abc,a,c->b', error, U[:, t], W[:, t])
                    dW[:, t] = np.einsum('abc,a,b->c', error, U[:, t], V[:, t])
                
                # Update
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
        return None

    # Try to find a decomposition with rank r
    # Start from 59 down to 20
    for r in range(59, 19, -1):
        if time.time() > deadline - 10:
            break
        result = try_rank(r)
        if result is not None:
            if verify(result['U'], result['V'], result['W']):
                if r < best_rank:
                    best = result
                    best_rank = r
                break

    return best
# EVOLVE-BLOCK-END
