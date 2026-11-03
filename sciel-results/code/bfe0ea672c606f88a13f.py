"""Matrix multiplication 3x3x5 rank search with structured construction and deadline guard."""
import time
import random
import math
import numpy as np

DIMENSIONS = [3, 3, 5]
_START = time.time()
_DEADLINE = 140.0

def _time_left():
    return _DEADLINE - (time.time() - _START)

def _validate(U, V, W):
    """Check exact tensor decomposition. Returns (rank, U, V, W) if valid, else None."""
    n, m, p = DIMENSIONS
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                ui = U[i * m + j]
                vk = V[j * p + k]
                wk = W[k * n + i]
                for c in range(r):
                    s += ui[c] * vk[c] * wk[c]
                if s != 1:
                    return None
    return (r, U, V, W)

def _schoolbook():
    """Schoolbook rank-45 decomposition."""
    n, m, p = DIMENSIONS
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
    return (rank, U, V, W)

def _gradient_descent(rank, max_iter=200, lr=0.01):
    """
    Try to find a decomposition of the given rank using gradient descent.
    """
    n, m, p = DIMENSIONS
    rng = random.Random(42)
    
    # Initialize with small random values
    U = np.random.uniform(-0.5, 0.5, (n * m, rank))
    V = np.random.uniform(-0.5, 0.5, (m * p, rank))
    W = np.random.uniform(-0.5, 0.5, (p * n, rank))
    
    # Target tensor
    # T[i, j, k] = 1
    
    for step in range(max_iter):
        if _time_left() < 5.0:
            break
            
        # Compute T
        T = np.zeros((n, m, p))
        for c in range(rank):
            u_col = U[:, c].reshape(n, m)
            v_col = V[:, c].reshape(m, p)
            w_col = W[:, c].reshape(p, n)
            # T[i,j,k] += U[i,j,c] * V[j,k,c] * W[k,i,c]
            # This is a tensor product sum.
            # We can use einsum or loops.
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i,j,k] += u_col[i,j] * v_col[j,k] * w_col[k,i]
        
        # Residual
        R = np.sum((T - 1.0) ** 2)
        
        if R < 1e-8:
            # Snap to half-integers
            U_snap = np.round(U * 2) / 2
            V_snap = np.round(V * 2) / 2
            W_snap = np.round(W * 2) / 2
            
            # Convert to list of lists for validation
            U_list = [[float(x) for x in row] for row in U_snap]
            V_list = [[float(x) for x in row] for row in V_snap]
            W_list = [[float(x) for x in row] for row in W_snap]
            
            if _validate(U_list, V_list, W_list) is not None:
                return (rank, U_list, V_list, W_list)
            
            # If snap fails, continue or break?
            # Maybe the real solution is valid but not half-integer?
            # The problem requires half-integers.
            # So we must find a half-integer solution.
            # Gradient descent finds real solutions.
            # We need to check if the real solution is close to a half-integer grid.
            # If not, we might need to continue or restart.
            # For now, if R is small, we return the snapped version if valid.
            # If not valid, we break and try another rank/init.
            break
        
        # Compute Gradients
        # dR/dU[i,j,c] = 2 * sum_k (T[i,j,k] - 1) * V[j,k,c] * W[k,i,c]
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for c in range(rank):
            u_col = U[:, c].reshape(n, m)
            v_col = V[:, c].reshape(m, p)
            w_col = W[:, c].reshape(p, n)
            
            for i in range(n):
                for j in range(m):
                    # dU[i,j,c]
                    grad_u = 0.0
                    for k in range(p):
                        grad_u += (T[i,j,k] - 1.0) * v_col[j,k] * w_col[k,i]
                    dU[i*m+j, c] = 2.0 * grad_u
                    
                    # dV[j,k,c]
                    for k in range(p):
                        grad_v = 0.0
                        for i2 in range(n):
                            grad_v += (T[i2,j,k] - 1.0) * u_col[i2,j] * w_col[k,i2]
                        dV[j*p+k, c] = 2.0 * grad_v
                        
                    # dW[k,i,c]
                    for k in range(p):
                        grad_w = 0.0
                        for j2 in range(m):
                            grad_w += (T[i,j2,k] - 1.0) * u_col[i,j2] * v_col[j2,k]
                        dW[k*n+i, c] = 2.0 * grad_w
                        
        # Update
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
    return None

def construct():
    """Main entry point. Returns {'U', 'V', 'W'} with the best valid decomposition found."""
    # Start with schoolbook as fallback
    best_rank, best_U, best_V, best_W = _schoolbook()
    
    # Try ranks 20, 25, 30
    for rank in [20, 25, 30]:
        if _time_left() < 10.0:
            break
            
        result = _gradient_descent(rank)
        if result is not None:
            r, U, V, W = result
            if _validate(U, V, W) is not None:
                return {"U": U, "V": V, "W": W}
                
    # Fallback to schoolbook
    return {"U": best_U, "V": best_V, "W": best_W}
