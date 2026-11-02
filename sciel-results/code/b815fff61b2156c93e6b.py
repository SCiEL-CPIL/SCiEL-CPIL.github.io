"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 4]

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

def _verify(U, V, W, n=4, m=4, p=4, tol=1e-9):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _residual(U, V, W, n=4, m=4, p=4):
    """Compute residual tensor."""
    rank = U.shape[1]
    res = np.zeros((n, m, p))
    for r in range(rank):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    res[i, j, k] += u[i, j] * v[j, k] * w[k, i]
    return res - 1.0

def _gradient_descent(U, V, W, n, m, p, lr=0.01, steps=200, deadline=None):
    """Simple gradient descent to minimize residual norm."""
    U = U.copy()
    V = V.copy()
    W = W.copy()
    rank = U.shape[1]
    
    for step in range(steps):
        if deadline is not None and time.time() > deadline:
            break
        res = _residual(U, V, W, n, m, p)
        loss = np.sum(res**2)
        
        # Compute gradients numerically or analytically
        # For each r, the gradient w.r.t. U[i*m+j, r] is:
        # sum_{k} res[i,j,k] * V[j*p+k, r] * W[k*n+i, r]
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for r in range(rank):
            u = U[:, r].reshape(n, m)
            v = V[:, r].reshape(m, p)
            w = W[:, r].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        dU[i*m+j, r] += res[i,j,k] * v[j,k] * w[k,i]
                        dV[j*p+k, r] += res[i,j,k] * u[i,j] * w[k,i]
                        dW[k*n+i, r] += res[i,j,k] * u[i,j] * v[j,k]
        
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
    
    return U, V, W

def _round_to_half(U, V, W):
    """Round all entries to nearest half-integer."""
    U_r = np.round(U * 2) / 2.0
    V_r = np.round(V * 2) / 2.0
    W_r = np.round(W * 2) / 2.0
    return U_r, V_r, W_r

def _try_rank49(deadline):
    """Try to find a rank-49 decomposition."""
    n, m, p = 4, 4, 4
    rank = 49
    
    for seed in range(100):
        if time.time() > deadline:
            break
        
        np.random.seed(seed)
        
        # Initialize
        U = np.random.randn(n*m, rank) * 0.3
        V = np.random.randn(m*p, rank) * 0.3
        W = np.random.randn(p*n, rank) * 0.3
        
        # Run gradient descent
        U, V, W = _gradient_descent(U, V, W, n, m, p, lr=0.005, steps=100, deadline=deadline)
        
        # Check residual
        res = _residual(U, V, W, n, m, p)
        max_res = np.max(np.abs(res))
        
        if max_res < 0.1:
            # Try rounding
            U_r, V_r, W_r = _round_to_half(U, V, W)
            res_r = _residual(U_r, V_r, W_r, n, m, p)
            if np.max(np.abs(res_r)) < 1e-9:
                # Convert to lists
                U_list = U_r.tolist()
                V_list = V_r.tolist()
                W_list = W_r.tolist()
                return {"U": U_list, "V": V_list, "W": W_list}
        
        if time.time() > deadline:
            break
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Leave 20s margin
    
    # Try to find a rank-49 decomposition
    result = _try_rank49(deadline)
    
    if result is not None:
        return result
    
    # Fall back to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
