"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly decomposes the matmul tensor."""
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_arr[i * m + j]
                v_val = V_arr[j * p + k]
                w_val = W_arr[k * n + i]
                s = np.dot(u_val, v_val * w_val)
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
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

def _random_search(n, m, p, target_rank, deadline, seed=42, max_restarts=5):
    """
    Randomized search for a rank-target_rank decomposition.
    Uses random initialization and gradient descent.
    """
    if time.time() > deadline:
        return None
    
    rng = np.random.RandomState(seed)
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Random initialization
        U = rng.randn(n*m, target_rank) * 0.5
        V = rng.randn(m*p, target_rank) * 0.5
        W = rng.randn(p*n, target_rank) * 0.5
        
        # Gradient descent
        lr = 0.01
        for it in range(200):
            if time.time() > deadline:
                break
            
            # Compute gradients
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_val = U[i*m+j]
                        v_val = V[j*p+k]
                        w_val = W[k*n+i]
                        s = np.dot(u_val, v_val * w_val)
                        err = s - 1.0
                        
                        # d/dU[i*m+j,c] = err * V[j*p+k,c] * W[k*n+i,c]
                        dU[i*m+j] += err * v_val * w_val
                        dV[j*p+k] += err * u_val * w_val
                        dW[k*n+i] += err * u_val * v_val
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
        
        # Check if we found a solution
        if _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # safety margin
    
    # Start with schoolbook
    result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find better decompositions with random search
    for target_rank in [71, 70, 69, 68]:
        if time.time() > deadline - 10:
            break
        candidate = _random_search(n, m, p, target_rank, deadline, seed=target_rank)
        if candidate is not None:
            result = candidate
            best_rank = target_rank
            break
    
    return result
# EVOLVE-BLOCK-END
