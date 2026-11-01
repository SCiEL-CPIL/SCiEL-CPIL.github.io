import numpy as np
import time
import random

DIMENSIONS = [2, 3, 3]

def _verify_factors(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is correct."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Build full tensor
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # where a = i*m+j, b = j*p+k, c = k*n+i
    # For each (i,j,k), check that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U[a] * V[b] * W[c])
                if abs(val - 1) > tol:
                    return False
    # Check that non-matching positions are 0
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                i1 = a // m
                j1 = a % m
                j2 = b // p
                k1 = b % p
                k2 = c // n
                i2 = c % n
                valid = (j1 == j2) and (i1 == i2) and (k1 == k2)
                val = np.sum(U[a] * V[b] * W[c])
                if valid:
                    if abs(val - 1) > tol:
                        return False
                else:
                    if abs(val) > tol:
                        return False
    return True

def _build_trivial(n, m, p):
    """Build the trivial rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _compute_error(U, V, W, n, m, p):
    """Compute the maximum error of the decomposition."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    max_err = 0.0
    # Check valid positions
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U[a] * V[b] * W[c])
                err = abs(val - 1)
                if err > max_err:
                    max_err = err
    # Check invalid positions
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                i1 = a // m
                j1 = a % m
                j2 = b // p
                k1 = b % p
                k2 = c // n
                i2 = c % n
                valid = (j1 == j2) and (i1 == i2) and (k1 == k2)
                if not valid:
                    val = np.sum(U[a] * V[b] * W[c])
                    err = abs(val)
                    if err > max_err:
                        max_err = err
    return max_err

def _try_rank_search(n, m, p, max_rank, min_rank, deadline, seed=42):
    """Try to find a decomposition with rank between min_rank and max_rank."""
    random.seed(seed)
    np.random.seed(seed)
    
    best_rank = max_rank + 1
    best_U, best_V, best_W = None, None, None
    
    # Start from max_rank and try to reduce
    for target_rank in range(min_rank, max_rank + 1):
        if time.time() > deadline - 30:
            break
            
        # Try multiple random initializations
        for attempt in range(20):
            if time.time() > deadline - 30:
                break
                
            # Initialize with random values
            U = np.random.randn(n * m, target_rank) * 0.1
            V = np.random.randn(m * p, target_rank) * 0.1
            W = np.random.randn(p * n, target_rank) * 0.1
            
            # Gradient descent to minimize error
            lr = 0.01
            for step in range(200):
                if time.time() > deadline - 30:
                    break
                    
                # Compute gradients
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                # For valid (i,j,k)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i * m + j
                            b = j * p + k
                            c = k * n + i
                            val = np.sum(U[a] * V[b] * W[c])
                            err = val - 1
                            for r in range(target_rank):
                                dU[a, r] += err * V[b, r] * W[c, r]
                                dV[b, r] += err * U[a, r] * W[c, r]
                                dW[c, r] += err * U[a, r] * V[b, r]
                
                # For invalid (a,b,c)
                for a in range(n * m):
                    for b in range(m * p):
                        for c in range(p * n):
                            i1 = a // m
                            j1 = a % m
                            j2 = b // p
                            k1 = b % p
                            k2 = c // n
                            i2 = c % n
                            valid = (j1 == j2) and (i1 == i2) and (k1 == k2)
                            if not valid:
                                val = np.sum(U[a] * V[b] * W[c])
                                for r in range(target_rank):
                                    dU[a, r] += val * V[b, r] * W[c, r]
                                    dV[b, r] += val * U[a, r] * W[c, r]
                                    dW[c, r] += val * U[a, r] * V[b, r]
                
                # Update
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                # Check convergence
                err = _compute_error(U, V, W, n, m, p)
                if err < 1e-9:
                    best_rank = target_rank
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    break
            
            if best_rank == target_rank:
                break
        
        if best_rank == target_rank:
            break
    
    return best_rank, best_U, best_V, best_W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Try to find a lower rank decomposition
    best_rank, best_U, best_V, best_W = _try_rank_search(n, m, p, 18, 10, deadline)
    
    if best_U is not None:
        # Convert to lists
        U_list = [[float(x) for x in row] for row in best_U]
        V_list = [[float(x) for x in row] for row in best_V]
        W_list = [[float(x) for x in row] for row in best_W]
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fall back to trivial decomposition
    U, V, W = _build_trivial(n, m, p)
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    return {"U": U_list, "V": V_list, "W": W_list}
