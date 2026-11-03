"""Rewritten program for matrix multiplication 2x5 by 5x5 tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 5, 5]

def _verify(U, V, W):
    """Check if the decomposition exactly decomposes the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = len(U[0]) if U else 0
    # Build the tensor T[i,j,k] = 1 if the multiplication uses a[i,j]*b[j,k]
    # The decomposition says: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == T[i,j,k]
    # For matrix multiplication, T[i,j,k] = 1 for all valid i,j,k (since C[i,k] = sum_j A[i,j]*B[j,k])
    # Actually, the tensor for matrix multiplication has T[i,j,k] = 1 for all i,j,k.
    # Let's verify: for each (i,j,k), sum over c should equal 1.
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # For each (i, j, k), compute sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # This should equal 1 for all i, j, k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(np.dot(U[i*m+j], V[j*p+k]), W[k*n+i])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook():
    """Return the schoolbook decomposition of rank n*m*p."""
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

def _try_random_search(rank, time_budget=60.0):
    """Try to find a decomposition of the given rank using random initialization and gradient descent."""
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + time_budget
    
    # The tensor T[i,j,k] = 1 for all i,j,k
    # We want U[i*m+j, c], V[j*p+k, c], W[k*n+i, c] such that
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
    
    # This is a system of n*m*p = 50 equations in rank*100 variables (roughly)
    # Let's use a random initialization approach
    
    best_result = None
    
    for attempt in range(100):
        if time.time() > deadline - 5:
            break
        
        # Random initialization
        np.random.seed(attempt)
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        
        # Simple gradient descent
        lr = 0.01
        for step in range(500):
            if time.time() > deadline - 5:
                break
            
            # Compute residuals
            residuals = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        residuals[i, j, k] = np.dot(np.dot(U[i*m+j], V[j*p+k]), W[k*n+i]) - 1.0
            
            err = np.sum(residuals**2)
            if err < 1e-12:
                # Convert to list format
                result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                if _verify(result["U"], result["V"], result["W"]):
                    return result
                best_result = result
                break
            
            # Compute gradients
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        r = residuals[i, j, k]
                        dU[i*m+j] += 2 * r * np.outer(V[j*p+k], W[k*n+i])
                        dV[j*p+k] += 2 * r * np.outer(U[i*m+j], W[k*n+i])
                        dW[k*n+i] += 2 * r * np.outer(U[i*m+j], V[j*p+k])
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
        
        # Check final result
        result = {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
        if _verify(result["U"], result["V"], result["W"]):
            return result
        if best_result is None:
            best_result = result
    
    return best_result

def _try_structured_search(rank, time_budget=30.0):
    """Try structured approaches for low-rank decomposition."""
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + time_budget
    
    # For rank = n*m*p = 50, we already have the schoolbook solution.
    # For lower ranks, we need to find non-trivial decompositions.
    # This is generally hard, so we try a few structured approaches.
    
    # Approach 1: Try to exploit the structure of 2x5 * 5x5
    # The key insight is that C[i,k] = sum_j A[i,j] * B[j,k]
    # We can write this as: C = A @ B where A is 2x5 and B is 5x5
    
    # One idea: use the fact that we only have 2 rows in A
    # For each output element C[i,k], we need sum_j A[i,j]*B[j,k]
    # This is a bilinear form in (A[i,:], B[:,k])
    
    # Let's try a block structure
    if rank >= n * m * p:
        return _schoolbook()
    
    # For now, return None if we can't find a better decomposition
    return None

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    
    # Start with schoolbook
    result = _schoolbook()
    
    # Try to find lower rank decompositions
    # The theoretical lower bound is at least max(n*m, m*p, p*n) = 25
    # But for matrix multiplication, the lower bound is higher
    
    # Try ranks from 25 upward
    for rank in range(25, n*m*p + 1):
        if time.time() > start + 100:
            break
        
        # Try random search for this rank
        res = _try_random_search(rank, time_budget=min(30.0, 100.0 - (time.time() - start)))
        if res is not None:
            if _verify(res["U"], res["V"], res["W"]):
                result = res
                break
    
    return result
