"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
import random
import math

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the tensor decomposition is exact."""
    rank = len(U[0]) if U else 0
    # Check U, V, W shapes
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    for r in range(rank):
        if len(U[r]) != 1:  # U is (n*m, rank), so U[row][col]
            pass
    # The decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # T[i,j,k] = 1 if all indices valid (which they always are for matrix mult tensor)
    # Actually for matrix multiplication, T[i,j,k] = delta_{i,k}... no.
    # Matrix mult tensor: (A_{n x m} * B_{m x p})_{i,k} = sum_j A_{i,j} B_{j,k}
    # So T[i,j,k] = 1 for all i,j,k? No, that's not right either.
    # The tensor T is defined such that C_{i,k} = sum_j T[i,j,k] * A_{i,j} * B_{j,k}
    # For standard matrix multiplication, T[i,j,k] = 1 for all valid i,j,k.
    # Wait, no. T[i,j,k] should be 1 if the (i,j) element of A and (j,k) element of B 
    # contribute to C[i,k]. That's always the case for matrix multiplication.
    # So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # Let me re-read the problem. "Matrix multiplication 2,5,6"
    # The tensor for matrix multiplication: T_{i,j,k} = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # Yes, T[i,j,k] = 1 for all valid indices.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Start with schoolbook as fallback
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # For small dimensions, try to find lower-rank decompositions
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = max(10, 30, 12) = 30
    # Upper bound from schoolbook: 60
    # Try ranks from lower bound up to schoolbook rank - 1
    
    # Use random search with verification
    # For rank r, we need U (n*m x r), V (m*p x r), W (p*n x r)
    # such that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # This is a system of equations. For each (i,j,k), we have:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Let's try a structured approach: fix some entries and solve for others.
    # Or use gradient-free random search.
    
    # For very small cases, we can try exhaustive/structured search.
    # n=2, m=5, p=6: 10*30*12 = 3600 tensor entries, each must equal 1.
    
    # Try rank from 30 to 59
    for target_rank in range(30, best_rank):
        if time.time() > deadline - 5.0:
            break
        
        # Try a few random attempts at this rank
        max_attempts = 20 if target_rank >= 40 else 5
        for attempt in range(max_attempts):
            if time.time() > deadline - 2.0:
                break
            
            # Random initialization with small integers/half-integers
            # Use a sparse-like initialization
            U = [[0.0] * target_rank for _ in range(n * m)]
            V = [[0.0] * target_rank for _ in range(m * p)]
            W = [[0.0] * target_rank for _ in range(p * n)]
            
            # Initialize with random small values
            for i in range(n * m):
                for r in range(target_rank):
                    U[i][r] = random.choice([-1, 0, 0, 0, 1, 0.5, -0.5])
            for j in range(m * p):
                for r in range(target_rank):
                    V[j][r] = random.choice([-1, 0, 0, 0, 1, 0.5, -0.5])
            for k in range(p * n):
                for r in range(target_rank):
                    W[k][r] = random.choice([-1, 0, 0, 0, 1, 0.5, -0.5])
            
            # Quick check: compute residual
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(target_rank):
                            s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                        err += (s - 1.0) ** 2
                        if err > 1.0:
                            break
                    if err > 1.0:
                        break
                if err > 1.0:
                    break
            
            # If error is small, do a local refinement
            if err < 1.0:
                # Simple gradient descent on the entries
                lr = 0.01
                for step in range(100):
                    if time.time() > deadline - 1.0:
                        break
                    # Compute gradients
                    dU = [[0.0] * target_rank for _ in range(n * m)]
                    dV = [[0.0] * target_rank for _ in range(m * p)]
                    dW = [[0.0] * target_rank for _ in range(p * n)]
                    
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                # d/dU[i*m+j, r] = V[j*p+k, r] * W[k*n+i, r] * (s-1)
                                s = 0.0
                                for r in range(target_rank):
                                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                                diff = s - 1.0
                                for r in range(target_rank):
                                    dU[i * m + j][r] += V[j * p + k][r] * W[k * n + i][r] * diff
                                    dV[j * p + k][r] += U[i * m + j][r] * W[k * n + i][r] * diff
                                    dW[k * n + i][r] += U[i * m + j][r] * V[j * p + k][r] * diff
                    
                    # Update
                    for i in range(n * m):
                        for r in range(target_rank):
                            U[i][r] -= lr * dU[i][r]
                    for j in range(m * p):
                        for r in range(target_rank):
                            V[j][r] -= lr * dV[j][r]
                    for k in range(p * n):
                        for r in range(target_rank):
                            W[k][r] -= lr * dW[k][r]
                    
                    # Check convergence
                    err2 = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = 0.0
                                for r in range(target_rank):
                                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                                err2 += (s - 1.0) ** 2
                    if err2 < 1e-12:
                        break
                
                # Verify exactness
                if _verify(U, V, W, n, m, p, tol=1e-9):
                    # Round to nearest half-integer if close
                    U2 = [[round(U[i][r] * 2) / 2 for r in range(target_rank)] for i in range(n * m)]
                    V2 = [[round(V[j][r] * 2) / 2 for r in range(target_rank)] for j in range(m * p)]
                    W2 = [[round(W[k][r] * 2) / 2 for r in range(target_rank)] for k in range(p * n)]
                    if _verify(U2, V2, W2, n, m, p, tol=1e-9):
                        best = {"U": U2, "V": V2, "W": W2}
                        best_rank = target_rank
                        break
                    else:
                        best = {"U": U, "V": V, "W": W}
                        best_rank = target_rank
                        break
    
    return best
# EVOLVE-BLOCK-END
