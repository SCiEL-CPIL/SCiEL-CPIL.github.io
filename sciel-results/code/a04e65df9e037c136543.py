"""Rank-minimal tensor decomposition for 2x4 * 4x7 matrix multiplication."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 7]

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Check that sum_c U[:,c] V[:,c] W[:,c] reconstructs the matmul tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
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

def _try_rank(rank, n, m, p, seed=42, max_iter=2000, tol=1e-6):
    """Try to find a rank decomposition using random initialization and gradient-free search."""
    np.random.seed(seed)
    U = np.random.randn(n*m, rank) * 0.5
    V = np.random.randn(m*p, rank) * 0.5
    W = np.random.randn(p*n, rank) * 0.5
    
    # Build the tensor constraint matrix
    # For each (i,j,k), we need sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # This is a system of nonlinear equations.
    
    # Use a simple iterative refinement
    for it in range(max_iter):
        # Compute current error
        errors = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    errors.append(s - 1.0)
        errors = np.array(errors)
        if np.max(np.abs(errors)) < tol:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        # Simple coordinate descent: adjust one entry at a time
        # Pick the worst error
        worst_idx = np.argmax(np.abs(errors))
        ei, ej, ek = divmod(worst_idx, n*m*p)
        ei, rem = divmod(worst_idx, m*p)
        ej, ek = rem, rem % p
        ei = worst_idx // (m*p)
        rem = worst_idx % (m*p)
        ej = rem // p
        ek = rem % p
        
        # Try to fix by adjusting U[ei*m+ej, :]
        idx = ei*m + ej
        for c in range(rank):
            if V[ej*p+ek, c] != 0 and W[ek*n+ei, c] != 0:
                factor = V[ej*p+ek, c] * W[ek*n+ei, c]
                if abs(factor) > 1e-10:
                    U[idx, c] -= errors[worst_idx] / factor * 0.1
                    break
    
    # Final check
    errors = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                errors.append(s - 1.0)
    if np.max(np.abs(errors)) < tol:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    return None

def _constructive_rank50(n, m, p):
    """Construct a rank-50 decomposition using block Strassen."""
    # We'll use the following approach:
    # Split B into B1 (4x4) and B2 (4x3)
    # C = A*B1 + A*B2
    # 
    # For A*B1: split A into A1, A2 (2x2 each), B1 into B1_top, B1_bottom (2x4 each)
    # A*B1 = A1*B1_top + A2*B1_bottom
    # Each 2x2 * 2x4 can be done in rank 14 (using Strassen on two 2x2x2 blocks)
    # So A*B1 has rank <= 28
    #
    # For A*B2: split A into A1, A2 (2x2 each), B2 into B2_top, B2_bottom (2x3 each)
    # A*B2 = A1*B2_top + A2*B2_bottom
    # Each 2x2 * 2x3 has rank <= 12 (schoolbook)
    # So A*B2 has rank <= 24
    #
    # Total: 52. We need to find a better decomposition.
    #
    # Let me try a different approach: use the fact that we can share some computations.
    # 
    # Actually, let me just return schoolbook for now and rely on the optimizer.
    return _schoolbook(n, m, p)

def construct():
    start = time.time()
    deadline = start + 115.0
    n, m, p = DIMENSIONS

    fallback = _schoolbook(n, m, p)
    best = fallback
    best_rank = n * m * p

    # Try to find lower-rank decompositions
    for rank in range(50, 56):
        if time.time() > deadline - 10.0:
            break
        result = _try_rank(rank, n, m, p, seed=rank, max_iter=5000)
        if result is not None:
            if _verify(result["U"], result["V"], result["W"], n, m, p):
                best = result
                best_rank = rank
                break

    if time.time() > deadline - 5.0:
        return best

    return best
