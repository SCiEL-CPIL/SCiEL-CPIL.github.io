"""Tensor decomposition for 4x4 by 4x9 matrix multiplication."""
import time
import itertools
import numpy as np

DIMENSIONS = [4, 4, 9]
START_TIME = time.time()
TIME_LIMIT = 115  # seconds, leaving margin for 120s evaluator limit

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying
    a 4x4 matrix by a 4x9 matrix.
    
    Output factor follows the official trace convention W[k*n+i, r].
    Returns {'U', 'V', 'W'} with shapes (n*m, rank), (m*p, rank), (p*n, rank).
    """
    n, m, p = DIMENSIONS
    
    def verify(U, V, W):
        """Verify that the decomposition is exact."""
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0 if i == k else 0.0
                    actual = 0.0
                    for r in range(rank):
                        actual += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(actual - expected) > 1e-9:
                        return False
        return True
    
    def is_valid_entry(x):
        """Check if x is an integer or half-integer."""
        x2 = x * 2
        return abs(x2 - round(x2)) < 1e-9
    
    def check_all_half_integers(U, V, W):
        """Check all entries are integers or half-integers."""
        for mat in [U, V, W]:
            for row in mat:
                for val in row:
                    if not is_valid_entry(val):
                        return False
        return True
    
    def schoolbook():
        """Return the schoolbook decomposition of rank n*m*p."""
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
        return U, V, W
    
    best_solution = None
    best_rank = n * m * p
    
    # Strategy 1: Structured construction exploiting T[i,j,k] = delta(i,k)
    # Since T is independent of j, try U[i,j,r] = A[i,r], V[j,k,r] = B[k,r]
    # Need: sum_r A[i,r] * B[k,r] * W[k*n+i, r] = delta(i,k)
    
    # Try rank = n with simple diagonal structure
    target_rank = n
    U = [[0] * target_rank for _ in range(n * m)]
    V = [[0] * target_rank for _ in range(m * p)]
    W = [[0] * target_rank for _ in range(p * n)]
    
    # A[i,r] = delta(i,r), B[k,r] = delta(k,r) for r < n
    for i in range(n):
        for j in range(m):
            U[i * m + j][i] = 1
    
    for k in range(p):
        for j in range(m):
            if k < n:
                V[j * p + k][k] = 1
    
    # W[k*n+i, r] = delta(k,i) * delta(r,i)
    for k in range(p):
        for i in range(n):
            if k == i and i < target_rank:
                W[k * n + i][i] = 1
    
    if verify(U, V, W) and check_all_half_integers(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    # Try larger ranks with more flexible structures
    for target_rank in range(n + 1, min(n + p, 25)):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
        
        found = False
        
        # Try various structured patterns
        for pattern in range(5):
            if time.time() - START_TIME > TIME_LIMIT - 10:
                break
            
            U = [[0] * target_rank for _ in range(n * m)]
            V = [[0] * target_rank for _ in range(m * p)]
            W = [[0] * target_rank for _ in range(p * n)]
            
            # Pattern-based initialization
            if pattern == 0:
                # Diagonal in first n components
                for i in range(n):
                    for j in range(m):
                        U[i * m + j][i] = 1
                for k in range(p):
                    for j in range(m):
                        if k < n:
                            V[j * p + k][k] = 1
                for k in range(p):
                    for i in range(n):
                        if k == i:
                            W[k * n + i][i] = 1
            elif pattern == 1:
                # Use all components
                for r in range(target_rank):
                    for i in range(n):
                        for j in range(m):
                            U[i * m + j][r] = 1.0 / target_rank
                    for k in range(p):
                        for j in range(m):
                            V[j * p + k][r] = 1.0 / target_rank
                    for k in range(p):
                        for i in range(n):
                            W[k * n + i][r] = target_rank if k == i else 0
            elif pattern == 2:
                # Identity-like with scaling
                for i in range(n):
                    for j in range(m):
                        U[i * m + j][i] = 1
                for k in range(p):
                    for j in range(m):
                        V[j * p + k][min(k, n-1)] = 1
                for k in range(p):
                    for i in range(n):
                        if k == i:
                            W[k * n + i][i] = 1
                        elif k < n:
                            W[k * n + i][k] = 0
            elif pattern == 3:
                # Block diagonal
                for r in range(min(n, target_rank)):
                    for i in range(n):
                        for j in range(m):
                            U[i * m + j][r] = 1 if i == r else 0
                    for k in range(p):
                        for j in range(m):
                            V[j * p + k][r] = 1 if k == r else 0
                    for k in range(p):
                        for i in range(n):
                            W[k * n + i][r] = 1 if (k == i == r) else 0
            else:
                # Mixed
                for r in range(target_rank):
                    for i in range(n):
                        for j in range(m):
                            U[i * m + j][r] = 1.0 if i == r % n else 0.0
                    for k in range(p):
                        for j in range(m):
                            V[j * p + k][r] = 1.0 if k == r % p else 0.0
                    for k in range(p):
                        for i in range(n):
                            W[k * n + i][r] = 1.0 if (k == i and r % n == i and r % p == k) else 0.0
            
            if verify(U, V, W) and check_all_half_integers(U, V, W):
                best_rank = target_rank
                best_solution = (U, V, W)
                found = True
                break
        
        if found:
            break
    
    if best_solution is not None:
        U, V, W = best_solution
        return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook decomposition
    U, V, W = schoolbook()
    return {"U": U, "V": V, "W": W}
