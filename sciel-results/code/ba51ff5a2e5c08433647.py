import numpy as np
from fractions import Fraction
import time
import random

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    # Target tensor: T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # U has shape (n*m, r), V has shape (m*p, r), W has shape (p*n, r)
    # Constraint: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    def verify(U, V, W):
        r = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(r):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    def verify_fraction(U, V, W):
        r = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for c in range(r):
                        s += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                    if s != 1:
                        return False
        return True
    
    # Start with schoolbook
    def schoolbook():
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W
    
    best_U, best_V, best_W = schoolbook()
    best_rank = n * m * p
    
    # Try to reduce rank using random search
    # Strategy: start with rank r, try to find a decomposition
    
    def try_find_decomposition(target_rank, max_iters=200):
        """Try to find a rank=target_rank decomposition using random search."""
        np.random.seed(42)
        for attempt in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Initialize with random values
            U = np.zeros((n*m, target_rank))
            V = np.zeros((m*p, target_rank))
            W = np.zeros((p*n, target_rank))
            
            for c in range(target_rank):
                U[:, c] = np.random.uniform(-1, 1, n*m)
                V[:, c] = np.random.uniform(-1, 1, m*p)
                W[:, c] = np.random.uniform(-1, 1, p*n)
            
            # Use gradient descent on the constraint
            # f(c) = sum over (i,j,k) of (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
            lr = 0.01
            for it in range(500):
                if time.time() > deadline:
                    return None
                
                # Compute gradients
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # d/dc of U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
                            for c in range(target_rank):
                                term = V[j*p+k][c] * W[k*n+i][c]
                                pred = 0
                                for cc in range(target_rank):
                                    pred += U[i*m+j][cc] * V[j*p+k][cc] * W[k*n+i][cc]
                                err = pred - 1
                                grad_U[i*m+j][c] += 2 * err * term
                                
                                term2 = U[i*m+j][c] * W[k*n+i][c]
                                grad_V[j*p+k][c] += 2 * err * term2
                                
                                term3 = U[i*m+j][c] * V[j*p+k][c]
                                grad_W[k*n+i][c] += 2 * err * term3
                
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
            
            # Check if we found a solution
            max_err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for c in range(target_rank):
                            s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                        max_err = max(max_err, abs(s - 1))
            
            if max_err < 1e-6:
                # Convert to fractions and verify
                U_frac = [[Fraction(U[i][c]).limit_denominator(100) for c in range(target_rank)] for i in range(n*m)]
                V_frac = [[Fraction(V[i][c]).limit_denominator(100) for c in range(target_rank)] for i in range(m*p)]
                W_frac = [[Fraction(W[i][c]).limit_denominator(100) for c in range(target_rank)] for i in range(p*n)]
                
                if verify_fraction(U_frac, V_frac, W_frac):
                    return U_frac, V_frac, W_frac
                elif verify(U_frac, V_frac, W_frac):
                    # Try to clean up
                    return U_frac, V_frac, W_frac
        
        return None
    
    # Search from low rank upward
    for target_rank in range(24, best_rank + 1):
        if time.time() > deadline:
            break
        result = try_find_decomposition(target_rank, max_iters=50)
        if result is not None:
            U, V, W = result
            if verify(U, V, W):
                best_U, best_V, best_W = U, V, W
                best_rank = target_rank
                break
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
