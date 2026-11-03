"""Tensor decomposition for 2x5x6 matrix multiplication using a ladder/shifted approach."""
import numpy as np
from itertools import product

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    
    # Build the multiplication tensor T with shape (n*m, m*p, p*n)
    # T[a, b, c] = 1 if there exists (i,j,k) such that:
    #   a = i*m + j, b = j*p + k, c = k*n + i
    # This is the standard matrix multiplication tensor.
    
    # The schoolbook rank is n*m*p = 60.
    # We attempt to find a lower rank exact decomposition using integer/half-integer entries.
    
    # Strategy: Try to find a rank-r decomposition by solving a system of linear equations.
    # For each rank r, we need to find U (n*m x r), V (m*p x r), W (p*n x r) such that:
    #   sum_s U[a,s] * V[b,s] * W[c,s] = T[a,b,c] for all (a,b,c)
    
    # This is a non-linear system (trilinear), so we use a different approach.
    # We can fix U and V and solve for W, or use alternating minimization.
    
    # However, for exact rational solutions, we can use the following:
    # The tensor T can be viewed as a linear map from (n*m, m*p) to (p*n) via the multiplication.
    
    # For n=2, there is a known technique: decompose the 2x5x6 multiplication
    # using the fact that C[0,k] and C[1,k] can be computed with shared terms.
    
    # Let's try a rank-59 or lower decomposition using a greedy/iterative approach.
    
    # Actually, let's use a more principled approach: 
    # The matrix multiplication tensor M_{2,5,6} has a known lower bound.
    # For now, let's implement a valid decomposition and try to optimize.
    
    # Start with schoolbook rank 60, then try to reduce.
    
    def schoolbook_decomposition():
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
        
        return U, V, W
    
    # Verify the tensor entries
    def tensor_entry(a, b, c):
        # a = i*m + j, b = j'*p + k, c = k'*n + i'
        i = a // m
        j = a % m
        j2 = b // p
        k = b % p
        k2 = c // n
        i2 = c % n
        # The tensor entry is 1 if i == i2 and j == j2
        return 1 if (i == i2 and j == j2) else 0
    
    # Check if a given (U, V, W) is valid
    def is_valid(U, V, W, tolerance=1e-9):
        rank = len(U[0])
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    expected = tensor_entry(a, b, c)
                    actual = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(actual - expected) > tolerance:
                        return False
        return True
    
    # Start with schoolbook
    U, V, W = schoolbook_decomposition()
    
    # Try to reduce rank by finding linear dependencies
    # This is a hard problem, so we use a heuristic:
    # Check if any two columns in the decomposition are linearly dependent
    # in a way that allows merging.
    
    # For a more effective approach, let's try random rank-r decompositions
    # and see if we can find a valid one with r < 60.
    
    import random
    random.seed(42)
    
    best_rank = n * m * p
    best_U, best_V, best_W = U, V, W
    
    # Try ranks from 59 down to some minimum
    for target_rank in range(n * m * p - 1, 30, -1):
        # Try a few random initializations
        for attempt in range(10):
            U_try = np.random.randn(n * m, target_rank)
            V_try = np.random.randn(m * p, target_rank)
            W_try = np.random.randn(p * n, target_rank)
            
            # Check if this is valid
            if is_valid(U_try.tolist(), V_try.tolist(), W_try.tolist()):
                best_rank = target_rank
                best_U, best_V, best_W = U_try.tolist(), V_try.tolist(), W_try.tolist()
                break
        if best_rank == target_rank:
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
