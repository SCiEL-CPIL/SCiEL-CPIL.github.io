import time
import random
import numpy as np

DIMENSIONS = [3, 4, 8]

def _verify(U, V, W, n, m, p):
    # Check if U, V, W decompose the tensor
    rank = len(U[0])
    # Construct the tensor from U, V, W
    # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We need to check if this equals the multiplication tensor
    # Multiplication tensor: T[i, j, k] = 1 if i, j, k are valid indices for C = A * B
    # C[i, k] = sum_j A[i, j] B[j, k]
    # So T[i, j, k] = 1 for all i, j, k? No.
    # The tensor is defined by T[i, j, k] = 1 if the term A[i,j]B[j,k] is part of the sum.
    # Actually, the tensor is T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # Wait, the tensor decomposition is for the map A, B -> C.
    # C[i, k] = sum_j A[i, j] B[j, k].
    # The tensor T is such that C[i, k] = sum_{i,j,k} T[i, j, k] A[i, j] B[j, k].
    # So T[i, j, k] = 1 if the indices match?
    # No, T[i, j, k] is 1 for all i, j, k in the schoolbook.
    # The decomposition is: C[i, k] = sum_c U[i, k, c] * (sum_j A[i, j] V[j, c]) * ... ?
    # No, the standard form is:
    # C[i, k] = sum_c (sum_j U[i, j, c] A[i, j]) * (sum_k V[j, k, c] B[j, k]) ?
    # The interface is U (n*m, rank), V (m*p, rank), W (p*n, rank).
    # The equation is:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i, j, k?
    # No, this is not the multiplication tensor.
    # The multiplication tensor is T[i, j, k] = 1 if i, j, k are such that A[i,j] and B[j,k] contribute to C[i,k].
    # In the schoolbook, each term A[i,j]B[j,k] is a separate multiplication.
    # So the tensor has 1s at (i, j, k) for all i, j, k.
    # And the decomposition is:
    # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Wait, if U, V, W are 1 at the schoolbook positions, then:
    # sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
    # For a fixed (i, j, k), only one c is non-zero (the one corresponding to i, j, k).
    # So the sum is 1.
    # So the condition is:
    # For all i, j, k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # This is the condition.
    
    # We can verify by constructing the full tensor and comparing.
    # But for speed, we can just check a few random (i, j, k) or all.
    # Since n, m, p are small, we can check all.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    # Start with schoolbook
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
    
    # Try to reduce rank by searching for a lower rank decomposition
    # We will try rank 48
    target_rank = 48
    if target_rank < rank:
        # Random search for rank 48
        start_time = time.time()
        max_time = 100  # Leave 20s for safety
        found = False
        while time.time() - start_time < max_time and not found:
            # Generate random integer/half-integer matrices
            # Use small integers to increase chance of exact match
            U_rand = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(n * m)]
            V_rand = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(m * p)]
            W_rand = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(p * n)]
            
            if _verify(U_rand, V_rand, W_rand, n, m, p):
                return {"U": U_rand, "V": V_rand, "W": W_rand}
    
    return {"U": U, "V": V, "W": W}
