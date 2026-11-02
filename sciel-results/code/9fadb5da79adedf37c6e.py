import numpy as np
import time
from itertools import product
# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 4, 7
    start_time = time.time()
    deadline = start_time + 120  # 120s safety margin

    # Schoolbook decomposition
    def schoolbook():
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

    def is_valid_decomposition(U, V, W, n, m, p, tol=1e-10):
        """Verify that the decomposition correctly represents matrix multiplication."""
        rank = len(U[0])
        # Check shapes
        if len(U) != n * m or len(V) != m * p or len(W) != p * n:
            return False
        if any(len(row) != rank for row in U):
            return False
        if any(len(row) != rank for row in V):
            return False
        if any(len(row) != rank for row in W):
            return False
        
        # Check that all entries are integers or half-integers
        for mat in [U, V, W]:
            for row in mat:
                for val in row:
                    if val != int(val) and val * 2 != int(val * 2):
                        return False
        
        # Verify tensor equality: T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        # For matrix multiplication, T[i,j,k] = 1 if j==k, else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0 if j == k else 0.0
                    actual = 0.0
                    for r in range(rank):
                        actual += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(actual - expected) > tol:
                        return False
        return True

    def try_structured_search(max_rank, time_budget):
        """Try to find a lower-rank decomposition using structured patterns."""
        # For now, this is a placeholder for more sophisticated search
        # In a real implementation, we would use SAT solvers or other methods
        return None

    # Return the schoolbook solution which is always valid
    return schoolbook()
# EVOLVE-BLOCK-END
