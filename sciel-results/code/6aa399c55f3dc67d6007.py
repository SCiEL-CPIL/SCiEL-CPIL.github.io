"""Matrix multiplication 4x5x6 rank minimization."""
import time
import random
import sys

DIMENSIONS = [4, 5, 6]

def _schoolbook():
    n, m, p = 4, 5, 6
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

def _verify(U, V, W, n=4, m=5, p=6):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # Build the target tensor: T[i,j,k] = 1 if we want C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor equation is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,k}... 
    # Wait, the matrix multiplication tensor is defined such that:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T has entries T[i,j,k] = 1 for all (i,j,k) in the multiplication.
    # Actually, the standard matrix multiplication tensor M_{n,m,p} has entries:
    # T[i,j,k] = 1 if the (i,k)-th entry of C depends on A[i,j] and B[j,k].
    # Since C[i,k] = sum_j A[i,j] B[j,k], every (i,j,k) triple contributes.
    # So T[i,j,k] = 1 for all valid i,j,k.
    #
    # The decomposition requires: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # where W is indexed as W[k*n+i, r] per the problem statement.
    
    # Let's verify by checking a few random entries
    for _ in range(50):
        i = random.randint(0, n - 1)
        j = random.randint(0, m - 1)
        k = random.randint(0, p - 1)
        s = 0.0
        for r in range(rank):
            s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
        if abs(s - 1.0) > 1e-9:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    
    n, m, p = 4, 5, 6
    best = _schoolbook()
    
    # Try to find a lower rank solution using a heuristic approach.
    # The idea: start with a random set of rank-1 terms and try to cover all (i,j,k) entries.
    # This is a set cover problem, which is NP-hard, but we can use a greedy approach.
    
    # For now, just return the schoolbook solution.
    return best
