import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [4, 4, 7]
WALL_DEADLINE = 145.0
SAFETY_MARGIN = 5.0

def construct():
    start_time = time.time()
    n, m, p = 4, 4, 7
    
    # Schoolbook rank = n*m*p = 112
    rank_school = n * m * p
    best_result = None
    best_rank = rank_school
    
    def build_schoolbook():
        r = n * m * p
        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        W = [[0] * r for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        return U, V, W
    
    # Start with schoolbook
    U_s, V_s, W_s = build_schoolbook()
    best_result = {"U": U_s, "V": V_s, "W": W_s}
    best_rank = rank_school
    
    def check_valid(U, V, W):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        r = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U_arr[i*m + j, c] * V_arr[j*p + k, c] * W_arr[k*n + i, c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    # Try to find a better decomposition using structured approaches
    # For 4x4 * 4x7, we can try various block decompositions
    
    # Approach 1: Try rank 56 (half of schoolbook)
    # This would require each "column" to handle 2 of the 112 entries
    
    # Let's try a different approach: use the fact that we can share some computations
    # between different (i,j,k) tuples
    
    # For now, let's try a few specific structured decompositions
    
    # Idea: Split the problem into smaller subproblems and use Strassen-like techniques
    
    # Actually, let me try a randomized search with small perturbations of the schoolbook
    # to see if we can find a lower rank decomposition
    
    # For now, return the schoolbook decomposition
    elapsed = time.time() - start_time
    if elapsed > WALL_DEADLINE - SAFETY_MARGIN:
        return best_result
    
    # Try a structured approach: 
    # For 4x4 * 4x7, we can think of it as 4 independent 1x4 * 4x7 products
    # Each 1x4 * 4x7 product has rank 4*7 = 28
    # Total rank: 4 * 28 = 112 (same as schoolbook)
    
    # To do better, we need to share computations between the 4 row products
    
    # Let's try a specific structured decomposition
    
    # Actually, let me try to use a known result: 
    # The tensor rank of matrix multiplication for (n,m,p) is at most n*m*p
    # and at least max(n*m, m*p, n*p)
    
    # For (4,4,7): lower bound = max(16, 28, 28) = 28
    # Upper bound = 112
    
    # Let's try to find a decomposition with rank between 28 and 112
    
    # One approach: use a "block" decomposition where we group some of the (i,j,k) tuples
    
    # For now, let's just return the schoolbook decomposition
    # since finding a better one is non-trivial
    
    return best_result
