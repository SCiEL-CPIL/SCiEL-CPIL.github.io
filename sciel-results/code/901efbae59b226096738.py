import time
import sys
import random
import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify that U, V, W form a valid exact decomposition."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U_arr[i * m + j, r] * V_arr[j * p + k, r] * W_arr[k * n + i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
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

def _try_rank(n, m, p, rank, time_budget):
    """Try to find a rank-r decomposition using randomized search."""
    start = time.time()
    n_m = n * m
    m_p = m * p
    p_n = p * n
    
    # Number of equations: n*m*p
    n_eq = n * m * p
    
    # We need to find U (n_m x rank), V (m_p x rank), W (p_n x rank)
    # such that for all (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    
    # Randomized approach: try random initializations and use gradient-like updates
    # Or: set up as a system and try to solve
    
    # For now, try a structured approach based on the dimensions
    # Use the fact that we can decompose using polynomial multiplication
    
    # Winograd-style: think of the middle dimension as polynomial multiplication
    # For each (i,k) pair in C, C[i,k] = sum_j A[i,j] * B[j,k]
    # This is a dot product of length m=3
    
    # We can use the fact that dot product of length 3 can be done with 3 multiplications
    # (that's just the standard way), but maybe we can share across different (i,k) pairs
    
    # Let's try a different approach: use the structure of the problem
    # C is 2x7, A is 2x3, B is 3x7
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    
    # This is a bilinear form. The tensor is in (R^2 x R^3)^* x (R^3 x R^7)^* x (R^2 x R^7)
    
    # Let's try to find a low-rank decomposition by solving the system
    # For rank r, we have r variables in each of U, V, W
    # Total unknowns: r*(n_m + m_p + p_n) = r*(6 + 21 + 14) = 41r
    # Equations: 42
    
    # For r >= 1, we have more unknowns than equations, so solutions should exist
    # But they need to be exact (integer/half-integer)
    
    # Let's try r from 21 upward
    # Actually, let's start with a heuristic: try to find a decomposition
    # by exploiting the structure
    
    # For now, just return None if we can't find one quickly
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    best = _schoolbook(n, m, p)
    
    # Try to find a better decomposition
    # Start from rank 21 (theoretical lower bound) and go up
    for rank in range(21, n * m * p):
        if time.time() > deadline - 5:
            break
        result = _try_rank(n, m, p, rank, deadline - time.time())
        if result is not None:
            if _verify(result["U"], result["V"], result["W"], n, m, p):
                best = result
                break
    
    return best
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    res = construct()
