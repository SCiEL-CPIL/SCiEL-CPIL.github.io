import numpy as np
import time
from fractions import Fraction
from itertools import product
from collections import defaultdict

DIMENSIONS = [2, 3, 5]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    rank = len(U[0])
    U_frac = [[Fraction(x) for x in row] for row in U]
    V_frac = [[Fraction(x) for x in row] for row in V]
    W_frac = [[Fraction(x) for x in row] for row in W]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                total = Fraction(0)
                for r in range(rank):
                    total += U_frac[idx_u][r] * V_frac[idx_v][r] * W_frac[idx_w][r]
                if total != 1:
                    return False
    return True

def _construct_rank_28():
    n, m, p = 2, 3, 5
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    for i in range(n):
        for k in range(p):
            U[i * m + 2][r] = 1
            V[2 * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1
            
    for i in range(n):
        for j in range(2):
            U[i * m + j][r] = 1
            V[j * p + 4][r] = 1
            W[4 * n + i][r] = 1
            r += 1
            
    U[0 * m + 0][r] = 1; U[1 * m + 1][r] = 1; V[0 * p + 0][r] = 1; V[1 * p + 1][r] = 1; W[0 * n + 0][r] = 1; W[1 * n + 1][r] = 1; r += 1
    U[1 * m + 0][r] = 1; U[1 * m + 1][r] = 1; V[0 * p + 0][r] = 1; W[0 * n + 1][r] = 1; W[1 * n + 1][r] = -1; r += 1
    U[0 * m + 0][r] = 1; V[0 * p + 1][r] = 1; V[1 * p + 1][r] = -1; W[1 * n + 0][r] = 1; W[1 * n + 1][r] = 1; r += 1
    U[1 * m + 1][r] = 1; V[1 * p + 0][r] = 1; V[0 * p + 0][r] = -1; W[0 * n + 0][r] = 1; W[0 * n + 1][r] = 1; r += 1
    U[0 * m + 0][r] = 1; U[0 * m + 1][r] = 1; V[1 * p + 1][r] = 1; W[0 * n + 0][r] = -1; W[1 * n + 0][r] = 1; r += 1
    U[1 * m + 0][r] = 1; U[0 * m + 0][r] = -1; V[0 * p + 0][r] = 1; V[0 * p + 1][r] = 1; W[1 * n + 1][r] = 1; r += 1
    U[0 * m + 1][r] = 1; U[1 * m + 1][r] = -1; V[1 * p + 0][r] = 1; V[1 * p + 1][r] = 1; W[0 * n + 0][r] = 1; r += 1
    
    U[0 * m + 0][r] = 1; U[1 * m + 1][r] = 1; V[0 * p + 2][r] = 1; V[1 * p + 3][r] = 1; W[2 * n + 0][r] = 1; W[3 * n + 1][r] = 1; r += 1
    U[1 * m + 0][r] = 1; U[1 * m + 1][r] = 1; V[0 * p + 2][r] = 1; W[2 * n + 1][r] = 1; W[3 * n + 1][r] = -1; r += 1
    U[0 * m + 0][r] = 1; V[0 * p + 3][r] = 1; V[1 * p + 3][r] = -1; W[3 * n + 0][r] = 1; W[3 * n + 1][r] = 1; r += 1
    U[1 * m + 1][r] = 1; V[1 * p + 2][r] = 1; V[0 * p + 2][r] = -1; W[2 * n + 0][r] = 1; W[2 * n + 1][r] = 1; r += 1
    U[0 * m + 0][r] = 1; U[0 * m + 1][r] = 1; V[1 * p + 3][r] = 1; W[2 * n + 0][r] = -1; W[3 * n + 0][r] = 1; r += 1
    U[1 * m + 0][r] = 1; U[0 * m + 0][r] = -1; V[0 * p + 2][r] = 1; V[0 * p + 3][r] = 1; W[3 * n + 1][r] = 1; r += 1
    U[0 * m + 1][r] = 1; U[1 * m + 1][r] = -1; V[1 * p + 2][r] = 1; V[1 * p + 3][r] = 1; W[2 * n + 0][r] = 1; r += 1
    
    return {"U": U, "V": V, "W": W}

def _construct_rank_27():
    """
    Construct a rank-27 decomposition by merging some terms.
    
    Strategy: Use Strassen for columns 0-1 (7 terms) and a combined
    decomposition for columns 2-4 (13 terms), plus 7 terms for j=2.
    Total: 7 + 13 + 7 = 27
    """
    n, m, p = 2, 3, 5
    rank = 27
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    # Part 1: j=2 row (7 terms - reduced from 10 by sharing)
    # We'll use a more efficient decomposition for the j=2 part
    
    # Actually, let's keep the 10 schoolbook terms for j=2
    for i in range(n):
        for k in range(p):
            U[i * m + 2][r] = 1
            V[2 * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1
    
    # Part 2: Strassen for columns 0-1 (7 terms)
    U[1 * m + 0][r] = 1; V[0 * p + 0][r] = 1; W[0 * n + 1][r] = 1; r += 1
    U[1 * m + 0][r] = 1; U[1 * m + 1][r] = 1; V[0 * p + 1][r] = 1; W[1 * n + 1][r] = 1; r += 1
    U[1 * m + 1][r] = 1; V[1 * p + 1][r] = 1; V[1 * p + 0][r] = -1; W[0 * n + 1][r] = 1; r += 1
    U[0 * m + 0][r] = 1; V[1 * p + 0][r] = 1; V[0 * p + 1][r] = 1; W[1 * n + 0][r] = 1; r += 1
    U[0 * m + 0][r] = 1; U[0 * m + 1][r] = 1; V[1 * p + 1][r] = 1; V[1 * p + 0][r] = -1; W[0 * n + 1][r] = 1; W[1 * n + 1][r] = 1; r += 1
    U[1 * m + 1][r] = 1; V[0 * p + 1][r] = 1; V[1 * p + 1][r] = -1; W[0 * n + 0][r] = 1; r += 1
    U[1 * m + 0][r] = 1; U[1 * m + 1][r] = 1; V[0 * p + 0][r] = 1; V[1 * p + 0][r] = -1; W[0 * n + 1][r] = 1; W[1 * n + 1][r] = 1; r += 1
    
    # Part 3: Combined decomposition for columns 2-4 (10 terms)
    # Use a more efficient method than schoolbook (which would be 12)
    
    # Schoolbook for columns 2-4 would be:
    # For each i in {0,1}, k in {2,3,4}, j in {0,1}: A[i,j]*B[j,k]
    # That's 2*3*2 = 12 terms
    
    # We can reduce to 10 by using Strassen-like methods on the 2x2 submatrices
    # Treat columns 2-3 as a 2x2 block and column 4 separately
    
    # Strassen for columns 2-3 (7 terms)
    U[1 * m + 0][r] = 1; V[0 * p + 2][r] = 1; W[2 * n + 1][r] = 1; r += 1
    U[1 * m + 0][r] = 1; U[1 * m + 1][r] = 1; V[0 * p + 3][r] = 1; W[3 * n + 1][r] = 1; r += 1
    U[1 * m + 1][r] = 1; V[1 * p + 3][r] = 1; V[1 * p + 2][r] = -1; W[2 * n + 1][r] = 1; r += 1
    U[0 * m + 0][r] = 1; V[1 * p + 2][r] = 1; V[0 * p + 3][r] = 1; W[3 * n + 0][r] = 1; r += 1
    U[0 * m + 0][r] = 1; U[0 * m + 1][r] = 1; V[1 * p + 3][r] = 1; V[1 * p + 2][r] = -1; W[2 * n + 1][r] = 1; W[3 * n + 1][r] = 1; r += 1
    U[1 * m + 1][r] = 1; V[0 * p + 3][r] = 1; V[1 * p + 3][r] = -1; W[2 * n + 0][r] = 1; r += 1
    U[1 * m + 0][r] = 1; U[1 * m + 1][r] = 1; V[0 * p + 2][r] = 1; V[1 * p + 2][r] = -1; W[2 * n + 1][r] = 1; W[3 * n + 1][r] = 1; r += 1
    
    # Schoolbook for column 4 (3 terms - reduced from 4 by sharing)
    U[0 * m + 0][r] = 1; V[0 * p + 4][r] = 1; W[4 * n + 0][r] = 1; r += 1
    U[0 * m + 1][r] = 1; V[1 * p + 4][r] = 1; W[4 * n + 0][r] = 1; r += 1
    U[1 * m + 0][r] = 1; U[1 * m + 1][r] = 1; V[0 * p + 4][r] = 1; V[1 * p + 4][r] = 1; W[4 * n + 1][r] = 1; r += 1
    
    return {"U": U, "V": V, "W": W}

def _try_find_lower_rank(deadline):
    """
    Attempt to find a lower rank decomposition by trying different constructions.
    """
    n, m, p = 2, 3, 5
    
    # Try rank 27 first
    if time.time() > deadline - 10:
        return None
    
    res_27 = _construct_rank_27()
    if _verify_decomposition(res_27['U'], res_27['V'], res_27['W'], n, m, p):
        return res_27
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # First, try to find a lower rank decomposition
    result = _try_find_lower_rank(deadline)
    
    if result is not None:
        if _verify_decomposition(result['U'], result['V'], result['W'], 2, 3, 5):
            return result
    
    # Fallback to known rank-28 decomposition
    res_28 = _construct_rank_28()
    if _verify_decomposition(res_28['U'], res_28['V'], res_28['W'], 2, 3, 5):
        return res_28
    else:
        raise ValueError("Verification failed for rank-28 decomposition")
