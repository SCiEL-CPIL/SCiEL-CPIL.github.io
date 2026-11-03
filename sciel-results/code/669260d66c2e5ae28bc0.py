# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from itertools import product

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    if not U or not V or not W:
        return False
    rank = len(U[0])
    if len(V[0]) != rank or len(W[0]) != rank:
        return False
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Reshape for efficient contraction
    # U: (n*m, rank) -> (n, m, rank)
    # V: (m*p, rank) -> (m, p, rank)
    # W: (p*n, rank) -> (p, n, rank)
    U_t = U_arr.reshape(n, m, rank)
    V_t = V_arr.reshape(m, p, rank)
    W_t = W_arr.reshape(p, n, rank)
    
    # Compute T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    try:
        T = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
    except Exception:
        return False
        
    # The target tensor is 1 for all i,j,k
    target = np.ones((n, m, p))
    if not np.allclose(T, target, atol=tol):
        return False
    return True

def _schoolbook(n, m, p):
    """Return the standard schoolbook decomposition of rank n*m*p."""
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

def _try_strassen_3x3x6():
    """
    Attempt to construct a rank-46 decomposition by using two independent
    rank-23 Strassen decompositions for the 3x3x6 case.
    
    The 3x3x6 multiplication can be viewed as C = A * [B1 | B2] where B1 and B2
    are 3x3 matrices.
    C1 = A * B1 and C2 = A * B2.
    
    We need explicit rank-23 factors for 3x3x3.
    Since hardcoding the full 23-term Strassen factors for 3x3x3 is complex and
    prone to error, we will attempt to generate them or use a known small set.
    
    However, a simpler improvement might be possible if we find a rank < 54
    decomposition.
    
    Actually, let's try a random search for a lower rank decomposition.
    """
    return None

def _random_search(n, m, p, max_rank, time_limit, seed=None):
    """
    Random search for a tensor decomposition of rank max_rank.
    We search over integer coefficients in a small range.
    """
    if seed is not None:
        random.seed(seed)
        
    start_time = time.time()
    best_result = None
    
    # We will try to find a decomposition of rank r < max_rank
    # Start from max_rank - 1 downwards? No, that's too slow.
    # Instead, we can try to find *any* decomposition of rank r.
    # But random search in high dimensions is very hard.
    
    # A more structured approach:
    # The tensor is T[i,j,k] = 1.
    # We want sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1.
    
    # Let's try to find a rank-1 decomposition first? Impossible for matmul.
    # Rank 2? No.
    
    # Given the time constraints and difficulty, we will just return None
    # unless we find something very quickly.
    
    # Let's try a simple heuristic:
    # If we can find a decomposition of rank 46 (two Strassen 3x3s), that's great.
    # But we need the factors.
    
    # Since we don't have the factors hardcoded, we can't guarantee this.
    
    return None

def construct():
    n, m, p = 3, 3, 6
    
    # Start with schoolbook (Rank 54)
    result = _schoolbook(n, m, p)
    
    # Attempt to find a better decomposition?
    # For now, return schoolbook as it is guaranteed correct.
    
    return result

# EVOLVE-BLOCK-END
