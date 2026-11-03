"""Rewritten program for matrix multiplication 3x3 by 3x4 tensor decomposition."""
import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 3, 4]

# Known lower bound for 3x3 * 3x4 is at least 20 (from literature on small matrix multiplication)
# Schoolbook gives 36. We'll try to find a rank-20 or lower decomposition.
# 
# The strategy: We'll use a randomized/structured search for a rank-r decomposition.
# For each candidate rank r, we set up the tensor equations and try to solve them.
#
# The tensor T has entries T[i,j,k] = 1 if the (i,k) entry of A*B uses A[i,j]*B[j,k].
# A decomposition of rank r means: T[i,j,k] = sum_{c=1}^{r} U[i,j,c] * V[j,k,c] * W[i,k,c]
# where U is (n*m) x r, V is (m*p) x r, W is (p*n) x r.
# 
# Note the W indexing: W[k*n+i, c] per the trace convention.
#
# We'll try ranks from 20 upward, using a structured approach:
# 1. Fix a sparse pattern for U, V, W (e.g., each column has few non-zeros)
# 2. Solve the resulting polynomial system

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    # Check shapes
    if len(U) != n * m:
        return False
    if len(V) != m * p:
        return False
    if len(W) != p * n:
        return False
    for c in range(rank):
        if len(U[c]) != rank or len(V[c]) != rank or len(W[c]) != rank:
            return False
    
    # Compute the tensor from the decomposition
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(rank):
                    val += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                expected = 1  # Every entry of the multiplication tensor is 1
                if val != expected:
                    return False
    return True


def _try_rank_random(r, n, m, p, max_iters=5000, seed=42):
    """Try to find a rank-r decomposition using randomized search."""
    rng = np.random.RandomState(seed)
    
    for it in range(max_iters):
        # Initialize U, V, W with random sparse structure
        # Each column c has a few non-zero entries in U, V, W
        U = np.zeros((n * m, r), dtype=object)
        V = np.zeros((m * p, r), dtype=object)
        W = np.zeros((p * n, r), dtype=object)
        
        for c in range(r):
            # Pick 2-3 indices for U
            u_idx = rng.choice(n * m, size=min(3, n * m), replace=False)
            v_idx = rng.choice(m * p, size=min(3, m * p), replace=False)
            w_idx = rng.choice(p * n, size=min(3, p * n), replace=False)
            
            for idx in u_idx:
                U[idx, c] = Fraction(rng.randint(-3, 4))
            for idx in v_idx:
                V[idx, c] = Fraction(rng.randint(-3, 4))
            for idx in w_idx:
                W[idx, c] = Fraction(rng.randint(-3, 4))
        
        # Check if this works
        if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return U.tolist(), V.tolist(), W.tolist()
    
    return None


def _try_rank_structured(r, n, m, p):
    """Try a structured decomposition based on known patterns."""
    # For 3x3 * 3x4, let's try to exploit the structure.
    # A[i,j] * B[j,k] where i in {0,1,2}, j in {0,1,2}, k in {0,1,2,3}
    # 
    # One approach: use the fact that B has 4 columns.
    # We can think of this as computing 4 separate 3x3 * 3x1 products,
    # but sharing structure.
    #
    # Let's try a systematic search over small integer coefficients.
    # For rank 20, we have 20 terms. Each term (c) has U_c, V_c, W_c.
    # 
    # A simpler structured approach: try to find a decomposition where
    # each column of U has exactly 2 non-zeros, etc.
    
    # This is complex to do systematically. Let's try a different approach:
    # Use the known result that 3x3 * 3x3 has rank at most 23 (Strassen-like).
    # For 3x3 * 3x4, we can extend this.
    
    # Actually, let's just return None and rely on the random search.
    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for the 150s limit
    
    # First, always have the schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_fb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_fb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_fb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_fb, "V": V_fb, "W": W_fb}
    best_rank = rank_schoolbook
    
    # Try to find a better decomposition
    # Try ranks from 20 up to 35
    for r in range(20, min(36, rank_schoolbook)):
        if time.time() > deadline:
            break
        
        result = _try_rank_random(r, n, m, p, max_iters=200, seed=r * 1000)
        if result is not None:
            U, V, W = result
            best_result = {"U": U, "V": V, "W": W}
            best_rank = r
            break  # Found the lowest rank we tried
    
    if time.time() > deadline:
        return best_result
    
    # If we didn't find anything better, try a more thorough search at rank 20
    if best_rank == rank_schoolbook:
        result = _try_rank_random(20, n, m, p, max_iters=5000, seed=99999)
        if result is not None:
            U, V, W = result
            best_result = {"U": U, "V": V, "W": W}
            best_rank = 20
    
    return best_result
