"""Rank-minimal exact tensor decomposition for 3x4 * 4x7 matrix multiplication."""
import time
import random
import sys
from fractions import Fraction

DIMENSIONS = [3, 4, 7]
n, m, p = DIMENSIONS

def _build_tensor_coeffs():
    """Return dict mapping (i,j,k) -> 1.0 for the multiplication tensor.
    The tensor is T[i,j,k] such that C[i,k] = sum_j A[i,j] B[j,k].
    We need U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] summed over c = T[i,j,k].
    """
    # T is indexed by (i, j, k)
    # U has rows indexed by (i,j), V by (j,k), W by (k,i)
    # Product: sum_c U[i,j,c] V[j,k,c] W[k,i,c] = delta(i,j,k) for the identity
    # Actually the multiplication tensor T[i,j,k] = 1 for all valid (i,j,k)
    # because C[i,k] = sum_j A[i,j]*B[j,k]
    pass

def _verify(U, V, W, rank):
    """Check if U,V,W with given rank exactly decompose the multiplication tensor.
    U: (n*m) x rank, V: (m*p) x rank, W: (p*n) x rank
    For each (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    """
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(rank):
                    total += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if total != 1:
                    return False
    return True

def _verify_frac(U, V, W, rank):
    """Exact verification using Fractions."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for c in range(rank):
                    total += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if total != 1:
                    return False
    return True

def _schoolbook():
    """Return the schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _random_integer_factors(rank, seed_val=0):
    """Generate random integer factor matrices of given rank."""
    rng = random.Random(seed_val)
    U = [[rng.randint(-2, 2) for _ in range(rank)] for _ in range(n*m)]
    V = [[rng.randint(-2, 2) for _ in range(rank)] for _ in range(m*p)]
    W = [[rng.randint(-2, 2) for _ in range(rank)] for _ in range(p*n)]
    return U, V, W

def _try_rank(rank, max_trials=200, seed_base=0):
    """Try to find an exact decomposition of given rank via random search."""
    for trial in range(max_trials):
        U, V, W = _random_integer_factors(rank, seed_base + trial)
        if _verify(U, V, W, rank):
            return {"U": U, "V": V, "W": W}
    return None

def _structured_search(rank, max_trials=100, seed_base=10000):
    """Try structured random factors with small integer entries."""
    rng = random.Random(seed_base)
    for trial in range(max_trials):
        # Use a mix of 0, 1, -1, 2, -2
        vals = [-2, -1, 0, 1, 2]
        U = [[rng.choice(vals) for _ in range(rank)] for _ in range(n*m)]
        V = [[rng.choice(vals) for _ in range(rank)] for _ in range(m*p)]
        W = [[rng.choice(vals) for _ in range(rank)] for _ in range(p*n)]
        if _verify(U, V, W, rank):
            return {"U": U, "V": V, "W": W}
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # Start with schoolbook (guaranteed valid)
    best = _schoolbook()
    best_rank = n * m * p  # 84
    
    # Try to reduce rank. Known lower bound is max(n*m, m*p, p*n) = max(12,28,21) = 28
    # But actual rank may be higher. Let's try from a reasonable starting point.
    # For 3x4x7, the rank is known to be at most 84 (schoolbook).
    # Let's try to find lower ranks.
    
    # Try ranks from 30 upward (below schoolbook)
    # We'll do a quick search for each rank
    for rank in range(30, best_rank):
        if time.time() > deadline - 5.0:
            break
        
        # Quick random search for this rank
        result = _try_rank(rank, max_trials=50, seed_base=rank * 100)
        if result is not None:
            best = result
            best_rank = rank
            break  # Found a lower rank, use it
        
        # Also try structured
        if time.time() > deadline - 5.0:
            break
        result = _structured_search(rank, max_trials=50, seed_base=rank * 100 + 5000)
        if result is not None:
            best = result
            best_rank = rank
            break
    
    # If we found something, verify it
    if best_rank < n * m * p:
        U, V, W = best["U"], best["V"], best["W"]
        if not _verify(U, V, W, best_rank):
            # Fall back to schoolbook
            best = _schoolbook()
            best_rank = n * m * p
    
    return best

# EVOLVE-BLOCK-END
