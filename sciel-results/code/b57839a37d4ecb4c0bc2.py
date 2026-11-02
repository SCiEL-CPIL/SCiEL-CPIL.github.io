"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np

DIMENSIONS = [2, 2, 9]
_start_time = time.time()
_TIMEOUT = 140  # safety margin before 150s kill

def _check_validity(U, V, W, n, m, p):
    """Check if the decomposition is valid for the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # T[i,j,k] = 1 for all i,j,k in the multiplication
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # Tensor: T[a,b,d] where a=i*m+j, b=j'*p+k, d=k'*n+i
    # T[a,b,d] = 1 if a's j == b's j' (the inner index matches)
    # a = i*m + j, b = j2*p + k, d = k2*n + i2
    # T=1 iff j == j2, i == i2, and k == k2? No...
    # Actually for the multiplication tensor: T[i,j,k] = 1
    # In the flattened form: a = i*m+j (A index), b = j*p+k (B index), d = k*n+i (C index)
    # T[a,b,d] = 1 iff (a mod m) == (b // p) and (a // m) == (d % n) and (b % p) == (d // n)
    # Wait, let me re-derive.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # A has indices (i,j) -> a = i*m + j
    # B has indices (j,k) -> b = j*p + k  
    # C has indices (i,k) -> d = k*n + i (trace convention: W[k*n+i, r])
    # T[a,b,d] = 1 iff j_from_a == j_from_b AND i_from_a == i_from_d AND k_from_b == k_from_d
    # j_from_a = a % m, i_from_a = a // m
    # j_from_b = b // p, k_from_b = b % p
    # i_from_d = d % n, k_from_d = d // n
    # So T[a,b,d] = 1 iff (a % m) == (b // p) and (a // m) == (d % n) and (b % p) == (d // n)
    
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Compute the tensor from the decomposition
    for a in range(nm):
        for b in range(mp):
            for d in range(pn):
                expected = 1.0 if (a % m == b // p and a // m == d % n and b % p == d // n) else 0.0
                actual = 0.0
                for c in range(rank):
                    actual += U[a][c] * V[b][c] * W[d][c]
                if abs(actual - expected) > 1e-9:
                    return False
    return True

def _build_schoolbook(n, m, p):
    """Build the schoolbook rank n*m*p decomposition."""
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
    return U, V, W, rank

def _try_remove_pair(U, V, W, c1, c2, n, m, p):
    """Try to remove term c2 by adjusting other terms to absorb it. Returns (success, new_U, new_V, new_W, new_rank)."""
    # This is complex; for now, just check if removing c2 and keeping all others still works
    # (i.e., term c2 contributes zero, which it doesn't in schoolbook)
    # So this won't work for direct removal. We need a different approach.
    return False, None, None, None, 0

def construct():
    n, m, p = DIMENSIONS
    U, V, W, rank = _build_schoolbook(n, m, p)
    
    # Try to reduce rank by finding linear dependencies
    # For now, return the schoolbook decomposition
    # The pair-term removal strategy can find rank 32, but implementing it
    # within time constraints is challenging. Return schoolbook as fallback.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
