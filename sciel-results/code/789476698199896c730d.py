"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W give an exact decomposition of the matrix multiplication tensor."""
    rank = len(U[0])
    # Check all entries of the tensor
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                i_a = a // m
                j_a = a % m
                j_b = b // p
                k_b = b % p
                k_c = c // n
                i_c = c % n
                
                is_valid = (j_a == j_b) and (i_a == i_c) and (k_b == k_c)
                
                val = 0
                for col in range(rank):
                    val += U[a][col] * V[b][col] * W[c][col]
                
                if is_valid:
                    if abs(val - 1) > 1e-9:
                        return False
                else:
                    if abs(val) > 1e-9:
                        return False
    return True


def _is_half_integer(val, tol=1e-9):
    """Check if val is an integer or half-integer."""
    val2 = val * 2
    return abs(val2 - round(val2)) < tol


def _round_to_half_int(val):
    """Round to nearest half-integer."""
    return round(val * 2) / 2.0


def _schoolbook(n, m, p):
    """Return the schoolbook (trivial) decomposition."""
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
    return U, V, W


def _try_structured_rank(n, m, p, target_rank, deadline):
    """
    Try to find a rank-`target_rank` decomposition using structured patterns.
    For matrix multiplication A(n×m) * B(m×p) = C(n×p),
    the tensor T[i,j,k] = 1 if indices are consistent.
    
    We try various algebraic structures that can reduce rank below n*m*p.
    """
    if time.time() > deadline:
        return None
    
    # Known lower bounds: rank >= max(n*m, m*p, n*p) = max(6, 18, 12) = 18
    # Schoolbook rank = 36
    # Try ranks from 18 upward
    
    # Strategy: Use the fact that for small dimensions, we can try
    # explicit algebraic identities.
    
    # For 2x3 * 3x6, one approach:
    # Group columns of A and rows of B to create shared products.
    
    # Let's try a block-based approach:
    # A is 2x3, B is 3x6, C is 2x6
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # Rank 18 is the trivial lower bound (max of dimensions).
    # Let's see if we can achieve rank < 36.
    
    # One known technique: if we can factor through intermediate sums.
    # For example, if B has structure, we can precompute sums.
    
    # Let's try rank = 18 first (the theoretical minimum)
    # This would require a very specific structure.
    
    # Actually, for general matrix multiplication, the rank is typically
    # close to n*m*p. But for small cases, there might be shortcuts.
    
    # Let's try a different approach: use the Strassen-like idea
    # but adapted to our dimensions.
    
    # For 2x3 * 3x6:
    # We have 2*3*6 = 36 products in schoolbook.
    # Can we do better?
    
    # One idea: if we group the 3 columns of A and 3 rows of B,
    # we might be able to use fewer than 9 products per (i,k) pair.
    
    # Let's try to find a rank-24 decomposition (50% reduction):
    # This would mean using 24 instead of 36 basis tensors.
    
    # Actually, let's think about this more carefully.
    # The tensor is T[(i,j), (j,k), (i,k)] for i in {0,1}, j in {0,1,2}, k in {0,...,5}
    
    # Total number of non-zero entries = 2 * 3 * 6 = 36
    
    # A rank-r decomposition means we can write T as sum of r rank-1 tensors.
    # Each rank-1 tensor is u ⊗ v ⊗ w.
    
    # The question is: what is the minimum r?
    
    # For now, let's try to find any decomposition with rank < 36.
    
    # Approach: Random search with constraint satisfaction
    # But this is hard to do exactly.
    
    # Alternative: Use known results. For 2x2 * 2x2, Strassen gives rank 7 < 8.
    # For our case, let's try to see if there's a similar shortcut.
    
    # Let's try rank = 30 as a first attempt (modest improvement)
    # We'll use a heuristic: some products can be shared.
    
    # Actually, let's just return the schoolbook for now,
    # and implement a more sophisticated search later.
    
    return None


def _try_rank(n, m, p, target_rank, deadline):
    """Try to find a decomposition of the given rank."""
    if time.time() > deadline:
        return None
    
    # Try structured approaches first
    result = _try_structured_rank(n, m, p, target_rank, deadline)
    if result is not None:
        return result
    
    # For now, return None to indicate failure
    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    schoolbook_rank = n * m * p  # 36
    
    # Try to find a decomposition with rank less than schoolbook
    best_rank = schoolbook_rank
    best_U, best_V, best_W = _schoolbook(n, m, p)
    
    # Try ranks from 18 (theoretical min) up to 35
    for target_rank in range(18, schoolbook_rank):
        if time.time() > deadline:
            break
        
        result = _try_rank(n, m, p, target_rank, deadline)
        if result is not None:
            U, V, W = result
            if _verify_decomposition(U, V, W, n, m, p):
                # Check half-integer constraint
                valid = True
                for matrix in [U, V, W]:
                    for row in matrix:
                        for val in row:
                            if not _is_half_integer(val):
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                if valid:
                    best_rank = target_rank
                    best_U, best_V, best_W = U, V, W
                    break  # Found a valid lower-rank decomposition
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
