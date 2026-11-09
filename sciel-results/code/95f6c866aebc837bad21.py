"""Program that attempts to find a rank-minimal exact tensor decomposition for 3x5x7 matrix multiplication."""
import time
import sys
import random
from fractions import Fraction
import numpy as np

DIMENSIONS = [3, 5, 7]
SEED_RANK = 105

# EVOLVE-BLOCK-START

def _validate(U, V, W, n=3, m=5, p=7):
    """
    Validate that the factor matrices U, V, W exactly decompose the matrix multiplication tensor.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    The condition is: for all i, j, k,
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    """
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1:
                    return False
    return True


def _schoolbook(n=3, m=5, p=7):
    """Return the schoolbook decomposition with rank n*m*p."""
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


def _try_merge_terms(U, V, W, n, m, p, deadline):
    """
    Try to merge two terms that have compatible support patterns.
    Two terms r1 and r2 can potentially be merged if their supports are disjoint
    in a way that allows a linear combination to cover the same tensor entries.
    """
    rank = len(U[0])
    best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    best_rank = rank
    
    # Try all pairs of terms to see if they can be merged
    for r1 in range(rank):
        if time.time() > deadline:
            break
        for r2 in range(r1 + 1, rank):
            if time.time() > deadline:
                break
            
            # Check if terms r1 and r2 can be merged
            # This is possible if there exists a scalar c such that
            # the combined term covers the same (i,j,k) positions
            # For simplicity, we'll try a greedy approach:
            # if the supports are disjoint, we can't merge directly
            # but if they overlap, we might be able to adjust
            
            # For now, just try zeroing out one term and see if the other
            # can be adjusted to compensate
            U_save = [row[:] for row in U]
            V_save = [row[:] for row in V]
            W_save = [row[:] for row in W]
            
            # Zero out term r2
            for i in range(n * m):
                U[i][r2] = 0
            for j in range(m * p):
                V[j][r2] = 0
            for k in range(p * n):
                W[k][r2] = 0
            
            # Check if still valid
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for rr in range(rank):
                            s += U[i*m+j][rr] * V[j*p+k][rr] * W[k*n+i][rr]
                        if s != 1:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                # Remove column r2
                U = [row[:r2] + row[r2+1:] for row in U]
                V = [row[:r2] + row[r2+1:] for row in V]
                W = [row[:r2] + row[r2+1:] for row in W]
                rank -= 1
                best_rank = rank
                best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
                break
            else:
                # Restore
                U, V, W = U_save, V_save, W_save
        if best_rank < rank:
            break
    
    if best_rank < rank:
        return {"U": best_U, "V": best_V, "W": best_W}
    return None


def _structured_decomposition(n=3, m=5, p=7, deadline=None):
    """
    Try to find a structured lower-rank decomposition using known
    matrix multiplication algorithm patterns.
    """
    if deadline is None:
        deadline = time.time() + 140
    
    best = _schoolbook(n, m, p)
    current = best
    
    # Iteratively try to reduce rank
    for iteration in range(50):
        if time.time() > deadline:
            break
        
        result = _try_merge_terms(current["U"], current["V"], current["W"], n, m, p, deadline)
        if result is not None:
            current = result
            if _validate(current["U"], current["V"], current["W"]):
                best = current
        else:
            break
    
    return best


def construct():
    """
    Returns the best tensor decomposition found for matrix multiplication.
    """
    deadline = time.time() + 140
    
    # First, return schoolbook as a safe fallback
    best = _schoolbook()
    
    # Attempt structured rank reduction
    if time.time() < deadline:
        result = _structured_decomposition(deadline=deadline)
        if result is not None:
            # Validate the result before accepting it
            if _validate(result["U"], result["V"], result["W"]):
                best = result
            else:
                # If validation fails, stick with schoolbook
                best = _schoolbook()
    
    return best

# EVOLVE-BLOCK-END
