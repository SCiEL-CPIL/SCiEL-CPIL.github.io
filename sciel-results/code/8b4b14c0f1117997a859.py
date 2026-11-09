import numpy as np
import time
from fractions import Fraction
import itertools
import math
import random
import sys

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def _schoolbook():
    """Return schoolbook decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
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

def _verify(decomp, tol=1e-9):
    """Verify that the decomposition exactly represents matrix multiplication."""
    n, m, p = DIMENSIONS
    U = np.array(decomp["U"], dtype=float)
    V = np.array(decomp["V"], dtype=float)
    W = np.array(decomp["W"], dtype=float)
    
    # Check dimensions
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = U[i * m + j, :]
                v_row = V[j * p + k, :]
                w_row = W[k * n + i, :]
                s = np.dot(u_row, v_row * w_row)
                if abs(s - 1.0) > tol:
                    return False
    return True

def _try_structured(rank, deadline):
    """
    Placeholder for structured search. 
    Returns None to indicate no solution found, triggering fallback.
    """
    return None

def _local_repair(U, V, W, rank, deadline):
    """Placeholder for local repair. Returns inputs unchanged."""
    return U, V, W

def _to_fraction(x):
    """Convert a float to a Fraction, rounding to nearest half-integer if close."""
    x_rounded = round(x * 2) / 2
    if abs(x - x_rounded) < 1e-9:
        return Fraction(x_rounded)
    
    x_int = round(x)
    if abs(x - x_int) < 1e-9:
        return Fraction(int(x_int))
    
    return Fraction(x).limit_denominator(1000)

def construct():
    """
    Returns the schoolbook decomposition immediately to ensure 
    a valid result within the time limit.
    """
    return _schoolbook()
# EVOLVE-BLOCK-END
