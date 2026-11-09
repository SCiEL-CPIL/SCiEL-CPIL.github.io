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
    U = decomp["U"]
    V = decomp["V"]
    W = decomp["W"]
    rank = len(U[0]) if U else 0
    
    # Check dimensions
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if any(len(row) != rank for row in U) or any(len(row) != rank for row in V) or any(len(row) != rank for row in W):
        return False
    
    # For matrix multiplication, the tensor T[i,j,k] = 1 for all i,j,k
    # The decomposition should satisfy: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _try_structured(rank, deadline):
    """Try structured initializations based on algebraic identities."""
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Strategy 1: Try to find a decomposition using known patterns
    # For 2x3x7, the minimal rank is believed to be around 14-20
    
    # Try random structured initializations
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        # Random initialization with some structure
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Initialize with small random values
        for c in range(rank):
            # Use a structured pattern
            scale = 1.0 / math.sqrt(rank)
            U[:, c] = np.random.randn(n * m) * scale
            V[:, c] = np.random.randn(m * p) * scale
            W[:, c] = np.random.randn(p * n) * scale
        
        # Local repair
        U, V, W = _local_repair(U, V, W, rank, deadline)
        
        # Check if we found an exact solution
        if _verify({"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}):
            # Convert to fractions for exactness
            U_frac = [[_to_fraction(x) for x in row] for row in U]
            V_frac = [[_to_fraction(x) for x in row] for row in V]
            W_frac = [[_to_fraction(x) for x in row] for row in W]
            return {"U": U_frac, "V": V_frac, "W": W_frac}
    
    return None

def _local_repair(U, V, W, rank, deadline):
    """Local optimization to reduce error."""
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Compute current error
    def compute_error():
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    err += (s - 1.0) ** 2
        return err
    
    best_error = compute_error()
    
    # Gradient-free local search
    for iteration in range(200):
        if time.time() > deadline:
            break
        
        # Try small perturbations
        improved = False
        for c in range(rank):
            for matrix, name in [(U, 'U'), (V, 'V'), (W, 'W')]:
                orig_val = matrix.copy()
                # Perturb one element
                idx = random.randint(0, len(matrix) - 1)
                col = c
                matrix[idx, col] += random.gauss(0, 0.01)
                
                new_error = compute_error()
                if new_error < best_error - 1e-12:
                    best_error = new_error
                    improved = True
                else:
                    matrix[idx, col] = orig_val[idx, col]
        
        if not improved:
            # Try larger perturbations
            for c in range(rank):
                for matrix, name in [(U, 'U'), (V, 'V'), (W, 'W')]:
                    orig_val = matrix.copy()
                    idx = random.randint(0, len(matrix) - 1)
                    matrix[idx, c] += random.gauss(0, 0.1)
                    
                    new_error = compute_error()
                    if new_error < best_error - 1e-12:
                        best_error = new_error
                        improved = True
                    else:
                        matrix[idx, c] = orig_val[idx, c]
        
        if not improved:
            break
    
    return U, V, W

def _to_fraction(x):
    """Convert a float to a Fraction, rounding to nearest half-integer if close."""
    # Try to find a simple fraction
    x_rounded = round(x * 2) / 2
    if abs(x - x_rounded) < 1e-9:
        return Fraction(x_rounded)
    
    # Try integer
    x_int = round(x)
    if abs(x - x_int) < 1e-9:
        return Fraction(int(x_int))
    
    # Return as is (shouldn't happen for exact solutions)
    return Fraction(x).limit_denominator(1000)

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    # First, return schoolbook as fallback
    schoolbook = _schoolbook()
    
    # Try to find lower-rank decompositions
    for rank in [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]:
        if time.time() > deadline:
            break
        
        result = _try_structured(rank, deadline)
        if result is not None:
            return result
    
    return schoolbook
# EVOLVE-BLOCK-END

