"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    return False
    return True

def _verify_fraction(U, V, W, n, m, p):
    """Verify using exact fractions."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                if s != 1:
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
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def _try_rank_r(n, m, p, r, time_limit=10):
    """Try to find a rank-r decomposition using randomized search."""
    start_time = time.time()
    um = n * m
    vp = m * p
    wn = p * n
    
    # Number of equations: n*m*p
    num_eq = n * m * p
    
    # We need: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    # Strategy: initialize with schoolbook and try to merge columns
    # Or: random initialization with rational entries
    
    # Try: start with a sparse random structure
    for attempt in range(200):
        if time.time() - start_time > time_limit:
            break
        
        # Random initialization with small integers
        U = [[random.choice([0, 1, -1]) for _ in range(r)] for _ in range(um)]
        V = [[random.choice([0, 1, -1]) for _ in range(r)] for _ in range(vp)]
        W = [[random.choice([0, 1, -1]) for _ in range(r)] for _ in range(wn)]
        
        # Check if it works
        if _verify(U, V, W, n, m, p):
            return U, V, W
    
    # Try structured approach: use blocks
    # Since n=2, we can think of this as two rows of the result
    # C[0,k] = sum_j A[0,j]*B[j,k]
    # C[1,k] = sum_j A[1,j]*B[j,k]
    
    # Try to find a decomposition where some columns handle multiple (i,j,k) tuples
    # This is equivalent to finding a set of r triples of vectors (u_c, v_c, w_c)
    # such that for each (i,j,k), the number of c where u_c[i*m+j]*v_c[j*p+k]*w_c[k*n+i] != 0
    # and the sum equals 1.
    
    # Alternative: use the fact that we can have cancellations
    # Try with entries from {0, 1, 2, -1, -2}
    for attempt in range(100):
        if time.time() - start_time > time_limit:
            break
        
        U = [[random.choice([0, 1, 2, -1, -2]) for _ in range(r)] for _ in range(um)]
        V = [[random.choice([0, 1, 2, -1, -2]) for _ in range(r)] for _ in range(vp)]
        W = [[random.choice([0, 1, 2, -1, -2]) for _ in range(r)] for _ in range(wn)]
        
        if _verify(U, V, W, n, m, p):
            return U, V, W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try lower ranks
    # Lower bound: at least max(n*m, m*p, p*n) = max(10, 30, 12) = 30
    # But actually the tensor rank lower bound is higher.
    # For [2x5][5x6], the rank is at least... let's try from 30 upward
    
    for r in range(30, best_rank):
        if time.time() > deadline - 20:  # Keep 20s safety margin
            break
        
        result = _try_rank_r(n, m, p, r, time_limit=5)
        if result is not None:
            U, V, W = result
            best_rank = r
            break
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
