"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time
import sys
from functools import lru_cache

_start_time = time.time()
TIME_LIMIT = 140.0  # Safety margin

def _time_check():
    if time.time() - _start_time > TIME_LIMIT:
        raise TimeoutError("Time limit exceeded")

@lru_cache(maxsize=None)
def _get_rank(n, m, p):
    _time_check()
    
    # Base case: Strassen for 2x2x2
    if n == 2 and m == 2 and p == 2:
        return 7
    
    # Base case: 1 in any dimension means schoolbook (rank = n*m*p)
    if n == 1 or m == 1 or p == 1:
        return n * m * p
        
    best = n * m * p
    
    # Try splitting n
    for n1 in range(1, n):
        n2 = n - n1
        r = _get_rank(n1, m, p) + _get_rank(n2, m, p)
        if r < best:
            best = r
            
    # Try splitting m
    for m1 in range(1, m):
        m2 = m - m1
        r = _get_rank(n, m1, p) + _get_rank(n, m2, p)
        if r < best:
            best = r
            
    # Try splitting p
    for p1 in range(1, p):
        p2 = p - p1
        r = _get_rank(n, m, p1) + _get_rank(n, m, p2)
        if r < best:
            best = r
            
    return best

def _strassen_matrices():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition.
    U, V, W are lists of lists (4x7).
    """
    # U, V, W columns correspond to the 7 products M1..M7
    # Rows correspond to matrix entries: 00, 01, 10, 11
    
    U = [
        [1, 0, 1, 0, 1, -1, 0],  # A00
        [0, 0, 0, 0, 1, 0, 1],   # A01
        [0, 1, 0, 0, 0, 1, 0],   # A10
        [1, 1, 0, 1, 0, 0, -1]   # A11
    ]
    
    V = [
        [1, 1, 0, -1, 0, 1, 0],  # B00
        [0, 0, 0, 1, 0, 1, 0],   # B01
        [0, 0, 1, 0, 0, 0, 1],   # B10
        [1, 0, -1, 0, 1, 0, 1]   # B11
    ]
    
    W = [
        [1, 0, 0, 1, -1, 0, 1],  # C00
        [0, 0, 1, 0, 1, 0, 0],   # C01
        [0, 1, 0, 1, 0, 0, 0],   # C10
        [1, -1, 1, 0, 0, 1, 0]   # C11
    ]
    
    return U, V, W

def _schoolbook_matrices(n, m, p):
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

def _construct(n, m, p):
    _time_check()
    
    # Base case: Strassen
    if n == 2 and m == 2 and p == 2:
        U, V, W = _strassen_matrices()
        return U, V, W, 7
        
    # Base case: 1 in any dimension
    if n == 1 or m == 1 or p == 1:
        return _schoolbook_matrices(n, m, p)
        
    best_rank = n * m * p
    best_split = None
    best_dim = None
    
    # Try splitting n
    for n1 in range(1, n):
        n2 = n - n1
        r1, r2 = _get_rank(n1, m, p), _get_rank(n2, m, p)
        r = r1 + r2
        if r < best_rank:
            best_rank = r
            best_split = (n1, n2)
            best_dim = 'n'
            
    # Try splitting m
    for m1 in range(1, m):
        m2 = m - m1
        r1, r2 = _get_rank(n, m1, p), _get_rank(n, m2, p)
        r = r1 + r2
        if r < best_rank:
            best_rank = r
            best_split = (m1, m2)
            best_dim = 'm'
            
    # Try splitting p
    for p1 in range(1, p):
        p2 = p - p1
        r1, r2 = _get_rank(n, m, p1), _get_rank(n, m, p2)
        r = r1 + r2
        if r < best_rank:
            best_rank = r
            best_split = (p1, p2)
            best_dim = 'p'
            
    if best_dim is None:
        # Fallback to schoolbook if no improvement found (shouldn't happen if rank > schoolbook)
        return _schoolbook_matrices(n, m, p)
        
    d1, d2 = best_split
    
    if best_dim == 'n':
        n1, n2 = d1, d2
        U1, V1, W1, r1 = _construct(n1, m, p)
        U2, V2, W2, r2 = _construct(n2, m, p)
        
        # Combine U: stack rows
        U = U1 + U2
        # Combine V: stack columns
        V = [[v1[i] for v1 in V1] + [v2[i] for v2 in V2] for i in range(m * p)]
        # Combine W: stack rows
        W = W1 + W2
        
        return U, V, W, r1 + r2
        
    elif best_dim == 'm':
        m1, m2 = d1, d2
        U1, V1, W1, r1 = _construct(n, m1, p)
        U2, V2, W2, r2 = _construct(n, m2, p)
        
        # Combine U: stack columns
        U = [[u1[i] for u1 in U1] + [u2[i] for u2 in U2] for i in range(n * m)]
        # Combine V: stack rows
        V = V1 + V2
        # Combine W: stack columns
        W = [[w1[i] for w1 in W1] + [w2[i] for w2 in W2] for i in range(p * n)]
        
        return U, V, W, r1 + r2
        
    elif best_dim == 'p':
        p1, p2 = d1, d2
        U1, V1, W1, r1 = _construct(n, m, p1)
        U2, V2, W2, r2 = _construct(n, m, p2)
        
        # Combine U: stack columns
        U = [[u1[i] for u1 in U1] + [u2[i] for u2 in U2] for i in range(n * m)]
        # Combine V: stack columns
        V = [[v1[i] for v1 in V1] + [v2[i] for v2 in V2] for i in range(m * p)]
        # Combine W: stack rows
        W = W1 + W2
        
        return U, V, W, r1 + r2

def construct():
    n, m, p = DIMENSIONS
    U, V, W, rank = _construct(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
