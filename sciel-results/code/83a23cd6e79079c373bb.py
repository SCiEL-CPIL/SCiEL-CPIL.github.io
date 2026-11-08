"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]
import time
import random
import numpy as np

def _verify_decomposition(U, V, W, n, m, p, rank):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    # For each (i, j, k), check that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    # And for all other index combinations, the sum should be 0
    # The tensor T[i,j,k] = 1 if this is a valid matrix mult entry
    # Actually, the tensor for matrix mult is: T[i, j, k] where i indexes A rows, j indexes B cols...
    # Let me think about the convention: W[k*n+i, r]
    # The decomposition is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta(i,j,k) for the MM tensor
    # The MM tensor T has T[i, j, k] = 1 if we're computing (A*B)[i, k] which involves A[i,j]*B[j,k]
    # So the tensor is: for each i in [0,n), j in [0,m), k in [0,p), the value is 1
    # Wait, that's not right. The matrix multiplication tensor T_{i,j,k} = 1 for all valid (i,j,k)
    # because (AB)_{i,k} = sum_j A_{i,j} B_{j,k}
    # So the tensor is all ones on the domain [0,n) x [0,m) x [0,p)
    
    # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1 for all i,j,k
    # And we need to make sure there are no spurious contributions... but actually the tensor
    # only has support on the valid indices, so we just need the sum to be 1 for each (i,j,k).
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _is_half_integer(x):
    """Check if x is an integer or half-integer (i.e., 2*x is an integer)."""
    return abs(2 * x - round(2 * x)) < 1e-9

def _round_to_half_int(x):
    """Round x to the nearest half-integer."""
    return round(2 * x) / 2.0

def _try_rank(rank, n, m, p, max_attempts=200, timeout=None):
    """Try to find a rank-rank exact decomposition using random search."""
    start_time = time.time()
    
    for attempt in range(max_attempts):
        if timeout is not None and time.time() - start_time > timeout:
            break
        
        # Random initialization with values in {-1, -0.5, 0, 0.5, 1}
        U = [[random.choice([-1, -0.5, 0, 0.5, 1]) for _ in range(rank)] for _ in range(n * m)]
        V = [[random.choice([-1, -0.5, 0, 0.5, 1]) for _ in range(rank)] for _ in range(m * p)]
        W = [[random.choice([-1, -0.5, 0, 0.5, 1]) for _ in range(rank)] for _ in range(p * n)]
        
        if _verify_decomposition(U, V, W, n, m, p, rank):
            return U, V, W
    
    return None

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Known lower bound: rank >= max(n*m, m*p, p*n) = max(6, 18, 12) = 18
    # Upper bound: n*m*p = 36
    # Try to find lower rank decompositions
    
    # Try ranks from 18 upward
    for rank in range(18, 37):
        if time.time() > deadline:
            break
        
        # Allocate time for this rank attempt
        remaining = deadline - time.time()
        if remaining < 2:
            break
        
        # More attempts for smaller ranks (harder to find)
        max_attempts = min(500, int(remaining * 50))
        timeout = remaining - 1
        
        result = _try_rank(rank, n, m, p, max_attempts=max_attempts, timeout=timeout)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
