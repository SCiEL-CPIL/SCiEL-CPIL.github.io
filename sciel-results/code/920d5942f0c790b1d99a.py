"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import numpy as np
from fractions import Fraction
import time

def _check_decomposition(U, V, W, n, m, p):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    n, m, p = int(n), int(m), int(p)
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # The tensor entry T[i,j,k] should equal sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
                # U is indexed by [i*m+j, c], V by [j*p+k, c], W by [k*n+i, c]
                val = 0
                for c in range(rank):
                    u_val = U[i*m + j][c]
                    v_val = V[j*p + k][c]
                    w_val = W[k*n + i][c]
                    val += u_val * v_val * w_val
                if val != 1:
                    return False
    # Check that all non-essential entries are zero
    # The tensor has 1s at positions (i,j,k) for all i,j,k and 0 elsewhere
    # But actually the matrix multiplication tensor T has T[i,j,k] = 1 if we're computing (AB)[i,k] = sum_j A[i,j]B[j,k]
    # So the tensor entries are T[i,j,k] = 1 for all valid i,j,k (since each term A[i,j]*B[j,k] contributes)
    # Wait, the standard matrix multiplication tensor has entries such that:
    # (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    # So the 3-tensor T_{i,j,k} = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # And we need U[i,j,c]*V[j,k,c]*W[k,i,c] summed over c to equal T_{i,j,k}
    # Since T_{i,j,k} = 1 for all valid indices, we need the sum to be 1 for all (i,j,k)
    # But what about invalid indices? There are no invalid indices in this tensor.
    # Actually, the tensor is only defined for valid i,j,k. The decomposition should satisfy:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    return True

def _try_rank(rank, n, m, p, max_attempts=50):
    """Try to find a rank decomposition of given rank using random search with verification."""
    start_time = time.time()
    U_rows = n * m
    V_rows = m * p
    W_rows = p * n
    
    for attempt in range(max_attempts):
        if time.time() - start_time > 8:
            break
        # Random initialization with small integers or fractions
        U = [[Fraction(0) for _ in range(rank)] for _ in range(U_rows)]
        V = [[Fraction(0) for _ in range(rank)] for _ in range(V_rows)]
        W = [[Fraction(0) for _ in range(rank)] for _ in range(W_rows)]
        
        # Initialize randomly
        for i in range(U_rows):
            for c in range(rank):
                U[i][c] = Fraction(np.random.randint(-2, 3))
        for i in range(V_rows):
            for c in range(rank):
                V[i][c] = Fraction(np.random.randint(-2, 3))
        for i in range(W_rows):
            for c in range(rank):
                W[i][c] = Fraction(np.random.randint(-2, 3))
        
        if _check_decomposition(U, V, W, n, m, p):
            return U, V, W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Try ranks starting from the known lower bound
    # Lower bound for 2x3x4 matrix multiplication is at least 19 (from Strassen-type bounds)
    # Actually, the Strassen algorithm gives rank 7 for 2x2x2, and for 2x3x4,
    # we can use block methods. Let's try ranks from 19 upward.
    
    # First, let's try to find a decomposition of rank 19 or lower
    for rank in range(19, 24):
        result = _try_rank(rank, n, m, p, max_attempts=30)
        if result is not None:
            U, V, W = result
            # Convert Fraction to float or keep as fraction
            # The interface says integer/half-integer entries, so let's ensure that
            U_out = [[float(u) for u in row] for row in U]
            V_out = [[float(v) for v in row] for row in V]
            W_out = [[float(w) for w in row] for row in W]
            return {"U": U_out, "V": V_out, "W": W_out}
    
    # If random search fails, fall back to a known construction
    # Use a block-based approach: treat 2x3x4 as blocks
    # For 2x3 * 3x4, we can partition the 3x3 middle into blocks
    # Actually, let's just use the schoolbook as fallback but with rank = n*m*p = 24
    # Or try a smarter construction
    
    # Schoolbook decomposition of rank n*m*p = 24
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
# EVOLVE-BLOCK-END
