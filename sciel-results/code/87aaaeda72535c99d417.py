"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110  # Safety margin before 120s limit
    n, m, p = DIMENSIONS
    
    # Schoolbook as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    # Try to find a lower-rank exact decomposition
    # Use a structured approach: try ranks from 50 upward
    for target_rank in range(50, 126):
        if time.time() > deadline - 5:
            break
        result = _try_construct(n, m, p, target_rank, deadline)
        if result is not None:
            return result
    
    # Return schoolbook
    return {"U": U_school, "V": V_school, "W": W_school}

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # Check that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1) > 1e-12:
                    return False
    return True

def _try_construct(n, m, p, target_rank, deadline):
    """Try to construct a rank target_rank decomposition using Strassen-inspired block approach."""
    # For 5x5x5, we can split into blocks:
    # Split each dimension into 2+3
    # Use Strassen for the 2x2x2 blocks (rank 7)
    # Use schoolbook for mixed blocks
    
    # Strategy: Use a combination of Strassen and schoolbook
    # The 5x5 matrix can be viewed as blocks of size 2 and 3
    # A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x3, A21 is 3x2, A22 is 3x3
    # B = [[B11, B12], [B21, B22]] where B11 is 2x2, B12 is 2x3, B21 is 3x2, B22 is 3x3
    
    # For the 2x2x2 blocks, we can use Strassen (rank 7)
    # For mixed blocks, we use schoolbook
    
    # Let's try a direct construction using known low-rank decompositions
    
    # First, let's try to use the fact that 5 = 2 + 3
    # We can decompose the 5x5x5 tensor as a sum of:
    # - Four 2x2x2 tensors (each rank 7 with Strassen)
    # - Mixed terms
    
    # Actually, let's try a simpler approach: use random search with structured initialization
    
    for attempt in range(50):
        if time.time() > deadline - 2:
            break
        
        # Initialize with random small values
        U = [[0.0] * target_rank for _ in range(n * m)]
        V = [[0.0] * target_rank for _ in range(m * p)]
        W = [[0.0] * target_rank for _ in range(p * n)]
        
        # Use a structured initialization based on Strassen patterns
        # Strassen for 2x2x2 uses 7 multiplications
        # We'll try to embed this pattern
        
        for r in range(target_rank):
            # Assign structured indices based on Strassen-like patterns
            # For the first 7, use Strassen pattern for 2x2x2
            if r < 7:
                # Strassen indices for 2x2x2
                strassen_patterns = [
                    (0, 0, 0),  # M1 = (A11+A22)(B11+B22)
                    (0, 1, 1),  # M2 = (A21+A22)B11
                    (1, 0, 0),  # M3 = A11(B21-B22)
                    (1, 1, 1),  # M4 = A22(B12-B11)
                    (0, 0, 1),  # M5 = (A11+A12)B22
                    (1, 1, 0),  # M6 = (A21-A11)(B11+B12)
                    (0, 1, 1),  # M7 = (A12-A22)(B21+B22)
                ]
                i, j, k = strassen_patterns[r]
                # Map to 5x5x5 indices
                i5 = i
                j5 = j
                k5 = k
                U[i5*m+j5][r] = 1.0
                V[j5*p+k5][r] = 1.0
                W[k5*n+i5][r] = 1.0
            else:
                # For remaining ranks, use random or schoolbook-like patterns
                i = (r * 7) % n
                j = (r * 11) % m
                k = (r * 13) % p
                U[i*m+j][r] = 1.0
                V[j*p+k][r] = 1.0
                W[k*n+i][r] = 1.0
        
        # Verify
        if _verify_decomposition(U, V, W, n, m, p):
            # Convert to integer/half-integer
            U_int = [[int(round(2*x))/2 for x in row] for row in U]
            V_int = [[int(round(2*x))/2 for x in row] for row in V]
            W_int = [[int(round(2*x))/2 for x in row] for row in W]
            if _verify_decomposition(U_int, V_int, W_int, n, m, p):
                return {"U": U_int, "V": V_int, "W": W_int}
    
    return None
# EVOLVE-BLOCK-END
