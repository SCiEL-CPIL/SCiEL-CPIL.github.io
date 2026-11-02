"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    # Schoolbook rank as fallback
    schoolbook_rank = n * m * p
    
    # Best known result (start with schoolbook)
    best_result = _schoolbook_decomposition(n, m, p)
    best_rank = schoolbook_rank
    
    # Try to find lower-rank decompositions
    # For 2x3 x 3x6, known bounds suggest rank could be lower
    # Try ranks from known lower bounds up to schoolbook
    # Lower bound for matmul (n,m,p) is max(n*m, m*p, p*n) = max(6, 18, 12) = 18
    min_possible = max(n * m, m * p, p * n)
    
    # Search for ranks between min_possible and schoolbook_rank
    # Use randomized approach with verification
    for target_rank in range(min_possible, schoolbook_rank + 1):
        if time.time() > deadline:
            break
        
        # Try multiple random attempts for each rank
        max_attempts = 50 if target_rank <= 25 else 20
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Generate random half-integer matrices
            scale = 2  # Use integers, will be half-integers when divided by 2
            U_rand = _random_half_integer_matrix(n * m, target_rank, scale)
            V_rand = _random_half_integer_matrix(m * p, target_rank, scale)
            
            # Check if we can find W that satisfies the decomposition
            # The tensor equation: sum_c U[i*m+j, c] * V[j*p+k, c] = delta[i,k] for matmul
            # Actually, the decomposition is: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            # This is a 3-tensor, so we need all three to satisfy it.
            
            # Alternative approach: use the fact that for matrix multiplication,
            # we can express it as a sum of rank-1 tensors
            # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
            
            # This is complex. Let's use a different strategy:
            # Try structured decompositions based on known algorithms
            
            # For now, try a simpler approach: check if random U,V can be extended
            pass
        
        # Also try structured approaches
        structured = _try_structured_decomposition(n, m, p, target_rank)
        if structured is not None:
            if target_rank < best_rank:
                best_rank = target_rank
                best_result = structured
                if time.time() > deadline:
                    break
    
    return best_result


def _schoolbook_decomposition(n, m, p):
    """Return the standard schoolbook decomposition."""
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


def _random_half_integer_matrix(rows, cols, scale=2):
    """Generate a random matrix with half-integer entries (integers/scale)."""
    # Use small integers to keep values manageable
    max_val = 3
    mat = []
    for _ in range(rows):
        row = []
        for _ in range(cols):
            val = random.randint(-max_val, max_val)
            row.append(val / scale)
        mat.append(row)
    return mat


def _try_structured_decomposition(n, m, p, target_rank):
    """
    Try to find a structured decomposition for the given rank.
    Uses known techniques for small matrix multiplication.
    Returns None if no decomposition found.
    """
    # For 2x3 x 3x6, we can try to exploit the structure
    # The multiplication tensor has entries T[i,j,k] = 1 if the (i,j) entry of A 
    # multiplies with (j,k) entry of B to contribute to (i,k) entry of C
    
    # A structured approach: try to find U, V, W such that
    # For all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta[i,k]
    # where delta[i,k] is 1 if i==k (for the identity-like structure of matmul)
    # Actually, the matmul tensor is: T[i,j,k] = 1 (it's always 1 for valid indices)
    # Wait, no. The matrix multiplication tensor is defined such that
    # (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T[i,j,k] = 1 for all valid i,j,k, and the decomposition
    # should satisfy: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
    
    # This is a 3-way tensor decomposition problem.
    
    # Try a simple structured approach for small ranks
    if target_rank >= n * m * p:
        # Schoolbook always works
        return _schoolbook_decomposition(n, m, p)
    
    # For now, return None to indicate no structured solution found yet
    # Future: implement more sophisticated search
    return None

# EVOLVE-BLOCK-END
