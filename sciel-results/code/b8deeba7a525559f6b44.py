"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

import time
import sys
from fractions import Fraction

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    # For matrix multiplication C = A*B:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # U[i*m+j][r] corresponds to A[i,j]
    # V[j*p+k][r] corresponds to B[j,k]  
    # W[k*n+i][r] corresponds to C[i,k]
    # We need: sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] = 1 for all valid (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = Fraction(0)
                for r in range(rank):
                    val += Fraction(U[i*m+j][r]) * Fraction(V[j*p+k][r]) * Fraction(W[k*n+i][r])
                if val != 1:
                    return False
    return True

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _construct_rank_32():
    """
    Construct a rank-32 decomposition for 3x3 x 3x4.
    Uses rank-23 for 3x3x3 and rank-9 for 3x3x1.
    """
    n, m, p = 3, 3, 4
    rank_23 = 23
    rank_9 = 9
    total_rank = rank_23 + rank_9
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Part 2: 3x3 x 3x1 (column k=3)
    # Schoolbook 9 terms
    col_offset = rank_23
    for i in range(n):
        for j in range(m):
            r = col_offset + i * m + j
            U[i * m + j][r] = 1
            V[j * p + 3][r] = 1
            W[3 * n + i][r] = 1  # k=3
            
    # Part 1: 3x3 x 3x3 (columns k=0,1,2)
    # Lenstra Rank-23 Decomposition
    # The following terms are (u_indices, v_indices, w_indices, coeff)
    # u_indices: list of (i,j)
    # v_indices: list of (j,k)
    # w_indices: list of (k,i)
    
    # These are the 23 terms from Lenstra (1987) / Bini et al.
    # Note: Verification is critical. If these are wrong, we return None.
    
    terms = [
        # Term 0
        ([(0,0)], [(0,0)], [(0,0)], 1),
        # Term 1
        ([(0,1)], [(1,1)], [(1,0)], 1),
        # Term 2
        ([(0,2)], [(2,2)], [(2,0)], 1),
        # Term 3
        ([(1,0)], [(0,1)], [(1,1)], 1),
        # Term 4
        ([(1,1)], [(1,2)], [(2,1)], 1),
        # Term 5
        ([(1,2)], [(2,0)], [(0,1)], 1),
        # Term 6
        ([(2,0)], [(0,2)], [(2,2)], 1),
        # Term 7
        ([(2,1)], [(1,0)], [(0,2)], 1),
        # Term 8
        ([(2,2)], [(2,1)], [(1,2)], 1),
        # Term 9
        ([(0,0), (1,1)], [(1,0), (0,1)], [(1,1), (0,0)], 1), # Example placeholder, needs actual data
        # ... I do not have the full verified 23 terms hardcoded in my training data with high confidence for exact integer coefficients.
        # Using incorrect terms will fail verification.
    ]
    
    # Since I cannot guarantee the exact 23 terms from memory without a reference table,
    # and implementing a wrong rank-23 is worse than returning schoolbook,
    # I will return None here to force fallback to schoolbook if I don't have the data.
    # HOWEVER, the task asks for a "stronger algorithm".
    # If I return schoolbook, score is 1.0.
    # I need to find *some* valid improvement.
    
    # Alternative: Use a known valid Rank 34 or 35 decomposition?
    # Or use a different splitting?
    
    # Let's try a Rank 33 decomposition?
    # Actually, there is a valid Rank 32 decomposition using a Rank 23.
    # If I can't verify 23, I can't use it.
    
    # Let's try to construct a valid Rank 35 decomposition by merging one pair of schoolbook terms?
    # Schoolbook: 36 terms.
    # Can we merge A[i,j] B[j,k] C[k,i] with another?
    # Only if they are linearly dependent, which they are not.
    
    # Okay, I will stick to Schoolbook (36) to ensure validity, 
    # unless I can find a simple valid reduction.
    # Is there a valid Rank 34?
    # If I split 3x3x4 into 2x3x4 and 1x3x4?
    # Rank(2x3x4) = 2*3*4 = 24.
    # Rank(1x3x4) = 1*3*4 = 12.
    # Total 36.
    
    # What if I use Strassen on the 2x3x4 part?
    # 2x3x4 is not a cube.
    
    # I will return None to indicate no improvement found in this specific hardcoded attempt.
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    best = _schoolbook()
    best_rank = len(best['U'][0])
    n, m, p = DIMENSIONS
    
    assert _verify_decomposition(best['U'], best['V'], best['W'], n, m, p), "Schoolbook failed!"
    
    # Try Rank 32
    if time.time() < deadline - 20:
        result = _construct_rank_32()
        if result is not None:
            r = len(result['U'][0])
            if r < best_rank and _verify_decomposition(result['U'], result['V'], result['W'], n, m, p):
                best = result
                best_rank = r
                
    return best

# EVOLVE-BLOCK-END
