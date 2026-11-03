<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the schoolbook tensor decomposition for matrix multiplication.
    Dimensions: n=3, m=3, p=8.
    Rank: n*m*p = 72.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize factor matrices
    # U: shape (n*m, rank) -> (9, 72)
    # V: shape (m*p, rank) -> (24, 72)
    # W: shape (p*n, rank) -> (24, 72)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                U[i * m + j][column] = 1
                # V index: j*p + k
                V[j * p + k][column] = 1
                # W index: k*n + i (Trace convention W[k*n+i, r])
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
=======
"""Improved program: Strassen-based rank-68 decomposition for 3x3 * 3x8."""
import time
import numpy as np

DIMENSIONS = [3, 3, 8]
_DEADLINE_SECONDS = 140.0
_start_time = time.time()

# EVOLVE-BLOCK-START
def _time_left():
    return _DEADLINE_SECONDS - (time.time() - _start_time)

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    # Convert to numpy for speed
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    rank = U_arr.shape[1]
    # Reshape: U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Check against identity: T should be 1 if (i,j,k) corresponds to valid product
    # For matrix mult (n x m) * (m x p) -> (n x p):
    # T[i,j,k] = 1 if j is the contraction index... 
    # Actually the tensor is: T[i, j, k] where i in [0,n), j in [0,m), k in [0,p)
    # and T[i,j,k] = 1 if this is a valid term in the product
    # The product (A * B)[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor has T[i,j,k] = 1 for all i,j,k (it's the full multiplication tensor)
    
    # Check: for each (i, j, k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1
    # But wait - the tensor for matrix multiplication is:
    # T[i, j, k] = 1 for all valid (i,j,k) triples
    # where i indexes rows of A (and output), j indexes columns of A / rows of B, k indexes columns of B (and output)
    
    # Let's verify by checking all entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U_arr[i * m + j, r] * V_arr[j * p + k, r] * W_arr[k * n + i, r]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook decomposition of rank n*m*p."""
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

def _strassen_2x2_2(n, m, p, i0, j0, k0):
    """
    Generate Strassen decomposition for a 2x2 * 2x2 sub-block.
    The sub-block uses rows i0, i0+1 of A, columns j0, j0+1 of A,
    rows j0, j0+1 of B, columns k0, k0+1 of B.
    
    Strassen's 7 multiplications for 2x2*2x2:
    M1 = (A00+A11)*(B00+B11)
    M2 = (A10+A11)*B00
    M3 = A00*(B01-B11)
    M4 = A11*(B10-B00)
    M5 = (A00+A01)*B11
    M6 = (A10-A00)*(B00+B01)
    M7 = (A01-A11)*(B10+B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    # Each Strassen term is a rank-1 term in the tensor
    # For term r, we need U[i*m+j], V[j*p+k], W[k*n+i]
    
    # The 7 Strassen formulas expressed as linear combinations:
    # Each formula defines coefficients for (i,j,k) triples
    
    # Let's define the 7 terms:
    # Term 0: M1 = (A00+A11)*(B00+B11)
    #   A part: (i0,j0) + (i0+1,j0+1)
    #   B part: (j0,k0) + (j0+1,k0+1)
    #   C part: contributes to C00, C01, C10, C11
    #   C00: +1, C01: 0, C10: 0, C11: +1
    #   Wait, M1 appears in C00 with +1 and C11 with +1
    
    # Actually, let me think about this more carefully.
    # The tensor decomposition means:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
    # where T[i,j,k] = 1 for all i,j,k (the full multiplication tensor)
    
    # For the 2x2 sub-block, we need to represent the 8 terms (2*2*2) 
    # using 7 rank-1 terms.
    
    # The 8 terms are: (i0,j0,k0), (i0,j0,k0+1), (i0,j0+1,k0), (i0,j0+1,k0+1),
    #                   (i0+1,j0,k0), (i0+1,j0,k0+1), (i0+1,j0+1,k0), (i0+1,j0+1,k0+1)
    
    # Strassen replaces these 8 with 7 combinations:
    
    # Let's define each of the 7 terms by their U, V, W support:
    
    terms = []
    
    # M1: (A[i0,j0] + A[i0+1,j0+1]) * (B[j0,k0] + B[j0+1,k0+1])
    # U support: (i0,j0):+1, (i0+1,j0+1):+1
    # V support: (j0,k0):+1, (j0+1,k0+1):+1
    # W support: (k0,i0):+1, (k0+1,i0+1):+1  [from C00 and C11]
    # But wait, M1 contributes to C00 (which is output[i0,k0]) and C11 (output[i0+1,k0+1])
    # So W[k0*n+i0] += 1 and W[(k0+1)*n+(i0+1)] += 1
    terms.append({
        'U': [(i0, j0, 1), (i0+1, j0+1, 1)],
        'V': [(j0, k0, 1), (j0+1, k0+1, 1)],
        'W': [(k0, i0, 1), (k0+1, i0+1, 1)]
    })
    
    # M2: (A[i0+1,j0] + A[i0+1,j0+1]) * B[j0,k0]
    # U support: (i0+1,j0):+1, (i0+1,j0+1):+1
    # V support: (j0,k0):+1
    # W support: (k0,i0+1):+1  [from C10]
    terms.append({
        'U': [(i0+1, j0, 1), (i0+1, j0+1, 1)],
        'V': [(j0, k0, 1)],
        'W': [(k0, i0+1, 1)]
    })
    
    # M3: A[i0,j0] * (B[j0,k0+1] - B[j0+1,k0+1])
    # U support: (i0,j0):+1
    # V support: (j0,k0+1):+1, (j0+1,k0+1):-1
    # W support: (k0+1,i0):+1  [from C01]
    terms.append({
        'U': [(i0, j0, 1)],
        'V': [(j0, k0+1, 1), (j0+1, k0+1, -1)],
        'W': [(k0+1, i0, 1)]
    })
    
    # M4: A[i0+1,j0+1] * (B[j0+1,k0] - B[j0,k0])
    # U support: (i0+1,j0+1):+1
    # V support: (j0+1,k0):+1, (j0,k0):-1
    # W support: (k0,i0):+1, (k0,i0+1):+1  [from C00 and C10]
    terms.append({
        'U': [(i0+1, j0+1, 1)],
        'V': [(j0+1, k0, 1), (j0, k0, -1)],
        'W': [(k0, i0, 1), (k0, i0+1, 1)]
    })
    
    # M5: (A[i0,j0] + A[i0,j0+1]) * B[j0+1,k0+1]
    # U support: (i0,j0):+1, (i0,j0+1):+1
    # V support: (j0+1,k0+1):+1
    # W support: (k0+1,i0):-1, (k0+1,i0+1):-1  [from C00(-M5) and C11(-M5)]
    # Wait, let me re-check: C00 = M1 + M4 - M5 + M7, C11 = M1 - M2 + M3 + M6
    # M5 is in C01: C01 = M3 + M5, and C00: C00 = M1 + M4 - M5 + M7
    # So M5 contributes +1 to C01 (output[i0,k0+1]) and -1 to C00 (output[i0,k0])
    terms.append({
        'U': [(i0, j0, 1), (i0, j0+1, 1)],
        'V': [(j0+1, k0+1, 1)],
        'W': [(k0, i0, -1), (k0+1, i0, 1)]
    })
    
    # M6: (A[i0+1,j0] - A[i0,j0]) * (B[j0,k0] + B[j0,k0+1])
    # U support: (i0+1,j0):+1, (i0,j0):-1
    # V support: (j0,k0):+1, (j0,k0+1):+1
    # W support: (k0+1,i0+1):+1  [from C11]
    terms.append({
        'U': [(i0+1, j0, 1), (i0, j0, -1)],
        'V': [(j0, k0, 1), (j0, k0+1, 1)],
        'W': [(k0+1, i0+1, 1)]
    })
    
    # M7: (A[i0,j0+1] - A[i0+1,j0+1]) * (B[j0+1,k0] + B[j0+1,k0+1])
    # U support: (i0,j0+1):+1, (i0+1,j0+1):-1
    # V support: (j0+1,k0):+1, (j0+1,k0+1):+1
    # W support: (k0,i0):+1, (k0+1,i0):-1  [from C00(+M7) and C01... wait]
    # C00 = M1 + M4 - M5 + M7, C01 = M3 + M5
    # M7 is only in C00: output[i0,k0]
    # Wait, that's not right either. Let me look at Strassen more carefully.
    # 
    # Standard Strassen:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5  
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # So M7 only appears in C00.
    terms.append({
        'U': [(i0, j0+1, 1), (i0+1, j0+1, -1)],
        'V': [(j0+1, k0, 1), (j0+1, k0+1, 1)],
        'W': [(k0, i0, 1)]
    })
    
    return terms

def _build_strassen_decomposition(n, m, p, block_positions):
    """
    Build a decomposition using Strassen for specified 2x2x2 blocks
    and schoolbook for the remaining entries.
    
    block_positions: list of (i0, j0, k0) for each 2x2x2 Strassen block
    """
    # Determine which (i,j,k) triples are covered by Strassen blocks
    covered = set()
    for (i0, j0, k0) in block_positions:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered.add((i0+di, j0+dj, k0+dk))
    
    # Start with schoolbook for uncovered entries
    rank_schoolbook = len([1 for i in range(n) for j in range(m) for k in range(p) 
                          if (i,j,k) not in covered])
    
    # Strassen blocks: 7 terms each
    rank_strassen = 7 * len(block_positions)
    
    total_rank = rank_schoolbook + rank_strassen
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    # First add Strassen terms
    for (i0, j0, k0) in block_positions:
        terms = _strassen_2x2_2(n, m, p, i0, j0, k0)
        for term in terms:
            for (i, j, val) in term['U']:
                U[i * m + j][col] = val
            for (j, k, val) in term['V']:
                V[j * p + k][col] = val
            for (k, i, val) in term['W']:
                W[k * n + i][col] = val
            col += 1
    
    # Then add schoolbook for uncovered entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
    
    return {"U": U, "V": V, "W": W}

def construct():
    """
    Returns a Strassen-based tensor decomposition for 3x3 * 3x8 matrix multiplication.
    Uses 4 disjoint 2x2x2 Strassen blocks to achieve rank 68 (vs schoolbook 72).
    """
    n, m, p = DIMENSIONS
    
    # Try different placements of 4 disjoint 2x2x2 blocks
    # For 3x3x8, we can place 2x2x2 blocks at various positions
    # i0 in {0,1}, j0 in {0,1}, k0 in {0,1,...,6}
    
    # We need 4 blocks that are disjoint (no overlapping (i,j,k) triples)
    # Each block covers 8 triples, so 4 blocks cover 32 triples
    # Total triples: 3*3*8 = 72, so 40 remain for schoolbook
    # Rank = 4*7 + 40 = 28 + 40 = 68
    
    # Let's try: blocks at (0,0,0), (0,0,2), (0,0,4), (0,0,6)
    # These use i in {0,1}, j in {0,1}, k in {0,1}, {2,3}, {4,5}, {6,7}
    # All disjoint!
    
    block_positions = [(0, 0, 0), (0, 0, 2), (0, 0, 4), (0, 0, 6)]
    
    result = _build_strassen_decomposition(n, m, p, block_positions)
    
    # Verify
    if _verify(result["U"], result["V"], result["W"], n, m, p):
        return result
    
    # Fallback: try other placements
    # Blocks at (0,0,0), (0,0,2), (1,0,0), (1,0,2)
    block_positions2 = [(0, 0, 0), (0, 0, 2), (1, 0, 0), (1, 0, 2)]
    result2 = _build_strassen_decomposition(n, m, p, block_positions2)
    if _verify(result2["U"], result2["V"], result2["W"], n, m, p):
        return result2
    
    # Fallback: schoolbook
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
>>>>>>> REPLACE

Now let me verify the Strassen decomposition is correct by tracing through the math. Let me reconsider the W coefficients more carefully.

For M1: contributes to C00 (+1) and C11 (+1). C00 is output[i0,k0], C11 is output[i0+1,k0+1].
- W[k0*n+i0] += 1 (for C00)
- W[(k0+1)*n+(i0+1)] += 1 (for C11)

For M2: contributes to C10 (+1). C10 is output[i0+1,k0].
- W[k0*n+(i0+1)] += 1

For M3: contributes to C01 (+1). C01 is output[i0,k0+1].
- W[(k0+1)*n+i0] += 1

For M4: contributes to C00 (+1) and C10 (+1).
- W[k0*n+i0] += 1 (for C00)
- W[k0*n+(i0+1)] += 1 (for C10)

For M5: contributes to C01 (+1) and C00 (-1).
- W[(k0+1)*n+i0] += 1 (for C01)
- W[k0*n+i0] += -1 (for C00)

For M6: contributes to C11 (+1). C11 is output[i0+1,k0+1].
- W[(k0+1)*n+(i0+1)] += 1

For M7: contributes to C00 (+1). C00 is output[i0,k0].
- W[k0*n+i0] += 1

Let me fix the W coefficients:

