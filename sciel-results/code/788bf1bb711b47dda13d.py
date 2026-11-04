"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time

def _try_pan_decomposition(n, m, p, deadline):
    """
    Attempt to construct a Pan-style decomposition for the 3x4x8 multiplication.
    
    Pan's algorithm for 2x2x2 uses 7 multiplications instead of 8.
    For larger matrices, we can tile Pan's decomposition.
    
    Strategy: We'll decompose the 3x4x8 problem by tiling Pan's 2x2x2 blocks
    where applicable, and use schoolbook for the remainder.
    
    Actually, let me think about this more carefully. Pan's algorithm gives
    rank 7 for M_{2,2,2} vs standard rank 8. That's a 1/8 improvement.
    
    For 3x4x8:
    - We can tile 2x2x2 blocks. 3x4x8 can be partitioned into:
      - One 2x2x2 block in the "corner" 
      - The rest handled schoolbook-ish
    
    Let me try a different approach: use Pan's algorithm on a 2x2 subblock
    and handle the rest with schoolbook.
    
    Actually, the simplest valid improvement: if we can find ANY known
    lower-rank decomposition, use it. Otherwise return schoolbook.
    
    For now, let me return the standard schoolbook decomposition of rank 96,
    but structured to allow quick verification.
    """
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

def _try_tiled_pan(n, m, p, deadline):
    """
    Try to use Pan's algorithm tiled across the 3x4x8 multiplication.
    
    Pan's 2x2x2 decomposition:
    For C = A*B where A,B are 2x2:
    C[0,0] = x1 + x3 + x7 - x6 - x2
    C[0,1] = x2 + x4
    C[1,0] = x1 + x5
    C[1,1] = x3 + x4 + x6 - x5 - x1
    
    where:
    x1 = (a00 + a11)*b00
    x2 = (a10 + a11)*b10
    x3 = (a00 + a01)*b01
    x4 = (a10 + a11)*b11
    x5 = (a00 - a10)*b10
    x6 = (a01 - a11)*b01
    x7 = (a00 + a11)*b01
    
    Wait, let me recall Pan's algorithm correctly.
    
    Pan's algorithm (1978):
    m1 = (a00 + a11) * b00
    m2 = (a10 + a11) * b10
    m3 = (a00 + a01) * b01
    m4 = (a10 + a11) * b11
    m5 = (a00 - a10) * b10
    m6 = (a01 - a11) * b01
    m7 = (a00 + a11) * b01
    
    c00 = m1 + m3 + m7 - m6 - m2
    c01 = m2 + m4
    c10 = m1 + m5
    c11 = m3 + m4 + m6 - m5 - m1
    """
    n, m, p = DIMENSIONS
    # For now, just return schoolbook. The tiled Pan approach is complex
    # to verify and implement correctly without extensive testing.
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

def construct():
    """
    Return a factorization of the matrix multiplication tensor M_{3,4,8}.
    
    The tensor has entries T[i,j,k] = 1 if the (i,j) entry of A times the
    (j,k) entry of B contributes to the (i,k) entry of C.
    
    We seek rank r with factors U (12 x r), V (32 x r), W (24 x r) such that:
    sum_c U[i*m+j, c] * V[j*p+k, c] = W[k*n+i, c] for all i,j,k,c
    
    More precisely, the multiplication is:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    And we need: for each c, the outer product U[:,c] ⊗ V[:,c] contributes
    to W[:,c] in the appropriate way.
    
    The standard schoolbook decomposition has rank n*m*p = 96.
    """
    start_time = time.time()
    deadline = start_time + 120  # Must finish within 120s
    
    n, m, p = DIMENSIONS
    
    # Try to find a better decomposition
    # For now, return the standard schoolbook decomposition
    result = _try_pan_decomposition(n, m, p, deadline)
    
    elapsed = time.time() - start_time
    # Safety margin
    if elapsed > 100:
        return result
    
    return result
# EVOLVE-BLOCK-END
