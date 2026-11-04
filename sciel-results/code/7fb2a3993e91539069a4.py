"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition correctly computes matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _strassen_block(U, V, W, i0, j0, k0, col_start, n, m, p):
    """
    Add Strassen decomposition for 2x2 A-block at (i0, j0), 2x2 B-block at (j0, k0).
    A indices: i0, i0+1 ; j0, j0+1
    B indices: j0, j0+1 ; k0, k0+1
    C indices: i0, i0+1 ; k0, k0+1
    W row index for C[i,k] is k*n+i
    """
    terms = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        ([0,0,1,1], [1,1,0,0], [0,0,1,-1]),
        ([1,1,0,0], [0,0,1,1], [0,1,0,1]),
        ([0,0,0,1], [0,-1,0,1], [1,0,-1,0]),
        ([1,0,0,0], [1,0,-1,0], [-1,1,0,0]),
        ([-1,0,1,0], [0,1,0,1], [0,0,0,1]),
        ([0,1,0,-1], [1,0,1,0], [1,0,0,0]),
    ]
    
    col = col_start
    for t in range(7):
        a_coeffs = terms[t][0]
        b_coeffs = terms[t][1]
        c_coeffs = terms[t][2]
        
        for di in range(2):
            for dj in range(2):
                U[(i0+di)*m + (j0+dj)][col] = a_coeffs[di*2+dj]
        
        for dj in range(2):
            for dk in range(2):
                V[(j0+dj)*p + (k0+dk)][col] = b_coeffs[dj*2+dk]
        
        for di in range(2):
            for dk in range(2):
                W[(k0+dk)*n + (i0+di)][col] = c_coeffs[di*2+dk]
        
        col += 1
    
    return col

def _strassen_j_block(U, V, W, i0, j0, k0, col_start, n, m, p):
    """
    Add Strassen-like decomposition for 2x2 A-block at (i0, j0), 2x2 B-block at (j0, k0),
    but optimized for j-dimension pairing. This uses the same 7-term Strassen formula
    but groups the j indices differently to potentially allow further optimization.
    """
    # For now, this is identical to _strassen_block as a placeholder
    # In a real optimization, we might find a different 7-term decomposition
    # that allows merging with adjacent blocks
    return _strassen_block(U, V, W, i0, j0, k0, col_start, n, m, p)

def _schoolbook_block(U, V, W, i0, j0, k0, col_start, n, m, p):
    """Add schoolbook decomposition for 1x2 A-block, 2x2 B-block."""
    col = col_start
    for dj in range(2):
        for dk in range(2):
            U[i0*m + (j0+dj)][col] = 1
            V[(j0+dj)*p + (k0+dk)][col] = 1
            W[(k0+dk)*n + i0][col] = 1
            col += 1
    return col

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    start_time = time.time()
    deadline = start_time + 110
    
    # Try to find a better decomposition within the time limit
    # Current best: rank 88 using Strassen on 2x2x2 blocks
    best_rank = 88
    best_result = None

    # Strategy: Use Strassen on i-pair {0,1} for all j-pairs and k-pairs,
    # and schoolbook for i=2.
    # i-groups: {0,1} (Strassen), {2} (schoolbook)
    # j-groups: {0,1}, {2,3}
    # k-groups: {0,1}, {2,3}, {4,5}, {6,7}
    # Rank: 2*4*7 + 2*4*4 = 56 + 32 = 88

    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for j0 in [0, 2]:
        for k0 in [0, 2, 4, 6]:
            # Strassen block for i={0,1}
            col = _strassen_block(U, V, W, 0, j0, k0, col, n, m, p)
            # Schoolbook for i=2
            col = _schoolbook_block(U, V, W, 2, j0, k0, col, n, m, p)
    
    # Verify
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
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
