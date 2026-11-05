import time
import sys
from fractions import Fraction

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U * V^T * W gives the correct matrix multiplication tensor."""
    rank = len(U[0])
    # U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
    # The tensor is T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We need T[i,j,k] = 1 if the entry A[i,j]*B[j,k] contributes to C[i,k]
    # i.e., T[i,j,k] = 1 for all valid i,j,k
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s != 1:
                    return False
    return True

def _add_strassen_block(U, V, W, r, i_off, j_off, k_off, n, m, p):
    """
    Add Strassen decomposition for a 2x2 * 2x2 -> 2x2 block.
    i_off: offset for A rows (i index)
    j_off: offset for A cols / B rows (j index)
    k_off: offset for B cols / C cols (k index)
    Returns next r.
    """
    # M1 = (A11+A22)(B11+B22)
    U[i_off * m + j_off][r] += 1
    U[(i_off+1) * m + (j_off+1)][r] += 1
    V[j_off * p + k_off][r] += 1
    V[(j_off+1) * p + (k_off+1)][r] += 1
    W[k_off * n + i_off][r] += 1
    W[(k_off+1) * n + (i_off+1)][r] += 1
    r += 1
    
    # M2 = (A21+A22)(B11+B12)
    U[(i_off+1) * m + j_off][r] += 1
    U[(i_off+1) * m + (j_off+1)][r] += 1
    V[j_off * p + k_off][r] += 1
    V[j_off * p + (k_off+1)][r] += 1
    W[(k_off+1) * n + i_off][r] += 1
    W[k_off * n + (i_off+1)][r] += 1
    r += 1
    
    # M3 = (A11-A22)(B21+B22)
    U[i_off * m + j_off][r] += 1
    U[(i_off+1) * m + (j_off+1)][r] -= 1
    V[(j_off+1) * p + k_off][r] += 1
    V[(j_off+1) * p + (k_off+1)][r] += 1
    W[(k_off+1) * n + i_off][r] += 1
    W[k_off * n + (i_off+1)][r] -= 1
    r += 1
    
    # M4 = (A12-A21)(B11+B22)
    U[i_off * m + (j_off+1)][r] += 1
    U[(i_off+1) * m + j_off][r] -= 1
    V[j_off * p + k_off][r] += 1
    V[(j_off+1) * p + (k_off+1)][r] += 1
    W[k_off * n + i_off][r] += 1
    W[(k_off+1) * n + i_off][r] -= 1
    W[(k_off+1) * n + (i_off+1)][r] += 1
    r += 1
    
    # M5 = (A11+A12)(B21-B22)
    U[i_off * m + j_off][r] += 1
    U[i_off * m + (j_off+1)][r] += 1
    V[(j_off+1) * p + k_off][r] += 1
    V[(j_off+1) * p + (k_off+1)][r] -= 1
    W[k_off * n + i_off][r] -= 1
    W[k_off * n + (i_off+1)][r] += 1
    r += 1
    
    # M6 = (A21+A11)(B12-B11)
    U[(i_off+1) * m + j_off][r] += 1
    U[i_off * m + j_off][r] += 1
    V[j_off * p + (k_off+1)][r] += 1
    V[j_off * p + k_off][r] -= 1
    W[(k_off+1) * n + i_off][r] -= 1
    W[(k_off+1) * n + (i_off+1)][r] += 1
    r += 1
    
    # M7 = (A12+A22)(B11-B21)
    U[i_off * m + (j_off+1)][r] += 1
    U[(i_off+1) * m + (j_off+1)][r] += 1
    V[j_off * p + k_off][r] += 1
    V[(j_off+1) * p + k_off][r] -= 1
    W[k_off * n + i_off][r] += 1
    W[k_off * n + (i_off+1)][r] -= 1
    r += 1
    
    return r

def construct():
    n, m, p = 2, 3, 5
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 6 x 28
    V = [[0] * rank for _ in range(m * p)]  # 15 x 28
    W = [[0] * rank for _ in range(p * n)]  # 10 x 28
    
    r = 0
    
    # Part 1: j=2, all k (schoolbook): A[:,2] * B[2,:]
    for i in range(n):
        for k in range(p):
            U[i * m + 2][r] = 1
            V[2 * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1
    # r = 10
    
    # Part 2: j={0,1}, k=4 (2x2 * 2x1 -> 2x1, schoolbook): 4 terms
    for i in range(n):
        for j in range(2):
            U[i * m + j][r] = 1
            V[j * p + 4][r] = 1
            W[4 * n + i][r] = 1
            r += 1
    # r = 14
    
    # Part 3: j={0,1}, k={0,1} (2x2 * 2x2 -> 2x2, Strassen): 7 terms
    r = _add_strassen_block(U, V, W, r, 0, 0, 0, n, m, p)
    # r = 21
    
    # Part 4: j={0,1}, k={2,3} (2x2 * 2x2 -> 2x2, Strassen): 7 terms
    r = _add_strassen_block(U, V, W, r, 0, 0, 2, n, m, p)
    # r = 28
    
    assert r == 28, f"Expected r=28, got r={r}"
    
    # Verify the decomposition
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback: return a known valid decomposition
        # Try rank 30 (pure schoolbook)
        rank = 30
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
                    r += 1
        return {"U": U, "V": V, "W": W}
    
    # Return the rank-28 decomposition
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
