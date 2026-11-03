"""Matrix multiplication 3x5x6: rank-minimal exact tensor decomposition."""
import numpy as np
from itertools import product
import time

DIMENSIONS = [3, 5, 6]

# Precomputed Strassen 2x2x2 decomposition (rank 7)
# For multiplying A(2x2) * B(2x2) = C(2x2):
# M1 = (A00+A11)(B00+B11)
# M2 = (A10+A11)B00
# M3 = A00(B01-B11)
# M4 = A11(B10-B00)
# M5 = (A00+A10)B10
# M6 = (A01-A11)(B10+B11)
# M7 = (A10-A00)(B00+B01)
# C00 = M1+M4-M5+M7
# C01 = M3+M5
# C10 = M2+M4
# C11 = M1-M2+M3+M6

def _strassen_2x2x2():
    """
    Return (U, V, W) for 2x2x2 Strassen with convention:
    U[i*m+j, c], V[j*p+k, c], W[k*n+i, c] where n=m=p=2.
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1  # A00, A11
    V[0, 0] = 1; V[3, 0] = 1  # B00, B11
    W[0, 0] = 1               # C00 += M1
    W[3, 0] = 1               # C11 += M1
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1  # A10, A11
    V[0, 1] = 1               # B00
    W[2, 1] = 1               # C10 += M2
    W[3, 1] = -1              # C11 -= M2
    
    # M3 = A00(B01-B11)
    U[0, 2] = 1               # A00
    V[1, 2] = 1; V[3, 2] = -1 # B01, -B11
    W[1, 2] = 1               # C01 += M3
    W[3, 2] = 1               # C11 += M3
    
    # M4 = A11(B10-B00)
    U[3, 3] = 1               # A11
    V[2, 3] = 1; V[0, 3] = -1 # B10, -B00
    W[0, 3] = 1               # C00 += M4
    W[2, 3] = 1               # C10 += M4
    
    # M5 = (A00+A10)B10
    U[0, 4] = 1; U[2, 4] = 1  # A00, A10
    V[2, 4] = 1               # B10
    W[0, 4] = -1              # C00 -= M5
    W[1, 4] = 1               # C01 += M5
    
    # M6 = (A01-A11)(B10+B11)
    U[1, 5] = 1; U[3, 5] = -1 # A01, -A11
    V[2, 5] = 1; V[3, 5] = 1  # B10, B11
    W[3, 5] = 1               # C11 += M6
    
    # M7 = (A10-A00)(B00+B01)
    U[2, 6] = 1; U[0, 6] = -1 # A10, -A00
    V[0, 6] = 1; V[1, 6] = 1  # B00, B01
    W[0, 6] = 1               # C00 += M7
    
    return U, V, W

def _build_block_decomposition(n, m, p):
    """
    Build a decomposition using Strassen for overlapping 2x2x2 blocks.
    We tile the 3x5x6 problem with 2x2x2 blocks.
    
    For a 3x5x6 matrix product:
    - i ranges over {0,1,2}
    - j ranges over {0,1,2,3,4}
    - k ranges over {0,1,2,3,4,5}
    
    We can cover with 2x2x2 blocks:
    Block (bi, bj, bk) covers i in {2*bi, 2*bi+1}, j in {2*bj, 2*bj+1}, k in {2*bk, 2*bk+1}
    
    For n=3: bi can be 0 (covers i=0,1) and we need i=2 covered separately
    For m=5: bj can be 0,1 (covers j=0,1,2,3) and j=4 separately
    For p=6: bk can be 0,1,2 (covers k=0,1,2,3,4,5)
    
    Strategy: Use Strassen for full 2x2x2 blocks, and schoolbook for remaining.
    
    Actually, let me think more carefully. The tensor is T[i,j,k] = sum over the product.
    We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta[i,j,k] (the tensor entry)
    
    Wait, the matrix multiplication tensor is: T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    So the tensor T_{i,j,k} = 1 (it's just the identity in the j-sum sense).
    
    Actually the tensor for matrix multiplication is:
    T[i, j, k] = 1 for all i, j, k (it's the "structure tensor" where each entry is 1)
    No wait. The tensor is defined such that:
    C[i,k] = sum_j A[i,j] * B[j,k]
    So the tensor T has T[i,j,k] = 1 for all valid i,j,k.
    
    A rank-r decomposition means:
    sum_{c=1}^{r} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    For schoolbook: one term per (i,j,k), so U[i*m+j, c]=1, V[j*p+k, c]=1, W[k*n+i, c]=1 for that c.
    
    For Strassen 2x2x2: we group 8 schoolbook terms into 7.
    """
    pass

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_strassen_tiling():
    """
    Try to build a decomposition using Strassen blocks where possible.
    
    For 3x5x6:
    - We can place 2x2x2 Strassen blocks at various positions.
    - Remaining (i,j,k) entries not covered by any Strassen block use schoolbook.
    
    A 2x2x2 Strassen block at offset (i0, j0, k0) covers:
    i in {i0, i0+1}, j in {j0, j0+1}, k in {k0, k0+1}
    
    It uses 7 terms instead of 8, saving 1 rank per block.
    
    We want to maximize the number of non-overlapping 2x2x2 blocks.
    For 3x5x6:
    - i: positions 0,1 (can have blocks at i0=0 covering i=0,1; i=2 left over)
    - j: positions 0,1,2,3,4 (blocks at j0=0,1 covering j=0,1,2,3; j=4 left over)
    - k: positions 0,1,2,3,4,5 (blocks at k0=0,1,2 covering all k)
    
    Max non-overlapping blocks:
    - i0=0: covers i=0,1
    - j0=0,1: covers j=0,1,2,3
    - k0=0,1,2: covers k=0,1,2,3,4,5
    
    So we can have 1 * 2 * 3 = 6 blocks of 2x2x2, each saving 1 rank.
    Remaining entries: those with i=2, or j=4.
    
    Total schoolbook terms: 3*5*6 = 90
    Strassen blocks: 6, each replacing 8 terms with 7, saving 6.
    Total rank: 90 - 6 = 84.
    """
    n, m, p = 3, 5, 6
    
    # Determine which (i,j,k) are covered by Strassen blocks
    # Blocks at (i0, j0, k0) where i0 in {0}, j0 in {0,1}, k0 in {0,1,2}
    block_positions = []
    for i0 in [0]:
        for j0 in [0, 1]:
            for k0 in [0, 1, 2]:
                block_positions.append((i0, j0, k0))
    
    # Check coverage
    covered = set()
    for i0, j0, k0 in block_positions:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    i, j, k = i0+di, j0+dj, k0+dk
                    if i < n and j < m and k < p:
                        covered.add((i, j, k))
    
    # Uncovered entries use schoolbook
    uncovered = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    uncovered.append((i, j, k))
    
    # Build U, V, W
    rank = 7 * len(block_positions) + len(uncovered)
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    # Strassen blocks
    for i0, j0, k0 in block_positions:
        sU, sV, sW = _strassen_2x2x2()
        for c in range(7):
            for di in range(2):
                for dj in range(2):
                    i, j = i0 + di, j0 + dj
                    if i < n and j < m:
                        U[i * m + j][col + c] += sU[di * 2 + dj][c]
            for dj in range(2):
                for dk in range(2):
                    j, k = j0 + dj, k0 + dk
                    if j < m and k < p:
                        V[j * p + k][col + c] += sV[dj * 2 + dk][c]
            for dk in range(2):
                for di in range(2):
                    k, i = k0 + dk, i0 + di
                    if k < p and i < n:
                        W[k * n + i][col + c] += sW[dk * 2 + di][c]
        col += 7
    
    # Schoolbook for uncovered
    for i, j, k in uncovered:
        U[i * m + j][col] = 1.0
        V[j * p + k][col] = 1.0
        W[k * n + i][col] = 1.0
        col += 1
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        return None
    
    return {"U": U, "V": V, "W": W}, rank

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0  # Safety margin before 150s limit
    
    # Try Strassen tiling approach
    result = _try_strassen_tiling()
    if result is not None:
        matrices, rank = result
        return matrices
    
    # Fallback: schoolbook
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
