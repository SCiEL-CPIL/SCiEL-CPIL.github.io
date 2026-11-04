import time
import numpy as np

def construct():
    n, m, p = 2, 4, 7
    start_time = time.time()
    
    # Rank 50 decomposition
    rank = 50
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    def add_strassen(i0, j0, k0):
        nonlocal col
        # A is 2x2 block starting at row i0 (always 0 or 1? No, A has 2 rows, so i0 is 0 for the block rows)
        # Wait, A is 2x4. The "i0" in the block refers to the row offset in A. Since A has only 2 rows, 
        # the block covers rows 0 and 1. So i0 is effectively 0 for the row dimension of the 2x2 block.
        # However, the column offset j0 varies (0 or 2).
        # B is 4x7. The row offset j0 varies (0 or 2). The column offset k0 varies (0, 2, 4, 6).
        
        # Indices in A:
        # a = A[i0, j0] -> U[i0*m + j0]
        # b = A[i0, j0+1] -> U[i0*m + j0+1]
        # c = A[i0+1, j0] -> U[(i0+1)*m + j0]
        # d = A[i0+1, j0+1] -> U[(i0+1)*m + j0+1]
        
        # Indices in B:
        # e = B[j0, k0] -> V[j0*p + k0]
        # f = B[j0, k0+1] -> V[j0*p + k0+1]
        # g = B[j0+1, k0] -> V[(j0+1)*p + k0]
        # h = B[j0+1, k0+1] -> V[(j0+1)*p + k0+1]
        
        # Indices in C:
        # C[i0, k0] -> W[k0*n + i0]
        # C[i0, k0+1] -> W[(k0+1)*n + i0]
        # C[i0+1, k0] -> W[k0*n + (i0+1)]
        # C[i0+1, k0+1] -> W[(k0+1)*n + (i0+1)]
        
        # M1 = (a+d)(e+h)
        U[i0*m + j0, col] += 1
        U[(i0+1)*m + (j0+1), col] += 1
        V[j0*p + k0, col] += 1
        V[(j0+1)*p + (k0+1), col] += 1
        W[k0*n + i0, col] += 1
        W[(k0+1)*n + (i0+1), col] += 1
        col += 1
        
        # M2 = (c+d)e
        U[(i0+1)*m + j0, col] += 1
        U[(i0+1)*m + (j0+1), col] += 1
        V[j0*p + k0, col] += 1
        W[k0*n + (i0+1), col] += 1
        W[(k0+1)*n + (i0+1), col] -= 1
        col += 1
        
        # M3 = a(f-h)
        U[i0*m + j0, col] += 1
        V[j0*p + (k0+1), col] += 1
        V[(j0+1)*p + (k0+1), col] -= 1
        W[(k0+1)*n + i0, col] += 1
        W[(k0+1)*n + (i0+1), col] += 1
        col += 1
        
        # M4 = d(g-e)
        U[(i0+1)*m + (j0+1), col] += 1
        V[(j0+1)*p + k0, col] += 1
        V[j0*p + k0, col] -= 1
        W[k0*n + i0, col] += 1
        W[k0*n + (i0+1), col] += 1
        col += 1
        
        # M5 = (a+b)h
        U[i0*m + j0, col] += 1
        U[i0*m + (j0+1), col] += 1
        V[(j0+1)*p + (k0+1), col] += 1
        W[k0*n + i0, col] -= 1
        W[(k0+1)*n + i0, col] += 1
        col += 1
        
        # M6 = (c-a)(e+f)
        U[(i0+1)*m + j0, col] += 1
        U[i0*m + j0, col] -= 1
        V[j0*p + k0, col] += 1
        V[j0*p + (k0+1), col] += 1
        W[(k0+1)*n + (i0+1), col] += 1
        col += 1
        
        # M7 = (b-d)(g+h)
        U[i0*m + (j0+1), col] += 1
        U[(i0+1)*m + (j0+1), col] -= 1
        V[(j0+1)*p + k0, col] += 1
        V[(j0+1)*p + (k0+1), col] += 1
        W[k0*n + i0, col] += 1
        col += 1

    def add_schoolbook_2x2x1(i0, j0, k0):
        # A is 2x2 (rows i0, i0+1; cols j0, j0+1)
        # B is 2x1 (rows j0, j0+1; col k0)
        # C is 2x1 (rows i0, i0+1; col k0)
        # C[i0, k0] = A[i0, j0]B[j0, k0] + A[i0, j0+1]B[j0+1, k0]
        # C[i0+1, k0] = A[i0+1, j0]B[j0, k0] + A[i0+1, j0+1]B[j0+1, k0]
        nonlocal col
        
        # Term 1: A[i0, j0] * B[j0, k0] -> C[i0, k0]
        U[i0*m + j0, col] = 1
        V[j0*p + k0, col] = 1
        W[k0*n + i0, col] = 1
        col += 1
        
        # Term 2: A[i0, j0+1] * B[j0+1, k0] -> C[i0, k0]
        U[i0*m + (j0+1), col] = 1
        V[(j0+1)*p + k0, col] = 1
        W[k0*n + i0, col] = 1
        col += 1
        
        # Term 3: A[i0+1, j0] * B[j0, k0] -> C[i0+1, k0]
        U[(i0+1)*m + j0, col] = 1
        V[j0*p + k0, col] = 1
        W[k0*n + (i0+1), col] = 1
        col += 1
        
        # Term 4: A[i0+1, j0+1] * B[j0+1, k0] -> C[i0+1, k0]
        U[(i0+1)*m + (j0+1), col] = 1
        V[(j0+1)*p + k0, col] = 1
        W[k0*n + (i0+1), col] = 1
        col += 1

    # Block 1: A_L (cols 0,1), B_T (rows 0,1)
    # i0=0, j0=0
    # Cols 0,1
    add_strassen(0, 0, 0)
    # Cols 2,3
    add_strassen(0, 0, 2)
    # Cols 4,5
    add_strassen(0, 0, 4)
    # Col 6
    add_schoolbook_2x2x1(0, 0, 6)
    
    # Block 2: A_R (cols 2,3), B_B (rows 2,3)
    # i0=0 (A still has 2 rows, so row index in A is 0,1), j0=2 (col offset in A)
    # j0_B=2 (row offset in B)
    # Cols 0,1
    add_strassen(0, 2, 0)
    # Cols 2,3
    add_strassen(0, 2, 2)
    # Cols 4,5
    add_strassen(0, 2, 4)
    # Col 6
    add_schoolbook_2x2x1(0, 2, 6)

    if time.time() - start_time > 100:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
