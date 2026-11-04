import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 140
    
    n, m, p = DIMENSIONS
    
    # Rank 50 decomposition
    rank = 50
    U = np.zeros((n * m, rank), dtype=np.int32)
    V = np.zeros((m * p, rank), dtype=np.int32)
    W = np.zeros((p * n, rank), dtype=np.int32)
    
    col = 0
    
    def emit_strassen(i0, j0, k0):
        nonlocal col
        
        U[i0*m + j0, col] = 1
        U[(i0+1)*m + (j0+1), col] = 1
        V[j0*p + k0, col] = 1
        V[(j0+1)*p + (k0+1), col] = 1
        W[k0*n + i0, col] = 1
        W[(k0+1)*n + (i0+1), col] = 1
        col += 1
        
        U[(i0+1)*m + j0, col] = 1
        U[(i0+1)*m + (j0+1), col] = 1
        V[j0*p + k0, col] = 1
        W[k0*n + (i0+1), col] = 1
        W[(k0+1)*n + (i0+1), col] = -1
        col += 1
        
        U[i0*m + j0, col] = 1
        V[j0*p + (k0+1), col] = 1
        V[(j0+1)*p + (k0+1), col] = -1
        W[(k0+1)*n + i0, col] = 1
        W[(k0+1)*n + (i0+1), col] = 1
        col += 1
        
        U[(i0+1)*m + (j0+1), col] = 1
        V[(j0+1)*p + k0, col] = 1
        V[j0*p + k0, col] = -1
        W[k0*n + (i0+1), col] = 1
        W[k0*n + i0, col] = 1
        col += 1
        
        U[i0*m + j0, col] = 1
        U[i0*m + (j0+1), col] = 1
        V[(j0+1)*p + (k0+1), col] = 1
        W[k0*n + i0, col] = -1
        W[(k0+1)*n + i0, col] = 1
        col += 1
        
        U[(i0+1)*m + j0, col] = 1
        U[i0*m + j0, col] = -1
        V[j0*p + k0, col] = 1
        V[j0*p + (k0+1), col] = 1
        W[(k0+1)*n + (i0+1), col] = 1
        col += 1
        
        U[i0*m + (j0+1), col] = 1
        U[(i0+1)*m + (j0+1), col] = -1
        V[(j0+1)*p + k0, col] = 1
        V[(j0+1)*p + (k0+1), col] = 1
        W[k0*n + i0, col] = 1
        col += 1

    def emit_schoolbook_2x2x1(i0, j0, k0):
        nonlocal col
        
        U[i0*m + j0, col] = 1
        V[j0*p + k0, col] = 1
        W[k0*n + i0, col] = 1
        col += 1
        
        U[i0*m + (j0+1), col] = 1
        V[(j0+1)*p + k0, col] = 1
        W[k0*n + i0, col] = 1
        col += 1
        
        U[(i0+1)*m + j0, col] = 1
        V[j0*p + k0, col] = 1
        W[k0*n + (i0+1), col] = 1
        col += 1
        
        U[(i0+1)*m + (j0+1), col] = 1
        V[(j0+1)*p + k0, col] = 1
        W[k0*n + (i0+1), col] = 1
        col += 1

    emit_strassen(0, 0, 0)
    emit_strassen(0, 0, 2)
    emit_strassen(0, 0, 4)
    emit_schoolbook_2x2x1(0, 0, 6)
    
    emit_strassen(0, 2, 0)
    emit_strassen(0, 2, 2)
    emit_strassen(0, 2, 4)
    emit_schoolbook_2x2x1(0, 2, 6)
    
    # Fast verification using int dot product
    # Convert to lists for return
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
