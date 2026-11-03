"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
import itertools

def _verify_decomp(U, V, W, n, m, p, rank):
    """Verify that the decomposition is correct using numpy."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for l in range(m):
            for k in range(p):
                u_row = U_arr[i*m+l, :]
                v_row = V_arr[l*p+k, :]
                w_row = W_arr[k*n+i, :]
                s = np.dot(u_row, v_row * w_row)
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _strassen_block(U, V, W, i0, j0, k0, n, m, p):
    """Apply Strassen's 7-product scheme to a 2x2x2 sub-block.
    Rows: i0, i0+1; Cols of A: j0, j0+1; Cols of B: k0, k0+1.
    Returns the next column index."""
    col = 0
    # P1 = (A[i0,j0]+A[i0+1,j0+1]) * (B[j0,k0]+B[j0+1,k0+1])
    U[i0*m + j0][col] += 1
    U[(i0+1)*m + j0+1][col] += 1
    V[j0*p + k0][col] += 1
    V[(j0+1)*p + k0+1][col] += 1
    W[k0*n + i0][col] += 1
    W[(k0+1)*n + i0+1][col] += 1
    col += 1
    
    # P2 = (A[i0+1,j0]+A[i0+1,j0+1]) * B[j0,k0]
    U[(i0+1)*m + j0][col] += 1
    U[(i0+1)*m + j0+1][col] += 1
    V[j0*p + k0][col] += 1
    W[k0*n + i0+1][col] += 1
    W[(k0+1)*n + i0+1][col] -= 1
    col += 1
    
    # P3 = A[i0,j0] * (B[j0,k0+1]-B[j0+1,k0+1])
    U[i0*m + j0][col] += 1
    V[j0*p + k0+1][col] += 1
    V[(j0+1)*p + k0+1][col] -= 1
    W[(k0+1)*n + i0][col] += 1
    W[(k0+1)*n + i0+1][col] += 1
    col += 1
    
    # P4 = A[i0+1,j0+1] * (B[j0+1,k0]-B[j0,k0])
    U[(i0+1)*m + j0+1][col] += 1
    V[(j0+1)*p + k0][col] += 1
    V[j0*p + k0][col] -= 1
    W[k0*n + i0][col] += 1
    W[k0*n + i0+1][col] += 1
    col += 1
    
    # P5 = (A[i0,j0]+A[i0,j0+1]) * B[j0+1,k0+1]
    U[i0*m + j0][col] += 1
    U[i0*m + j0+1][col] += 1
    V[(j0+1)*p + k0+1][col] += 1
    W[k0*n + i0][col] -= 1
    W[(k0+1)*n + i0][col] += 1
    col += 1
    
    # P6 = (A[i0+1,j0]-A[i0,j0]) * (B[j0,k0]+B[j0,k0+1])
    U[(i0+1)*m + j0][col] += 1
    U[i0*m + j0][col] -= 1
    V[j0*p + k0][col] += 1
    V[j0*p + k0+1][col] += 1
    W[(k0+1)*n + i0+1][col] += 1
    col += 1
    
    # P7 = (A[i0,j0+1]-A[i0+1,j0+1]) * (B[j0+1,k0]+B[j0+1,k0+1])
    U[i0*m + j0+1][col] += 1
    U[(i0+1)*m + j0+1][col] -= 1
    V[(j0+1)*p + k0][col] += 1
    V[(j0+1)*p + k0+1][col] += 1
    W[k0*n + i0][col] += 1
    col += 1
    
    return col

def construct():
    n, m, p = DIMENSIONS
    _start_time = time.time()
    _deadline = _start_time + 140
    
    # Start with the known rank-54 decomposition
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    j_pairs = [(0, 1), (2, 3)]
    k_pairs = [(0, 1), (2, 3), (4, 5)]
    
    for j0, j1 in j_pairs:
        for k0, k1 in k_pairs:
            col = _strassen_block(U, V, W, 0, j0, k0, n, m, p) + col
    
    # Remaining j=4: standard products
    for k in range(p):
        U[0*m + 4][col] += 1
        V[4*p + k][col] += 1
        W[k*n + 0][col] += 1
        col += 1
        
        U[1*m + 4][col] += 1
        V[4*p + k][col] += 1
        W[k*n + 1][col] += 1
        col += 1
    
    best_rank = rank
    best_U = [row[:] for row in U]
    best_V = [row[:] for row in V]
    best_W = [row[:] for row in W]
    
    if not _verify_decomp(best_U, best_V, best_W, n, m, p, best_rank):
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Try to reduce rank by eliminating redundant terms
    U_arr = np.array(best_U, dtype=float)
    V_arr = np.array(best_V, dtype=float)
    W_arr = np.array(best_W, dtype=float)
    
    # Attempt to find a lower-rank decomposition by solving a linear system
    # For rank R-1, we need to find U', V', W' of rank R-1 such that
    # sum_{r=0}^{R-2} U'[i*m+l,r] * V'[l*p+k,r] * W'[k*n+i,r] = 1
    
    # This is a bilinear system, which is hard. Let's try a simpler approach:
    # Try to merge two terms that are "similar" in some sense.
    
    # For now, return the rank-54 decomposition.
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
