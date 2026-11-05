"""Matrix multiplication 2,5,6 rank-minimal tensor decomposition."""
import numpy as np
import time
import itertools
import random

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS  # 2, 5, 6
    
    # Build rank-54 Strassen decomposition
    rank_base = 54
    U = [[0] * rank_base for _ in range(n * m)]
    V = [[0] * rank_base for _ in range(m * p)]
    W = [[0] * rank_base for _ in range(p * n)]
    col = 0
    
    def add_strassen_block(i_base, j_base, k_base):
        nonlocal col
        # M0: (A00+A11)(B00+B11)
        U[i_base*m+j_base][col] += 1
        U[(i_base+1)*m+(j_base+1)][col] += 1
        V[j_base*p+k_base][col] += 1
        V[(j_base+1)*p+(k_base+1)][col] += 1
        W[k_base*n+i_base][col] += 1
        W[(k_base+1)*n+(i_base+1)][col] += 1
        col += 1
        # M1: (A10+A11)B00
        U[(i_base+1)*m+j_base][col] += 1
        U[(i_base+1)*m+(j_base+1)][col] += 1
        V[j_base*p+k_base][col] += 1
        W[k_base*n+(i_base+1)][col] += 1
        W[(k_base+1)*n+(i_base+1)][col] -= 1
        col += 1
        # M2: A00(B01-B11)
        U[i_base*m+j_base][col] += 1
        V[j_base*p+(k_base+1)][col] += 1
        V[(j_base+1)*p+(k_base+1)][col] -= 1
        W[(k_base+1)*n+i_base][col] += 1
        W[(k_base+1)*n+(i_base+1)][col] += 1
        col += 1
        # M3: A11(B10-B00)
        U[(i_base+1)*m+(j_base+1)][col] += 1
        V[(j_base+1)*p+k_base][col] += 1
        V[j_base*p+k_base][col] -= 1
        W[k_base*n+i_base][col] += 1
        W[k_base*n+(i_base+1)][col] += 1
        col += 1
        # M4: (A00+A01)B11
        U[i_base*m+j_base][col] += 1
        U[i_base*m+(j_base+1)][col] += 1
        V[(j_base+1)*p+(k_base+1)][col] += 1
        W[k_base*n+i_base][col] -= 1
        W[(k_base+1)*n+i_base][col] += 1
        col += 1
        # M5: (A10-A00)(B00+B01)
        U[(i_base+1)*m+j_base][col] += 1
        U[i_base*m+j_base][col] -= 1
        V[j_base*p+k_base][col] += 1
        V[j_base*p+(k_base+1)][col] += 1
        W[(k_base+1)*n+(i_base+1)][col] += 1
        col += 1
        # M6: (A01-A11)(B10+B11)
        U[i_base*m+(j_base+1)][col] += 1
        U[(i_base+1)*m+(j_base+1)][col] -= 1
        V[(j_base+1)*p+k_base][col] += 1
        V[(j_base+1)*p+(k_base+1)][col] += 1
        W[k_base*n+i_base][col] += 1
        col += 1

    def add_naive_block(i_base, j_base, k_base):
        nonlocal col
        U[i_base*m+j_base][col] += 1
        V[j_base*p+k_base][col] += 1
        W[k_base*n+i_base][col] += 1
        col += 1
        U[i_base*m+j_base][col] += 1
        V[j_base*p+(k_base+1)][col] += 1
        W[(k_base+1)*n+i_base][col] += 1
        col += 1
        U[(i_base+1)*m+j_base][col] += 1
        V[j_base*p+k_base][col] += 1
        W[k_base*n+(i_base+1)][col] += 1
        col += 1
        U[(i_base+1)*m+j_base][col] += 1
        V[j_base*p+(k_base+1)][col] += 1
        W[(k_base+1)*n+(i_base+1)][col] += 1
        col += 1

    add_strassen_block(0, 0, 0)
    add_strassen_block(0, 0, 2)
    add_strassen_block(0, 0, 4)
    add_strassen_block(0, 2, 0)
    add_strassen_block(0, 2, 2)
    add_strassen_block(0, 2, 4)
    add_naive_block(0, 4, 0)
    add_naive_block(0, 4, 2)
    add_naive_block(0, 4, 4)
    
    if col != rank_base:
        rank = n*m*p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i*m*p + j*p + k
                    U[i*m+j][idx] = 1
                    V[j*p+k][idx] = 1
                    W[k*n+i][idx] = 1
        return {"U": U, "V": V, "W": W}
    
    # Convert to numpy for verification
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Build target tensor
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    
    # Verify base decomposition
    T_dec = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    
    if not np.allclose(T_dec, T, atol=1e-10):
        # Fallback to naive
        rank = n*m*p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i*m*p + j*p + k
                    U[i*m+j][idx] = 1
                    V[j*p+k][idx] = 1
                    W[k*n+i][idx] = 1
        return {"U": U, "V": V, "W": W}
    
    best_rank = rank_base
    best_U = [row[:] for row in U]
    best_V = [row[:] for row in V]
    best_W = [row[:] for row in W]
    
    # Symmetry-based pruning: explore rank-53 candidates using structural perturbation
    # Strategy: try merging pairs of columns that might be linearly dependent under symmetry
    # Apply a "symmetric group action" by trying small integer perturbations to entries
    
    if time.time() < deadline - 30.0:
        # Try local search for rank reduction
        # Attempt: set one column to a linear combination of others and see if tensor matches
        # Use symmetry: try permutations of output indices k
        
        # Try all pairs of k values that could be merged
        # For rank-53, we need to find if any two "directions" in the tensor are dependent
        
        # Simple heuristic: try each column, zero it out, and see if we can compensate
        # by adjusting other columns. This is expensive, so limit attempts.
        
        # Alternative: try all 2x2 rotations within the 7 Strassen terms
        # For each strassen block, try to find a rank-6 variant (if one exists)
        
        # For now, stick with rank 54 but try to find equivalent forms with different
        # integer representations that might help with numerical stability
        
        return {"U": best_U, "V": best_V, "W": best_W}
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
