"""Matrix multiplication 5x5x5 tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition.
    U, V, W are lists of 7 vectors, each of length 4.
    Indexing: A[i,j] -> i*2+j, B[j,k] -> j*2+k, C[k,i] -> k*2+i
    """
    U = [
        [1, 0, 0, 1],   # A00+A11
        [0, 0, 1, 1],   # A10+A11
        [1, 0, 0, 0],   # A00
        [0, 0, 0, 1],   # A11
        [1, 0, 1, 0],   # A00+A10
        [-1, 1, 0, 0],  # A01-A00
        [0, 0, 1, -1],  # A10-A11
    ]
    V = [
        [1, 0, 0, 1],   # B00+B11
        [1, 0, 0, 0],   # B00
        [0, 0, 1, -1],  # B10-B11
        [-1, 1, 0, 0],  # B01-B00
        [0, 0, 0, 1],   # B11
        [1, 0, 1, 0],   # B00+B10
        [0, 1, 0, 1],   # B01+B11
    ]
    W = [
        [1, 0, 0, 1],   # C00+C11
        [0, 0, 1, -1],  # C10-C11
        [0, 1, 0, 1],   # C01+C11
        [1, 0, 1, 0],   # C00+C10
        [-1, 1, 0, 0],  # -C00+C01
        [0, 0, 0, 1],   # C11
        [1, 0, 0, 0],   # C00
    ]
    return U, V, W

def _schoolbook(n, m, p):
    """
    Returns U, V, W for n x m x p schoolbook decomposition.
    Rank = n*m*p.
    U shape: (n*m, rank)
    V shape: (m*p, rank)
    W shape: (p*n, rank)
    """
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
                
    return U, V, W

def _get_decomp(n, m, p):
    """
    Returns U, V, W for n x m x p decomposition.
    Uses Strassen for 2x2x2, schoolbook otherwise.
    """
    if n == 2 and m == 2 and p == 2:
        return _strassen_2x2()
    else:
        return _schoolbook(n, m, p)

def _verify_float(n, m, p, U, V, W, tol=1e-9):
    """Verify using floating point."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                dot = np.dot(U_np[i * m + j], V_np[j * p + k] * W_np[k * n + i])
                if abs(dot - 1.0) > tol:
                    return False
    return True

def _verify_exact(n, m, p, U, V, W):
    """Exact verification using Fraction arithmetic."""
    rank = len(U[0]) if U else 0
    U_f = [[Fraction(x) for x in row] for row in U]
    V_f = [[Fraction(x) for x in row] for row in V]
    W_f = [[Fraction(x) for x in row] for row in W]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for c in range(rank):
                    total += U_f[i * m + j][c] * V_f[j * p + k][c] * W_f[k * n + i][c]
                if total != Fraction(1):
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin
    
    # Fallback: Schoolbook
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = n * m * p
    
    # Strategy: Block decomposition with partition 2, 2, 1
    partition = [2, 2, 1]
    offsets = [0, 2, 4]
    num_blocks = 3
    
    # Pre-compute decompositions for each block size combination
    decomp_cache = {}
    for i in range(num_blocks):
        for j in range(num_blocks):
            for k in range(num_blocks):
                ni, nj, nk = partition[i], partition[j], partition[k]
                if (ni, nj, nk) not in decomp_cache:
                    decomp_cache[(ni, nj, nk)] = _get_decomp(ni, nj, nk)
    
    # Calculate total rank
    total_rank = 0
    for i in range(num_blocks):
        for j in range(num_blocks):
            for k in range(num_blocks):
                ni, nj, nk = partition[i], partition[j], partition[k]
                U_sub, V_sub, W_sub = decomp_cache[(ni, nj, nk)]
                total_rank += len(U_sub[0])
                
    # Initialize global matrices
    U5 = [[0] * total_rank for _ in range(n * m)]
    V5 = [[0] * total_rank for _ in range(m * p)]
    W5 = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    for i in range(num_blocks):
        for j in range(num_blocks):
            for k in range(num_blocks):
                ni, nj, nk = partition[i], partition[j], partition[k]
                U_sub, V_sub, W_sub = decomp_cache[(ni, nj, nk)]
                rank_sub = len(U_sub[0])
                
                off_i = offsets[i]
                off_j = offsets[j]
                off_k = offsets[k]
                
                for c in range(rank_sub):
                    # Map A
                    for local_idx in range(ni * nj):
                        val = U_sub[local_idx][c]
                        if val != 0:
                            li, lj = divmod(local_idx, nj)
                            gi = off_i + li
                            gj = off_j + lj
                            U5[gi * m + gj][col_offset + c] = val
                            
                    # Map B
                    for local_idx in range(nj * nk):
                        val = V_sub[local_idx][c]
                        if val != 0:
                            lj, lk = divmod(local_idx, nk)
                            gj = off_j + lj
                            gk = off_k + lk
                            V5[gj * p + gk][col_offset + c] = val
                            
                    # Map C
                    for local_idx in range(nk * ni):
                        val = W_sub[local_idx][c]
                        if val != 0:
                            lk, li = divmod(local_idx, ni)
                            gk = off_k + lk
                            gi = off_i + li
                            W5[gk * n + gi][col_offset + c] = val
                            
                col_offset += rank_sub
                
    # Verify
    if _verify_float(n, m, p, U5, V5, W5):
        if _verify_exact(n, m, p, U5, V5, W5):
            if total_rank < best_rank:
                best = {"U": U5, "V": V5, "W": W5}
                best_rank = total_rank
                
    return best

# EVOLVE-BLOCK-END
