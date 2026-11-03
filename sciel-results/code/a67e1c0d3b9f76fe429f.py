"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

import numpy as np
import time
from itertools import product

# EVOLVE-BLOCK-START

def _build_strassen_2x2x2():
    """
    Build the 2x2x2 matrix multiplication tensor (Strassen, rank 7) and its decomposition.
    Returns (U, V, W) where U, V, W are 4x7 matrices (since 2*2=4 entries per matrix).
    U[i,j] for i in 0..1, j in 0..1 -> index i*2+j in U
    V[j,k] for j in 0..1, k in 0..1 -> index j*2+k in V
    W[k,i] for k in 0..1, i in 0..1 -> index k*2+i in W
    """
    # Strassen formulas for 2x2x2:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # Column 0: M1 = (A11 + A22)(B11 + B22)
    U[0, 0] = 1  # A11
    U[3, 0] = 1  # A22
    V[0, 0] = 1  # B11
    V[3, 0] = 1  # B22
    W[0, 0] = 1  # C11 (k=0, i=0 -> index 0*2+0=0)
    W[3, 0] = 1  # C22 (k=1, i=1 -> index 1*2+1=3)
    
    # Column 1: M2 = (A21 + A22)B11
    U[2, 1] = 1  # A21 (i=1, j=0 -> index 1*2+0=2)
    U[3, 1] = 1  # A22 (i=1, j=1 -> index 1*2+1=3)
    V[0, 1] = 1  # B11 (j=0, k=0 -> index 0*2+0=0)
    W[1, 1] = 1  # C21 (k=0, i=1 -> index 0*2+1=1)
    W[3, 1] = 1  # C22 (k=1, i=1 -> index 1*2+1=3)
    
    # Column 2: M3 = A11(B21 - B22)
    U[0, 2] = 1  # A11
    V[2, 2] = 1  # B21 (j=1, k=0 -> index 1*2+0=2)
    V[3, 2] = -1 # B22 (j=1, k=1 -> index 1*2+1=3)
    W[1, 2] = 1  # C12 (k=1, i=0 -> index 1*2+0=2)
    W[3, 2] = 1  # C22 (k=1, i=1 -> index 1*2+1=3)
    
    # Column 3: M4 = A22(B12 - B11)
    U[3, 3] = 1  # A22
    V[1, 3] = 1  # B12 (j=0, k=1 -> index 0*2+1=1)
    V[0, 3] = -1 # B11 (j=0, k=0 -> index 0*2+0=0)
    W[0, 3] = 1  # C11 (k=0, i=0 -> index 0*2+0=0)
    W[1, 3] = 1  # C21 (k=0, i=1 -> index 0*2+1=1)
    
    # Column 4: M5 = (A11 + A12)B22
    U[0, 4] = 1  # A11
    U[1, 4] = 1  # A12 (i=0, j=1 -> index 0*2+1=1)
    V[3, 4] = 1  # B22 (j=1, k=1 -> index 1*2+1=3)
    W[0, 4] = -1 # C11 (k=0, i=0 -> index 0*2+0=0)
    W[2, 4] = 1  # C12 (k=1, i=0 -> index 1*2+0=2)
    
    # Column 5: M6 = (A21 - A11)(B11 + B12)
    U[2, 5] = 1  # A21 (i=1, j=0 -> index 1*2+0=2)
    U[0, 5] = -1 # A11
    V[0, 5] = 1  # B11 (j=0, k=0 -> index 0*2+0=0)
    V[1, 5] = 1  # B12 (j=0, k=1 -> index 0*2+1=1)
    W[3, 5] = 1  # C22 (k=1, i=1 -> index 1*2+1=3)
    
    # Column 6: M7 = (A12 - A22)(B21 + B22)
    U[1, 6] = 1  # A12 (i=0, j=1 -> index 0*2+1=1)
    U[3, 6] = -1 # A22
    V[2, 6] = 1  # B21 (j=1, k=0 -> index 1*2+0=2)
    V[3, 6] = 1  # B22 (j=1, k=1 -> index 1*2+1=3)
    W[0, 6] = 1  # C11 (k=0, i=0 -> index 0*2+0=0)
    
    return U, V, W

def _build_schoolbook(n, m, p):
    """Build schoolbook decomposition for n x m x p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _build_block_decomposition(n, m, p, partition_n, partition_m, partition_p):
    """
    Build a block decomposition using Strassen for 2x2x2 blocks and schoolbook for others.
    
    partition_n: list of block sizes for n dimension
    partition_m: list of block sizes for m dimension  
    partition_p: list of block sizes for p dimension
    
    Each block (i_block, j_block, k_block) corresponds to:
    - A block of size n_i x m_j
    - B block of size m_j x p_k
    - C block of size p_k x n_i
    
    We use Strassen if n_i=2, m_j=2, p_k=2, otherwise schoolbook.
    """
    # Calculate total rank
    total_rank = 0
    for ni in partition_n:
        for mj in partition_m:
            for pk in partition_p:
                if ni == 2 and mj == 2 and pk == 2:
                    total_rank += 7  # Strassen
                else:
                    total_rank += ni * mj * pk  # Schoolbook
    
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    col = 0
    # Track current positions in U, V, W
    i_offset = 0
    for ni in partition_n:
        j_offset = 0
        for mj in partition_m:
            k_offset = 0
            for pk in partition_p:
                if ni == 2 and mj == 2 and pk == 2:
                    # Use Strassen for this 2x2x2 block
                    Us, Vs, Ws = _build_strassen_2x2x2()
                    for c in range(7):
                        for ui in range(4):
                            U[i_offset + ui, col + c] = Us[ui, c]
                        for vi in range(4):
                            V[j_offset + vi, col + c] = Vs[vi, c]
                        for wi in range(4):
                            W[k_offset + wi, col + c] = Ws[wi, c]
                    col += 7
                else:
                    # Use schoolbook for this block
                    rank = ni * mj * pk
                    for ii in range(ni):
                        for jj in range(mj):
                            for kk in range(pk):
                                U[i_offset + ii, col + (ii * mj + jj) * pk + kk] = 1
                                V[j_offset + jj, col + (ii * mj + jj) * pk + kk] = 1
                                W[k_offset + kk, col + (ii * mj + jj) * pk + kk] = 1
                    col += rank
                k_offset += pk
            j_offset += mj
        i_offset += ni
    
    return U, V, W

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3 by 4 matrix by a 4 by 7 matrix.
    Uses block decomposition with Strassen for 2x2x2 blocks.
    """
    n, m, p = DIMENSIONS  # 3, 4, 7
    
    # Try various partitions and pick the one with minimum rank
    best_rank = n * m * p  # Start with schoolbook
    best_U, best_V, best_W = None, None, None
    
    # Generate all possible partitions
    def get_partitions(total, max_part=4):
        """Generate all partitions of total into parts of size 1 to max_part."""
        if total == 0:
            return [()]
        result = []
        for first in range(1, min(total, max_part) + 1):
            for rest in get_partitions(total - first, max_part):
                result.append((first,) + rest)
        return result
    
    # Try partitions of n=3, m=4, p=7
    partitions_n = get_partitions(n, 3)
    partitions_m = get_partitions(m, 4)
    partitions_p = get_partitions(p, 4)
    
    for pn in partitions_n:
        for pm in partitions_m:
            for pp in partitions_p:
                U, V, W = _build_block_decomposition(n, m, p, pn, pm, pp)
                rank = U.shape[1]
                if rank < best_rank:
                    best_rank = rank
                    best_U, best_V, best_W = U, V, W
    
    # Convert to lists for the return format
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }

# EVOLVE-BLOCK-END
