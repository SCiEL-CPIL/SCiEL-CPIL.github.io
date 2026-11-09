"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 24 decomposition using 3x2x2 blocks (rank 6 each)
    # A is 3x4, B is 4x4, C is 3x4
    # Split j into {0,1} and {2,3}, k into {0,1} and {2,3}
    # 4 sub-problems of 3x2x2, each rank 6, total rank 24
    rank = 24
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    
    # Helper: add a 3x2x2 block
    # A block: rows i_off..i_off+2, cols j_off..j_off+1 (3x2)
    # B block: rows j_off..j_off+1, cols k_off..k_off+1 (2x2)
    # C block: rows i_off..i_off+2, cols k_off..k_off+1 (3x2)
    #
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # Decomposition: 6 terms (3 for A[i,0]*B[0,k], 3 for A[i,1]*B[1,k])
    def add_322(i_off, j_off, k_off):
        nonlocal col
        # Term 1-3: A[i,0]*B[0,k] for i=0,1,2
        for i in range(3):
            # U: A[i_off+i, j_off+0]
            U[(i_off + i) * m + (j_off + 0), col] = 1
            # V: B[j_off+0, k_off+0] and B[j_off+0, k_off+1]
            V[(j_off + 0) * p + (k_off + 0), col] = 1
            V[(j_off + 0) * p + (k_off + 1), col] = 1
            # W: C[i_off+i, k_off+0] and C[i_off+i, k_off+1]
            # W index: k*n + i, so W[(k_off+0)*n + (i_off+i)] and W[(k_off+1)*n + (i_off+i)]
            W[(k_off + 0) * n + (i_off + i), col] = 1
            W[(k_off + 1) * n + (i_off + i), col] = 1
            col += 1
        
        # Term 4-6: A[i,1]*B[1,k] for i=0,1,2
        for i in range(3):
            # U: A[i_off+i, j_off+1]
            U[(i_off + i) * m + (j_off + 1), col] = 1
            # V: B[j_off+1, k_off+0] and B[j_off+1, k_off+1]
            V[(j_off + 1) * p + (k_off + 0), col] = 1
            V[(j_off + 1) * p + (k_off + 1), col] = 1
            # W: C[i_off+i, k_off+0] and C[i_off+i, k_off+1]
            W[(k_off + 0) * n + (i_off + i), col] = 1
            W[(k_off + 1) * n + (i_off + i), col] = 1
            col += 1
    
    # 4 sub-problems:
    # j_off=0, k_off=0: A[:,0:2] x B[0:2,0:2]
    add_322(0, 0, 0)
    # j_off=0, k_off=2: A[:,0:2] x B[0:2,2:4]
    add_322(0, 0, 2)
    # j_off=2, k_off=0: A[:,2:4] x B[2:4,0:2]
    add_322(0, 2, 0)
    # j_off=2, k_off=2: A[:,2:4] x B[2:4,2:4]
    add_322(0, 2, 2)
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
