"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # We construct a decomposition of rank 72
    # Split m=5 into 2+3
    # Split p=6 into 2+2+2 for the first part, 3+3 for the second part
    
    # Part 1: A1 (3x2) * B1 (2x6)
    # B1 is split into three 2x2 blocks
    # Each 3x2x2 block uses rank 6
    # Total rank for Part 1: 3 * 6 = 18
    
    # Part 2: A2 (3x3) * B2 (3x6)
    # B2 is split into two 3x3 blocks
    # Each 3x3x3 block uses schoolbook rank 27
    # Total rank for Part 2: 2 * 27 = 54
    
    total_rank = 18 + 54
    U = np.zeros((nm, total_rank))
    V = np.zeros((mp, total_rank))
    W = np.zeros((pn, total_rank))
    
    col = 0
    
    # Part 1: i in 0..2, j in 0..1, k in 0..5
    # j is global index 0, 1
    # k is global index 0..5
    for block_id in range(3):
        # k_global starts at 2 * block_id
        k_start = 2 * block_id
        # Use rank 6 decomposition for 3x2x2
        # Terms 0-2: j=0, k=k_start
        for i in range(n):
            U[i*m, col] = 1  # U index is i*m + j? No.
            # Wait, U shape is (n*m, rank).
            # U index is i*m + j_A.
            # In Part 1, j_A is 0 or 1.
            # But the term is defined by j_B (which is j_A) and k_B.
            # The tensor entry is T_{i, j, k}.
            # U corresponds to A indices (i, j).
            # V corresponds to B indices (j, k).
            # W corresponds to C indices (k, i) -> index k*n + i.
            
            # For term with j=0, k=k_start:
            # U[i*m + 0, col] = 1
            # V[0*p + k_start, col] = 1
            # W[k_start*n + i, col] = 1
            U[i*m + 0, col] = 1
            V[0*p + k_start, col] = 1
            W[k_start*n + i, col] = 1
            col += 1
            
        # Terms 3-5: j=1, k=k_start+1
        for i in range(n):
            U[i*m + 1, col] = 1
            V[1*p + (k_start + 1), col] = 1
            W[(k_start + 1)*n + i, col] = 1
            col += 1
            
    # Part 2: i in 0..2, j in 2..4, k in 0..5
    # j is global index 2, 3, 4
    # k is global index 0..5
    # Split k into two blocks of 3
    for block_id in range(2):
        k_start = 3 * block_id
        # Use schoolbook for 3x3x3
        for j_local in range(3):
            j_global = 2 + j_local
            for k_local in range(3):
                k_global = k_start + k_local
                for i in range(n):
                    U[i*m + j_global, col] = 1
                    V[j_global*p + k_global, col] = 1
                    W[k_global*n + i, col] = 1
                    col += 1
                    
    # Verify col == total_rank
    if col != total_rank:
        raise ValueError(f"Column count mismatch: {col} vs {total_rank}")
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
