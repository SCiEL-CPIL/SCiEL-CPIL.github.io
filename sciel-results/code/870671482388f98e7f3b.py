"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS
    # We know that M_{2,3,9} is the direct sum of three M_{2,3,3} tensors.
    # The rank of M_{2,3,3} is 9 (schoolbook, as lower bound max(2*3, 3*3)=9).
    # So total rank is 3 * 9 = 27.
    
    rank_total = 27
    
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
    
    col_offset = 0
    for block in range(3):
        # This block corresponds to columns 3*block to 3*block+2 of B
        # and rows 3*block to 3*block+2 of C (conceptually, though C rows are i, so k index shifts)
        
        # Standard schoolbook for 2x3x3
        for i in range(n):
            for j in range(m):
                for k_local in range(3):
                    k_global = block * 3 + k_local
                    
                    # U index: i*m + j
                    U[i * m + j][col_offset] = 1
                    
                    # V index: j*p + k_global
                    V[j * p + k_global][col_offset] = 1
                    
                    # W index: k_global*n + i
                    W[k_global * n + i][col_offset] = 1
                    
                    col_offset += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
