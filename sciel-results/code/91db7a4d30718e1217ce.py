"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    n = 10
    # Generate two mutually orthogonal Latin squares of order 10
    # L1[i][j] = (i + j) % n
    # L2[i][j] = (i + 2*j) % n
    # These are orthogonal because the mapping (i,j) -> (i+j, i+2j) is a bijection on Z_n x Z_n
    # for any n, as long as the determinant (2-1=1) is coprime to n? 
    # Actually, the system:
    # i + j = u
    # i + 2j = v
    # j = v - u
    # i = u - j = 2u - v
    # This uniquely determines i and j for any u, v in Z_n.
    # So yes, they are orthogonal for any n.
    
    sq1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    sq2 = [[(i + 2 * j) % n for j in range(n)] for i in range(n)]
    
    # Ensure strict Python types
    result = []
    for sq in [sq1, sq2]:
        clean_sq = []
        for row in sq:
            clean_row = [int(x) for x in row]
            clean_sq.append(clean_row)
        result.append(clean_sq)
        
    return result
# EVOLVE-BLOCK-END
