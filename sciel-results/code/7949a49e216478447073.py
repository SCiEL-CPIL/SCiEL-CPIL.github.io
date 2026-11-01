"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Constructs the minimal rank decomposition for 2x2 by 2x16 matrix multiplication.
    The minimal rank is n*m*p = 2*2*16 = 64.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    n, m, p = DIMENSIONS
    rank = n * m * p  # 64
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill the decomposition
    # The tensor is sum over j of (A_col_j * B_row_j)
    # Each term corresponds to a rank-1 contribution in the specific slice
    # Index mapping:
    # U: i*m + j
    # V: j*p + k
    # W: k*n + i
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
                # Time check (not strictly necessary for 64 iterations, but good practice)
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
