"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 54 decomposition:
    # Two 2x2x6 blocks (rank 21 each) and one 2x1x6 block (rank 12)
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen 2x2x2 decomposition for a single 2x2x2 block
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    def add_strassen_block(u_offset, v_offset, w_offset, j_start):
        nonlocal col
        # j_start is the index in the middle dimension (0, 2, etc)
        # A is 2x2, B is 2x6 (split into three 2x2s)
        # We need 3 Strassen instances for the 6 columns of B
        
        for k_block in range(3):
            # k_block is 0, 1, 2. Corresponds to B columns 2*k_block, 2*k_block+1
            # Strassen coefficients for U (A part)
            # M1: A00, A11
            # M2: A10, A11
            # M3: A00
            # M4: A11
            # M5: A00, A01
            # M6: A10, A00
            # M7: A01, A11
            
            # U indices: i*m + j. i in 0,1. j in j_start, j_start+1
            u00 = 0 * m + j_start
            u01 = 0 * m + j_start + 1
            u10 = 1 * m + j_start
            u11 = 1 * m + j_start + 1
            
            # V indices: j*p + k. j in j_start, j_start+1. k in 2*k_block, 2*k_block+1
            v00 = j_start * p + 2 * k_block
            v01 = j_start * p + 2 * k_block + 1
            v10 = (j_start + 1) * p + 2 * k_block
            v11 = (j_start + 1) * p + 2 * k_block + 1
            
            # W indices: k*n + i. k in 2*k_block, 2*k_block+1. i in 0,1
            w00 = 2 * k_block * n + 0
            w01 = 2 * k_block * n + 1
            w10 = (2 * k_block + 1) * n + 0
            w11 = (2 * k_block + 1) * n + 1
            
            # M1
            U[u00][col] = 1
            U[u11][col] = 1
            V[v00][col] = 1
            V[v11][col] = 1
            W[w00][col] = 1
            W[w01][col] = 1
            W[w10][col] = 1
            W[w11][col] = 1
            col += 1
            
            # M2
            U[u10][col] = 1
            U[u11][col] = 1
            V[v00][col] = 1
            W[w10][col] = 1
            W[w11][col] = 1
            col += 1
            
            # M3
            U[u00][col] = 1
            V[v01][col] = 1
            V[v11][col] = -1
            W[w01][col] = 1
            W[w11][col] = 1
            col += 1
            
            # M4
            U[u11][col] = 1
            V[v10][col] = 1
            V[v00][col] = -1
            W[w10][col] = 1
            W[w11][col] = 1
            col += 1
            
            # M5
            U[u00][col] = 1
            U[u01][col] = 1
            V[v11][col] = 1
            W[w00][col] = 1
            W[w01][col] = 1
            col += 1
            
            # M6
            U[u10][col] = 1
            U[u00][col] = -1
            V[v00][col] = 1
            V[v01][col] = 1
            W[w10][col] = 1
            W[w11][col] = 1
            col += 1
            
            # M7
            U[u01][col] = 1
            U[u11][col] = -1
            V[v10][col] = 1
            V[v11][col] = 1
            W[w00][col] = 1
            W[w01][col] = 1
            col += 1

    # Add first 2x2x6 block (j=0,1)
    add_strassen_block(0, 0, 0, 0)
    
    # Add second 2x2x6 block (j=2,3)
    add_strassen_block(0, 12, 0, 2)
    
    # Add third 2x1x6 block (j=4)
    # Rank 12: 2 * 6 terms
    for i in range(2):
        for k in range(6):
            # U[i*m + 4] = 1
            U[i * m + 4][col] = 1
            # V[4*p + k] = 1
            V[4 * p + k][col] = 1
            # W[k*n + i] = 1
            W[k * n + i][col] = 1
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
