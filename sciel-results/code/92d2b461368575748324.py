"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Decompose B (3x7) into B1 (3x3), B2 (3x3), B3 (3x1)
    # C = A*B1 + A*B2 + A*B3
    
    # Block 1: 2x3x3 schoolbook (rank 9)
    # Block 2: 2x3x3 schoolbook (rank 9)
    # Block 3: 2x3x1 schoolbook (rank 6)
    
    rank1 = 9
    rank2 = 9
    rank3 = 6
    total_rank = rank1 + rank2 + rank3
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Helper to fill schoolbook 2x3x3
    def fill_2x3x3(offset_u, offset_v, offset_w, start_col):
        # U: 6x9, V: 9x9, W: 6x9
        # U[i*3+j, r] = 1 if r == i*3+j
        # V[j*3+k, r] = 1 if r == j*3+k
        # W[k*2+i, r] = 1 if r == j*3+k  (Wait, W depends on j too? No, W is output coeff)
        # In schoolbook, term r corresponds to (i, j, k).
        # U[i*3+j, r] = 1 if (i,j) matches r's (i,j)
        # V[j*3+k, r] = 1 if (j,k) matches r's (j,k)
        # W[k*2+i, r] = 1 if (k,i) matches r's (k,i)
        
        r = 0
        for i in range(2):
            for j in range(3):
                for k in range(3):
                    U[i * 3 + j][start_col + r] = 1
                    V[j * 3 + k][start_col + r] = 1
                    W[k * 2 + i][start_col + r] = 1
                    r += 1

    # Block 1: k=0,1,2
    # V rows: j*7 + k. For k=0,1,2, rows are 0,1,2; 7,8,9; 14,15,16.
    # We need to map local V indices (j*3+k_local) to global V indices.
    # Local V index l = j*3 + k_local.
    # Global V index g = j*7 + k_global.
    # k_global = k_local + offset_k.
    
    def fill_2x3x3_mapped(offset_k, start_col):
        r = 0
        for i in range(2):
            for j in range(3):
                for k_local in range(3):
                    k_global = k_local + offset_k
                    U[i * 3 + j][start_col + r] = 1
                    V[j * 7 + k_global][start_col + r] = 1
                    W[k_global * 2 + i][start_col + r] = 1
                    r += 1

    # Block 3: 2x3x1 schoolbook
    def fill_2x3x1_mapped(offset_k, start_col):
        r = 0
        for i in range(2):
            for j in range(3):
                k_global = offset_k
                U[i * 3 + j][start_col + r] = 1
                V[j * 7 + k_global][start_col + r] = 1
                W[k_global * 2 + i][start_col + r] = 1
                r += 1

    # Fill Block 1
    fill_2x3x3_mapped(0, 0)
    
    # Fill Block 2
    fill_2x3x3_mapped(3, rank1)
    
    # Fill Block 3
    fill_2x3x1_mapped(6, rank1 + rank2)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
