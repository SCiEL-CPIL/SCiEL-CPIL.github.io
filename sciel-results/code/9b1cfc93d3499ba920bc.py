"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen's rank-23 decomposition for 3x3x3 matrix multiplication
    # We use the known rank-23 algorithm for 3x3 matrices
    # The 23 terms are given by specific linear combinations
    
    # Known rank-23 decomposition for 3x3x3 (from Bini et al. / Stott)
    # U has 9 entries (3x3), V has 9 entries (3x3), W has 9 entries (3x3)
    # Each of the 23 rank-1 terms
    
    # Using the standard rank-23 decomposition
    # Let's define the 23 terms explicitly
    
    # Term structure: (a_i, b_j, c_k) where a_i is coeff for A[i,j], b_j for B[j,k], c_k for C[k,i]
    # Using the known decomposition:
    
    strassen_23 = [
        # (U_row, V_col, W_row, U_val, V_val, W_val)
        # This is the standard 23-term decomposition for 3x3x3
        # Using the formulation from "The rank of the 3x3x3 tensor"
        
        # I'll use the explicit rank-23 decomposition
        # Reference: Bini, Capovani, Lotti, Romani (1978)
        
        # Term 1: A[0,0]*B[0,0] -> C[0,0]
        (0, 0, 0, 1, 1, 1),
        # Term 2: A[0,1]*B[1,0] -> C[0,0]
        (1, 3, 0, 1, 1, 1),
        # Term 3: A[1,0]*B[0,1] -> C[1,0]
        (3, 1, 3, 1, 1, 1),
        # Term 4: A[1,1]*B[1,1] -> C[1,1]
        (4, 4, 4, 1, 1, 1),
        # Term 5: A[2,2]*B[2,2] -> C[2,2]
        (8, 8, 8, 1, 1, 1),
        # Term 6: (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
        (0, 0, 0, 1, 1, 0),  # This won't work directly, need proper formulation
        
    ]
    
    # Let me use a cleaner approach with the explicit rank-23 coefficients
    # The 23 rank-1 terms for 3x3x3:
    
    # Using the formulation where we have 23 triples (u, v, w) in R^9 x R^9 x R^9
    # such that sum_t u_t v_t w_t^T = tensor T
    
    # I'll use the known decomposition from the literature
    # For now, let's use a simpler approach: use Strassen-style for 3x3
    
    # Actually, let me use the explicit rank-23 decomposition:
    # The 23 terms can be written as:
    
    # Using the standard notation where A is 3x3, B is 3x3, C = AB
    
    # Term definitions (u_idx, v_idx, w_idx, u_val, v_val, w_val):
    # u_idx is row of U (0-8), v_idx is col of V (0-8), w_idx is row of W (0-8)
    
    terms_3x3x3 = [
        # Diagonal terms
        (0, 0, 0, 1, 1, 1),      # A00*B00 -> C00
        (1, 3, 3, 1, 1, 1),      # A01*B10 -> C00... wait this is wrong
        
    ]
    
    # Let me restart with a proper rank-23 decomposition
    # The known 23-term decomposition for 3x3x3:
    
    # I'll use the following explicit decomposition:
    # Each term is (i, j, k, a, b, c) meaning:
    # U[i][t] += a, V[j][t] += b, W[k][t] += c
    
    # Using the decomposition from https://en.wikipedia.org/wiki/Strassen_algorithm
    # extended to 3x3:
    
    # Actually, for 3x3x3, the minimal known rank is 23.
    # Let me use the explicit coefficients:
    
    # The 23 terms (from Bini et al.):
    # I'll define them as (u_row, v_col, w_row, u_coeff, v_coeff, w_coeff)
    
    # Using a known rank-23 decomposition:
    rank23_terms = [
        (0, 0, 0, 1, 1, 1),
        (1, 3, 0, 1, 1, 1),
        (2, 6, 0, 1, 1, 1),
        (3, 1, 3, 1, 1, 1),
        (4, 4, 3, 1, 1, 1),
        (5, 7, 3, 1, 1, 1),
        (6, 2, 6, 1, 1, 1),
        (7, 5, 6, 1, 1, 1),
        (8, 8, 6, 1, 1, 1),
        (0, 3, 0, 1, 1, -1),
        (1, 6, 0, 1, 1, -1),
        (3, 4, 3, 1, 1, -1),
        (4, 7, 3, 1, 1, -1),
        (6, 5, 6, 1, 1, -1),
        (7, 8, 6, 1, 1, -1),
        (0, 6, 3, 1, 1, 1),
        (3, 7, 6, 1, 1, 1),
        (6, 8, 0, 1, 1, 1),
        (1, 0, 0, 1, 1, 1),
        (4, 1, 3, 1, 1, 1),
        (7, 2, 6, 1, 1, 1),
        (2, 0, 3, 1, 1, 1),
        (5, 1, 6, 1, 1, 1),
    ]
    
    # This doesn't look right. Let me use a verified rank-23 decomposition.
    # For now, let's just use schoolbook for the 3x3 blocks but try to be smarter.
    
    # Actually, let me just use the straightforward schoolbook approach for now
    # since I need to ensure correctness. The rank will be 63.
    
    rank = 63
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    # First 3x3 block: i in 0..2, j in 0..2, k in 0..2
    for i in range(n):
        for j in range(m):
            for k in range(3):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    
    # Second 3x3 block: i in 0..2, j in 0..2, k in 3..5
    for i in range(n):
        for j in range(m):
            for k in range(3, 6):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    
    # Third 3x1 block: i in 0..2, j in 0..2, k in 6
    for i in range(n):
        for j in range(m):
            k = 6
            U[i*m + j][col] = 1
            V[j*p + k][col] = 1
            W[k*n + i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
