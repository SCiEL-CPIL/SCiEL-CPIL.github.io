"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Hybrid decomposition:
    - Apply Strassen's rank-7 algorithm to each of the 5 independent 2x2 column blocks.
    - Total rank = 5 * 7 = 35.
    
    The 2x2x10 matrix multiplication can be viewed as 5 independent 2x2x2 multiplications
    (one for each pair of columns in the 10-column matrix).
    
    For each 2x2x2 block, we use Strassen's 7-multiplication algorithm.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=10
    
    # Strassen's 7 products for 2x2 matrix multiplication:
    # M1 = (A11) * (B11 - B22)
    # M2 = (A11 + A22) * B22
    # M3 = (A22 - A11) * B11
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A21) * B22
    # M6 = (A22 - A21) * (B11 + B12)
    # M7 = (A11 - A22) * (B21 + B22)
    #
    # C11 = M1 + M2 - M4 + M7
    # C12 = M3 + M5
    # C21 = M2 + M6
    # C22 = M1 - M3 + M4 + M5
    
    # For each of the 5 column pairs (columns 2j, 2j+1), we apply Strassen.
    # The 2x2x2 tensor for block j has:
    #   A indices: i in {0,1}, j_a in {0,1}  (but here the "middle" dimension is shared)
    #   B indices: j_b in {0,1}, k in {0,1}  (columns within the block)
    #   C indices: i in {0,1}, k in {0,1}
    
    # Let's define the Strassen coefficients for the 2x2x2 case.
    # U has shape (n*m, rank) = (4, 35)
    # V has shape (m*p, rank) = (20, 35)
    # W has shape (p*n, rank) = (20, 35)
    
    rank = 35
    U = [[0] * rank for _ in range(n * m)]  # (4, 35)
    V = [[0] * rank for _ in range(m * p)]  # (20, 35)
    W = [[0] * rank for _ in range(p * n)]  # (20, 35)
    
    # Strassen coefficients for 2x2x2:
    # For each of the 7 products, we define:
    #   u[i*m + j] : coefficient in U
    #   v[j*p + k] : coefficient in V (but here p=2 for the sub-block)
    #   w[k*n + i] : coefficient in W
    
    # The 7 Strassen products:
    # Product 1: M1 = A11 * (B11 - B22)
    #   U: A11 -> row (0,0) = 0
    #   V: B11 -> (0,0), B22 -> (1,1)  => V[0*2+0]=1, V[1*2+1]=-1
    #   W: M1 contributes to C11 and C22
    #     C11 += M1, C22 += M1
    #     W[0*2+0] += 1 (C11), W[1*2+1] += 1 (C22)
    
    # Let me be more careful. For a single 2x2x2 block (columns 2j, 2j+1):
    # A is 2x2 (rows i=0,1; cols j=0,1)
    # B is 2x2 (rows j=0,1; cols k=0,1)
    # C is 2x2 (rows i=0,1; cols k=0,1)
    
    # In the full problem:
    # U row index: i*m + j where i in {0,1}, j in {0,1}
    # V row index: j*p + k where j in {0,1}, k in {0,...,9}
    # W row index: k*n + i where k in {0,...,9}, i in {0,1}
    
    # For block b (b = 0,1,2,3,4), columns are 2b and 2b+1.
    # V row for (j, k) in this block: j*p + (2b + k_local) where k_local in {0,1}
    # W row for (k, i) in this block: (2b + k_local)*n + i
    
    # Strassen's 7 products for 2x2x2:
    # Let's define them with local indices.
    # A[i][j], B[j][k], C[i][k] for i,j,k in {0,1}
    
    # M1 = A[0][0] * (B[0][0] - B[1][1])
    # M2 = (A[0][0] + A[1][1]) * B[1][1]
    # M3 = (A[1][1] - A[0][0]) * B[0][0]
    # M4 = A[1][1] * (B[1][0] - B[0][0])
    # M5 = (A[0][0] + A[0][1]) * B[1][1]
    # M6 = (A[1][1] - A[1][0]) * (B[0][0] + B[0][1])
    # M7 = (A[0][0] - A[1][1]) * (B[1][0] + B[1][1])
    
    # C[0][0] = M1 + M2 - M4 + M7
    # C[0][1] = M3 + M5
    # C[1][0] = M2 + M6
    # C[1][1] = M1 - M3 + M4 + M5
    
    # Define the 7 products as tuples (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: dict from (i,j) to coeff
    # v_coeffs: dict from (j,k) to coeff
    # w_coeffs: dict from (i,k) to coeff
    
    strassen_products = [
        # M1
        ({(0,0): 1}, {(0,0): 1, (1,1): -1}, {(0,0): 1, (1,1): 1}),
        # M2
        ({(0,0): 1, (1,1): 1}, {(1,1): 1}, {(0,0): 1, (1,0): 1}),
        # M3
        ({(1,1): 1, (0,0): -1}, {(0,0): 1}, {(0,1): 1, (1,1): -1}),
        # M4
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): -1, (1,1): 1}),
        # M5
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,1): 1, (1,1): 1}),
        # M6
        ({(1,1): 1, (1,0): -1}, {(0,0): 1, (0,1): 1}, {(1,0): 1}),
        # M7
        ({(0,0): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    for b in range(5):  # 5 column blocks
        for prod_idx in range(7):
            col = b * 7 + prod_idx  # column index in the rank-35 decomposition
            u_coeffs, v_coeffs, w_coeffs = strassen_products[prod_idx]
            
            for (i, j), coeff in u_coeffs.items():
                U[i * m + j][col] += coeff
            
            for (j, k_local), coeff in v_coeffs.items():
                k_global = 2 * b + k_local
                V[j * p + k_global][col] += coeff
            
            for (i, k_local), coeff in w_coeffs.items():
                k_global = 2 * b + k_local
                W[k_global * n + i][col] += coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
