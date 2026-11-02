"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    # Strassen-style block decomposition exploiting the 2x2 block structure.
    # A is 2x2, B is 2x10. Split B into B_left (2x2) and B_right (2x8).
    # Product = A @ B = [A@B_left, A@B_right].
    # 
    # For A@B_left (2x2 by 2x2), use Strassen's algorithm (rank 7).
    # For A@B_right (2x2 by 2x8), we can split B_right into 4 blocks of 2x2:
    #   B_right = [B1, B2, B3, B4] each 2x2.
    #   A@B_right = [A@B1, A@B2, A@B3, A@B4]
    #   Each A@Bi can use Strassen (rank 7 each), total 28.
    #   Plus A@B_left uses Strassen (rank 7).
    #   Total rank = 7 + 4*7 = 35.
    #
    # Actually, let's think more carefully. The standard Strassen decomposition for 2x2 x 2x2
    # has rank 7. For 2x2 x 2x10, we can view it as 2x2 x (2x5 blocks of 2x2).
    # That gives 5 * 7 = 35.
    #
    # Alternatively, we can use a more direct approach. Let me implement Strassen for the 2x2x2x2 part
    # and replicate it for each 2x2 block of the 10-wide matrix.
    #
    # Strassen's 7 multiplications for 2x2 x 2x2:
    # Let A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # M1 = (a11 + a22)(b11 + b22)
    # M2 = (a21 + a22) b11
    # M3 = a11 (b12 - b22)
    # M4 = a22 (b21 - b11)
    # M5 = (a11 + a12) b22
    # M6 = (a21 - a11) (b11 + b12)
    # M7 = (a12 - a22) (b21 + b22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # For the full problem:
    # n=2, m=2, p=10
    # U has shape (n*m, rank) = (4, rank)
    # V has shape (m*p, rank) = (20, rank)
    # W has shape (p*n, rank) = (20, rank)
    #
    # U rows: (0,0), (0,1), (1,0), (1,1) -> indices 0,1,2,3
    # V rows: (0,0)..(0,9), (1,0)..(1,9) -> indices 0..19
    # W rows: (0,0),(0,1),(1,0),(1,1),...,(9,0),(9,1) -> indices 0..19 (k*n+i)
    #
    # For each of the 5 blocks of 2 columns in B, apply Strassen.
    # Block t (t=0..4) covers B columns 2t, 2t+1.
    #
    # For block t, the 7 Strassen terms contribute to:
    # U factors (4 entries each), V factors (4 entries each for the 2x2 block), W factors (4 entries each).
    
    n, m, p = DIMENSIONS
    num_blocks = p // 2  # 5 blocks of 2x2
    
    # Strassen coefficients for the 7 multiplications
    # Each multiplication t has (U_vec, V_vec, W_vec) of length 4
    # U_vec: coefficients for a11, a12, a21, a22
    # V_vec: coefficients for b11, b12, b21, b22 (within the 2x2 block)
    # W_vec: coefficients for C11, C12, C21, C22 (within the 2x2 output block)
    
    strassen_terms = [
        # M1 = (a11 + a22)(b11 + b22)
        # U: [1, 0, 0, 1], V: [1, 0, 0, 1], W: [1, 0, 0, 1]
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2 = (a21 + a22) b11
        # U: [0, 0, 1, 1], V: [1, 0, 0, 0], W: [0, 0, 1, 0]
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        # M3 = a11 (b12 - b22)
        # U: [1, 0, 0, 0], V: [0, 1, 0, -1], W: [0, 1, 0, 0]
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),
        # M4 = a22 (b21 - b11)
        # U: [0, 0, 0, 1], V: [-1, 0, 1, 0], W: [1, 0, 0, 0]
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 0, 0]),
        # M5 = (a11 + a12) b22
        # U: [1, 1, 0, 0], V: [0, 0, 0, 1], W: [-1, 1, 0, 0]
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # M6 = (a21 - a11) (b11 + b12)
        # U: [-1, 0, 1, 0], V: [1, 1, 0, 0], W: [0, 0, 0, 1]
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # M7 = (a12 - a22) (b21 + b22)
        # U: [0, 1, 0, -1], V: [0, 0, 1, 1], W: [1, 0, 0, 0]
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    
    rank = num_blocks * 7  # 35
    U = [[0] * rank for _ in range(n * m)]  # (4, 35)
    V = [[0] * rank for _ in range(m * p)]  # (20, 35)
    W = [[0] * rank for _ in range(p * n)]  # (20, 35)
    
    col = 0
    for t in range(num_blocks):
        for s in range(7):
            u_vec, v_vec, w_vec = strassen_terms[s]
            # U indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
            U[0][col] += u_vec[0]
            U[1][col] += u_vec[1]
            U[2][col] += u_vec[2]
            U[3][col] += u_vec[3]
            
            # V indices for block t: rows 0..3 correspond to B[0, 2t], B[0, 2t+1], B[1, 2t], B[1, 2t+1]
            # V row index = j * p + k, where j in {0,1}, k in {0,...,9}
            # For block t, k = 2t, 2t+1
            v_base = 0 * p + 2 * t  # row for j=0, k=2t
            V[v_base][col] += v_vec[0]       # b11
            V[v_base + 1][col] += v_vec[1]   # b12
            V[v_base + p][col] += v_vec[2]   # b21 (j=1, k=2t)
            V[v_base + p + 1][col] += v_vec[3]  # b22 (j=1, k=2t+1)
            
            # W indices: k*n + i, where k in {0,...,9}, i in {0,1}
            # For block t, output columns 2t, 2t+1
            # W rows: (2t)*n + 0, (2t)*n + 1, (2t+1)*n + 0, (2t+1)*n + 1
            w_base = (2 * t) * n
            W[w_base + 0][col] += w_vec[0]   # C11 at (2t, 0)
            W[w_base + 1][col] += w_vec[1]   # C12 at (2t, 1)
            W[w_base + n + 0][col] += w_vec[2]  # C21 at (2t+1, 0)
            W[w_base + n + 1][col] += w_vec[3]  # C22 at (2t+1, 1)
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
