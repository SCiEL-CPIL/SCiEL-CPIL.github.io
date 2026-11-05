"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-28 decomposition for 2x3 by 3x5 matrix multiplication.
    
    Strategy:
    1. Split A (2x3) into A_L (2x1) and A_R (2x2)
    2. Split B (3x5) into B_top (1x5) and B_bottom (2x5)
    3. C = A_L * B_top + A_R * B_bottom
    
    Term 1: A_L (2x1) * B_top (1x5) -> rank 10 (schoolbook)
    Term 2: A_R (2x2) * B_bottom (2x5) -> rank 18
      - Split B_bottom columns into B_12 (2x2), B_34 (2x2), B_5 (2x1)
      - A_R * B_12: Strassen (7 mults)
      - A_R * B_34: Strassen (7 mults)  
      - A_R * B_5: Schoolbook (4 mults)
    
    Total rank = 10 + 7 + 7 + 4 = 28
    
    The output factor follows W[k*n+i, r] convention.
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=5
    
    rank = 28
    U = [[0.0] * rank for _ in range(n * m)]  # 6 x 28
    V = [[0.0] * rank for _ in range(m * p)]  # 15 x 28
    W = [[0.0] * rank for _ in range(p * n)]  # 10 x 28
    
    def add_factor(col, u_entries, v_entries, w_entries):
        """Add one rank-1 factor at column col."""
        for (row, val) in u_entries:
            U[row][col] += val
        for (row, val) in v_entries:
            V[row][col] += val
        for (row, val) in w_entries:
            W[row][col] += val
    
    # A[i,j], B[j,k], C[i,k]
    # U indices: i*m+j for i in [0,1], j in [0,1,2]
    # V indices: j*p+k for j in [0,1,2], k in [0,1,2,3,4]
    # W indices: k*n+i for k in [0,1,2,3,4], i in [0,1]
    
    col = 0
    
    # ====================
    # Term 1: A_L (2x1) * B_top (1x5) - schoolbook
    # A_L = A[:, 0] = [a00, a10]
    # B_top = B[0, :] = [b00, b01, b02, b03, b04]
    # C[:, 0:5] += A_L * B_top
    # For each k in 0..4: C[i,k] += a_i0 * b_0k
    # ====================
    for k in range(5):
        # C[0,k] += a00 * b0k
        add_factor(col,
                   [(0, 1.0)],       # U[0,0] = a00
                   [(0*5+k, 1.0)],   # V[0,k] = b0k
                   [(k*n+0, 1.0)])   # W[k,0]
        col += 1
        # C[1,k] += a10 * b0k
        add_factor(col,
                   [(1, 1.0)],       # U[1,0] = a10
                   [(0*5+k, 1.0)],   # V[0,k] = b0k
                   [(k*n+1, 1.0)])   # W[k,1]
        col += 1
    
    # ====================
    # Term 2: A_R (2x2) * B_bottom (2x5)
    # A_R = A[:, 1:3] = [[a01, a02], [a11, a12]]
    # B_bottom = B[1:3, :] = [[b10, b11, b12, b13, b14],
    #                          [b20, b21, b22, b23, b24]]
    # C[:, :] += A_R * B_bottom
    #
    # Split B_bottom columns: B_12 (cols 0,1), B_34 (cols 2,3), B_5 (col 4)
    # ====================
    
    # A_R = [[a01, a02],
    #        [a11, a12]]
    
    # ---- A_R * B_12 using Strassen ----
    # B_12 = [[b10, b11],
    #         [b20, b21]]
    #
    # Strassen for 2x2 * 2x2:
    # M1 = (a01 + a12) * (b20 + b21)  -- wait, let me be careful
    #
    # Standard Strassen:
    # C = A*B where A = [[p, q], [r, s]], B = [[w, x], [y, z]]
    # M1 = (p+s)*(w+z)
    # M2 = (r+s)*w
    # M3 = p*(x-z)
    # M4 = s*(y-w)
    # M5 = (p+q)*z
    # M6 = (r-q)*(w+x)
    # M7 = (q-s)*(y+z)
    # C11 = M1 - M4 + M5 + M6
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M2 - M7
    
    # Here: p=a01, q=a02, r=a11, s=a12
    #        w=b10, x=b11, y=b20, z=b21
    
    # M1 = (a01+a12)*(b10+b21)
    add_factor(col,
               [(0*3+1, 1.0), (1*3+2, 1.0)],  # U[a01], U[a12]
               [(1*5+0, 1.0), (2*5+1, 1.0)],  # V[b10], V[b21]
               [(0*n+0, 1.0), (0*n+1, 1.0), (1*n+0, 1.0), (1*n+1, 1.0)])  # all C entries for col 0,1 rows 0,1
    col += 1
    
    # M2 = (a11+a12)*b10
    add_factor(col,
               [(1*3+1, 1.0), (1*3+2, 1.0)],  # U[a11], U[a12]
               [(1*5+0, 1.0)],                 # V[b10]
               [(0*n+1, 1.0), (1*n+1, 1.0)])  # C[0,1], C[1,1]
    col += 1
    
    # M3 = a01*(b11-b20)
    add_factor(col,
               [(0*3+1, 1.0)],                 # U[a01]
               [(1*5+1, 1.0), (2*5+0, -1.0)],  # V[b11] - V[b20]
               [(0*n+1, 1.0), (1*n+1, 1.0)])  # C[0,1], C[1,1]
    col += 1
    
    # M4 = a12*(b20-b10)
    add_factor(col,
               [(1*3+2, 1.0)],                 # U[a12]
               [(2*5+0, 1.0), (1*5+0, -1.0)],  # V[b20] - V[b10]
               [(0*n+1, 1.0), (1*n+1, 1.0)])  # C[0,1], C[1,1]
    col += 1
    
    # M5 = (a01+a02)*b21
    add_factor(col,
               [(0*3+1, 1.0), (0*3+2, 1.0)],  # U[a01], U[a02]
               [(2*5+1, 1.0)],                 # V[b21]
               [(0*n+0, 1.0), (0*n+1, 1.0), (1*n+0, 1.0), (1*n+1, 1.0)])
    col += 1
    
    # M6 = (a11-a02)*(b10+b11)
    add_factor(col,
               [(1*3+1, 1.0), (0*3+2, -1.0)],  # U[a11] - U[a02]
               [(1*5+0, 1.0), (1*5+1, 1.0)],   # V[b10] + V[b11]
               [(0*n+0, 1.0), (0*n+1, 1.0), (1*n+0, 1.0), (1*n+1, 1.0)])
    col += 1
    
    # M7 = (a02-a12)*(b20+b21)
    add_factor(col,
               [(0*3+2, 1.0), (1*3+2, -1.0)],  # U[a02] - U[a12]
               [(2*5+0, 1.0), (2*5+1, 1.0)],   # V[b20] + V[b21]
               [(0*n+0, 1.0), (0*n+1, 1.0), (1*n+0, 1.0), (1*n+1, 1.0)])
    col += 1
    
    # ---- A_R * B_34 using Strassen ----
    # B_34 = [[b12, b13],
    #         [b22, b23]]
    # w=b12, x=b13, y=b22, z=b23
    # Output goes to C[:, 2:4]
    
    # M1' = (a01+a12)*(b12+b23)
    add_factor(col,
               [(0*3+1, 1.0), (1*3+2, 1.0)],  # U[a01], U[a12]
               [(1*5+2, 1.0), (2*5+3, 1.0)],  # V[b12], V[b23]
               [(2*n+0, 1.0), (3*n+0, 1.0), (2*n+1, 1.0), (3*n+1, 1.0)])
    col += 1
    
    # M2' = (a11+a12)*b12
    add_factor(col,
               [(1*3+1, 1.0), (1*3+2, 1.0)],  # U[a11], U[a12]
               [(1*5+2, 1.0)],                 # V[b12]
               [(2*n+1, 1.0), (3*n+1, 1.0)])  # C[0,3], C[1,3]
    col += 1
    
    # M3' = a01*(b13-b22)
    add_factor(col,
               [(0*3+1, 1.0)],                 # U[a01]
               [(1*5+3, 1.0), (2*5+2, -1.0)],  # V[b13] - V[b22]
               [(2*n+1, 1.0), (3*n+1, 1.0)])  # C[0,3], C[1,3]
    col += 1
    
    # M4' = a12*(b22-b12)
    add_factor(col,
               [(1*3+2, 1.0)],                 # U[a12]
               [(2*5+2, 1.0), (1*5+2, -1.0)],  # V[b22] - V[b12]
               [(2*n+1, 1.0), (3*n+1, 1.0)])  # C[0,3], C[1,3]
    col += 1
    
    # M5' = (a01+a02)*b23
    add_factor(col,
               [(0*3+1, 1.0), (0*3+2, 1.0)],  # U[a01], U[a02]
               [(2*5+3, 1.0)],                 # V[b23]
               [(2*n+0, 1.0), (3*n+0, 1.0), (2*n+1, 1.0), (3*n+1, 1.0)])
    col += 1
    
    # M6' = (a11-a02)*(b12+b13)
    add_factor(col,
               [(1*3+1, 1.0), (0*3+2, -1.0)],  # U[a11] - U[a02]
               [(1*5+2, 1.0), (1*5+3, 1.0)],   # V[b12] + V[b13]
               [(2*n+0, 1.0), (3*n+0, 1.0), (2*n+1, 1.0), (3*n+1, 1.0)])
    col += 1
    
    # M7' = (a02-a12)*(b22+b23)
    add_factor(col,
               [(0*3+2, 1.0), (1*3+2, -1.0)],  # U[a02] - U[a12]
               [(2*5+2, 1.0), (2*5+3, 1.0)],   # V[b22] + V[b23]
               [(2*n+0, 1.0), (3*n+0, 1.0), (2*n+1, 1.0), (3*n+1, 1.0)])
    col += 1
    
    # ---- A_R * B_5 using schoolbook ----
    # B_5 = [b14, b24]^T (2x1)
    # C[:, 4] += A_R * B_5
    # C[0,4] = a01*b14 + a02*b24
    # C[1,4] = a11*b14 + a12*b24
    
    # a01 * b14
    add_factor(col,
               [(0*3+1, 1.0)],
               [(1*5+4, 1.0)],
               [(4*n+0, 1.0)])
    col += 1
    
    # a02 * b24
    add_factor(col,
               [(0*3+2, 1.0)],
               [(2*5+4, 1.0)],
               [(4*n+0, 1.0)])
    col += 1
    
    # a11 * b14
    add_factor(col,
               [(1*3+1, 1.0)],
               [(1*5+4, 1.0)],
               [(4*n+1, 1.0)])
    col += 1
    
    # a12 * b24
    add_factor(col,
               [(1*3+2, 1.0)],
               [(2*5+4, 1.0)],
               [(4*n+1, 1.0)])
    col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    # Convert to integer entries where possible
    U = [[int(round(v)) if abs(v - round(v)) < 1e-9 else v for v in row] for row in U]
    V = [[int(round(v)) if abs(v - round(v)) < 1e-9 else v for v in row] for row in V]
    W = [[int(round(v)) if abs(v - round(v)) < 1e-9 else v for v in row] for row in W]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

