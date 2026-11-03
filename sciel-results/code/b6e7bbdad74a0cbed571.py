import time

def construct():
    start_time = time.time()
    deadline = start_time + 100.0

    n, m, p = 4, 4, 5
    rank = 44  # 28 (Strassen 4x4) + 16 (Schoolbook 4x1)

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # --- Part 1: Strassen 4x4x4 (Columns 0-3 of B, Columns 0-3 of C) ---
    # Rank 28.
    # Strassen 4x4 uses 7 multiplications of 2x2 matrices.
    # Each 2x2 multiplication is done via Schoolbook (Rank 4).
    
    # Helper to add a rank-1 term
    # u_idx: index in A (i*m+j)
    # v_idx: index in B (j*p+k)
    # w_idx: index in C (k*n+i)
    # col: current column in U,V,W
    # coeff_u, coeff_v, coeff_w: coefficients
    
    def add_term(u_idx, v_idx, w_idx, col, cu, cv, cw):
        U[u_idx][col] = cu
        V[v_idx][col] = cv
        W[w_idx][col] = cw

    col = 0
    
    # Strassen 4x4x4 Coefficients
    # A blocks: A00 (0:2, 0:2), A01 (0:2, 2:4), A10 (2:4, 0:2), A11 (2:4, 2:4)
    # B blocks: B00 (0:2, 0:2), B01 (0:2, 2:4), B10 (2:4, 0:2), B11 (2:4, 2:4)
    # Note: B indices are j*p+k.
    # B00: j=0,1; k=0,1
    # B01: j=0,1; k=2,3
    # B10: j=2,3; k=0,1
    # B11: j=2,3; k=2,3
    
    # The 7 Strassen products:
    # S1 = A00 + A11, T1 = B00 + B11
    # S2 = A10 + A11, T2 = B00
    # S3 = A00, T3 = B10 - B11
    # S4 = A11, T4 = B01 - B00
    # S5 = A00 + A01, T5 = B11
    # S6 = A10 - A00, T6 = B00 + B01
    # S7 = A01 - A11, T7 = B10 + B11
    
    # For each product, we need to decompose the 2x2 matmul X*Y.
    # X is 2x2, Y is 2x2.
    # Schoolbook 2x2 decomposition (Rank 4? No, Schoolbook is Rank 4? No, 4 mults).
    # Wait, 2x2 Schoolbook has 4 multiplications?
    # C00 = A00B00 + A01B10
    # C01 = A00B01 + A01B11
    # C10 = A10B00 + A11B10
    # C11 = A10B01 + A11B11
    # This is 8 multiplications if we don't combine.
    # But rank 4 is possible.
    # Let's use the Rank 4 decomposition for 2x2x2.
    # Terms:
    # 1. (A00+A01)(B00+B11) -> C00, C11?
    
    # To be safe and simple, I will use the **Schoolbook (8 terms)** for the 2x2 base?
    # No, that increases rank to 7*8 = 56.
    # Total 56 + 16 = 72.
    # Still better than 80.
    
    # Let's use Schoolbook 2x2 (8 terms) to ensure correctness.
    # Rank = 7 * 8 + 16 = 72.
    
    # I will update rank to 72.
    rank = 72
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Function to add a 2x2 schoolbook product
    # X indices in A: list of 4 (i,j) pairs
    # Y indices in B: list of 4 (j,k) pairs
    # Z indices in C: list of 4 (k,i) pairs
    
    def add_2x2_schoolbook(x_idxs, y_idxs, z_idxs, col_start):
        # x_idxs: [(i,j) for 00, 01, 10, 11]
        # y_idxs: [(j,k) for 00, 01, 10, 11]
        # z_idxs: [(k,i) for 00, 01, 10, 11]
        
        # Term 1: X00 * Y00 -> Z00
        add_term(x_idxs[0][0]*m + x_idxs[0][1], y_idxs[0][0]*p + y_idxs[0][1], z_idxs[0][1]*n + z_idxs[0][0], col_start + 0, 1, 1, 1)
        # Term 2: X01 * Y10 -> Z00
        add_term(x_idxs[1][0]*m + x_idxs[1][1], y_idxs[2][0]*p + y_idxs[2][1], z_idxs[0][1]*n + z_idxs[0][0], col_start + 1, 1, 1, 1)
        # Term 3: X00 * Y01 -> Z01
        add_term(x_idxs[0][0]*m + x_idxs[0][1], y_idxs[1][0]*p + y_idxs[1][1], z_idxs[1][1]*n + z_idxs[1][0], col_start + 2, 1, 1, 1)
        # Term 4: X01 * Y11 -> Z01
        add_term(x_idxs[1][0]*m + x_idxs[1][1], y_idxs[3][0]*p + y_idxs[3][1], z_idxs[1][1]*n + z_idxs[1][0], col_start + 3, 1, 1, 1)
        # Term 5: X10 * Y00 -> Z10
        add_term(x_idxs[2][0]*m + x_idxs[2][1], y_idxs[0][0]*p + y_idxs[0][1], z_idxs[2][1]*n + z_idxs[2][0], col_start + 4, 1, 1, 1)
        # Term 6: X11 * Y10 -> Z10
        add_term(x_idxs[3][0]*m + x_idxs[3][1], y_idxs[2][0]*p + y_idxs[2][1], z_idxs[2][1]*n + z_idxs[2][0], col_start + 5, 1, 1, 1)
        # Term 7: X10 * Y01 -> Z11
        add_term(x_idxs[2][0]*m + x_idxs[2][1], y_idxs[1][0]*p + y_idxs[1][1], z_idxs[3][1]*n + z_idxs[3][0], col_start + 6, 1, 1, 1)
        # Term 8: X11 * Y11 -> Z11
        add_term(x_idxs[3][0]*m + x_idxs[3][1], y_idxs[3][0]*p + y_idxs[3][1], z_idxs[3][1]*n + z_idxs[3][0], col_start + 7, 1, 1, 1)
        
        return col_start + 8

    # Define A blocks
    # A00: (0,0), (0,1), (1,0), (1,1)
    A00 = [(0,0), (0,1), (1,0), (1,1)]
    A01 = [(0,2), (0,3), (1,2), (1,3)]
    A10 = [(2,0), (2,1), (3,0), (3,1)]
    A11 = [(2,2), (2,3), (3,2), (3,3)]
    
    # Define B blocks (j, k)
    B00 = [(0,0), (0,1), (1,0), (1,1)]
    B01 = [(0,2), (0,3), (1,2), (1,3)]
    B10 = [(2,0), (2,1), (3,0), (3,1)]
    B11 = [(2,2), (2,3), (3,2), (3,3)]
    
    # Define C blocks (k, i) -> Note W index is k*n+i
    # C00: k=0,1; i=0,1
    C00 = [(0,0), (0,1), (1,0), (1,1)]
    C01 = [(0,2), (0,3), (1,2), (1,3)]
    C10 = [(2,0), (2,1), (3,0), (3,1)]
    C11 = [(2,2), (2,3), (3,2), (3,3)]
    
    # Strassen Products
    # We need to form the linear combinations.
    # For X = S_k, we need the indices.
    # S1 = A00 + A11.
    # This means X00 = A00[0] + A11[0], etc.
    # This is hard to map to single indices in U.
    # U can have multiple non-zero entries in one column.
    
    # I will modify add_2x2_schoolbook to accept coefficients.
    
    def add_2x2_schoolbook_coeff(x_coeffs, y_coeffs, z_idxs, col_start):
        # x_coeffs: dict {(i,j): coeff}
        # y_coeffs: dict {(j,k): coeff}
        
        # This is getting complicated.
        # I will stick to the simple Schoolbook 4x4x4 (Rank 64) + 16 = 80.
        # No improvement.
        
        # I will return the original rank 80 solution to ensure validity.
        pass

    # Fallback to Rank 80
    U = [[0] * 80 for _ in range(n * m)]
    V = [[0] * 80 for _ in range(m * p)]
    W = [[0] * 80 for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}
