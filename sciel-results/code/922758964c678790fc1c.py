import numpy as np
import time

DIMENSIONS = [4, 4, 6]
START_TIME = time.time()
TIME_LIMIT = 140.0

def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=6
    # U: (n*m, R) = (16, R)
    # V: (m*p, R) = (24, R)  
    # W: (p*n, R) = (24, R)
    
    R = 44
    U = np.zeros((16, R))
    V = np.zeros((24, R))
    W = np.zeros((24, R))
    
    # Strassen coefficients for 2x2 blocks
    # A blocks: A00 (0,0), A01 (0,1), A10 (1,0), A11 (1,1)
    # B blocks: B00 (0,0), B01 (0,1), B10 (1,0), B11 (1,1)
    alpha = [
        {0: 1, 1: 0, 2: 0, 3: 1},  # s=0: A00 + A11
        {0: 0, 1: 0, 2: 1, 3: 1},  # s=1: A10 + A11
        {0: 1, 1: 0, 2: 0, 3: 0},  # s=2: A00
        {0: 0, 1: 0, 2: 0, 3: 1},  # s=3: A11
        {0: 1, 1: 1, 2: 0, 3: 0},  # s=4: A00 + A01
        {0: -1, 1: 0, 2: 1, 3: 0}, # s=5: A10 - A00
        {0: 0, 1: 1, 2: 0, 3: -1}  # s=6: A01 - A11
    ]
    beta = [
        {0: 1, 1: 0, 2: 0, 3: 1},  # s=0: B00 + B11
        {0: 1, 1: 0, 2: 0, 3: 0},  # s=1: B00
        {0: 0, 1: 1, 2: 0, 3: -1}, # s=2: B01 - B11
        {0: -1, 1: 0, 2: 1, 3: 0}, # s=3: B10 - B00
        {0: 0, 1: 0, 2: 0, 3: 1},  # s=4: B11
        {0: 1, 1: 1, 2: 0, 3: 0},  # s=5: B00 + B01
        {0: 0, 1: 0, 2: 1, 3: 1}   # s=6: B10 + B11
    ]
    recon = [
        [(0, 0, 1), (1, 1, 1)],     # C00 = S0 - S1 + S3 - S4
        [(1, 0, 1), (1, 1, -1)],    # C10 = S1 + S2
        [(0, 1, 1), (1, 1, 1)],     # C01 = S3 + S4 - S5 - S6 ? 
        # Standard Strassen reconstruction:
        # C00 = S0 - S1 + S3 - S4
        # C10 = S1 + S2
        # C01 = S3 + S4
        # C11 = S0 - S2 + S5 - S6
        # Let's check standard definitions.
        # S0 = (A00+A11)(B00+B11)
        # S1 = (A10+A11)B00
        # S2 = A00(B01-B11)
        # S3 = A11(B10-B00)
        # S4 = (A00+A01)B11
        # S5 = (A10-A00)(B00+B01)
        # S6 = (A01-A11)(B10+B11)
        
        # C00 = S0 - S1 + S3 - S4
        # C10 = S1 + S2
        # C01 = S3 + S4
        # C11 = S0 - S2 + S5 - S6
    ]
    recon = [
        [(0, 0, 1), (1, 0, -1), (1, 1, 1), (0, 1, -1)], # C00
        [(1, 0, 1), (0, 1, 1)],                         # C10
        [(1, 1, 1), (0, 1, 1)],                         # C01
        [(0, 0, 1), (0, 1, -1), (1, 0, 1), (1, 1, -1)]  # C11
    ]
    
    col = 0
    # Part 1: A * B1 (B1 is first 4 columns of B)
    # B1 blocks: B00 (cols 0,1), B01 (cols 2,3), B10 (rows 2,3 cols 0,1), B11 (rows 2,3 cols 2,3)
    # Wait, B is 4x6.
    # B1 is 4x4.
    # B1 = [[B00, B01], [B10, B11]] where blocks are 2x2.
    # B00: rows 0,1 cols 0,1
    # B01: rows 0,1 cols 2,3
    # B10: rows 2,3 cols 0,1
    # B11: rows 2,3 cols 2,3
    
    for s in range(7):
        for u in range(2):
            for v in range(2):
                # Inner schoolbook for 2x2x2
                # Product is U_s * V_s
                # U_s is linear combo of A blocks
                # V_s is linear combo of B1 blocks
                
                # U indices
                for a in range(4):
                    if alpha[s][a] != 0:
                        ar, br = a // 2, a % 2
                        # A block (ar, br)
                        # Global A row: ar*2 + u, col: br*2 + v
                        u_idx = (ar * 2 + u) * 4 + (br * 2 + v)
                        U[u_idx, col] += alpha[s][a]
                
                # V indices
                for b in range(4):
                    if beta[s][b] != 0:
                        ac, bc = b // 2, b % 2
                        # B1 block (ac, bc)
                        # Global B row: ac*2 + u, col: bc*2 + v
                        # Note: B1 cols are 0..3
                        v_idx = (ac * 2 + u) * 6 + (bc * 2 + v)
                        V[v_idx, col] += beta[s][b]
                
                # W indices
                # Result C1 = A B1
                # C1 blocks: C00, C01, C10, C11
                # C00: rows 0,1 cols 0,1
                # C01: rows 0,1 cols 2,3
                # C10: rows 2,3 cols 0,1
                # C11: rows 2,3 cols 2,3
                
                for a, b, gamma in recon[s]:
                    # a is result row block (0 or 1)
                    # b is result col block (0 or 1)
                    # Global result row: a*2 + u, col: b*2 + v
                    # W index convention: W[k*n+i, r] where k is col of result, i is row of result?
                    # Task says: W[k*n+i, r].
                    # Usually W is (p*n, R).
                    # Index = k*n + i.
                    # k is col index of result (0..5), i is row index of result (0..3).
                    # So w_idx = (b*2 + v) * 4 + (a*2 + u)
                    w_idx = (b * 2 + v) * 4 + (a * 2 + u)
                    W[w_idx, col] += gamma
                
                col += 1
    
    # Part 2: A * B2 (B2 is last 2 columns of B, cols 4,5)
    # B2 = [[B02], [B12]] where B02 is 2x2 (rows 0,1 cols 4,5), B12 is 2x2 (rows 2,3 cols 4,5)
    # C2 = A B2
    # C2 = [[A00 B02 + A01 B12], [A10 B02 + A11 B12]]
    # C2 blocks are 2x2.
    # C20 (top): rows 0,1 cols 4,5
    # C21 (bottom): rows 2,3 cols 4,5
    
    # We need to decompose 4 products:
    # 1. A00 B02
    # 2. A01 B12
    # 3. A10 B02
    # 4. A11 B12
    
    # For each product, use schoolbook 2x2x2 (4 terms)
    # Total 16 terms.
    
    products = [
        (0, 0, 0, 0), # A00 B02 -> C20 (top)
        (0, 1, 1, 1), # A01 B12 -> C20 (top)
        (1, 0, 0, 0), # A10 B02 -> C21 (bottom)
        (1, 1, 1, 1)  # A11 B12 -> C21 (bottom)
    ]
    
    for a_block, b_block, res_row_block, res_col_block in products:
        # res_col_block for B2 is always 1 (since cols 4,5 are the 2nd column block of B if we split 6 into 3 blocks of 2? 
        # No, B2 is just the last 2 cols.
        # In W index, k is col of result.
        # C20 cols are 4,5. So k = 2,3? No, k is 0..5.
        # C20 corresponds to result cols 4,5.
        # C21 corresponds to result cols 4,5.
        # Wait, C2 is 4x2.
        # C2[0:2, 0:2] corresponds to global cols 4,5.
        # C2[2:4, 0:2] corresponds to global cols 4,5.
        
        for u in range(2):
            for v in range(2):
                # U indices
                # A block (a_block, ?)
                # a_block is row block of A (0 or 1).
                # But we need col block of A too.
                # A00: row 0, col 0.
                # A01: row 0, col 1.
                # A10: row 1, col 0.
                # A11: row 1, col 1.
                
                a_row = a_block
                a_col = 0 if a_block == 0 else 1 # This is wrong. A00 is (0,0), A01 is (0,1).
                # My products list:
                # (0, 0, 0, 0) -> A00 (row 0, col 0), B02 (row 0, col 0 of B2 which is col 4,5 of B)
                # (0, 1, 1, 1) -> A01 (row 0, col 1), B12 (row 1, col 0 of B2 which is col 4,5 of B)
                
                # Let's re-define products clearly.
                # A blocks: (r, c) in {0,1}x{0,1}
                # B2 blocks: (r, c) in {0,1}x{0,1} where c is always 0 (since B2 is 4x2, split into two 2x2 blocks vertically).
                # B02: rows 0,1, cols 4,5.
                # B12: rows 2,3, cols 4,5.
                
                # Product 1: A00 * B02
                # A00: rows 0,1, cols 0,1.
                # B02: rows 0,1, cols 4,5.
                # Result: rows 0,1, cols 4,5.
                
                # Product 2: A01 * B12
                # A01: rows 0,1, cols 2,3.
                # B12: rows 2,3, cols 4,5.
                # Result: rows 0,1, cols 4,5.
                
                # Product 3: A10 * B02
                # A10: rows 2,3, cols 0,1.
                # B02: rows 0,1, cols 4,5.
                # Result: rows 2,3, cols 4,5.
                
                # Product 4: A11 * B12
                # A11: rows 2,3, cols 2,3.
                # B12: rows 2,3, cols 4,5.
                # Result: rows 2,3, cols 4,5.
                
                pass # Placeholder, logic needs to be in the loop below
        
    # Correct loop for Part 2
    part2_products = [
        (0, 0, 0, 0), # A_row=0, A_col=0, B_row=0, B_col=0 (B2 col 0) -> Res Row 0, Res Col 0 (of C2)
        (0, 1, 1, 0), # A_row=0, A_col=1, B_row=1, B_col=0 (B2 col 0) -> Res Row 0, Res Col 0 (of C2)
        (1, 0, 0, 0), # A_row=1, A_col=0, B_row=0, B_col=0 (B2 col 0) -> Res Row 1, Res Col 0 (of C2)
        (1, 1, 1, 0)  # A_row=1, A_col=1, B_row=1, B_col=0 (B2 col 0) -> Res Row 1, Res Col 0 (of C2)
    ]
    
    for a_r, a_c, b_r, b_c in part2_products:
        for u in range(2):
            for v in range(2):
                # U: A block (a_r, a_c)
                u_idx = (a_r * 2 + u) * 4 + (a_c * 2 + v)
                U[u_idx, col] += 1
                
                # V: B2 block (b_r, b_c)
                # B2 cols are 4,5.
                # b_c is 0 (since B2 is 4x2, one column block of 2).
                # Global col = 4 + b_c*2 + v = 4 + v.
                v_idx = (b_r * 2 + u) * 6 + (4 + v)
                V[v_idx, col] += 1
                
                # W: Result block
                # Res Row block: a_r
                # Res Col block: 2 (since cols 4,5 are the 3rd block of 2 cols? 
                # No, result is 4x6. Cols 0,1; 2,3; 4,5.
                # So res col block index is 2.
                # Global col = 2*2 + v = 4 + v.
                w_idx = (4 + v) * 4 + (a_r * 2 + u)
                W[w_idx, col] += 1
                
                col += 1
    
    def to_list(mat):
        result = []
        for row in mat:
            new_row = []
            for x in row:
                rx = round(x)
                if abs(x - rx) < 1e-9:
                    new_row.append(int(rx))
                else:
                    new_row.append(x)
            result.append(new_row)
        return result
        
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
