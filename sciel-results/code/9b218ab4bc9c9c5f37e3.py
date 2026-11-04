# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = 4, 4, 9
    start_time = time.time()
    
    # We will build the decomposition in two parts:
    # 1. 4x4x4 part using 8 Strassen 2x2x2 blocks (rank 56)
    # 2. 4x4x5 part using schoolbook (rank 80)
    # Total rank 136.
    
    rank = 136
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    
    # Part 1: 4x4x4
    # A is 4x4, B_first is 4x4 (cols 0..3 of B)
    # Split into 2x2 blocks
    # A00 (rows 0-1, cols 0-1), A01 (rows 0-1, cols 2-3)
    # A10 (rows 2-3, cols 0-1), A11 (rows 2-3, cols 2-3)
    # B00 (rows 0-1, cols 0-1), B01 (rows 0-1, cols 2-3)
    # B10 (rows 2-3, cols 0-1), B11 (rows 2-3, cols 2-3)
    
    # Define Strassen 2x2x2 terms
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # We need to add 7 terms for each of the 8 block products?
    # No, we just use the 7 terms for each 2x2x2 multiplication.
    # The 8 products are:
    # P1 = A00 * B00
    # P2 = A01 * B10
    # P3 = A00 * B01
    # P4 = A01 * B11
    # P5 = A10 * B00
    # P6 = A11 * B10
    # P7 = A10 * B01
    # P8 = A11 * B11
    
    # Each Pi is a 2x2x2 multiplication.
    # We can use Strassen for each Pi.
    
    def add_strassen_block(r, a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        """
        Adds 7 rank-1 terms for a 2x2x2 multiplication.
        a_rows, a_cols, b_rows, b_cols, c_rows, c_cols are lists of global indices.
        """
        # M1 = (A00+A11)(B00+B11)
        # A00 is at (a_rows[0], a_cols[0]) and (a_rows[0], a_cols[1])? 
        # No, A00 is a 2x2 block.
        # a_rows[0], a_rows[1] are the row indices.
        # a_cols[0], a_cols[1] are the col indices.
        
        # Term 1: (A00 + A11) * (B00 + B11)
        # This expands to A00B00 + A00B11 + A11B00 + A11B11
        # Wait, Strassen combines these into one multiplication.
        # The rank-1 terms are:
        # 1. (A00+A11) * (B00+B11)
        # 2. (A10+A11) * B00
        # 3. A00 * (B10-B11)
        # 4. A11 * (B01-B00)
        # 5. (A00+A01) * B11
        # 6. (A10-A00) * (B00+B01)
        # 7. (A01-A11) * (B10+B11)
        
        # We need to add these to U, V, W.
        # U index: i*m + j
        # V index: j*p + k
        # W index: k*n + i
        
        # Let's define the local indices for the 2x2 block
        # Local A: rows 0,1; cols 0,1
        # Local B: rows 0,1; cols 0,1
        # Local C: rows 0,1; cols 0,1
        
        # Global mapping:
        # A_local[0,0] -> U[a_rows[0]*m + a_cols[0]]
        # A_local[0,1] -> U[a_rows[0]*m + a_cols[1]]
        # A_local[1,0] -> U[a_rows[1]*m + a_cols[0]]
        # A_local[1,1] -> U[a_rows[1]*m + a_cols[1]]
        
        # B_local[0,0] -> V[b_rows[0]*p + b_cols[0]]
        # ...
        
        # C_local[0,0] -> W[c_rows[0]*n + c_cols[0]]
        # ...
        
        # Term 1: (A00+A11)(B00+B11)
        # U: A00=1, A11=1
        # V: B00=1, B11=1
        # W: C00=1, C01=1, C10=1, C11=1? 
        # No, W collects the result.
        # The rank-1 term is $u \cdot v \cdot w$.
        # $u$ is the vector of coefficients for A.
        # $v$ is the vector of coefficients for B.
        # $w$ is the vector of coefficients for C.
        
        # For M1:
        # u: A00=1, A11=1
        # v: B00=1, B11=1
        # w: C00=1, C01=1, C10=1, C11=1? 
        # Wait, the product is a 2x2 matrix.
        # $M_1 = (A_{00}+A_{11})(B_{00}+B_{11})$
        # This contributes to all 4 entries of C?
        # No, $M_1$ is a 2x2 matrix.
        # $C_{00} = M_{1,00} + \dots$
        # $C_{01} = M_{1,01} + \dots$
        # So W should have 1s for all 4 entries of the local C block.
        
        # Let's verify Strassen formulas:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M1 - M2 + M6
        # C10 = M2 + M4 - M1 + M3
        # C11 = M5 + M7 - M4 + M1
        
        # So for M1, W has 1 for C00, C01, C10, C11.
        # For M4, W has 1 for C00, C10, -1 for C11, 1 for C01?
        # M4 is A11(B01-B00).
        # C00 += M4
        # C01 -= M4
        # C10 += M4
        # C11 -= M4
        
        # So W for M4 is: C00=1, C01=-1, C10=1, C11=-1.
        
        # Let's define the 7 terms explicitly.
        
        terms = [
            # (u_coeffs, v_coeffs, w_coeffs)
            # u: [A00, A01, A10, A11]
            # v: [B00, B01, B10, B11]
            # w: [C00, C01, C10, C11]
            
            # M1 = (A00+A11)(B00+B11)
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 1, 1, 1]),
            
            # M2 = (A10+A11)B00
            ([0, 0, 1, 1], [1, 0, 0, 0], [-1, 1, 1, 0]), # C00-=M2, C01+=M2, C10+=M2
            
            # M3 = A00(B10-B11)
            ([1, 0, 0, 0], [0, 0, 1, -1], [1, 1, 0, 0]), # C00+=M3, C01+=M3
            
            # M4 = A11(B01-B00)
            ([0, 0, 0, 1], [0, 1, 0, -1], [1, -1, 1, -1]), # C00+=M4, C01-=M4, C10+=M4, C11-=M4
            
            # M5 = (A00+A01)B11
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 0, 1]), # C00-=M5, C11+=M5
            
            # M6 = (A10-A00)(B00+B01)
            ([0, 0, 1, -1], [1, 1, 0, 0], [0, 1, -1, 0]), # C01+=M6, C10-=M6
            
            # M7 = (A01-A11)(B10+B11)
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 1]), # C00+=M7, C11+=M7
        ]
        
        for u_c, v_c, w_c in terms:
            # u_c[0] is A00, u_c[1] is A01, u_c[2] is A10, u_c[3] is A11
            # v_c[0] is B00, v_c[1] is B01, v_c[2] is B10, v_c[3] is B11
            # w_c[0] is C00, w_c[1] is C01, w_c[2] is C10, w_c[3] is C11
            
            # Add to U
            U[a_rows[0] * m + a_cols[0]][r] += u_c[0]
            U[a_rows[0] * m + a_cols[1]][r] += u_c[1]
            U[a_rows[1] * m + a_cols[0]][r] += u_c[2]
            U[a_rows[1] * m + a_cols[1]][r] += u_c[3]
            
            # Add to V
            V[b_rows[0] * p + b_cols[0]][r] += v_c[0]
            V[b_rows[0] * p + b_cols[1]][r] += v_c[1]
            V[b_rows[1] * p + b_cols[0]][r] += v_c[2]
            V[b_rows[1] * p + b_cols[1]][r] += v_c[3]
            
            # Add to W
            W[c_rows[0] * n + c_cols[0]][r] += w_c[0]
            W[c_rows[0] * n + c_cols[1]][r] += w_c[1]
            W[c_rows[1] * n + c_cols[0]][r] += w_c[2]
            W[c_rows[1] * n + c_cols[1]][r] += w_c[3]
            
            r += 1
        
        return r

    # Define blocks
    # A00: rows 0-1, cols 0-1
    # A01: rows 0-1, cols 2-3
    # A10: rows 2-3, cols 0-1
    # A11: rows 2-3, cols 2-3
    
    # B00: rows 0-1, cols 0-1
    # B01: rows 0-1, cols 2-3
    # B10: rows 2-3, cols 0-1
    # B11: rows 2-3, cols 2-3
    
    # C00: rows 0-1, cols 0-1
    # C01: rows 0-1, cols 2-3
    # C10: rows 2-3, cols 0-1
    # C11: rows 2-3, cols 2-3
    
    # P1 = A00 * B00 -> C00
    r = add_strassen_block(r, [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1])
    
    # P2 = A01 * B10 -> C00
    # Wait, C00 = A00B00 + A01B10
    # So P2 contributes to C00.
    # A01 is rows 0-1, cols 2-3
    # B10 is rows 2-3, cols 0-1
    r = add_strassen_block(r, [0, 1], [2, 3], [2, 3], [0, 1], [0, 1], [0, 1])
    
    # P3 = A00 * B01 -> C01
    # A00 is rows 0-1, cols 0-1
    # B01 is rows 0-1, cols 2-3
    # C01 is rows 0-1, cols 2-3
    r = add_strassen_block(r, [0, 1], [0, 1], [0, 1], [2, 3], [0, 1], [2, 3])
    
    # P4 = A01 * B11 -> C01
    # A01 is rows 0-1, cols 2-3
    # B11 is rows 2-3, cols 2-3
    # C01 is rows 0-1, cols 2-3
    r = add_strassen_block(r, [0, 1], [2, 3], [2, 3], [2, 3], [0, 1], [2, 3])
    
    # P5 = A10 * B00 -> C10
    # A10 is rows 2-3, cols 0-1
    # B00 is rows 0-1, cols 0-1
    # C10 is rows 2-3, cols 0-1
    r = add_strassen_block(r, [2, 3], [0, 1], [0, 1], [0, 1], [2, 3], [0, 1])
    
    # P6 = A11 * B10 -> C10
    # A11 is rows 2-3, cols 2-3
    # B10 is rows 2-3, cols 0-1
    # C10 is rows 2-3, cols 0-1
    r = add_strassen_block(r, [2, 3], [2, 3], [2, 3], [0, 1], [2, 3], [0, 1])
    
    # P7 = A10 * B01 -> C11
    # A10 is rows 2-3, cols 0-1
    # B01 is rows 0-1, cols 2-3
    # C11 is rows 2-3, cols 2-3
    r = add_strassen_block(r, [2, 3], [0, 1], [0, 1], [2, 3], [2, 3], [2, 3])
    
    # P8 = A11 * B11 -> C11
    # A11 is rows 2-3, cols 2-3
    # B11 is rows 2-3, cols 2-3
    # C11 is rows 2-3, cols 2-3
    r = add_strassen_block(r, [2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3])
    
    # Part 2: 4x4x5 (cols 4..8 of B)
    # Schoolbook
    for i in range(n):
        for j in range(m):
            for k in range(4, p):
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
