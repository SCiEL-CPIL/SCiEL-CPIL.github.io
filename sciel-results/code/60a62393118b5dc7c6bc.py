import numpy as np

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    # Total rank calculated as 99
    rank = 99
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Helper to add a schoolbook block
    # a_rows, a_cols (inner), b_rows (inner), b_cols
    # A is a_rows x a_cols, B is b_rows x b_cols
    # C is a_rows x b_cols
    # Indices: A starts at (a_r, a_c) in global A?
    # We need to map local indices to global indices.
    
    # Global A is 3x5. Global B is 5x7. Global C is 3x7.
    # A indices: i in 0..2, j in 0..4.
    # B indices: k in 0..4, l in 0..6.
    # C indices: i in 0..2, l in 0..6.
    # U index: i*m + j
    # V index: k*p + l  (Wait, V is m x p? No, V is m*p x rank. Index is k*p + l? 
    # Task says V is (m*p, rank). B is m x p. So row index is k*p + l.
    # W is (p*n, rank). C is n x p. Output factor W[k*n+i, r].
    # So W row index is l*n + i.
    
    # Let's define a function to add a block
    # block_a_r, block_a_c: start row, start col of A in global A
    # block_b_r, block_b_c: start row, start col of B in global B
    # block_c_r, block_c_c: start row, start col of C in global C
    # a_h, a_w: height, width of A block
    # b_h, b_w: height, width of B block
    # c_h, c_w: height, width of C block
    
    # Actually, the block decomposition is fixed.
    # I will just hardcode the contributions for each of the 11 sub-problems.
    
    col_offset = 0
    
    # Strassen 2x2x2 generator
    def add_strassen_222(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col_offset
        # Local indices 0,1
        # M1: (A00+A11)(B00+B11)
        # A part: A00, A11
        # B part: B00, B11
        # C part: C00, C11
        
        # We need to add 7 columns to U, V, W
        
        # M1
        # U: A00, A11
        U[a_r*m + a_c, col_offset] = 1
        U[(a_r+1)*m + (a_c+1), col_offset] = 1
        # V: B00, B11
        V[b_r*p + b_c, col_offset] = 1
        V[(b_r+1)*p + (b_c+1), col_offset] = 1
        # W: C00 (+), C11 (+)
        W[(b_c)*n + (a_r), col_offset] = 1
        W[((b_c+1))*n + ((a_r+1)), col_offset] = 1
        col_offset += 1
        
        # M2: (A10+A11)B00
        U[(a_r+1)*m + a_c, col_offset] = 1
        U[(a_r+1)*m + (a_c+1), col_offset] = 1
        V[b_r*p + b_c, col_offset] = 1
        # W: C10 (+)
        W[(b_c)*n + ((a_r+1)), col_offset] = 1
        col_offset += 1
        
        # M3: A00(B01-B11)
        U[a_r*m + a_c, col_offset] = 1
        V[b_r*p + (b_c+1), col_offset] = 1
        V[(b_r+1)*p + (b_c+1), col_offset] = -1
        # W: C01 (+)
        W[((b_c+1))*n + (a_r), col_offset] = 1
        col_offset += 1
        
        # M4: A11(B10-B00)
        U[(a_r+1)*m + (a_c+1), col_offset] = 1
        V[(b_r+1)*p + b_c, col_offset] = 1
        V[b_r*p + b_c, col_offset] = -1
        # W: C00 (+), C10 (+)
        W[(b_c)*n + (a_r), col_offset] = 1
        W[(b_c)*n + ((a_r+1)), col_offset] = 1
        col_offset += 1
        
        # M5: (A00+A01)B11
        U[a_r*m + a_c, col_offset] = 1
        U[a_r*m + (a_c+1), col_offset] = 1
        V[(b_r+1)*p + (b_c+1), col_offset] = 1
        # W: C00 (-), C11 (-)
        W[(b_c)*n + (a_r), col_offset] = -1
        W[((b_c+1))*n + ((a_r+1)), col_offset] = -1
        col_offset += 1
        
        # M6: (A10-A00)(B00+B01)
        U[(a_r+1)*m + a_c, col_offset] = 1
        U[a_r*m + a_c, col_offset] = -1
        V[b_r*p + b_c, col_offset] = 1
        V[b_r*p + (b_c+1), col_offset] = 1
        # W: C11 (+)
        W[((b_c+1))*n + ((a_r+1)), col_offset] = 1
        col_offset += 1
        
        # M7: (A01-A11)(B10+B11)
        U[a_r*m + (a_c+1), col_offset] = 1
        U[(a_r+1)*m + (a_c+1), col_offset] = -1
        V[(b_r+1)*p + b_c, col_offset] = 1
        V[(b_r+1)*p + (b_c+1), col_offset] = 1
        # W: C00 (+)
        W[(b_c)*n + (a_r), col_offset] = 1
        col_offset += 1

    def add_schoolbook(a_r, a_c, b_r, b_c, c_r, c_c, h_a, w_a, h_b, w_b):
        nonlocal col_offset
        for i in range(h_a):
            for j in range(w_a):
                for k in range(h_b):
                    # A[i, j] * B[k, l] -> C[i, l]
                    # But B is h_b x w_b.
                    # Wait, schoolbook for A (ha x wb_a) and B (hb x wb_b)
                    # Inner dim must match. w_a == h_b.
                    for l in range(w_b):
                        # This is not efficient, but rank is just counting.
                        # We need to add one column for each (i, j, l) if we treat each product separately?
                        # No, schoolbook rank is h_a * w_a * w_b.
                        # Each term is A[i, j] * B[j, l] contributing to C[i, l].
                        # So for each i, j, l, we have a column.
                        
                        # U: A[i, j]
                        U[(a_r + i)*m + (a_c + j), col_offset] = 1
                        # V: B[j, l]
                        V[(b_r + j)*p + (b_c + l), col_offset] = 1
                        # W: C[i, l]
                        W[(c_c + l)*n + (c_r + i), col_offset] = 1
                        col_offset += 1

    # 1. C_1a_1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    add_strassen_222(0, 0, 0, 0, 0, 0)
    
    # 2. C_1a_2: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    add_strassen_222(0, 0, 0, 2, 0, 2)
    
    # 3. C_1a_3: A[0:2, 0:2] * B[0:2, 4:6] -> C[0:2, 4:6]
    add_strassen_222(0, 0, 0, 4, 0, 4)
    
    # 4. C_1a_4: A[0:2, 0:2] * B[0:2, 6:7] -> C[0:2, 6:7]
    # A is 2x2, B is 2x1. Inner 2.
    # Schoolbook: 2*2*1 = 4.
    add_schoolbook(0, 0, 0, 6, 0, 6, 2, 2, 2, 1)
    
    # 5. C_1b: A[2:3, 0:2] * B[0:2, 0:7] -> C[2:3, 0:7]
    # A is 1x2, B is 2x7. Inner 2.
    # Schoolbook: 1*2*7 = 14.
    add_schoolbook(2, 0, 0, 0, 2, 0, 1, 2, 2, 7)
    
    # 6. C_2a_1_1: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    # A is 2x2, B is 2x2.
    add_strassen_222(0, 2, 2, 0, 0, 0)
    
    # 7. C_2a_1_2: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    add_strassen_222(0, 2, 2, 2, 0, 2)
    
    # 8. C_2a_1_3: A[0:2, 2:4] * B[2:4, 4:6] -> C[0:2, 4:6]
    add_strassen_222(0, 2, 2, 4, 0, 4)
    
    # 9. C_2a_1_4: A[0:2, 2:4] * B[2:4, 6:7] -> C[0:2, 6:7]
    # A 2x2, B 2x1.
    add_schoolbook(0, 2, 2, 6, 0, 6, 2, 2, 2, 1)
    
    # 10. C_2a_2: A[0:2, 4:5] * B[4:5, 0:7] -> C[0:2, 0:7]
    # A is 2x1, B is 1x7. Inner 1.
    # Schoolbook: 2*1*7 = 14.
    add_schoolbook(0, 4, 4, 0, 0, 0, 2, 1, 1, 7)
    
    # 11. C_2b: A[2:3, 2:5] * B[2:5, 0:7] -> C[2:3, 0:7]
    # A is 1x3, B is 3x7. Inner 3.
    # Schoolbook: 1*3*7 = 21.
    add_schoolbook(2, 2, 2, 0, 2, 0, 1, 3, 3, 7)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
