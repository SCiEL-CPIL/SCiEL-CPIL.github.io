"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    # Block decomposition: split 5 into 2+2+1
    # A is n×m = 5×5, B is m×p = 5×5, C is n×p = 5×5
    # A rows: [0,1], [2,3], [4]  (sizes 2, 2, 1)
    # A cols: [0,1], [2,3], [4]  (sizes 2, 2, 1)
    # B rows: [0,1], [2,3], [4]  (sizes 2, 2, 1)
    # B cols: [0,1], [2,3], [4]  (sizes 2, 2, 1)
    # C rows: [0,1], [2,3], [4]  (sizes 2, 2, 1)
    # C cols: [0,1], [2,3], [4]  (sizes 2, 2, 1)
    
    row_sizes = [2, 2, 1]  # for A rows and C rows
    col_sizes = [2, 2, 1]  # for A cols, B rows, and C cols
    
    # Block structure:
    # A = [[A00(2x2), A01(2x2), A02(2x1)],
    #      [A10(2x2), A11(2x2), A12(2x1)],
    #      [A20(1x2), A21(1x2), A22(1x1)]]
    # B = [[B00(2x2), B01(2x2), B02(2x1)],
    #      [B10(2x2), B11(2x2), B12(2x1)],
    #      [B20(1x2), B21(1x2), B22(1x1)]]
    # C = [[C00(2x2), C01(2x2), C02(2x1)],
    #      [C10(2x2), C11(2x2), C12(2x1)],
    #      [C20(1x2), C21(1x2), C22(1x1)]]
    
    # C[i][j] = sum_k A[i][k] * B[k][j]
    # C00 = A00*B00 + A01*B10 + A02*B20
    # C01 = A00*B01 + A01*B11 + A02*B21
    # C02 = A00*B02 + A01*B12 + A02*B22
    # C10 = A10*B00 + A11*B10 + A12*B20
    # C11 = A10*B01 + A11*B11 + A12*B21
    # C12 = A10*B02 + A11*B12 + A12*B22
    # C20 = A20*B00 + A21*B10 + A22*B20
    # C21 = A20*B01 + A21*B11 + A22*B21
    # C22 = A20*B02 + A21*B12 + A22*B22
    
    # For each term, the rank depends on the dimensions:
    # 2x2 * 2x2: rank 7 (Strassen)
    # 2x2 * 2x1: rank 4 (schoolbook: 2*2*1)
    # 2x1 * 2x2: rank 4 (schoolbook: 2*1*2)
    # 2x1 * 2x1: rank 2 (schoolbook: 2*1*1)
    # 1x2 * 2x2: rank 4 (schoolbook: 1*2*2)
    # 1x2 * 2x1: rank 2 (schoolbook: 1*2*1)
    # 1x1 * 2x2: rank 2 (schoolbook: 1*1*2)
    # 1x1 * 2x1: rank 1 (schoolbook: 1*1*1)
    # 2x2 * 1x2: rank 4 (schoolbook: 2*2*1... wait, B02 is 2x1, so 2x2*2x1 = 2*2*1 = 4)
    # 2x2 * 1x1: rank 2 (schoolbook: 2*2*1... B22 is 1x1, so 2x2*1x1 = 2*1*1 = 2)
    # 1x2 * 1x2: rank 2 (schoolbook: 1*2*1 = 2)
    # 1x2 * 1x1: rank 1 (schoolbook: 1*2*1 = 2... wait, 1x2 * 1x1: inner dim is 2? No.
    #   A20 is 1x2, B00 is 2x2: 1*2*2 = 4
    #   A20 is 1x2, B02 is 2x1: 1*2*1 = 2
    #   A22 is 1x1, B20 is 1x2: 1*1*2 = 2
    #   A22 is 1x1, B22 is 1x1: 1*1*1 = 1
    
    # Let me recount:
    # C00: A00*B00 (2x2*2x2, rank 7) + A01*B10 (2x2*2x2, rank 7) + A02*B20 (2x1*1x2, rank 2)
    #       = 7 + 7 + 2 = 16
    # C01: A00*B01 (2x2*2x2, rank 7) + A01*B11 (2x2*2x2, rank 7) + A02*B21 (2x1*1x2, rank 2)
    #       = 7 + 7 + 2 = 16
    # C02: A00*B02 (2x2*2x1, rank 4) + A01*B12 (2x2*2x1, rank 4) + A02*B22 (2x1*1x1, rank 2)
    #       = 4 + 4 + 2 = 10
    # C10: A10*B00 (2x2*2x2, rank 7) + A11*B10 (2x2*2x2, rank 7) + A12*B20 (2x1*1x2, rank 2)
    #       = 7 + 7 + 2 = 16
    # C11: A10*B01 (2x2*2x2, rank 7) + A11*B11 (2x2*2x2, rank 7) + A12*B21 (2x1*1x2, rank 2)
    #       = 7 + 7 + 2 = 16
    # C12: A10*B02 (2x2*2x1, rank 4) + A11*B12 (2x2*2x1, rank 4) + A12*B22 (2x1*1x1, rank 2)
    #       = 4 + 4 + 2 = 10
    # C20: A20*B00 (1x2*2x2, rank 4) + A21*B10 (1x2*2x2, rank 4) + A22*B20 (1x1*1x2, rank 2)
    #       = 4 + 4 + 2 = 10
    # C21: A20*B01 (1x2*2x2, rank 4) + A21*B11 (1x2*2x2, rank 4) + A22*B21 (1x1*1x2, rank 2)
    #       = 4 + 4 + 2 = 10
    # C22: A20*B02 (1x2*2x1, rank 2) + A21*B12 (1x2*2x1, rank 2) + A22*B22 (1x1*1x1, rank 1)
    #       = 2 + 2 + 1 = 5
    
    # Total: 16 + 16 + 10 + 16 + 16 + 10 + 10 + 10 + 5 = 109
    
    # That's better than 125! Let me implement this.
    
    rank = 109
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper: add a Strassen decomposition for a 2x2 * 2x2 product
    # A_sub is 2x2 with rows (ar1, ar2) and cols (ac1, ac2)
    # B_sub is 2x2 with rows (br1, br2) and cols (bc1, bc2)
    # C_sub is 2x2 with rows (cr1, cr2) and cols (cc1, cc2)
    # Strassen's 7 multiplications:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B00)
    # M4 = A11(B11-B01)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    def add_strassen(ar1, ar2, ac1, ac2, br1, br2, bc1, bc2, cr1, cr2, cc1, cc2):
        nonlocal col
        # M1 = (A00+A11)(B00+B11)
        # A00 = A[ar1, ac1], A11 = A[ar2, ac2]
        # B00 = B[br1, bc1], B11 = B[br2, bc2]
        # U[A[ar1,ac1]] += 1, U[A[ar2,ac2]] += 1
        # V[B[br1,bc1]] += 1, V[B[br2,bc2]] += 1
        # W[C[cr1,cc1]] += 1, W[C[cr2,cc2]] += 1
        # W[C[cr1,cc2]] += 1, W[C[cr2,cc1]] += 1, W[C[cr2,cc2]] += 1
        
        # Actually, for Strassen, each multiplication is a rank-1 term
        # M1: U has +1 at (ar1,ac1) and (ar2,ac2), V has +1 at (br1,bc1) and (br2,bc2)
        # But this is a product of sums, so it's not a single rank-1 term.
        
        # Let me think more carefully. Strassen's algorithm computes:
        # M1 = (A00+A11) * (B00+B11)
        # This is a scalar multiplication of two linear combinations.
        # In the tensor decomposition, each "multiplication" corresponds to one column.
        # For M1:
        #   U[ar1*2+ac1] += 1  (A00)
        #   U[ar2*2+ac2] += 1  (A11)
        #   V[br1*2+bc1] += 1  (B00)
        #   V[br2*2+bc2] += 1  (B11)
        #   The product M1 contributes to C00, C01, C10, C11 with coefficients.
        
        # Actually, I need to be more careful. Let me index the 2x2 blocks.
        # A_sub = [[a00, a01], [a10, a11]]
        # B_sub = [[b00, b01], [b10, b11]]
        # C_sub = [[c00, c01], [c10, c11]]
        
        # a00 = A[ar1, ac1], a01 = A[ar1, ac2]
        # a10 = A[ar2, ac1], a11 = A[ar2, ac2]
        # b00 = B[br1, bc1], b01 = B[br1, bc2]
        # b10 = B[br2, bc1], b11 = B[br2, bc2]
        # c00 = C[cr1, cc1], c01 = C[cr1, cc2]
        # c10 = C[cr2, cc1], c11 = C[cr2, cc2]
        
        # M1 = (a00+a11)(b00+b11)
        # U: a00+=1, a11+=1
        # V: b00+=1, b11+=1
        # W: c00+=1, c01+=1, c10+=1, c11+=1  (with appropriate signs)
        
        # Wait, M1 contributes to:
        # c00: +M1
        # c01: +M1 (no, c01 = M3+M5, not M1)
        # Let me re-derive:
        # c00 = M1 + M4 - M5 + M7
        # c01 = M3 + M5
        # c10 = M2 + M4
        # c11 = M1 - M2 + M3 + M6
        
        # So M1 contributes to c00 (+1) and c11 (+1)
        # M2 contributes to c10 (+1) and c11 (-1)
        # M3 contributes to c01 (+1) and c11 (+1)
        # M4 contributes to c00 (+1) and c10 (+1)
        # M5 contributes to c00 (-1) and c01 (+1)
        # M6 contributes to c11 (+1)
        # M7 contributes to c00 (+1)
        
        # Let me add each multiplication as a separate column.
        
        # M1 = (a00+a11)(b00+b11)
        U[ar1 * m + ac1][col] += 1
        U[ar2 * m + ac2][col] += 1
        V[br1 * p + bc1][col] += 1
        V[br2 * p + bc2][col] += 1
        W[cc1 * n + cr1][col] += 1  # c00
        W[cc2 * n + cr2][col] += 1  # c11
        col += 1
        
        # M2 = (a10+a11)b00
        U[ar2 * m + ac1][col] += 1
        U[ar2 * m + ac2][col] += 1
        V[br1 * p + bc1][col] += 1
        W[cc1 * n + cr2][col] += 1  # c10
        W[cc2 * n + cr2][col] += -1  # c11
        col += 1
        
        # M3 = a00(b10-b00)
        U[ar1 * m + ac1][col] += 1
        V[br2 * p + bc1][col] += 1
        V[br1 * p + bc1][col] += -1
        W[cc2 * n + cr1][col] += 1  # c01
        W[cc2 * n + cr2][col] += 1  # c11
        col += 1
        
        # M4 = a11(b11-b01)
        U[ar2 * m + ac2][col] += 1
        V[br2 * p + bc2][col] += 1
        V[br1 * p + bc2][col] += -1
        W[cc1 * n + cr1][col] += 1  # c00
        W[cc1 * n + cr2][col] += 1  # c10
        col += 1
        
        # M5 = (a00+a01)b11
        U[ar1 * m + ac1][col] += 1
        U[ar1 * m + ac2][col] += 1
        V[br2 * p + bc2][col] += 1
        W[cc1 * n + cr1][col] += -1  # c00
        W[cc2 * n + cr1][col] += 1   # c01
        col += 1
        
        # M6 = (a10-a00)(b00+b01)
        U[ar2 * m + ac1][col] += 1
        U[ar1 * m + ac1][col] += -1
        V[br1 * p + bc1][col] += 1
        V[br1 * p + bc2][col] += 1
        W[cc2 * n + cr2][col] += 1  # c11
        col += 1
        
        # M7 = (a01-a11)(b10+b11)
        U[ar1 * m + ac2][col] += 1
        U[ar2 * m + ac2][col] += -1
        V[br2 * p + bc1][col] += 1
        V[br2 * p + bc2][col] += 1
        W[cc1 * n + cr1][col] += 1  # c00
        col += 1
    
    # Helper: add a schoolbook decomposition for an (r1×c1) * (r2×c2) product
    # where r1=r2 (inner dimension matches)
    def add_schoolbook(ar1, ar2, ac1, ac2, br1, br2, bc1, bc2, cr1, cr2, cc1, cc2, rows_a, cols_a, rows_b, cols_b):
        nonlocal col
        for i in range(rows_a):
            for k in range(cols_a):
                for j in range(cols_b):
                    U[(ar1 + i) * m + (ac1 + k)][col] += 1
                    V[(br1 + k) * p + (bc1 + j)][col] += 1
                    W[(cc1 + j) * n + (cr1 + i)][col] += 1
                    col += 1
    
    # Now build the decomposition
    # Row/col offsets for the 3 blocks:
    # Block 0: rows/cols 0,1 (size 2)
    # Block 1: rows/cols 2,3 (size 2)
    # Block 2: row/col 4 (size 1)
    
    # C00 = A00*B00 + A01*B10 + A02*B20
    # A00: rows 0-1, cols 0-1; B00: rows 0-1, cols 0-1; C00: rows 0-1, cols 0-1
    add_strassen(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)  # A00*B00
    # A01: rows 0-1, cols 2-3; B10: rows 2-3, cols 0-1; C00: rows 0-1, cols 0-1
    add_strassen(0, 1, 2, 3, 2, 3, 0, 1, 0, 1, 0, 1)  # A01*B10
    # A02: rows 0-1, col 4; B20: row 4, cols 0-1; C00: rows 0-1, cols 0-1
    # A02 is 2x1, B20 is 1x2: schoolbook rank 2
    add_schoolbook(0, 1, 4, 4, 4, 4, 0, 1, 0, 1, 0, 1, 2, 1, 1, 2)
    
    # C01 = A00*B01 + A01*B11 + A02*B21
    add_strassen(0, 1, 0, 1, 0, 1, 2, 3, 0, 1, 2, 3)  # A00*B01
    add_strassen(0, 1, 2, 3, 2, 3, 2, 3, 0, 1, 2, 3)  # A01*B11
    add_schoolbook(0, 1, 4, 4, 4, 4, 2, 3, 0, 1, 2, 3, 2, 1, 1, 2)  # A02*B21
    
    # C02 = A00*B02 + A01*B12 + A02*B22
    # A00: 2x2, B02: 2x1: schoolbook rank 4
    add_schoolbook(0, 1, 0, 1, 0, 1, 4, 4, 0, 1, 4, 4, 2, 2, 2, 1)
    # A01: 2x2, B12: 2x1: schoolbook rank 4
    add_schoolbook(0, 1, 2, 3, 2, 3, 4, 4, 0, 1, 4, 4, 2, 2, 2, 1)
    # A02: 2x1, B22: 1x1: schoolbook rank 2
    add_schoolbook(0, 1, 4, 4, 4, 4, 4, 4, 0, 1, 4, 4, 2, 1, 1, 1)
    
    # C10 = A10*B00 + A11*B10 + A12*B20
    add_strassen(2, 3, 0, 1, 0, 1, 0, 1, 2, 3, 0, 1)  # A10*B00
    add_strassen(2, 3, 2, 3, 2, 3, 0, 1, 2, 3, 0, 1)  # A11*B10
    add_schoolbook(2, 3, 4, 4, 4, 4, 0, 1, 2, 3, 0, 1, 2, 1, 1, 2)  # A12*B20
    
    # C11 = A10*B01 + A11*B11 + A12*B21
    add_strassen(2, 3, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3)  # A10*B01
    add_strassen(2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3)  # A11*B11
    add_schoolbook(2, 3, 4, 4, 4, 4, 2, 3, 2, 3, 2, 3, 2, 1, 1, 2)  # A12*B21
    
    # C12 = A10*B02 + A11*B12 + A12*B22
    add_schoolbook(2, 3, 0, 1, 0, 1, 4, 4, 2, 3, 4, 4, 2, 2, 2, 1)  # A10*B02
    add_schoolbook(2, 3, 2, 3, 2, 3, 4, 4, 2, 3, 4, 4, 2, 2, 2, 1)  # A11*B12
    add_schoolbook(2, 3, 4, 4, 4, 4, 4, 4, 2, 3, 4, 4, 2, 1, 1, 1)  # A12*B22
    
    # C20 = A20*B00 + A21*B10 + A22*B20
    add_schoolbook(4, 4, 0, 1, 0, 1, 0, 1, 4, 4, 0, 1, 1, 2, 2, 2)  # A20*B00
    add_schoolbook(4, 4, 2, 3, 2, 3, 0, 1, 4, 4, 0, 1, 1, 2, 2, 2)  # A21*B10
    add_schoolbook(4, 4, 4, 4, 4, 4, 0, 1, 4, 4, 0, 1, 1, 1, 1, 2)  # A22*B20
    
    # C21 = A20*B01 + A21*B11 + A22*B21
    add_schoolbook(4, 4, 0, 1, 0, 1, 2, 3, 4, 4, 2, 3, 1, 2, 2, 2)  # A20*B01
    add_schoolbook(4, 4, 2, 3, 2, 3, 2, 3, 4, 4, 2, 3, 1, 2, 2, 2)  # A21*B11
    add_schoolbook(4, 4, 4, 4, 4, 4, 2, 3, 4, 4, 2, 3, 1, 1, 1, 2)  # A22*B21
    
    # C22 = A20*B02 + A21*B12 + A22*B22
    add_schoolbook(4, 4, 0, 1, 0, 1, 4, 4, 4, 4, 4, 4, 1, 2, 2, 1)  # A20*B02
    add_schoolbook(4, 4, 2, 3, 2, 3, 4, 4, 4, 4, 4, 4, 1, 2, 2, 1)  # A21*B12
    add_schoolbook(4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 1)  # A22*B22
    
    # Verify rank
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline - start_time:
        pass  # Should not happen for this static construction
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END

