"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 7
    
    # We'll build the decomposition by combining:
    # - Two 2x2x7 blocks (splitting the middle dimension into two halves of size 2)
    # - Each 2x2x7 uses Strassen for 3 column-pairs (rank 7 each) + schoolbook for 1 column (rank 4)
    # - Total rank: 2 * (3*7 + 4) = 2 * 25 = 50
    
    rank = 50
    U = [[0.0] * rank for _ in range(n * m)]  # 8 rows
    V = [[0.0] * rank for _ in range(m * p)]  # 28 rows
    W = [[0.0] * rank for _ in range(p * n)]  # 14 rows
    
    col = 0  # current column in the factor matrices
    
    # Strassen multipliers for 2x2: 
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each of the two 2x2x7 blocks:
    # Block a: rows 0-1 of A, cols 0-1 of B
    # Block b: rows 0-1 of A, cols 2-3 of B
    
    for block in range(2):
        # For each block, handle 3 column-pairs with Strassen and 1 column with schoolbook
        for cp in range(4):  # column pair index: 0,1,2 use Strassen; 3 uses schoolbook
            if cp < 3:
                # Strassen for 2x2x2: columns cp*2 and cp*2+1 of B
                b_col_offset = block * 2 + cp * 2
                # U: 2x2 part of A (rows i*m+j for i in 0,1; j in 0,1)
                # V: 2x2 part of B (rows j*p+k for j in block*2..block*2+1; k in b_col_offset..b_col_offset+1)
                
                # M1 = (A00+A11)(B00+B11)
                U[0*2+0][col] = 1.0  # A[0,0]
                U[1*2+1][col] = 1.0  # A[1,1]
                V[(block*2+0)*7 + b_col_offset][col] = 1.0  # B[0, col0]
                V[(block*2+1)*7 + b_col_offset+1][col] = 1.0  # B[1, col1]
                W[b_col_offset*2 + 0][col] = 1.0  # C[0,0]
                W[b_col_offset+1*2 + 1][col] = 1.0  # C[1,1]
                col += 1
                
                # M2 = (A10+A11)B00
                U[1*2+0][col] = 1.0  # A[1,0]
                U[1*2+1][col] = 1.0  # A[1,1]
                V[(block*2+0)*7 + b_col_offset][col] = 1.0  # B[0, col0]
                W[b_col_offset*2 + 1][col] = 1.0  # C[1,0]
                col += 1
                
                # M3 = A00(B10-B11)
                U[0*2+0][col] = 1.0  # A[0,0]
                V[(block*2+1)*7 + b_col_offset][col] = 1.0  # B[1, col0]
                V[(block*2+1)*7 + b_col_offset+1][col] = -1.0  # B[1, col1]
                W[b_col_offset*2 + 0][col] = 1.0  # C[0,0]
                W[b_col_offset+1*2 + 1][col] = 1.0  # C[1,1]
                col += 1
                
                # M4 = A11(B01-B00)
                U[1*2+1][col] = 1.0  # A[1,1]
                V[(block*2+0)*7 + b_col_offset+1][col] = 1.0  # B[0, col1]
                V[(block*2+0)*7 + b_col_offset][col] = -1.0  # B[0, col0]
                W[b_col_offset*2 + 0][col] = 1.0  # C[0,0]
                W[b_col_offset*2 + 1][col] = 1.0  # C[1,0]
                col += 1
                
                # M5 = (A00+A01)B11
                U[0*2+0][col] = 1.0  # A[0,0]
                U[0*2+1][col] = 1.0  # A[0,1]
                V[(block*2+1)*7 + b_col_offset+1][col] = 1.0  # B[1, col1]
                W[b_col_offset*2 + 0][col] = -1.0  # C[0,0]
                W[b_col_offset+1*2 + 1][col] = 1.0  # C[1,1]
                col += 1
                
                # M6 = (A10-A00)(B00+B10)
                U[1*2+0][col] = 1.0  # A[1,0]
                U[0*2+0][col] = -1.0  # A[0,0]
                V[(block*2+0)*7 + b_col_offset][col] = 1.0  # B[0, col0]
                V[(block*2+1)*7 + b_col_offset][col] = 1.0  # B[1, col0]
                W[b_col_offset*2 + 1][col] = 1.0  # C[1,0]
                col += 1
                
                # M7 = (A01-A11)(B01+B11)
                U[0*2+1][col] = 1.0  # A[0,1]
                U[1*2+1][col] = -1.0  # A[1,1]
                V[(block*2+0)*7 + b_col_offset+1][col] = 1.0  # B[0, col1]
                V[(block*2+1)*7 + b_col_offset+1][col] = 1.0  # B[1, col1]
                W[b_col_offset*2 + 0][col] = 1.0  # C[0,0]
                W[b_col_offset+1*2 + 1][col] = 1.0  # C[1,1]
                col += 1
            else:
                # Schoolbook for 2x2x1: column cp (which is column 6)
                b_col = block * 2 + 6
                for i in range(2):
                    for j in range(2):
                        U[i*2+j][col] = 1.0
                        V[(block*2+j)*7 + b_col][col] = 1.0
                        W[b_col*2 + i][col] = 1.0
                        col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
