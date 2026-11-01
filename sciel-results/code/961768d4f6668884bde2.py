"""
Rank-minimal tensor decomposition for 2x2 by 2x8 matrix multiplication.

Uses the standard Strassen-based rank-28 decomposition:
  - A is 2x2, B is 2x8.
  - Split B into 4 blocks B1,B2,B3,B4 each of size 2x2.
  - For each block, use 7 Strassen multiplications.
  - Total rank = 7 * 4 = 28.

Conventions:
  U has shape (n*m, rank) = (4, 28):  U[i*m+j, c]
  V has shape (m*p, rank) = (16, 28): V[j*p+k, c]
  W has shape (p*n, rank) = (16, 28): W[k*n+i, c]

The tensor identity is:
  sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T_{(i,j),(j,k),(k,i)}

For matrix multiplication, T_{(i,j),(j,k),(k,i)} = 1 if the indices form a valid
multiplication path, 0 otherwise.
"""

def construct():
    n, m, p = 2, 2, 8
    rank = 28  # 7 Strassen terms per 2x2 block, 4 blocks
    
    # U: shape (4, 28)
    # Indices: row = i*m + j where i in {0,1}, j in {0,1}
    # Columns: 0-6 for block 0, 7-13 for block 1, 14-20 for block 2, 21-27 for block 3
    
    # V: shape (16, 28)
    # Indices: row = j*p + k where j in {0,1}, k in {0,...,7}
    
    # W: shape (16, 28)
    # Indices: row = k*n + i where k in {0,...,7}, i in {0,1}
    
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    # Strassen 7 terms for a 2x2 block.
    # For block b (b in 0..3), the B indices are k = 2*b .. 2*b+1.
    # B is 2x8, so B[j, k] with j in {0,1}, k in {0..7}.
    # For block b: B_top_left = B[0, 2b], B_top_right = B[0, 2b+1]
    #               B_bottom_left = B[1, 2b], B_bottom_right = B[1, 2b+1]
    
    # Strassen M's for a 2x2 block (A is 2x2, B_block is 2x2):
    # M1 = A[0,0] * (B[0,0] + B[1,1])
    # M2 = (A[0,0] + A[1,0]) * B[1,1]
    # M3 = (A[0,1] + A[1,1]) * B[0,0]
    # M4 = A[1,1] * (B[1,0] - B[0,0])
    # M5 = (A[0,0] + A[0,1]) * B[1,0]
    # M6 = (A[1,0] - A[0,0]) * (B[0,1] + B[1,1])
    # M7 = (A[0,1] - A[1,0]) * (B[0,0] + B[1,0])
    
    # For each block b and each Strassen term t (0..6):
    #   column = 7*b + t
    
    for b in range(4):
        for t in range(7):
            col = 7 * b + t
            
            # B indices within this block:
            # B[0,0] -> j=0, k=2b    -> V row = 0*8 + 2b = 2b
            # B[0,1] -> j=0, k=2b+1  -> V row = 0*8 + (2b+1) = 2b+1
            # B[1,0] -> j=1, k=2b    -> V row = 1*8 + 2b = 8+2b
            # B[1,1] -> j=1, k=2b+1  -> V row = 1*8 + (2b+1) = 8+2b+1
            
            b00 = 2 * b        # V row for B[0,0]
            b01 = 2 * b + 1    # V row for B[0,1]
            b10 = 8 + 2 * b    # V row for B[1,0]
            b11 = 8 + 2 * b + 1  # V row for B[1,1]
            
            if t == 0:
                # M1 = A[0,0] * (B[0,0] + B[1,1])
                # U: A[0,0] -> row 0*2+0 = 0
                U[0][col] = 1
                # V: B[0,0] + B[1,1]
                V[b00][col] = 1
                V[b11][col] = 1
                # W: output C[0,0] = M1 + ... ; W[k*n+i, c] where this contributes to C[0,0]
                # C[0,0] corresponds to output index (k=0, i=0) -> W row = 0*2+0 = 0
                # But wait: the output C is 2x8. C[i, k] where i in {0,1}, k in {0..7}.
                # The task says W[k*n+i, r], so W row = k*n + i.
                # For C[0,0]: k=0, i=0 -> W row = 0
                # For C[0,1]: k=1, i=0 -> W row = 1
                # For C[1,0]: k=0, i=1 -> W row = 2
                # For C[1,1]: k=1, i=1 -> W row = 3
                # For C[0,2b]: k=2b, i=0 -> W row = 2b*2 + 0 = 4b
                # For C[0,2b+1]: k=2b+1, i=0 -> W row = (2b+1)*2 + 0 = 4b+2
                # For C[1,2b]: k=2b, i=1 -> W row = 2b*2 + 1 = 4b+1
                # For C[1,2b+1]: k=2b+1, i=1 -> W row = (2b+1)*2 + 1 = 4b+3
                
                # M1 contributes to C[0,0] in the block, i.e., C[0, 2b]
                W[4 * b][col] = 1
                
            elif t == 1:
                # M2 = (A[0,0] + A[1,0]) * B[1,1]
                U[0][col] = 1
                U[2][col] = 1  # A[1,0] -> row 1*2+0 = 2
                V[b11][col] = 1
                # M2 contributes to C[1,1] in the block, i.e., C[1, 2b+1]
                W[4 * b + 3][col] = 1
                
            elif t == 2:
                # M3 = (A[0,1] + A[1,1]) * B[0,0]
                U[1][col] = 1  # A[0,1] -> row 0*2+1 = 1
                U[3][col] = 1  # A[1,1] -> row 1*2+1 = 3
                V[b00][col] = 1
                # M3 contributes to C[0,1] in the block, i.e., C[0, 2b+1]
                W[4 * b + 2][col] = 1
                
            elif t == 3:
                # M4 = A[1,1] * (B[1,0] - B[0,0])
                U[3][col] = 1
                V[b10][col] = 1
                V[b00][col] = -1
                # M4 contributes to C[1,0] in the block, i.e., C[1, 2b]
                W[4 * b + 1][col] = 1
                
            elif t == 4:
                # M5 = (A[0,0] + A[0,1]) * B[1,0]
                U[0][col] = 1
                U[1][col] = 1
                V[b10][col] = 1
                # M5 contributes to C[1,0] in the block, i.e., C[1, 2b]
                W[4 * b + 1][col] = 1
                
            elif t == 5:
                # M6 = (A[1,0] - A[0,0]) * (B[0,1] + B[1,1])
                U[2][col] = 1
                U[0][col] = -1
                V[b01][col] = 1
                V[b11][col] = 1
                # M6 contributes to C[1,1] in the block, i.e., C[1, 2b+1]
                W[4 * b + 3][col] = 1
                
            elif t == 6:
                # M7 = (A[0,1] - A[1,0]) * (B[0,0] + B[1,0])
                U[1][col] = 1
                U[2][col] = -1
                V[b00][col] = 1
                V[b10][col] = 1
                # M7 contributes to C[0,1] in the block, i.e., C[0, 2b+1]
                W[4 * b + 2][col] = 1
    
    return {"U": U, "V": V, "W": W}
