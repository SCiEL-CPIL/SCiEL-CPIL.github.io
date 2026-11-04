import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 2, 3, 5
    
    # Strategy: Split B (3x5) into B1 (3x2) and B2 (3x3)
    # C = A*B1 + A*B2
    # A*B1: 2x3 x 3x2, use schoolbook rank 12
    # A*B2: 2x3 x 3x3, use rank 14 decomposition
    # Total rank: 26
    
    rank = 26
    U = [[0] * rank for _ in range(n * m)]  # (6, 26)
    V = [[0] * rank for _ in range(m * p)]  # (15, 26)
    W = [[0] * rank for _ in range(p * n)]  # (10, 26)
    
    col = 0
    
    # Part 1: Schoolbook for 2x3 x 3x2 (columns 0,1 of B)
    # A is 2x3, B1 is 3x2 (B rows 0,1,2; cols 0,1)
    # C1[i,k] = sum_j A[i,j]*B[j,k] for k=0,1
    for i in range(n):
        for j in range(m):
            for k in range(2):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    # col is now 12
    
    # Part 2: Rank 14 for 2x3 x 3x3 (columns 2,3,4 of B)
    # A is 2x3, B2 is 3x3 (B rows 0,1,2; cols 2,3,4)
    # C2[i,k] = sum_j A[i,j]*B[j,k] for k=2,3,4
    # 
    # Use decomposition: split into 2x2x2 (Strassen rank 7) + corrections
    # A[0:2, 0:2] * B2[0:2, 0:3] -> C2[0:2, 0:3]
    # A[0:2, 2] * B2[0:2, 0:3] -> C2[0:2, 0:3]  
    # A[0:2, 0:2] * B2[2, 0:3] -> C2[0:2, 0:3]
    # A[0:2, 2] * B2[2, 0:3] -> C2[0:2, 0:3]
    #
    # 2x2x2 Strassen: rank 7
    # 2x1x3: rank 6 (but we can do better with sharing)
    # 2x2x1: rank 4
    # 2x1x1: rank 2
    # Total: 7+6+4+2 = 19, not 14.
    #
    # Let me use a direct rank-14 construction for 2x3x3.
    # Known: rank(2x3x3) = 14.
    # I'll use a verified decomposition.
    
    # For 2x3 x 3x3 -> 2x3:
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    # 
    # Rank 14 decomposition (based on known results):
    # We'll use the following 14 terms.
    
    # Indices: A[i,j] with i in {0,1}, j in {0,1,2}
    #           B[j,k] with j in {0,1,2}, k in {0,1,2} (local, maps to global k+2)
    #           C[i,k] with i in {0,1}, k in {0,1,2} (local, maps to global k+2)
    
    # U row index: i*m + j = i*3 + j
    # V row index: j*p + (k+2) = j*5 + k + 2
    # W row index: (k+2)*n + i = (k+2)*2 + i
    
    def add_term(u_rows, v_rows, w_rows, c):
        """Add a rank-1 term. u_rows: list of (row, coeff), etc."""
        for row, coeff in u_rows:
            U[row][c] += coeff
        for row, coeff in v_rows:
            V[row][c] += coeff
        for row, coeff in w_rows:
            W[row][c] += coeff
    
    # Rank 14 decomposition for 2x3x3
    # Term 1: A[0,0]*B[0,0] -> C[0,0]
    add_term([(0, 1)], [(0*5+2, 1)], [(0*2+0, 1)], col); col += 1
    # Term 2: A[0,0]*B[0,1] -> C[0,1]
    add_term([(0, 1)], [(0*5+3, 1)], [(1*2+0, 1)], col); col += 1
    # Term 3: A[0,0]*B[0,2] -> C[0,2]
    add_term([(0, 1)], [(0*5+4, 1)], [(2*2+0, 1)], col); col += 1
    # Term 4: A[0,1]*B[1,0] -> C[0,0]
    add_term([(1, 1)], [(1*5+2, 1)], [(0*2+0, 1)], col); col += 1
    # Term 5: A[0,1]*B[1,1] -> C[0,1]
    add_term([(1, 1)], [(1*5+3, 1)], [(1*2+0, 1)], col); col += 1
    # Term 6: A[0,1]*B[1,2] -> C[0,2]
    add_term([(1, 1)], [(1*5+4, 1)], [(2*2+0, 1)], col); col += 1
    # Term 7: A[0,2]*B[2,0] -> C[0,0]
    add_term([(2, 1)], [(2*5+2, 1)], [(0*2+0, 1)], col); col += 1
    # Term 8: A[0,2]*B[2,1] -> C[0,1]
    add_term([(2, 1)], [(2*5+3, 1)], [(1*2+0, 1)], col); col += 1
    # Term 9: A[0,2]*B[2,2] -> C[0,2]
    add_term([(2, 1)], [(2*5+4, 1)], [(2*2+0, 1)], col); col += 1
    # Term 10: A[1,0]*B[0,0] -> C[1,0]
    add_term([(3, 1)], [(0*5+2, 1)], [(0*2+1, 1)], col); col += 1
    # Term 11: A[1,0]*B[0,1] -> C[1,1]
    add_term([(3, 1)], [(0*5+3, 1)], [(1*2+1, 1)], col); col += 1
    # Term 12: A[1,0]*B[0,2] -> C[1,2]
    add_term([(3, 1)], [(0*5+4, 1)], [(2*2+1, 1)], col); col += 1
    # Term 13: A[1,1]*B[1,0] -> C[1,0]
    add_term([(4, 1)], [(1*5+2, 1)], [(0*2+1, 1)], col); col += 1
    # Term 14: A[1,1]*B[1,1] -> C[1,1]
    add_term([(4, 1)], [(1*5+3, 1)], [(1*2+1, 1)], col); col += 1
    # Term 15: A[1,1]*B[1,2] -> C[1,2]
    add_term([(4, 1)], [(1*5+4, 1)], [(2*2+1, 1)], col); col += 1
    # Term 16: A[1,2]*B[2,0] -> C[1,0]
    add_term([(5, 1)], [(2*5+2, 1)], [(0*2+1, 1)], col); col += 1
    # Term 17: A[1,2]*B[2,1] -> C[1,1]
    add_term([(5, 1)], [(2*5+3, 1)], [(1*2+1, 1)], col); col += 1
    # Term 18: A[1,2]*B[2,2] -> C[1,2]
    add_term([(5, 1)], [(2*5+4, 1)], [(2*2+1, 1)], col); col += 1
    
    # That's 18 terms for the 2x3x3 part, total 30. Not improved.
    # Let me reset and use a proper rank-14 decomposition.
    
    # Reset
    rank = 26
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    # Part 1: Schoolbook for 2x3 x 3x2
    for i in range(n):
        for j in range(m):
            for k in range(2):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    # col = 12
    
    # Part 2: Rank 14 for 2x3 x 3x3
    # Using a known rank-14 decomposition
    # C[i,k] = sum_j A[i,j]*B[j,k]
    
    # I'll use the following approach:
    # Let M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1]) -> contributes to C[0,0],C[0,1],C[1,0],C[1,1]
    # etc. But this is complex. Let me just verify with a simpler approach.
    
    # Actually, let me just return the schoolbook rank 30 to be safe,
    # but try to use Strassen for the 2x2x2 blocks to get rank 26.
    
    # Reset again
    rank = 26
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    # Use Strassen for 2x2x2 sub-blocks
    # Split A into A_top (2x3) and split B columns into groups of 2
    
    # For each pair of output columns (0,1), (2,3), and single column 4:
    # Columns 0,1: A * B[:,0:2] (2x3 x 3x2)
    # Columns 2,3: A * B[:,2:4] (2x3 x 3x2)  
    # Column 4: A * B[:,4:5] (2x3 x 3x1)
    
    # For 2x3 x 3x2:
    # Split A into A1 (2x2, cols 0,1) and a (2x1, col 2)
    # Split B into B1 (2x2, rows 0,1) and b (1x2, row 2)
    # A1*B1: 2x2x2, Strassen rank 7
    # A1*b: 2x2x1, rank 4
    # a*B1: 2x1x2, rank 4
    # a*b: 2x1x1, rank 2
    # Total: 7+4+4+2 = 17 per 2x3x2 block. That's worse than schoolbook 12.
    
    # So Strassen doesn't help for 2x3x2. Let me just use schoolbook everywhere.
    # Rank 30 is the schoolbook. Can I do better?
    
    # The minimum rank for 2x3x5 is at most 26 (known bounds).
    # Let me just return schoolbook rank 30 for now, as I can't verify a lower rank.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    elapsed = time.time() - start_time
    if elapsed > 110:
        pass  # Already done, just return
    
    return {"U": U, "V": V, "W": W}
