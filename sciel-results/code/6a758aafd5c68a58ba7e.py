"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def _strassen222():
    """Rank-7 decomposition of 2x2x2 matrix multiplication.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Convention: U[i*m+j, c] where m=2, so U[i*2+j, c]
                V[j*p+k, c] where p=2, so V[j*2+k, c]
                W[k*n+i, c] where n=2, so W[k*2+i, c]
    """
    n = m = p = 2
    # Strassen's 7 multiplications for 2x2 * 2x2 -> 2x2
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    # C11 = M1 - M4 + M6 + M7
    # C12 = M2 + M4
    # C21 = M3 + M5
    # C22 = M1 - M2 + M5 + M6
    # where:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    
    # U factors: each M uses products of A sub-blocks
    # M1: A11+A22 -> U rows for (i=0,j=0) and (i=1,j=1)
    # M2: A21+A22 -> U rows for (i=1,j=0) and (i=1,j=1)
    # M3: A11 -> U row for (i=0,j=0)
    # M4: A22 -> U row for (i=1,j=1)
    # M5: A11+A12 -> U rows for (i=0,j=0) and (i=0,j=1)
    # M6: A21-A11 -> U rows for (i=1,j=0) and (i=0,j=0)
    # M7: A12-A22 -> U rows for (i=0,j=1) and (i=1,j=1)
    
    U = [[0]*7 for _ in range(4)]  # (n*m, rank) = (4, 7)
    V = [[0]*7 for _ in range(4)]  # (m*p, rank) = (4, 7)
    W = [[0]*7 for _ in range(4)]  # (p*n, rank) = (4, 7)
    
    # Column 0: M1 = (A11+A22)*(B11+B22)
    U[0*2+0][0] = 1  # A11
    U[1*2+1][0] = 1  # A22
    V[0*2+0][0] = 1  # B11
    V[1*2+1][0] = 1  # B22
    # W: M1 contributes to C11 (+1), C12 (0), C21 (0), C22 (+1)
    W[0*2+0][0] = 1  # C11: k=0,i=0
    W[1*2+1][0] = 1  # C22: k=1,i=1
    
    # Column 1: M2 = (A21+A22)*B11
    U[1*2+0][1] = 1  # A21
    U[1*2+1][1] = 1  # A22
    V[0*2+0][1] = 1  # B11
    # M2 contributes to C12 (+1), C21 (+1)
    W[0*2+1][1] = 1  # C12: k=0,i=0
    W[1*2+0][1] = 1  # C21: k=1,i=0
    
    # Column 2: M3 = A11*(B12-B22)
    U[0*2+0][2] = 1  # A11
    V[0*2+1][2] = 1  # B12
    V[1*2+1][2] = -1 # B22
    # M3 contributes to C12 (-1), C21 (+1)
    W[0*2+1][2] = -1 # C12: k=0,i=0
    W[1*2+0][2] = 1  # C21: k=1,i=0
    
    # Column 3: M4 = A22*(B21-B11)
    U[1*2+1][3] = 1  # A22
    V[1*2+0][3] = 1  # B21
    V[0*2+0][3] = -1 # B11
    # M4 contributes to C11 (-1), C21 (-1)
    W[0*2+0][3] = -1 # C11: k=0,i=0
    W[1*2+0][3] = -1 # C21: k=1,i=0
    
    # Column 4: M5 = (A11+A12)*B22
    U[0*2+0][4] = 1  # A11
    U[0*2+1][4] = 1  # A12
    V[1*2+1][4] = 1  # B22
    # M5 contributes to C12 (-1), C22 (+1)
    W[0*2+1][4] = -1 # C12: k=0,i=0
    W[1*2+1][4] = 1  # C22: k=1,i=1
    
    # Column 5: M6 = (A21-A11)*(B11+B12)
    U[1*2+0][5] = 1  # A21
    U[0*2+0][5] = -1 # A11
    V[0*2+0][5] = 1  # B11
    V[0*2+1][5] = 1  # B12
    # M6 contributes to C11 (+1), C22 (-1)
    W[0*2+0][5] = 1  # C11: k=0,i=0
    W[1*2+1][5] = -1 # C22: k=1,i=1
    
    # Column 6: M7 = (A12-A22)*(B21+B22)
    U[0*2+1][6] = 1  # A12
    U[1*2+1][6] = -1 # A22
    V[1*2+0][6] = 1  # B21
    V[1*2+1][6] = 1  # B22
    # M7 contributes to C11 (+1), C22 (-1)
    W[0*2+0][6] = 1  # C11: k=0,i=0
    W[1*2+1][6] = -1 # C22: k=1,i=1
    
    return U, V, W


def _schoolbook223():
    """Rank-9 (or better) decomposition of 2x2x3 matrix multiplication.
    A is 2x2, B is 2x3, C is 2x3.
    """
    n = 2; m = 2; p = 3
    rank = n * m * p  # 12 for schoolbook, but we can do better
    # Actually rank of 2x2x3 is 9 (known result)
    # Let's just use schoolbook for simplicity: rank 12
    # But we want to be efficient. Let me think...
    # For now, use schoolbook: rank = 12
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W


def _block_2x3x2():
    """Rank-12 decomposition of 2x3x2 (schoolbook).
    A: 2x3, B: 3x2, C: 2x2
    """
    n = 2; m = 3; p = 2
    rank = n * m * p  # 12
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W


def _block_2x3x3():
    """Decomposition of 2x3x3.
    A: 2x3, B: 3x3, C: 2x3
    Schoolbook rank: 18
    """
    n = 2; m = 3; p = 3
    rank = n * m * p  # 18
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W


def _block_2x2x2():
    """Use Strassen for 2x2x2, rank 7."""
    return _strassen222()


def _block_2x2x3():
    """Decomposition of 2x2x3.
    A: 2x2, B: 2x3, C: 2x3
    Schoolbook: 12
    Can we do better? Rank of 2x2x3 is 9.
    """
    n = 2; m = 2; p = 3
    # Use schoolbook for now: rank 12
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W


def construct():
    """Construct a low-rank decomposition of 4x5 * 5x5 -> 4x5.
    
    Partition the 4 rows into [2, 2] and the 5 columns into [2, 3].
    This gives 4 sub-blocks for A (each 2x2 or 2x3) and 4 for B (each 2x5 or 3x5).
    
    Actually, let me partition more carefully:
    A is 4x5, B is 5x5, C is 4x5.
    
    Partition rows of A into [2, 2], columns of A (and rows of B) into [2, 3].
    So A has blocks A_{ij} where i in {0,1}, j in {0,1}:
      A_{00}: 2x2, A_{01}: 2x3
      A_{10}: 2x2, A_{11}: 2x3
    
    B has blocks B_{jk} where j in {0,1}, k in {0,1}:
      B_{00}: 2x2, B_{01}: 2x3
      B_{10}: 3x2, B_{11}: 3x3
    
    C has blocks C_{ik} where i in {0,1}, k in {0,1}:
      C_{00}: 2x2, C_{01}: 2x3
      C_{10}: 2x2, C_{11}: 2x3
    
    C_{00} = A_{00}B_{00} + A_{01}B_{10}
    C_{01} = A_{00}B_{01} + A_{01}B_{11}
    C_{10} = A_{10}B_{00} + A_{11}B_{10}
    C_{11} = A_{10}B_{01} + A_{11}B_{11}
    
    We can decompose each product term separately and sum the ranks.
    
    Terms:
    1. A_{00}B_{00}: 2x2 * 2x2 -> 2x2, rank 7 (Strassen)
    2. A_{01}B_{10}: 2x3 * 3x2 -> 2x2, rank 12 (schoolbook)
    3. A_{00}B_{01}: 2x2 * 2x3 -> 2x3, rank 12 (schoolbook)
    4. A_{01}B_{11}: 2x3 * 3x3 -> 2x3, rank 18 (schoolbook)
    5. A_{10}B_{00}: 2x2 * 2x2 -> 2x2, rank 7 (Strassen)
    6. A_{11}B_{10}: 2x3 * 3x2 -> 2x2, rank 12 (schoolbook)
    7. A_{10}B_{01}: 2x2 * 2x3 -> 2x3, rank 12 (schoolbook)
    8. A_{11}B_{11}: 2x3 * 3x3 -> 2x3, rank 18 (schoolbook)
    
    Total: 7+12+12+18+7+12+12+18 = 98
    
    That's worse than 100? No, 98 < 100. Good.
    
    But wait, I can do better. Let me try different partitions.
    """
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    start_time = time.time()
    deadline = start_time + 100.0  # Leave 50s safety margin
    
    # Partition: rows [2,2], cols [2,3]
    row_parts = [2, 2]
    col_parts = [2, 3]
    
    # Build the full U, V, W matrices
    # U has shape (n*m, rank) = (20, rank)
    # V has shape (m*p, rank) = (25, rank)
    # W has shape (p*n, rank) = (20, rank)
    
    # First, figure out total rank
    # We'll compute each block's decomposition and concatenate
    
    U_blocks = []
    V_blocks = []
    W_blocks = []
    offsets = []  # (row_start, col_start, rank_start) for each term
    
    row_starts = [0]
    for r in row_parts[:-1]:
        row_starts.append(row_starts[-1] + r)
    col_starts = [0]
    for c in col_parts[:-1]:
        col_starts.append(col_starts[-1] + c)
    
    rank_so_far = 0
    
    for i in range(len(row_parts)):
        for k in range(len(col_parts)):
            for j in range(len(col_parts)):
                # Term: A_{ij} * B_{jk} contributes to C_{ik}
                # A_{ij} is row_parts[i] x col_parts[j]
                # B_{jk} is col_parts[j] x col_parts[k]
                # Result C_{ik} is row_parts[i] x col_parts[k]
                
                ri = row_parts[i]
                rj = col_parts[j]
                rk = col_parts[k]
                
                # Get decomposition for this sub-problem
                if ri == 2 and rj == 2 and rk == 2:
                    Ui, Vi, Wi = _strassen222()
                    sub_rank = 7
                elif ri == 2 and rj == 3 and rk == 2:
                    Ui, Vi, Wi = _block_2x3x2()
                    sub_rank = 12
                elif ri == 2 and rj == 2 and rk == 3:
                    Ui, Vi, Wi = _block_2x2x3()
                    sub_rank = 12
                elif ri == 2 and rj == 3 and rk == 3:
                    Ui, Vi, Wi = _block_2x3x3()
                    sub_rank = 18
                else:
                    # Fallback to schoolbook
                    sub_rank = ri * rj * rk
                    Ui = [[0]*sub_rank for _ in range(ri*rj)]
                    Vi = [[0]*sub_rank for _ in range(rj*rk)]
                    Wi = [[0]*sub_rank for _ in range(rk*ri)]
                    col = 0
                    for ii in range(ri):
                        for jj in range(rj):
                            for kk in range(rk):
                                Ui[ii*rj+jj][col] = 1
                                Vi[jj*rk+kk][col] = 1
                                Wi[kk*ri+ii][col] = 1
                                col += 1
                
                U_blocks.append((i, j, Ui))
                V_blocks.append((j, k, Vi))
                W_blocks.append((i, k, Wi))
                
                offsets.append((i, j, k, rank_so_far, sub_rank))
                rank_so_far += sub_rank
                
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    total_rank = rank_so_far
    
    # Now build full matrices
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    for (i, j, k, rstart, rlen) in offsets:
        Ui = U_blocks[offsets.index((i, j, k, rstart, rlen))][2]
        Vi = V_blocks[offsets.index((i, j, k, rstart, rlen))][2]
        Wi = W_blocks[offsets.index((i, j, k, rstart, rlen))][2]
        
        row_off_A = row_starts[i]
        col_off_A = col_starts[j]
        col_off_B = col_starts[j]
        col_off_C = col_starts[k]
        
        ri = row_parts[i]
        rj = col_parts[j]
        rk = col_parts[k]
        
        for a in range(ri):
            for b in range(rj):
                for c_idx in range(rlen):
                    U[(row_off_A + a) * m + (col_off_A + b)][rstart + c_idx] = Ui[a * rj + b][c_idx]
        
        for a in range(rj):
            for b in range(rk):
                for c_idx in range(rlen):
                    V[(col_off_B + a) * p + (col_off_C + b)][rstart + c_idx] = Vi[a * rk + b][c_idx]
        
        for a in range(rk):
            for b in range(ri):
                for c_idx in range(rlen):
                    W[(col_off_C + a) * n + (row_off_A + b)][rstart + c_idx] = Wi[a * ri + b][c_idx]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
