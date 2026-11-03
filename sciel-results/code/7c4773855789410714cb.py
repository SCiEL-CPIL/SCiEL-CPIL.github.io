"""Matrix multiplication 2x4x5 low-rank decomposition."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import time

def _strassen_2x2x2():
    """Return (U, V, W) for 2x2x2 multiplication, rank 7.
    U shape (4,7), V shape (4,7), W shape (4,7) with W indexed [k*n+i, r] = [2*k+i, r].
    """
    # Strassen: 
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A10)B11
    # M6 = (A11-A01)(B00+B10)
    # M7 = (A01-A00)(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    # A indices: A00=0, A01=1, A10=2, A11=3 (row-major: i*m+j, m=2)
    # B indices: B00=0, B01=1, B10=2, B11=3 (j*p+k, p=2)
    # C indices: C00=0, C01=1, C10=2, C11=3 (k*n+i, n=2)
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1  # C00 and C11
    
    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1  # C10 and C11
    
    # M3 = A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1  # C01 and C11
    
    # M4 = A11(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1  # C00 and C10
    
    # M5 = (A00+A10)B11
    U[0][4] = 1; U[2][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1  # C00 and C01
    
    # M6 = (A11-A01)(B00+B10)
    U[3][5] = 1; U[1][5] = -1
    V[0][5] = 1; V[2][5] = 1
    W[3][5] = 1  # C11
    
    # M7 = (A01-A00)(B01+B11)
    U[1][6] = 1; U[0][6] = -1
    V[1][6] = 1; V[3][6] = 1
    W[0][6] = 1  # C00
    
    return U, V, W

def _schoolbook_2x2x1():
    """Return (U, V, W) for 2x2x1 multiplication, rank 4.
    U shape (4,4), V shape (2,4), W shape (2,4).
    A is 2x2 (indices 0,1,2,3), B is 2x1 (indices 0,1), C is 2x1 (indices 0,1).
    W indexed [k*n+i, r] = [2*k+i, r], k=0,1; n=2.
    """
    U = [[0]*4 for _ in range(4)]
    V = [[0]*4 for _ in range(2)]
    W = [[0]*4 for _ in range(2)]
    
    # C0 = A00*B0 + A01*B1
    U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
    U[1][1] = 1; V[1][1] = 1; W[0][1] = 1
    # C1 = A10*B0 + A11*B1
    U[2][2] = 1; V[0][2] = 1; W[1][2] = 1
    U[3][3] = 1; V[1][3] = 1; W[1][3] = 1
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    # Full decomposition: 2x4x5
    # A is 2x4, B is 4x5, C is 2x5
    # Split: A = [A1 | A2] where A1, A2 are 2x2
    # B = [B1; B2] where B1, B2 are 2x5
    # C = A1*B1 + A2*B2
    # Each A_i * B_i is 2x2 * 2x5 = 2x5
    # Split B_i (2x5) into [B_i1 | B_i2 | B_i3] where B_i1, B_i2 are 2x2 and B_i3 is 2x1
    # So each 2x2*2x5 = 2x2*2x2 + 2x2*2x2 + 2x2*2x1
    # Rank per 2x2*2x5 = 7 + 7 + 4 = 18
    # Total rank = 2 * 18 = 36
    
    n, m, p = DIMENSIONS  # 2, 4, 5
    
    # U has shape (n*m, rank) = (8, 36)
    # V has shape (m*p, rank) = (20, 36)  
    # W has shape (p*n, rank) = (10, 36)
    
    total_rank = 36
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # A indices (n*m = 8): row i, col j -> i*4+j, i in [0,1], j in [0,3]
    # B indices (m*p = 20): row j, col k -> j*5+k, j in [0,3], k in [0,4]
    # C indices (p*n = 10): W[k*n+i, r] = k*2+i, k in [0,4], i in [0,1]
    
    col_offset = 0
    for block in range(2):  # block 0: A1*B1, block 1: A2*B2
        # A block is columns 2*block to 2*block+1 of A
        # B block is rows 2*block to 2*block+1 of B
        
        # Sub-decompose 2x2 * 2x5 into three parts
        for sub in range(3):  # sub 0: 2x2*2x2, sub 1: 2x2*2x2, sub 2: 2x2*2x1
            if sub < 2:
                # 2x2 * 2x2 using Strassen, rank 7
                sU, sV, sW = _strassen_2x2x2()
                rank_sub = 7
                # A sub-block: rows 0,1 of A, cols 2*block+2*sub to 2*block+2*sub+1
                # A local indices: a_row*2 + a_col, a_row in [0,1], a_col in [0,1]
                # A global: i*4 + (2*block + 2*sub + a_col)
                for a_row in range(2):
                    for a_col in range(2):
                        a_local = a_row * 2 + a_col
                        a_global = a_row * 4 + (2 * block + 2 * sub + a_col)
                        for r in range(rank_sub):
                            U[a_global][col_offset + r] += sU[a_local][r]
                
                # B sub-block: rows 2*block to 2*block+1, cols 2*sub to 2*sub+1
                # B local indices: b_row*2 + b_col, b_row in [0,1], b_col in [0,1]
                # B global: (2*block + b_row)*5 + (2*sub + b_col)
                for b_row in range(2):
                    for b_col in range(2):
                        b_local = b_row * 2 + b_col
                        b_global = (2 * block + b_row) * 5 + (2 * sub + b_col)
                        for r in range(rank_sub):
                            V[b_global][col_offset + r] += sV[b_local][r]
                
                # C sub-block: k in [2*sub, 2*sub+1], i in [0,1]
                # C global: k*2 + i
                for k_local in range(2):
                    for i_local in range(2):
                        c_local = k_local * 2 + i_local
                        c_global = (2 * sub + k_local) * 2 + i_local
                        for r in range(rank_sub):
                            W[c_global][col_offset + r] += sW[c_local][r]
            else:
                # 2x2 * 2x1 using schoolbook, rank 4
                sU, sV, sW = _schoolbook_2x2x1()
                rank_sub = 4
                # A sub-block: rows 0,1 of A, cols 2*block to 2*block+1
                for a_row in range(2):
                    for a_col in range(2):
                        a_local = a_row * 2 + a_col
                        a_global = a_row * 4 + (2 * block + a_col)
                        for r in range(rank_sub):
                            U[a_global][col_offset + r] += sU[a_local][r]
                
                # B sub-block: rows 2*block to 2*block+1, col 4
                # B local indices: b_row*1 + b_col = b_row (since p=1 for this sub)
                # Actually B is 2x1, so b_local = b_row*1 + 0 = b_row
                # B global: (2*block + b_row)*5 + 4
                for b_row in range(2):
                    b_local = b_row  # 2x1: index is just row
                    b_global = (2 * block + b_row) * 5 + 4
                    for r in range(rank_sub):
                        V[b_global][col_offset + r] += sV[b_local][r]
                
                # C sub-block: k=4, i in [0,1]
                # C global: 4*2 + i = 8 + i
                for i_local in range(2):
                    c_local = i_local  # 2x1: index is just row (k*n+i with k=0 local)
                    c_global = 4 * 2 + i_local
                    for r in range(rank_sub):
                        W[c_global][col_offset + r] += sW[c_local][r]
            
            col_offset += rank_sub
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
