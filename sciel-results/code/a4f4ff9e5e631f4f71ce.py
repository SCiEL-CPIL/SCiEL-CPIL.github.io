"""Matrix multiplication 2x2x12 via block-Strassen decomposition."""
import time

DIMENSIONS = [2, 2, 12]

# Strassen rank-7 decomposition for 2x2x2:
# M1 = (a12)(b21 - b22)
# M2 = (a11 + a12)(b22)
# M3 = (a21 - a22)(b11)
# M4 = (a22)(b12 - b11)
# M5 = (a11)(b11 + b22)
# M6 = (a12 - a22)(b21 + b22)
# M7 = (a11 - a21)(b11 + b12)
# C11 = M5 + M4 - M2 - M7
# C12 = M1 + M2
# C21 = M3 + M5
# C22 = M1 - M3 + M4 + M7

def _strassen_2x2x2():
    """Return U,V,W for 2x2x2 Strassen (rank 7).
    U shape (4,7): rows = a11,a12,a21,a22
    V shape (4,7): cols = b11,b12,b21,b22
    W shape (4,7): rows = c11,c12,c21,c22 (trace convention W[k*n+i])
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (a12)(b21 - b22)
    U[1][0] = 1       # a12
    V[2][0] = 1       # b21
    V[3][0] = -1      # b22
    W[1][0] = 1       # c12
    W[3][0] = 1       # c22
    
    # M2 = (a11 + a12)(b22)
    U[0][1] = 1       # a11
    U[1][1] = 1       # a12
    V[3][1] = 1       # b22
    W[0][1] = -1      # c11
    W[1][1] = 1       # c12
    
    # M3 = (a21 - a22)(b11)
    U[2][2] = 1       # a21
    U[3][2] = -1      # a22
    V[0][2] = 1       # b11
    W[2][2] = 1       # c21
    W[3][2] = -1      # c22
    
    # M4 = (a22)(b12 - b11)
    U[3][3] = 1       # a22
    V[1][3] = 1       # b12
    V[0][3] = -1      # b11
    W[0][3] = 1       # c11
    W[3][3] = 1       # c22
    
    # M5 = (a11)(b11 + b22)
    U[0][4] = 1       # a11
    V[0][4] = 1       # b11
    V[3][4] = 1       # b22
    W[0][4] = 1       # c11
    W[2][4] = 1       # c21
    
    # M6 = (a12 - a22)(b21 + b22)
    U[1][5] = 1       # a12
    U[3][5] = -1      # a22
    V[2][5] = 1       # b21
    V[3][5] = 1       # b22
    # M6 is not used in C? Wait, let me recheck Strassen formulas.
    # Standard Strassen:
    # p1 = (a12)(b21 - b22)
    # p2 = (a11 + a12)(b22)
    # p3 = (a21 - a22)(b11)
    # p4 = (a22)(b12 - b11)
    # p5 = (a11)(b11 + b22)
    # p6 = (a12 - a22)(b21 + b22)
    # p7 = (a11 - a21)(b11 + b12)
    # c11 = p5 + p4 - p2 - p7
    # c12 = p1 + p2
    # c21 = p3 + p5
    # c22 = p1 - p3 + p4 + p7
    # So p6 is NOT used in the standard 7-product Strassen for 2x2. 
    # Actually wait - that's wrong. Let me recount.
    # Standard Strassen uses 7 products p1..p7 and they ALL appear:
    # c11 = p5 + p4 - p2 - p7
    # c12 = p1 + p2
    # c21 = p3 + p5
    # c22 = p1 - p3 + p4 + p7
    # p6 doesn't appear! That can't be right for rank 7.
    
    # Let me use the correct Strassen decomposition:
    # p1 = (a11 + a22)(b11 + b22)
    # p2 = (a21 + a22)b11
    # p3 = a11(b21 - b22)
    # p4 = a22(b12 - b11)
    # p5 = (a11 + a12)b22
    # p6 = (a21 - a11)(b11 + b12)
    # p7 = (a12 + a22)(b21 + b22)
    # c11 = p1 + p4 - p5 + p7
    # c12 = p3 + p5
    # c21 = p2 + p4
    # c22 = p1 - p2 + p3 + p6
    
    # Let me redo with this correct version.
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # p1 = (a11 + a22)(b11 + b22)
    U[0][0] = 1
    U[3][0] = 1
    V[0][0] = 1
    V[3][0] = 1
    W[0][0] = 1   # c11
    W[3][0] = 1   # c22
    
    # p2 = (a21 + a22)b11
    U[2][1] = 1
    U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1   # c21
    W[3][1] = -1  # c22
    
    # p3 = a11(b21 - b22)
    U[0][2] = 1
    V[2][2] = 1
    V[3][2] = -1
    W[1][2] = 1   # c12
    W[3][2] = 1   # c22
    
    # p4 = a22(b12 - b11)
    U[3][3] = 1
    V[1][3] = 1
    V[0][3] = -1
    W[0][3] = 1   # c11
    W[2][3] = 1   # c21
    
    # p5 = (a11 + a12)b22
    U[0][4] = 1
    U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1  # c11
    W[1][4] = 1   # c12
    
    # p6 = (a21 - a11)(b11 + b12)
    U[2][5] = 1
    U[0][5] = -1
    V[0][5] = 1
    V[1][5] = 1
    W[3][5] = 1   # c22
    
    # p7 = (a12 + a22)(b21 + b22)
    U[1][6] = 1
    U[3][6] = 1
    V[2][6] = 1
    V[3][6] = 1
    W[0][6] = 1   # c11
    
    return U, V, W


def construct():
    """Construct rank-42 decomposition for 2x2x12 matrix multiplication.
    
    Strategy: Split the 12 columns of B into 6 blocks of 2 columns each.
    Each block uses the Strassen rank-7 decomposition for 2x2x2.
    Total rank = 6 * 7 = 42.
    
    A is 2x2, B is 2x12, C = A*B is 2x12.
    U shape: (n*m, rank) = (4, 42)
    V shape: (m*p, rank) = (24, 42)  
    W shape: (p*n, rank) = (24, 42) with trace convention W[k*n+i, r]
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    
    num_blocks = p // 2  # 6 blocks of 2 columns
    block_rank = 7
    rank = num_blocks * block_rank  # 42
    
    U_s, V_s, W_s = _strassen_2x2x2()
    
    U = [[0] * rank for _ in range(n * m)]   # (4, 42)
    V = [[0] * rank for _ in range(m * p)]   # (24, 42)
    W = [[0] * rank for _ in range(p * n)]   # (24, 42)
    
    for blk in range(num_blocks):
        # Block blk handles columns 2*blk and 2*blk+1 of B
        col_offset = 2 * blk  # offset in B's columns
        rank_offset = blk * block_rank
        
        for s in range(block_rank):
            r = rank_offset + s
            
            # U: same A-factors for all blocks (shared)
            for i in range(n):
                for j in range(m):
                    # U row index: i*m + j
                    U[i * m + j][r] = U_s[i * m + j][s]
            
            # V: B-factors, offset by column block
            for j in range(m):
                for k in range(2):  # k=0,1 within block
                    # V row index: j*p + (col_offset + k)
                    # V_s row index: j*2 + k (since m=2, block has 2 cols)
                    V[j * p + col_offset + k][r] = V_s[j * 2 + k][s]
            
            # W: C-factors, trace convention W[k*n + i, r]
            # C row within block: k (0 or 1), corresponds to column col_offset+k of C
            for k in range(2):
                for i in range(n):
                    # W row index: (col_offset + k) * n + i
                    # W_s row index: k * n + i
                    W[(col_offset + k) * n + i][r] = W_s[k * n + i][s]
    
    return {"U": U, "V": V, "W": W}
