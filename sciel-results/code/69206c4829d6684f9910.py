import time

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=8
    # Rank 44
    rank = 44
    U = [[0] * rank for _ in range(n * m)]  # 6 x 44
    V = [[0] * rank for _ in range(m * p)]  # 24 x 44
    W = [[0] * rank for _ in range(p * n)]  # 16 x 44
    
    col_idx = 0
    
    # Helper to add Strassen factors for A_top (2x2) * B_sub (2x2) -> C_sub (2x2)
    # U indices for A_top:
    # x00: U[0], x01: U[1], x10: U[3], x11: U[4]
    # V indices for B_sub (rows 0,1 of B, cols c0, c1):
    # y00: V[c0], y01: V[c1], y10: V[8+c0], y11: V[8+c1]
    # W indices for C_sub (cols c0, c1):
    # z00: W[c0*2+0], z01: W[c0*2+1] -- WAIT.
    # W index is k*n + i. n=2.
    # C[0, c0] -> k=c0, i=0 -> c0*2 + 0
    # C[1, c0] -> k=c0, i=1 -> c0*2 + 1
    # C[0, c1] -> k=c1, i=0 -> c1*2 + 0
    # C[1, c1] -> k=c1, i=1 -> c1*2 + 1
    
    def add_strassen(b_col_offset, w_col_offset):
        nonlocal col_idx
        # V cols are b_col_offset, b_col_offset+1
        # W cols are w_col_offset, w_col_offset+1
        
        # M1 = (x00 + x11)(y00 + y11)
        c = col_idx
        U[0][c] = 1; U[4][c] = 1
        V[b_col_offset][c] = 1; V[8+b_col_offset+1][c] = 1
        W[w_col_offset*2][c] = 1; W[w_col_offset+1*2+1][c] = 1
        col_idx += 1

        # M2 = (x10 + x11)y00
        c = col_idx
        U[3][c] = 1; U[4][c] = 1
        V[b_col_offset][c] = 1
        W[w_col_offset*2+1][c] = 1; W[w_col_offset+1*2+1][c] = -1
        col_idx += 1

        # M3 = x00(y01 - y11)
        c = col_idx
        U[0][c] = 1
        V[b_col_offset+1][c] = 1; V[8+b_col_offset+1][c] = -1
        W[w_col_offset*2+1][c] = 1; W[w_col_offset+1*2+1][c] = 1
        col_idx += 1

        # M4 = x11(y10 - y00)
        c = col_idx
        U[4][c] = 1
        V[8+b_col_offset][c] = 1; V[b_col_offset][c] = -1
        W[w_col_offset*2][c] = 1; W[w_col_offset*2+1][c] = 1
        col_idx += 1

        # M5 = (x00 + x01)y11
        c = col_idx
        U[0][c] = 1; U[1][c] = 1
        V[8+b_col_offset+1][c] = 1
        W[w_col_offset*2][c] = -1; W[w_col_offset*2+1][c] = 1
        col_idx += 1

        # M6 = (x10 - x00)(y00 + y01)
        c = col_idx
        U[3][c] = 1; U[0][c] = -1
        V[b_col_offset][c] = 1; V[b_col_offset+1][c] = 1
        W[w_col_offset+1*2+1][c] = 1
        col_idx += 1

        # M7 = (x01 - x11)(y10 + y11)
        c = col_idx
        U[1][c] = 1; U[4][c] = -1
        V[8+b_col_offset][c] = 1; V[8+b_col_offset+1][c] = 1
        W[w_col_offset*2][c] = 1; W[w_col_offset*2+1][c] = 1
        col_idx += 1

    def add_direct(b_col_offset, w_col_offset):
        nonlocal col_idx
        # A_bot: U[2] (A[0,2]), U[5] (A[1,2])
        # B_bot: V[16+b_col_offset] (B[2, b_col_offset]), V[16+b_col_offset+1]
        
        # C[0, w_col_offset] += A[0,2] B[2, b_col_offset]
        c = col_idx
        U[2][c] = 1
        V[16+b_col_offset][c] = 1
        W[w_col_offset*2][c] = 1
        col_idx += 1
        
        # C[1, w_col_offset] += A[1,2] B[2, b_col_offset]
        c = col_idx
        U[5][c] = 1
        V[16+b_col_offset][c] = 1
        W[w_col_offset*2+1][c] = 1
        col_idx += 1
        
        # C[0, w_col_offset+1] += A[0,2] B[2, b_col_offset+1]
        c = col_idx
        U[2][c] = 1
        V[16+b_col_offset+1][c] = 1
        W[w_col_offset+1*2][c] = 1
        col_idx += 1
        
        # C[1, w_col_offset+1] += A[1,2] B[2, b_col_offset+1]
        c = col_idx
        U[5][c] = 1
        V[16+b_col_offset+1][c] = 1
        W[w_col_offset+1*2+1][c] = 1
        col_idx += 1

    # Block 0: B cols 0-3, C cols 0-3
    # Split B_top (rows 0,1) into L (cols 0,1) and R (cols 2,3)
    add_strassen(0, 0) # B cols 0,1 -> C cols 0,1
    add_strassen(2, 2) # B cols 2,3 -> C cols 2,3
    add_direct(0, 0)   # B col 0,1 -> C col 0,1
    add_direct(2, 2)   # B col 2,3 -> C col 2,3

    # Block 1: B cols 4-7, C cols 4-7
    add_strassen(4, 4) # B cols 4,5 -> C cols 4,5
    add_strassen(6, 6) # B cols 6,7 -> C cols 6,7
    add_direct(4, 4)   # B col 4,5 -> C col 4,5
    add_direct(6, 6)   # B col 6,7 -> C col 6,7

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
