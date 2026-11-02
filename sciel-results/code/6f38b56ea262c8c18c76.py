# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # Rank 18 decomposition:
    # 2 blocks of Strassen (rank 7 each) for cols 0-1, 2-3
    # 1 block of Schoolbook (rank 4) for col 4
    rank = 18
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(offset, col_start):
        # Strassen for 2x2x2
        # P1 = (a00+a11)(b00+b11)
        U[0][offset] = 1; U[3][offset] = 1
        V[col_start][offset] = 1; V[col_start+5+1][offset] = 1 # b00, b11
        W[0][offset] = 1; W[3][offset] = 1 # c00, c11
        
        # P2 = (a10+a11)b00
        U[2][offset+1] = 1; U[3][offset+1] = 1
        V[col_start][offset+1] = 1
        W[2][offset+1] = 1 # c10
        
        # P3 = a00(b10-b11)
        U[0][offset+2] = 1
        V[col_start+5][offset+2] = 1; V[col_start+5+1][offset+2] = -1
        W[1][offset+2] = 1; W[3][offset+2] = 1 # c01, c11
        
        # P4 = a11(b01-b00)
        U[3][offset+3] = 1
        V[col_start+1][offset+3] = 1; V[col_start][offset+3] = -1
        W[0][offset+3] = 1; W[2][offset+3] = 1 # c00, c10
        
        # P5 = (a00+a01)b11
        U[0][offset+4] = 1; U[1][offset+4] = 1
        V[col_start+5+1][offset+4] = 1
        W[0][offset+4] = -1; W[1][offset+4] = 1 # c00, c01
        
        # P6 = (a10-a00)(b00+b01)
        U[2][offset+5] = 1; U[0][offset+5] = -1
        V[col_start][offset+5] = 1; V[col_start+1][offset+5] = 1
        W[3][offset+5] = 1 # c11
        
        # P7 = (a01-a11)(b10+b11)
        U[1][offset+6] = 1; U[3][offset+6] = -1
        V[col_start+5][offset+6] = 1; V[col_start+5+1][offset+6] = 1
        W[1][offset+6] = 1; W[3][offset+6] = 1 # c01, c11

    def add_schoolbook_block(offset, col_start):
        # Schoolbook for 2x2x1
        # R1 = a00 b0k
        U[0][offset] = 1
        V[col_start][offset] = 1
        W[col_start*2+0][offset] = 1 # c0k
        
        # R2 = a01 b1k
        U[1][offset+1] = 1
        V[col_start+5][offset+1] = 1
        W[col_start*2+0][offset+1] = 1 # c0k
        
        # R3 = a10 b0k
        U[2][offset+2] = 1
        V[col_start][offset+2] = 1
        W[col_start*2+1][offset+2] = 1 # c1k
        
        # R4 = a11 b1k
        U[3][offset+3] = 1
        V[col_start+5][offset+3] = 1
        W[col_start*2+1][offset+3] = 1 # c1k

    # Block 1: cols 0, 1
    add_strassen_block(0, 0)
    # Block 2: cols 2, 3
    add_strassen_block(7, 2)
    # Block 3: col 4
    add_schoolbook_block(14, 4)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
