"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    # Block schoolbook: split into 2x2 blocks
    # A is 4x4, B is 4x8, C is 4x8
    # C = A * B, split into 2x2 blocks of size 2
    # A = [[A11, A12], [A21, A22]] (each 2x2)
    # B = [[B11, B12, B13, B14], [B21, B22, B23, B24]] (each 2x2, so B is 2x4 blocks)
    # Actually B is 4x8, so split into 2x4 blocks: B_top (2x4), B_bot (2x4)
    # C = [[A11*B_top + A12*B_bot], [A21*B_top + A22*B_bot]]
    # Each 2x2 x 2x4 schoolbook has rank 2*2*4 = 8
    # Total rank = 4 * 8 = 32
    
    rank = 32
    U = [[0] * rank for _ in range(n * m)]  # 16 x 32
    V = [[0] * rank for _ in range(m * p)]  # 32 x 32
    W = [[0] * rank for _ in range(p * n)]  # 32 x 32
    
    col = 0
    
    # For each block multiplication, enumerate all (i,j,k) triples
    # Block (arow0, acol0, brow0, bcol0, crow0, ccol0) with 2x2 A block, 2x4 B block, 2x4 C block
    def add_block(arow0, acol0, brow0, bcol0, crow0, ccol0, col):
        for di in range(2):
            for dj in range(2):
                for dk in range(4):
                    a_idx = (arow0 + di) * m + (acol0 + dj)
                    b_idx = (brow0 + dj) * p + (bcol0 + dk)
                    c_idx = (ccol0 + dk) * n + (crow0 + di)
                    U[a_idx][col] = 1
                    V[b_idx][col] = 1
                    W[c_idx][col] = 1
                    col += 1
        return col
    
    # Block 1: A11 * B_top -> C top
    col = add_block(0, 0, 0, 0, 0, 0, col)
    # Block 2: A12 * B_bot -> C top
    col = add_block(0, 2, 2, 0, 0, 0, col)
    # Block 3: A21 * B_top -> C bottom
    col = add_block(2, 0, 0, 0, 2, 0, col)
    # Block 4: A22 * B_bot -> C bottom
    col = add_block(2, 2, 2, 0, 2, 0, col)
    
    assert col == rank, f"Expected {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
