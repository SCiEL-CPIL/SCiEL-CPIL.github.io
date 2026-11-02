import time

DIMENSIONS = [3, 4, 8]
_DEADLINE = time.time() + 145.0

def _idx(row, row_size, col, col_size):
    return row * col_size + col

def construct():
    # Strassen recursive decomposition for 3x4x8
    # Use 2x2 Strassen (rank 7) to factor the 4 and 8 dimensions
    # Then combine with 3x4 structure
    n, m, p = 3, 4, 8
    
    # Build using recursive Strassen for 4x4 and 8x4 blocks
    # 4 = 2*2, 8 = 2*4, so:
    # - Split m=4 into two 2x2 blocks -> use Strassen (rank 7)
    # - Split p=8 into two 4x4 blocks -> use Strassen (rank 7)
    # - For each i in range(n=3), j in range(m/2=2), k in range(p/2=4):
    #   We need rank-7 decomposition for the 2x2 sub-blocks
    
    # Actually, let's think of it as:
    # Matrix A (3x4), B (4x8)
    # Split A = [A_top | A_bot] where A_top is 3x2, A_bot is 3x2
    # Split B = [B_top; B_bot] where B_top is 2x8, B_bot is 2x8
    # C = A_top @ B_top + A_bot @ B_bot
    # Each 3x2 @ 2x8 product has rank 3*2*8 = 48 in schoolbook
    # But we can use Strassen on the 2x2 sub-blocks
    
    # Better approach: use 2x2 Strassen recursively
    # For 4x4 multiplication: rank 28 (via 4x4 = 2x2 Strassen with 7 mults of 2x2)
    # For 4x8: split 8 into two 4s, so 4x8 = 4x4 + 4x4, each with rank 28
    # Total for 4x8: rank 56
    # For 3x4x8: rank = 3 * 56 = 168 (worse than 96!)
    
    # The schoolbook rank 96 is actually quite good.
    # Let me try a different Strassen approach:
    # 3x4x8: keep the 3 as is, use Strassen on 4x8
    # 4x8: split 4 into 2+2, use Strassen
    # Strassen 2x2: 7 multiplications
    # For 4x8: think of it as computing A(4x4) @ B(4x8)
    # Split A into 2x2 blocks, B into 2x4 blocks
    # Use Strassen: need 7 products of 2x4 matrices
    # Each 2x4 product has rank 2*4 = 8 in schoolbook
    # Total rank for 4x8: 7*8 = 56
    # For 3x4x8: 3*56 = 168 (still worse than 96)
    
    # The issue is that Strassen doesn't help for rectangular matrices in this way.
    # Let me just use the schoolbook decomposition which is rank 96.
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
    return {"U": U, "V": V, "W": W}
