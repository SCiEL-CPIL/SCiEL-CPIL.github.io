"""Matrix multiplication 2x3x10 rank-minimal decomposition using Strassen."""
import time
import sys

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-55 decomposition of the 2x3x10 matrix multiplication tensor.
    
    Strategy: Split A (2x3) into [A_L (2x2), A_R (2x1)] and 
    B (3x10) into [B_L (2x10), B_R (1x10)].
    
    C = A*B = A_L * B_L + A_R * B_R
    
    Use Strassen's 7-multiplication algorithm for A_L * B_L (2x2 * 2x10 -> 2x10)
    Use standard outer product for A_R * B_R (2x1 * 1x10 -> 2x10)
    
    Total rank = 7 + 4 = 11 per column of B, times 10 columns = 110... 
    Wait, that's not right. Let me reconsider.
    
    Actually, for each pair of columns (k1, k2) in B_L, we can apply Strassen 
    to a 2x2x2 sub-problem. But B_L is 2x10, so we have 10 columns.
    
    Better approach: Group the 10 columns of B into 5 pairs. For each pair 
    (col_a, col_b), we have a 2x2 block of B. Apply Strassen to A_L (2x2) 
    times this 2x2 block -> 2x2 result. That's 7 multiplications per pair.
    Plus 4 standard multiplications for A_R * B_R per pair.
    Total per pair: 11 multiplications.
    5 pairs * 11 = 55 total rank.
    """
    n, m, p = DIMENSIONS  # 2, 3, 10
    
    # We'll build U, V, W matrices
    rank = 55
    U = [[0] * rank for _ in range(n * m)]   # (6, 55)
    V = [[0] * rank for _ in range(m * p)]   # (30, 55)
    W = [[0] * rank for _ in range(p * n)]   # (20, 55)
    
    # Indexing:
    # A is 2x3: A[i,j] where i in [0,1], j in [0,1,2]
    #   U index: i*m + j = i*3 + j
    # B is 3x10: B[j,k] where j in [0,1,2], k in [0,1,9]
    #   V index: j*p + k = j*10 + k
    # C is 2x10: C[i,k] where i in [0,1], k in [0,1,9]
    #   W index: k*n + i = k*2 + i (trace convention)
    
    # Split A into A_L (2x2, columns 0,1) and A_R (2x1, column 2)
    # Split B into B_L (2x10, rows 0,1) and B_R (1x10, row 2)
    
    # For each pair of columns (k1, k2) in B_L, we form a 2x2 block:
    #   B_block = [[B[0,k1], B[0,k2]],
    #              [B[1,k1], B[1,k2]]]
    # 
    # Strassen for A_L * B_block:
    # A_L = [[a, b], [c, d]] where a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
    # B_block = [[x, y], [u, v]] where x=B[0,k1], y=B[0,k2], u=B[1,k1], v=B[1,k2]
    #
    # Strassen products:
    # M1 = (a+d)(x+u)
    # M2 = (c+d)x
    # M3 = a(y-u)
    # M4 = d(v-u)
    # M5 = (a+b)u
    # M6 = (c-a)(x+y)
    # M7 = (b-d)(v+y)
    #
    # C_block = [[M4+M5+M7, M3+M5],
    #            [M2+M4, M1+M6]]
    #
    # Then add A_R * B_R contributions:
    # A_R = [A[0,2], A[1,2]]
    # B_R row = [B[2,k1], B[2,k2]]
    # 
    # C[0,k1] += A[0,2]*B[2,k1]  -> M8
    # C[0,k2] += A[0,2]*B[2,k2]  -> M9
    # C[1,k1] += A[1,2]*B[2,k1]  -> M10
    # C[1,k2] += A[1,2]*B[2,k2]  -> M11
    
    col = 0
    for pair_idx in range(5):
        k1 = pair_idx * 2
        k2 = pair_idx * 2 + 1
        
        # M1 = (a+d)(x+u)
        U[0*3+0][col] = 1  # a = A[0,0]
        U[1*3+1][col] = 1  # d = A[1,1]
        V[0*10+k1][col] = 1  # x = B[0,k1]
        V[1*10+k1][col] = 1  # u = B[1,k1]
        # W: contributes to C[0,k1] and C[1,k1] with +1? 
        # C_block[0,0] = M4+M5+M7, C_block[1,1] = M1+M6
        # So M1 contributes to C[1,k2] with +1
        W[k2*2+1][col] = 1  # C[1, k2]
        col += 1
        
        # M2 = (c+d)x
        U[1*3+0][col] = 1  # c = A[1,0]
        U[1*3+1][col] = 1  # d = A[1,1]
        V[0*10+k1][col] = 1  # x = B[0,k1]
        # C_block[1,0] = M2+M4
        W[k1*2+1][col] = 1  # C[1, k1]
        col += 1
        
        # M3 = a(y-u)
        U[0*3+0][col] = 1  # a = A[0,0]
        V[0*10+k2][col] = 1  # y = B[0,k2]
        V[1*10+k1][col] = -1  # u = B[1,k1]
        # C_block[0,1] = M3+M5
        W[k2*2+0][col] = 1  # C[0, k2]
        col += 1
        
        # M4 = d(v-u)
        U[1*3+1][col] = 1  # d = A[1,1]
        V[1*10+k2][col] = 1  # v = B[1,k2]
        V[1*10+k1][col] = -1  # u = B[1,k1]
        # C_block[0,0] = M4+M5+M7, C_block[1,0] = M2+M4
        W[k1*2+0][col] = 1  # C[0, k1]
        W[k1*2+1][col] = 1  # C[1, k1]
        col += 1
        
        # M5 = (a+b)u
        U[0*3+0][col] = 1  # a = A[0,0]
        U[0*3+1][col] = 1  # b = A[0,1]
        V[1*10+k1][col] = 1  # u = B[1,k1]
        # C_block[0,0] = M4+M5+M7, C_block[0,1] = M3+M5
        W[k1*2+0][col] = 1  # C[0, k1]
        W[k2*2+0][col] = 1  # C[0, k2]
        col += 1
        
        # M6 = (c-a)(x+y)
        U[1*3+0][col] = 1  # c = A[1,0]
        U[0*3+0][col] = -1  # a = A[0,0]
        V[0*10+k1][col] = 1  # x = B[0,k1]
        V[0*10+k2][col] = 1  # y = B[0,k2]
        # C_block[1,1] = M1+M6
        W[k2*2+1][col] = 1  # C[1, k2]
        col += 1
        
        # M7 = (b-d)(v+y)
        U[0*3+1][col] = 1  # b = A[0,1]
        U[1*3+1][col] = -1  # d = A[1,1]
        V[1*10+k2][col] = 1  # v = B[1,k2]
        V[0*10+k2][col] = 1  # y = B[0,k2]
        # C_block[0,0] = M4+M5+M7
        W[k1*2+0][col] = 1  # C[0, k1]
        col += 1
        
        # M8 = A[0,2] * B[2,k1]
        U[0*3+2][col] = 1  # A[0,2]
        V[2*10+k1][col] = 1  # B[2,k1]
        W[k1*2+0][col] = 1  # C[0, k1]
        col += 1
        
        # M9 = A[0,2] * B[2,k2]
        U[0*3+2][col] = 1  # A[0,2]
        V[2*10+k2][col] = 1  # B[2,k2]
        W[k2*2+0][col] = 1  # C[0, k2]
        col += 1
        
        # M10 = A[1,2] * B[2,k1]
        U[1*3+2][col] = 1  # A[1,2]
        V[2*10+k1][col] = 1  # B[2,k1]
        W[k1*2+1][col] = 1  # C[1, k1]
        col += 1
        
        # M11 = A[1,2] * B[2,k2]
        U[1*3+2][col] = 1  # A[1,2]
        V[2*10+k2][col] = 1  # B[2,k2]
        W[k2*2+1][col] = 1  # C[1, k2]
        col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
