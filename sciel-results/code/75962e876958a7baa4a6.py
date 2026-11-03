"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Decompose 3x4 * 4x4 matrix multiplication using Winograd's F(2) scheme
    applied to overlapping 2x2 blocks of the matrices.
    
    Strategy: Apply Winograd F(2) to pairs of adjacent elements in the
    summation index. For each output element C[i,k] = sum_j A[i,j]*B[j,k],
    we group the 4 terms into two pairs: (j=0,1) and (j=2,3).
    
    For each pair (j, j+1), we use Winograd F(2) which requires 4 multiplications
    instead of 2, but with a clever combination we can share computations.
    
    Actually, let me use a cleaner approach: decompose using the fact that
    we can apply Strassen-like ideas to 2x2 sub-blocks.
    
    For the 3x4 * 4x4 case, let's split B into two 2x4 blocks (rows 0-1 and 2-3)
    and A into columns 0-1 and 2-3.
    
    C = A[:, 0:2] @ B[0:2, :] + A[:, 2:4] @ B[2:4, :]
    
    Each term is a 3x2 * 2x4 product. The rank of 3x2*2x4 is 12 (schoolbook)
    but can be reduced. Actually, 3x2*2x4 = 3*(1x2*2x4). 1x2*2x4 has rank 8
    schoolbook, but using Winograd on the 2x2 sub-blocks...
    
    Let me try a different decomposition. Split the 4x4 B matrix into four
    2x2 blocks. Split A (3x4) into three 1x4 rows, each split into two 1x2
    halves.
    
    For row i of A, and each 2x2 block of B:
    C[i, block] = A[i, half] @ B[block]
    
    This is 1x2 * 2x2 for each combination. There are 3 rows * 4 blocks = 12
    such products. Each 1x2*2x2 has schoolbook rank 4, but Winograd can do it
    in 3 multiplications.
    
    Wait, 1x2 * 2x2: C[0] = a0*b00 + a1*b10, C[1] = a0*b01 + a1*b11
    This is just two dot products of length 2. Winograd F(2) for dot product
    of length 2: m1 = (a0+a1)*(b0+b1), then C[0] = (m1 + (a0-a1)*(b0-b1))/2
    That's 2 multiplications, same as schoolbook. No gain.
    
    Let me think differently. The total schoolbook rank is 48. I need to find
    a valid decomposition with rank < 48.
    
    Known: The rank of 2x2*2x2 is 7 (Strassen). 
    If I can tile the 3x4*4x4 with 2x2*2x2 blocks and use Strassen, I can
    reduce the rank.
    
    Split A (3x4) into blocks: top 2x4 and bottom 1x4.
    Split B (4x4) into blocks: left 4x2 and right 4x2.
    
    C_top_left (2x2) = A_top (2x4) @ B_left (4x2)
    C_top_right (2x2) = A_top (2x4) @ B_right (4x2)
    C_bot_left (1x2) = A_bot (1x4) @ B_left (4x2)
    C_bot_right (1x2) = A_bot (1x4) @ B_right (4x2)
    
    Each 2x4*4x2 can be seen as 2x2 blocks. Let's use Strassen on 2x2 sub-blocks.
    
    Actually, this is getting complex. Let me just implement a valid rank-40
    decomposition using a combination of direct terms and some Winograd-style
    sharing.
    
    For now, let me provide a correct schoolbook decomposition as a baseline,
    and then try to improve it.
    """
    n, m, p = DIMENSIONS  # 3, 4, 4
    
    rank = 48
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
