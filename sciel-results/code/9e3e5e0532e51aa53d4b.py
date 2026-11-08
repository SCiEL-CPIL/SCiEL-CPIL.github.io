"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-42 decomposition of 2x4 * 4x6 matrix multiplication using Strassen
    applied to the two 2x2 blocks along the inner dimension.
    
    We split the 4-dimensional inner product into two 2-dimensional halves.
    The problem becomes: C = A1*B1 + A2*B2 where A1,A2 are 2x2, B1,B2 are 2x2.
    
    Each 2x2 matrix product can be computed with 7 Strassen multiplications,
    but since we're doing a sum of two such products, we can interleave them.
    
    Actually, let's think more carefully. The full 2x4 * 4x6 product is:
    C[i][k] = sum_{j=0}^{3} A[i][j] * B[j][k]
    
    Split j into j=0,1 (first half) and j=2,3 (second half):
    C[i][k] = sum_{j=0}^{1} A[i][j] * B[j][k] + sum_{j=0}^{1} A[i][j+2] * B[j+2][k]
    
    So C = C1 + C2 where C1 = A1*B1 and C2 = A2*B2, with A1,A2 being 2x2 and B1,B2 being 2x6.
    
    For each 2x2 * 2x6 product, we can use Strassen-like decomposition.
    A 2x2 * 2x6 can be decomposed by splitting the 6 columns into three 2-column groups.
    So 2x2 * 2x6 = three instances of 2x2 * 2x2, each with rank 7 (Strassen).
    Total for one 2x2*2x6: 3*7 = 21
    Two such: 42 total.
    
    Let's implement this explicitly.
    
    U has shape (8, 42), V has shape (24, 42), W has shape (12, 42).
    
    The Strassen decomposition for 2x2 * 2x2:
    Input: A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    Output: C = A*B = [[c00, c01], [c10, c11]]
    
    Strassen:
    m0 = (a00 + a11) * (b00 + b11)
    m1 = (a10 + a11) * b00
    m2 = a00 * (b10 - b11)
    m3 = a11 * (b01 - b00)
    m4 = (a00 + a01) * b11
    m5 = (a10 - a00) * (b00 + b10)
    m6 = (a01 - a11) * (b01 + b11)
    
    c00 = m0 - m3 + m5 - m6
    c01 = m2 + m4
    c10 = m1 + m3
    c11 = m0 - m2 + m4 - m5
    
    For each such 2x2*2x2 block, we need 7 rank-1 terms.
    
    Let me index the columns of W by (k_block, i) where k_block in {0,1,2} for the three 2-column groups, i in {0,1}.
    Row of U: (i, j_half) where i in {0,1}, j_half in {0,1} for the two halves of the 4-dim inner product.
    Actually, U row index: i*m + j where m=4, so row = i*4 + j, i in {0,1}, j in {0,1,2,3}.
    V row index: j*p + k where p=6, so row = j*6 + k, j in {0,1,2,3}, k in {0,1,2,3,4,5}.
    W column index: k*n + i where n=2, so col = k*2 + i, k in {0,1,2,3,4,5}, i in {0,1}.
    
    Let me structure this differently. For each of the two halves (h in {0,1}) of the inner dimension,
    and for each of the three 2-column groups (g in {0,1,2}) of the output,
    we have a 2x2 * 2x2 Strassen block.
    
    For half h, the inner indices are j = 2*h and j = 2*h+1.
    For column group g, the output columns are k = 2*g and k = 2*g+1.
    
    In the Strassen block for (h, g):
    A_block[i][j'] where i in {0,1}, j' in {0,1} maps to U[i*4 + (2*h + j')]
    B_block[j'][k'] where j' in {0,1}, k' in {0,1} maps to V[(2*h + j')*6 + (2*g + k')]
    C_block[i][k'] maps to W[(2*g + k')*2 + i]
    
    The 7 Strassen terms for this block contribute to 7 columns of the decomposition.
    
    Let's code it up.
    """
    n, m, p = DIMENSIONS  # 2, 4, 6
    rank = 42
    U = [[0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 12 x 42
    
    col = 0
    for h in range(2):  # two halves of inner dimension
        for g in range(3):  # three 2-column groups of output
            # Strassen indices for this 2x2*2x2 block
            # A indices: i in {0,1}, j' in {0,1} -> U row: i*4 + (2*h + j')
            # B indices: j' in {0,1}, k' in {0,1} -> V row: (2*h + j')*6 + (2*g + k')
            # C indices: i in {0,1}, k' in {0,1} -> W col: (2*g + k')*2 + i
            
            # m0 = (a00 + a11) * (b00 + b11)
            U[0*4 + (2*h + 0)][col] += 1
            U[1*4 + (2*h + 1)][col] += 1
            V[(2*h + 0)*6 + (2*g + 0)][col] += 1
            V[(2*h + 1)*6 + (2*g + 1)][col] += 1
            W[(2*g + 0)*2 + 0][col] += 1
            W[(2*g + 1)*2 + 0][col] += 0
            W[(2*g + 0)*2 + 1][col] += 0
            W[(2*g + 1)*2 + 1][col] += 1
            col += 1
            
            # m1 = (a10 + a11) * b00
            U[1*4 + (2*h + 0)][col] += 1
            U[1*4 + (2*h + 1)][col] += 1
            V[(2*h + 0)*6 + (2*g + 0)][col] += 1
            W[(2*g + 0)*2 + 1][col] += 1
            W[(2*g + 1)*2 + 1][col] += 1
            col += 1
            
            # m2 = a00 * (b10 - b11)
            U[0*4 + (2*h + 0)][col] += 1
            V[(2*h + 1)*6 + (2*g + 0)][col] += 1
            V[(2*h + 1)*6 + (2*g + 1)][col] += -1
            W[(2*g + 1)*2 + 0][col] += 1
            col += 1
            
            # m3 = a11 * (b01 - b00)
            U[1*4 + (2*h + 1)][col] += 1
            V[(2*h + 0)*6 + (2*g + 1)][col] += 1
            V[(2*h + 0)*6 + (2*g + 0)][col] += -1
            W[(2*g + 0)*2 + 0][col] += -1
            W[(2*g + 1)*2 + 1][col] += 1
            col += 1
            
            # m4 = (a00 + a01) * b11
            U[0*4 + (2*h + 0)][col] += 1
            U[0*4 + (2*h + 1)][col] += 1
            V[(2*h + 1)*6 + (2*g + 1)][col] += 1
            W[(2*g + 1)*2 + 0][col] += 1
            W[(2*g + 1)*2 + 1][col] += 1
            col += 1
            
            # m5 = (a10 - a00) * (b00 + b10)
            U[1*4 + (2*h + 0)][col] += 1
            U[0*4 + (2*h + 0)][col] += -1
            V[(2*h + 0)*6 + (2*g + 0)][col] += 1
            V[(2*h + 1)*6 + (2*g + 0)][col] += 1
            W[(2*g + 0)*2 + 0][col] += 1
            W[(2*g + 1)*2 + 1][col] += -1
            col += 1
            
            # m6 = (a01 - a11) * (b01 + b11)
            U[0*4 + (2*h + 1)][col] += 1
            U[1*4 + (2*h + 1)][col] += -1
            V[(2*h + 0)*6 + (2*g + 1)][col] += 1
            V[(2*h + 1)*6 + (2*g + 1)][col] += 1
            W[(2*g + 1)*2 + 0][col] += -1
            W[(2*g + 1)*2 + 1][col] += -1
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
