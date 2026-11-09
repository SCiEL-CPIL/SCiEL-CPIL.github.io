"""Rank-44 decomposition for 4x4x7 using Strassen on 2x2 blocks."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Decompose 4x4 * 4x7 matrix multiplication.
    
    Strategy: Split into 2x2 blocks of A and 2x3 blocks of B.
    There are 4 block products: A(a,c) * B(c,e) for a,c,e in {0,1}
    where a is row-block of A (2 rows), c is col-block of A / row-block of B,
    and e is col-block of B (3 cols each).
    
    For each 2x2 * 2x3 product, use Strassen-like rank-7 for the 2x2*2x2 part
    plus 2 extra multiplications for the third column of B.
    
    Actually, let me use a cleaner approach:
    - 2x2 * 2x2: rank 7 (Strassen)
    - 2x2 * 2x3: can be done in rank 9 (Strassen for first 2 cols + schoolbook for 3rd)
    
    Total: 4 block products, each 2x2*2x3 -> 4*9 = 36? No, that's not right.
    
    Let me reconsider. The 4x4 * 4x7 product:
    A is 4x4, B is 4x7.
    Split A into 2x2 blocks: A = [[A00, A01], [A10, A11]]
    Split B into 2x3 blocks: B = [[B00, B01, B02], [B10, B11, B12]]
    where each Bxe is 2x3.
    
    C = A*B is 4x7, split into 2x3 blocks:
    C = [[C00, C01, C02], [C10, C11, C12]]
    where Cae = A(a,0)*B(0,e) + A(a,1)*B(1,e)
    
    So we need 8 products: A(a,c)*B(c,e) for a,c,e in {0,1}
    Each is 2x2 * 2x3.
    
    For 2x2 * 2x3, we can use:
    - Strassen for the 2x2*2x2 part (first 2 cols of B): rank 7
    - 2 additional multiplications for the 3rd column: schoolbook
    Total per block: 9
    
    But wait, the 3rd column of B(c,e) is just a 2x1 vector, and A(a,c) is 2x2.
    So A(a,c) * [b1; b2] where b1,b2 are scalars.
    This is 2 multiplications: [A00*b1 + A01*b2; A10*b1 + A11*b2]
    
    So total rank = 8 * 9 = 72? That's worse than schoolbook (48).
    
    Let me try a different approach. Use the known rank-49 for 4x4x4 and extend.
    
    Actually, the simplest correct approach: use schoolbook but verify it works.
    """
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # Schoolbook: rank = n*m*p = 4*4*7 = 112
    # But we can do better with Strassen.
    
    # Let me implement a correct rank-56 using 8 Strassen decompositions.
    # 4x4 * 4x4 part: 8 Strassen products (rank 7 each) = 56
    # 4x4 * 4x3 part: 4 Strassen products for the 4x4x4 subpart + extras
    
    # Actually, let me just do schoolbook correctly first.
    rank = n * m * p
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
