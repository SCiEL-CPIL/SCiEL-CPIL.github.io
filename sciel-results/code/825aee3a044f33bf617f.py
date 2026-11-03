"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-80 exact tensor decomposition for 2x4 by 4x6 matrix multiplication.
    
    Strategy: Partition the 4x6 matrix into two 4x3 blocks (left and right).
    Compute two 2x3 by 3x3 matrix multiplications in parallel using the known
    rank-40 decomposition for 2x3x3 matrix multiplication.
    
    Total rank = 2 * 40 = 80.
    
    The 2x3x3 rank-40 decomposition:
    We use the known optimal rank for 2x3x3 matrix multiplication, which is 40.
    
    For 2x3 by 3x3 matrix multiplication, we can use a decomposition based on
    the fact that rank(2,3,3) = 40. One way to construct this is by using
    a combination of Strassen-like methods and direct terms.
    
    Actually, let me think more carefully. The rank of 2x3x3 matrix multiplication:
    The trivial rank is 2*3*3 = 18.
    
    Wait, I need to reconsider. The known results for small matrix multiplication ranks:
    - rank(2,2,2) = 7 (Strassen)
    - rank(2,2,3) = ?
    - rank(2,3,3) = ?
    
    Let me look at this differently. For 2x3 by 3x3:
    The output is 2x3, so there are 6 output entries.
    Each output C[i,k] = sum_j A[i,j] * B[j,k]
    
    Actually, I recall that the rank of 2x3x3 matrix multiplication is not 40.
    Let me reconsider the approach.
    
    The trivial rank for 2x4 by 4x6 is 2*4*6 = 48.
    
    Wait, I made an error in my reasoning. Let me restart.
    
    For 2x4 by 4x6 matrix multiplication:
    - A is 2x4
    - B is 4x6
    - C = A*B is 2x6
    
    The trivial/schoolbook rank is 2*4*6 = 48.
    
    Can we do better than 48?
    
    Using Strassen-like blocking:
    If we could use rank-7 for 2x2x2, we might be able to reduce the rank.
    
    Let's try to block the matrices:
    A is 2x4, B is 4x6.
    
    One approach: Split B into two 4x3 blocks: B = [B1, B2] where B1, B2 are 4x3.
    Then C = [A*B1, A*B2].
    
    Each A*B1 and A*B2 is a 2x4 by 4x3 multiplication.
    
    For 2x4 by 4x3:
    Trivial rank = 2*4*3 = 24.
    
    Can we do better?
    
    Alternatively, split A into two 2x2 blocks and B into two 2x3 blocks (vertically):
    A = [A1; A2] where A1, A2 are 2x2
    B = [B1; B2] where B1, B2 are 2x3
    
    Then C = A1*B1 + A2*B2, which is 2x3.
    
    Each A1*B1 and A2*B2 is 2x2 by 2x3.
    
    For 2x2 by 2x3:
    Trivial rank = 2*2*3 = 12.
    
    Using Strassen for the 2x2 part:
    A1*B1 where A1 is 2x2 and B1 is 2x3.
    We can view this as multiplying a 2x2 matrix by a 2x3 matrix.
    
    Actually, let me think about this more systematically.
    
    The key insight from the directive is to use a rank-40 decomposition for 2x3x3.
    But I'm not sure rank(2,3,3) = 40. Let me verify.
    
    Actually, I think I was confused. Let me just use a straightforward approach.
    
    For 2x4 by 4x6:
    Trivial rank = 48.
    
    Let me try to use Strassen's method to reduce this.
    
    Approach: Block A as 2x4 = [A_left | A_right] where A_left, A_right are 2x2.
    Block B as 4x6 = [B_top; B_bottom] where B_top, B_bottom are 2x6.
    
    Then C = A_left * B_top + A_right * B_bottom.
    
    Each term is 2x2 by 2x6.
    
    For 2x2 by 2x6:
    Trivial rank = 2*2*6 = 24.
    
    We can further block B_top and B_bottom into three 2x2 blocks each:
    B_top = [Bt1, Bt2, Bt3] where each is 2x2
    B_bottom = [Bb1, Bb2, Bb3] where each is 2x2
    
    Then C = A_left*Bt1 + A_right*Bb1, A_left*Bt2 + A_right*Bb2, A_left*Bt3 + A_right*Bb3
    
    Each output block is 2x2.
    
    For each 2x2 output block: C_j = A_left*Bt_j + A_right*Bb_j
    This is a sum of two 2x2 by 2x2 products.
    
    Using Strassen for each 2x2x2 product (rank 7), we'd need 7+7=14 per block,
    but we can share some computations since A_left and A_right are the same across blocks.
    
    Actually, this is getting complex. Let me just use the trivial decomposition with rank 48,
    which is already better than the current 48... wait, the current is also 48.
    
    Let me re-read the current program. The current rank is n*m*p = 2*4*6 = 48.
    
    So I need to find something better than 48.
    
    Let me try a different blocking strategy.
    
    Actually, I recall that for 2x4 by 4x6, one can potentially use a rank lower than 48.
    
    Let me try: Split the inner dimension (4) into two parts of 2.
    A = [A1 | A2], A1, A2 are 2x2
    B = [B1; B2], B1, B2 are 2x6
    
    C = A1*B1 + A2*B2
    
    Now, A1*B1 is 2x2 by 2x6, and A2*B2 is 2x2 by 2x6.
    
    For a 2x2 by 2x6 multiplication, we can use Strassen-like methods.
    
    Actually, let's think about 2x2 by 2x2 first (rank 7 with Strassen).
    
    For 2x2 by 2x6, we can view it as three 2x2 by 2x2 multiplications (one for each column pair of the 2x6 matrix).
    
    So A1*B1 can be decomposed into 3 Strassen products, each of rank 7, total rank 21.
    Similarly, A2*B2 has rank 21.
    
    But we need to sum A1*B1 + A2*B2. If we do them independently, total rank = 21+21 = 42.
    
    Can we do better by sharing?
    
    Actually, the sum A1*B1 + A2*B2 is a 2x6 matrix.
    
    Let me define:
    M1 = A1*B1 (2x6)
    M2 = A2*B2 (2x6)
    C = M1 + M2
    
    If we compute M1 and M2 independently using Strassen (rank 21 each), total rank = 42.
    
    42 < 48, so this is an improvement!
    
    Let me implement this.
    
    Strassen for 2x2 by 2x2:
    Given X (2x2) and Y (2x2):
    X = [[x1, x2], [x3, x4]]
    Y = [[y1, y2], [y3, y4]]
    
    Z1 = (x1+x4)*(y1+y4)
    Z2 = (x3+x4)*y1
    Z3 = x1*(y2-y4)
    Z4 = x4*(y3-y1)
    Z5 = (x1+x2)*y4
    Z6 = (x3-x1)*(y1+y2)
    Z7 = (x2-x4)*(y3+y4)
    
    Result:
    [[Z1+Z4-Z5+Z7, Z3+Z5],
     [Z2+Z4, Z1-Z2+Z3+Z6]]
    
    For 2x2 by 2x6, we split the 2x6 into three 2x2 blocks.
    
    Let me code this up.
    """
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    # We'll use rank 42: split inner dim into two 2s, use Strassen for each 2x2x2 sub-product
    # A (2x4) = [A1 | A2], A1, A2 are 2x2
    # B (4x6) = [B1; B2], B1, B2 are 2x6
    # C = A1*B1 + A2*B2
    # Each A1*B1 is 2x2 by 2x6 = 3 Strassen products (rank 7 each) = rank 21
    # Each A2*B2 is 2x2 by 2x6 = 3 Strassen products (rank 7 each) = rank 21
    # Total rank = 42
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 12 x 42
    
    # Helper: add a Strassen decomposition for 2x2 by 2x2
    # A_sub is 2x2 (indices in U), B_sub is 2x2 (indices in V)
    # Output is 2x2 (indices in W)
    # Returns the column offset used
    
    def add_strassen(a_rows, b_rows, w_rows, col_offset):
        """
        a_rows: list of 4 row indices in U for the 2x2 matrix A_sub
                 [a00, a01, a10, a11]
        b_rows: list of 4 row indices in V for the 2x2 matrix B_sub
                 [b00, b01, b10, b11]
        w_rows: list of 4 row indices in W for the 2x2 output
                 [w00, w01, w10, w11]
        col_offset: starting column
        """
        a00, a01, a10, a11 = a_rows
        b00, b01, b10, b11 = b_rows
        w00, w01, w10, w11 = w_rows
        
        # Z1 = (a00+a11)*(b00+b11)
        col = col_offset
        U[a00][col] += 1
        U[a11][col] += 1
        V[b00][col] += 1
        V[b11][col] += 1
        W[w00][col] += 1
        W[w11][col] += 1
        col += 1
        
        # Z2 = (a10+a11)*b00
        U[a10][col] += 1
        U[a11][col] += 1
        V[b00][col] += 1
        W[w10][col] += 1
        W[w11][col] -= 1
        col += 1
        
        # Z3 = a00*(b01-b11)
        U[a00][col] += 1
        V[b01][col] += 1
        V[b11][col] -= 1
        W[w01][col] += 1
        W[w10][col] += 1
        col += 1
        
        # Z4 = a11*(b10-b00)
        U[a11][col] += 1
        V[b10][col] += 1
        V[b00][col] -= 1
        W[w00][col] += 1
        W[w10][col] += 1
        col += 1
        
        # Z5 = (a00+a01)*b11
        U[a00][col] += 1
        U[a01][col] += 1
        V[b11][col] += 1
        W[w00][col] -= 1
        W[w01][col] += 1
        col += 1
        
        # Z6 = (a10-a00)*(b00+b01)
        U[a10][col] += 1
        U[a00][col] -= 1
        V[b00][col] += 1
        V[b01][col] += 1
        W[w11][col] += 1
        W[w10][col] -= 1
        col += 1
        
        # Z7 = (a01-a11)*(b10+b11)
        U[a01][col] += 1
        U[a11][col] -= 1
        V[b10][col] += 1
        V[b11][col] += 1
        W[w01][col] += 1
        W[w11][col] += 1
        col += 1
        
        return col
    
    # U indices for A (2x4):
    # A = [[A00, A01, A02, A03],
    #      [A10, A11, A12, A13]]
    # U row indices: Aij -> i*4 + j
    # A1 (left 2x2): rows 0,1,4,5 -> [A00, A01, A10, A11]
    # A2 (right 2x2): rows 2,3,6,7 -> [A02, A03, A12, A13]
    
    # V indices for B (4x6):
    # B = [[B00, B01, B02, B03, B04, B05],
    #      [B10, B11, B12, B13, B14, B15],
    #      [B20, B21, B22, B23, B24, B25],
    #      [B30, B31, B32, B33, B34, B35]]
    # V row indices: Bij -> i*6 + j
    # B1 (top 2x6): rows 0-5, 6-11
    # B2 (bottom 2x6): rows 12-17, 18-23
    
    # W indices for C (2x6):
    # C = [[C00, C01, C02, C03, C04, C05],
    #      [C10, C11, C12, C13, C14, C15]]
    # W row indices: Cik -> k*2 + i (trace convention: W[k*n+i, r])
    # So C00 -> W[0], C01 -> W[1], ..., C05 -> W[5], C10 -> W[6], ..., C15 -> W[11]
    
    col_offset = 0
    
    # Process A1*B1 (A1 is left 2x2 of A, B1 is top 2x6 of B)
    # A1 rows in U: [0, 1, 4, 5] -> [A00, A01, A10, A11]
    a1_rows = [0, 1, 4, 5]
    
    # B1 is 2x6, split into three 2x2 blocks:
    # B1_block0: columns 0,1 -> V rows [0, 1, 6, 7] -> [B00, B01, B10, B11]
    # B1_block1: columns 2,3 -> V rows [2, 3, 8, 9] -> [B02, B03, B12, B13]
    # B1_block2: columns 4,5 -> V rows [4, 5, 10, 11] -> [B04, B05, B14, B15]
    
    # Output C columns 0,1 -> W rows [0, 1, 6, 7] -> [C00, C01, C10, C11]
    w_block0 = [0, 1, 6, 7]
    col_offset = add_strassen(a1_rows, [0, 1, 6, 7], w_block0, col_offset)
    
    # Output C columns 2,3 -> W rows [2, 3, 8, 9] -> [C02, C03, C12, C13]
    w_block1 = [2, 3, 8, 9]
    col_offset = add_strassen(a1_rows, [2, 3, 8, 9], w_block1, col_offset)
    
    # Output C columns 4,5 -> W rows [4, 5, 10, 11] -> [C04, C05, C14, C15]
    w_block2 = [4, 5, 10, 11]
    col_offset = add_strassen(a1_rows, [4, 5, 10, 11], w_block2, col_offset)
    
    # Process A2*B2 (A2 is right 2x2 of A, B2 is bottom 2x6 of B)
    # A2 rows in U: [2, 3, 6, 7] -> [A02, A03, A12, A13]
    a2_rows = [2, 3, 6, 7]
    
    # B2 is 2x6, split into three 2x2 blocks:
    # B2_block0: columns 0,1 -> V rows [12, 13, 18, 19] -> [B20, B21, B30, B31]
    # B2_block1: columns 2,3 -> V rows [14, 15, 20, 21] -> [B22, B23, B32, B33]
    # B2_block2: columns 4,5 -> V rows [16, 17, 22, 23] -> [B24, B25, B34, B35]
    
    # Output C columns 0,1 -> W rows [0, 1, 6, 7] -> [C00, C01, C10, C11]
    col_offset = add_strassen(a2_rows, [12, 13, 18, 19], w_block0, col_offset)
    
    # Output C columns 2,3 -> W rows [2, 3, 8, 9] -> [C02, C03, C12, C13]
    col_offset = add_strassen(a2_rows, [14, 15, 20, 21], w_block1, col_offset)
    
    # Output C columns 4,5 -> W rows [4, 5, 10, 11] -> [C04, C05, C14, C15]
    col_offset = add_strassen(a2_rows, [16, 17, 22, 23], w_block2, col_offset)
    
    assert col_offset == rank, f"Expected rank {rank}, got {col_offset}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
