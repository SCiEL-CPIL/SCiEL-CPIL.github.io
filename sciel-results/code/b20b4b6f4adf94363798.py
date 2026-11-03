"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-40 exact tensor decomposition for 2x4 * 4x6 matrix multiplication.
    
    Strategy: Use Strassen-like blocking to reduce rank below the trivial 48.
    Split the inner dimension (4) into two groups of 2.
    A = [A1 A2] where A1, A2 are 2x2.
    B = [B1; B2] where B1, B2 are 2x6.
    C = A1*B1 + A2*B2.
    
    Each sub-problem is 2x2 * 2x6.
    We can decompose 2x2 * 2x6 by treating the 6 columns of B as three groups of 2.
    For each group of 2 columns, we use Strassen's rank-7 algorithm for 2x2 * 2x2.
    
    So for each 2x2 * 2x6, we have 3 * 7 = 21 multiplications.
    Total rank = 21 + 21 = 42.
    
    Can we do better?
    Actually, the 2x2 * 2x6 tensor can be decomposed with rank 12 if we don't use Strassen,
    but Strassen gives 21 which is worse than trivial 24? No, 21 < 24.
    
    Wait, trivial rank for 2x2 * 2x6 is 2*2*6 = 24.
    Strassen-based: 3 * 7 = 21.
    So 21 < 24.
    Total rank = 21 + 21 = 42.
    
    Is there a better decomposition for 2x2 * 2x6?
    The rank of M_{2,2,6} is known to be 12? No, that's too low.
    Lower bound: max(4, 12, 12) = 12.
    Upper bound: 24.
    
    Actually, for M_{2,2,n}, the rank is 2n for n=1, and 7 for n=2.
    For n=6, it's not obvious.
    
    Let's try a different approach.
    Use the rank-40 decomposition mentioned in the inspiration.
    
    The inspiration says: "Apply the known rank-40 exact tensor decomposition for 2x3x3".
    This is confusing. 2*3*3=18. Rank 40 is impossible.
    
    Maybe it means 4x4x4? Rank 40 is better than 64.
    Or maybe 3x3x3? Rank 21.
    
    Let's ignore the specific number and just use a valid decomposition.
    
    I will use the Strassen-based blocking with rank 42.
    """
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    # We'll construct the decomposition by combining two 2x2*2x6 sub-problems.
    # Each 2x2*2x6 uses 3 Strassen 2x2*2x2 blocks.
    
    rank = 42  # 2 * 3 * 7
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 12 x 42
    
    # Helper function to add a Strassen 2x2*2x2 block
    def add_strassen_block(a_offset, b_offset, c_offset, col_start):
        """
        Add a Strassen 2x2*2x2 block.
        a_offset: (row_offset, col_offset) for A sub-block (2x2)
        b_offset: (row_offset, col_offset) for B sub-block (2x2)
        c_offset: (row_offset, col_offset) for C sub-block (2x2)
        col_start: starting column index in the decomposition
        """
        ar, ac = a_offset
        br, bc = b_offset
        cr, cc = c_offset
        
        # Strassen's 7 multiplications:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 - M4 + M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # M1: (A00+A11)(B00+B11)
        U[ar + ac][col_start] = 1       # A00
        U[ar + 1 + ac + 1][col_start] = 1  # A11
        V[br + bc][col_start] = 1       # B00
        V[br + 1 + bc + 1][col_start] = 1  # B11
        W[cr + cc][col_start] = 1       # C00
        W[cr + 1 + cc + 1][col_start] = 1  # C11
        
        # M2: (A10+A11)B00
        U[ar + 1 + ac][col_start + 1] = 1   # A10
        U[ar + 1 + ac + 1][col_start + 1] = 1  # A11
        V[br + bc][col_start + 1] = 1       # B00
        W[cr + 1 + cc][col_start + 1] = 1   # C10
        
        # M3: A00(B01-B11)
        U[ar + ac][col_start + 2] = 1       # A00
        V[br + bc + 1][col_start + 2] = 1   # B01
        V[br + 1 + bc + 1][col_start + 2] = -1  # -B11
        W[cr + cc + 1][col_start + 2] = 1   # C01
        W[cr + 1 + cc + 1][col_start + 2] = 1  # C11
        
        # M4: A11(B10-B00)
        U[ar + 1 + ac + 1][col_start + 3] = 1  # A11
        V[br + 1 + bc][col_start + 3] = 1      # B10
        V[br + bc][col_start + 3] = -1         # -B00
        W[cr + cc][col_start + 3] = -1         # -C00
        W[cr + 1 + cc][col_start + 3] = 1      # C10
        
        # M5: (A00+A01)B11
        U[ar + ac][col_start + 4] = 1       # A00
        U[ar + ac + 1][col_start + 4] = 1   # A01
        V[br + 1 + bc + 1][col_start + 4] = 1  # B11
        W[cr + cc][col_start + 4] = 1       # C00
        W[cr + cc + 1][col_start + 4] = 1   # C01
        
        # M6: (A10-A00)(B00+B01)
        U[ar + 1 + ac][col_start + 5] = 1   # A10
        U[ar + ac][col_start + 5] = -1      # -A00
        V[br + bc][col_start + 5] = 1       # B00
        V[br + bc + 1][col_start + 5] = 1   # B01
        W[cr + 1 + cc + 1][col_start + 5] = 1  # C11
        
        # M7: (A01-A11)(B10+B11)
        U[ar + ac + 1][col_start + 6] = 1   # A01
        U[ar + 1 + ac + 1][col_start + 6] = -1  # -A11
        V[br + 1 + bc][col_start + 6] = 1   # B10
        V[br + 1 + bc + 1][col_start + 6] = 1  # B11
        W[cr + cc][col_start + 6] = 1       # C00
    
    # A is 2x4: indices (i, j) where i in {0,1}, j in {0,1,2,3}
    # U index: i*4 + j
    # A1 = A[:, 0:2], A2 = A[:, 2:4]
    
    # B is 4x6: indices (j, k) where j in {0,1,2,3}, k in {0,1,2,3,4,5}
    # V index: j*6 + k
    # B1 = B[0:2, :], B2 = B[2:4, :]
    
    # C is 2x6: indices (i, k) where i in {0,1}, k in {0,1,2,3,4,5}
    # W index: k*2 + i (trace convention W[k*n+i, r])
    
    # Sub-problem 1: A1 (2x2) * B1 (2x6)
    # A1 is at A[:, 0:2], so a_offset = (0, 0)
    # B1 is at B[0:2, :], so b_offset = (0, 0)
    # C is the full 2x6, so c_offset = (0, 0)
    
    # B1 has 6 columns, so 3 groups of 2 columns each.
    # Group 0: columns 0,1 -> b_offset col = 0
    # Group 1: columns 2,3 -> b_offset col = 2
    # Group 2: columns 4,5 -> b_offset col = 4
    
    add_strassen_block((0, 0), (0, 0), (0, 0), 0)
    add_strassen_block((0, 0), (0, 2), (0, 2), 7)
    add_strassen_block((0, 0), (0, 4), (0, 4), 14)
    
    # Sub-problem 2: A2 (2x2) * B2 (2x6)
    # A2 is at A[:, 2:4], so a_offset = (0, 2)
    # B2 is at B[2:4, :], so b_offset = (2, 0)
    # C is the full 2x6, so c_offset = (0, 0)
    
    add_strassen_block((0, 2), (2, 0), (0, 0), 21)
    add_strassen_block((0, 2), (2, 2), (0, 2), 28)
    add_strassen_block((0, 2), (2, 4), (0, 4), 35)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
