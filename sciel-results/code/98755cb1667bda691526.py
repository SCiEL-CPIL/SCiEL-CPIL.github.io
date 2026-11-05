import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def _schoolbook_factors(n, m, p):
    """Return (U, V, W) as lists for schoolbook decomposition of n x m by m x p."""
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
    return U, V, W

def _strassen_factors():
    """Return (U, V, W) as lists for Strassen decomposition of 2x2 by 2x2, rank 7.
    Indices: U[i*m+j, c] with m=2 so U has 4 rows.
             V[j*p+k, c] with p=2 so V has 4 rows.
             W[k*n+i, c] with n=2 so W has 4 rows.
    """
    # Strassen 7 products for 2x2
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need to express this as tensor decomposition.
    # Each product Mk contributes to C elements with coefficients.
    # U factors extract linear combinations from A.
    # V factors extract linear combinations from B.
    # W factors map products to C elements.
    
    # Let's define the 7 products explicitly.
    # A indices: (0,0)=A11, (0,1)=A12, (1,0)=A21, (1,1)=A22
    # B indices: (0,0)=B11, (0,1)=B12, (1,0)=B21, (1,1)=B22
    # C indices: (0,0)=C11, (0,1)=C12, (1,0)=C21, (1,1)=C22
    
    # Product 1: (A11+A22)(B11+B22)
    # A part: A11+A22 -> U[:,0] = [1, 0, 0, 1]
    # B part: B11+B22 -> V[:,0] = [1, 0, 0, 1]
    # C part: C11 += 1, C22 += 1 -> W[:,0] = [1, 0, 0, 1]
    
    # Product 2: (A21+A22)B11
    # A part: A21+A22 -> U[:,1] = [0, 0, 1, 1]
    # B part: B11 -> V[:,1] = [1, 0, 0, 0]
    # C part: C21 += 1, C22 -= 1 -> W[:,1] = [0, 0, 1, -1]
    
    # Product 3: A11(B21-B22)
    # A part: A11 -> U[:,2] = [1, 0, 0, 0]
    # B part: B21-B22 -> V[:,2] = [0, 0, 1, -1]
    # C part: C12 += 1, C22 += 1 -> W[:,2] = [0, 1, 0, 1]
    
    # Product 4: A22(B12-B11)
    # A part: A22 -> U[:,3] = [0, 0, 0, 1]
    # B part: B12-B11 -> V[:,3] = [0, 1, -1, 0]
    # C part: C11 += 1, C21 += 1 -> W[:,3] = [1, 0, 1, 0]
    
    # Product 5: (A11+A12)B22
    # A part: A11+A12 -> U[:,4] = [1, 1, 0, 0]
    # B part: B22 -> V[:,4] = [0, 0, 0, 1]
    # C part: C11 -= 1, C12 += 1 -> W[:,4] = [-1, 1, 0, 0]
    
    # Product 6: (A21-A11)(B11+B12)
    # A part: A21-A11 -> U[:,5] = [-1, 0, 1, 0]
    # B part: B11+B12 -> V[:,5] = [1, 1, 0, 0]
    # C part: C22 += 1 -> W[:,5] = [0, 0, 0, 1]
    
    # Product 7: (A12-A22)(B21+B22)
    # A part: A12-A22 -> U[:,6] = [0, 1, 0, -1]
    # B part: B21+B22 -> V[:,6] = [0, 0, 1, 1]
    # C part: C11 += 1 -> W[:,6] = [1, 0, 0, 0]
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # U[i*m+j, c] -> U[i*2+j, c]
    U[0][0] = 1; U[3][0] = 1  # A11+A22
    U[2][1] = 1; U[3][1] = 1  # A21+A22
    U[0][2] = 1               # A11
    U[3][3] = 1               # A22
    U[0][4] = 1; U[1][4] = 1  # A11+A12
    U[0][5] = -1; U[2][5] = 1 # A21-A11
    U[1][6] = 1; U[3][6] = -1 # A12-A22
    
    # V[j*p+k, c] -> V[j*2+k, c]
    V[0][0] = 1; V[3][0] = 1  # B11+B22
    V[0][1] = 1               # B11
    V[2][2] = 1; V[3][2] = -1 # B21-B22
    V[1][3] = 1; V[0][3] = -1 # B12-B11
    V[3][4] = 1               # B22
    V[0][5] = 1; V[1][5] = 1  # B11+B12
    V[2][6] = 1; V[3][6] = 1  # B21+B22
    
    # W[k*n+i, c] -> W[k*2+i, c]
    W[0][0] = 1; W[3][0] = 1  # C11+C22
    W[2][1] = 1; W[3][1] = -1 # C21-C22
    W[1][2] = 1; W[3][2] = 1  # C12+C22
    W[0][3] = 1; W[2][3] = 1  # C11+C21
    W[0][4] = -1; W[1][4] = 1 # -C11+C12
    W[3][5] = 1               # C22
    W[0][6] = 1               # C11
    
    return U, V, W

def _block_decompose_5x5x5():
    """
    Decompose 5x5 into 3x3 + 2x2 blocks.
    A = [[A33, A32], [A23, A22]]
    B = [[B33, B32], [B23, B22]]
    C = [[C33, C32], [C23, C22]]
    
    C33 = A33*B33 + A32*B23
    C32 = A33*B32 + A32*B22
    C23 = A23*B33 + A22*B23
    C22 = A23*B32 + A22*B22
    
    For 3x3, use block decomposition 2+1:
    A33 = [[A22, A21], [A12, A11]] (2x2, 2x1, 1x2, 1x1)
    B33 = [[B22, B21], [B12, B11]]
    
    Actually, let's use the standard block decomposition for 3x3:
    Split 3 into 2+1.
    A33 = [[A_2x2, A_2x1], [A_1x2, A_1x1]]
    B33 = [[B_2x2, B_2x1], [B_1x2, B_1x1]]
    
    C33_2x2 = A_2x2 * B_2x2 + A_2x1 * B_1x2
    C33_2x1 = A_2x2 * B_2x1 + A_2x1 * B_1x1
    C33_1x2 = A_1x2 * B_2x2 + A_1x1 * B_1x2
    C33_1x1 = A_1x2 * B_2x1 + A_1x1 * B_1x1
    
    Using Strassen for 2x2 (rank 7) and schoolbook for others:
    C33_2x2: 7 (Strassen) + 4 (2x1 * 1x2 schoolbook) = 11
    C33_2x1: 4 (2x2 * 2x1 schoolbook) + 2 (2x1 * 1x1 schoolbook) = 6
    C33_1x2: 4 (1x2 * 2x2 schoolbook) + 2 (1x1 * 1x2 schoolbook) = 6
    C33_1x1: 2 (1x2 * 2x1 schoolbook) + 1 (1x1 * 1x1 schoolbook) = 3
    Total for 3x3: 11 + 6 + 6 + 3 = 26
    
    Wait, earlier I calculated 21. Let's re-verify.
    C33_2x2: A_2x2*B_2x2 (rank 7) + A_2x1*B_1x2 (rank 4). Total 11.
    C33_2x1: A_2x2*B_2x1 (rank 4) + A_2x1*B_1x1 (rank 2). Total 6.
    C33_1x2: A_1x2*B_2x2 (rank 4) + A_1x1*B_1x2 (rank 2). Total 6.
    C33_1x1: A_1x2*B_2x1 (rank 2) + A_1x1*B_1x1 (rank 1). Total 3.
    Sum: 11+6+6+3 = 26.
    
    Can we do better for 3x3?
    Maybe use Strassen for more blocks?
    A_2x2 * B_2x1: Strassen? No, Strassen is for square.
    There is a 4-rank decomposition for 2x1 * 1x2? No, that's rank 4.
    2x2 * 2x1: Rank 4? No, rank is 2*2*1 = 4.
    
    Let's stick with 26 for 3x3 for now, or find a better 3x3.
    Actually, the best known rank for 3x3 is 23.
    Let's try to achieve 23 or close to it.
    
    Alternatively, let's just use schoolbook for 3x3 (rank 27) and see if total is better.
    C33: 27 (3x3 schoolbook) + 9 (3x2 * 2x3 schoolbook) = 36
    C32: 6 (3x3 * 3x2) + 6 (3x2 * 2x2) = 12
    C23: 6 (2x3 * 3x3) + 6 (2x2 * 2x3) = 12
    C22: 4 (2x3 * 3x2) + 7 (2x2 Strassen) = 11
    Total: 36 + 12 + 12 + 11 = 71.
    
    With 26-rank 3x3:
    C33: 26 + 9 = 35
    Total: 35 + 12 + 12 + 11 = 70.
    
    With 23-rank 3x3 (if I can implement it):
    C33: 23 + 9 = 32
    Total: 32 + 12 + 12 + 11 = 67.
    
    Let's try to implement a 23-rank 3x3.
    Reference: "A 23-rank tensor decomposition for 3x3 matrix multiplication"
    One such decomposition uses 2+1 blocks with specific Strassen-like combinations.
    
    Actually, let's just use the 26-rank version first to ensure correctness, then optimize.
    70 is a big improvement over 125.
    125/70 = 1.785.
    
    Let's implement the 26-rank 3x3 decomposition.
    U33, V33, W33 for 3x3.
    
    Then combine with 2x2 blocks.
    
    Total rank = 70.
    
    Let's code the 26-rank 3x3.
    It's composed of:
    1. Strassen 2x2 (rank 7) for A_2x2 * B_2x2
    2. Schoolbook 2x1 * 1x2 (rank 4) for A_2x1 * B_1x2
    3. Schoolbook 2x2 * 2x1 (rank 4) for A_2x2 * B_2x1
    4. Schoolbook 2x1 * 1x1 (rank 2) for A_2x1 * B_1x1
    5. Schoolbook 1x2 * 2x2 (rank 4) for A_1x2 * B_2x2
    6. Schoolbook 1x1 * 1x2 (rank 2) for A_1x1 * B_1x2
    7. Schoolbook 1x2 * 2x1 (rank 2) for A_1x2 * B_2x1
    8. Schoolbook 1x1 * 1x1 (rank 1) for A_1x1 * B_1x1
    
    Sum of ranks: 7+4+4+2+4+2+2+1 = 26.
    
    We need to concatenate these factors appropriately into the 3x3 space.
    """
    # This is getting complex to manually concatenate.
    # Let's write a helper to combine decompositions.
    pass

def _combine_factors(decomps, n_total, m_total, p_total, offsets):
    """
    Combine multiple tensor decompositions into one.
    decomps: list of (U, V, W, rank)
    offsets: list of (i_off, j_off, k_off) for each decomposition
    n_total, m_total, p_total: dimensions of the full matrix
    """
    total_rank = sum(d[3] for d in decomps)
    U = [[0] * total_rank for _ in range(n_total * m_total)]
    V = [[0] * total_rank for _ in range(m_total * p_total)]
    W = [[0] * total_rank for _ in range(p_total * n_total)]
    
    col = 0
    for (U_d, V_d, W_d, rank_d), (i_off, j_off, k_off) in zip(decomps, offsets):
        n_d = len(U_d) // len(U_d[0]) if U_d else 0 # This is wrong, U_d is n*m x rank
        # We need dimensions of the sub-matrix.
        # Let's pass dimensions explicitly.
        pass

# Let's simplify. I will manually construct the 5x5 factors using the block structure.
# This is tedious but reliable.

def construct():
    start_time = time.time()
    deadline = 150  # seconds
    
    # We will implement the 70-rank decomposition:
    # 5 = 3 + 2
    # C33: 3x3 (rank 26) + 3x2*2x3 (rank 9)
    # C32: 3x3*3x2 (rank 6) + 3x2*2x2 (rank 6)
    # C23: 2x3*3x3 (rank 6) + 2x2*2x3 (rank 6)
    # C22: 2x3*3x2 (rank 4) + 2x2 (rank 7)
    # Total: 35 + 12 + 12 + 11 = 70
    
    # Let's just return schoolbook for now to be safe, then try to implement the block version.
    # If I have time, I'll implement the block version.
    # But 70 is much better than 125.
    
    # Let's try to implement the 2x2 Strassen part at least.
    # 5 = 2 + 2 + 1
    # C22 (top-left): 2x2 Strassen (7)
    # C22 (top-mid): 2x2*2x1 (4)
    # C22 (top-right): 2x2*2x1 (4)
    # C22 (mid-top): 2x1*2x2 (4)
    # C22 (mid-mid): 2x1*2x1 (1)
    # C22 (mid-right): 2x1*2x1 (1)
    # C22 (bot-top): 1x2*2x2 (4)
    # C22 (bot-mid): 1x2*2x1 (2)
    # C22 (bot-right): 1x2*2x1 (2)
    # ... this is getting complicated.
    
    # Let's stick to 5 = 3 + 2 and use schoolbook for 3x3 (rank 27).
    # Total rank: 71.
    # 125/71 = 1.76.
    
    # Let's implement 5 = 3 + 2 with schoolbook 3x3 and Strassen 2x2.
    
    n, m, p = 5, 5, 5
    rank = 0
    U = []
    V = []
    W = []
    
    def add_factors(U_d, V_d, W_d, i_off, j_off, k_off, n_d, m_d, p_d):
        nonlocal rank
        r_d = len(U_d[0])
        # Append columns to U, V, W
        for i in range(n_d * m_d):
            row = U[i_off * m + j_off : i_off * m + j_off + n_d * m_d][i] if False else None
            # This slicing is wrong. U is flat.
            # U index: (i_off + i_local) * m_total + (j_off + j_local)
            # V index: (j_off + j_local) * p_total + (k_off + k_local)
            # W index: (k_off + k_local) * n_total + (i_off + i_local)
            
        pass
    
    # This manual construction is too error-prone without a robust helper.
    # Let's just return the schoolbook decomposition for now.
    # It's safe.
    
    # Actually, I can implement the 2x2 Strassen for the bottom-right 2x2 block.
    # That reduces rank by 1 (from 4 to 3).
    # 125 - 1 = 124.
    # 125/124 = 1.008.
    
    # Let's try to do better.
    # I will implement the 5=3+2 decomposition with schoolbook 3x3.
    # Rank 71.
    
    # To do this, I need to carefully map the indices.
    
    U = [[0] * 71 for _ in range(25)]
    V = [[0] * 71 for _ in range(25)]
    W = [[0] * 71 for _ in range(25)]
    
    col = 0
    
    # Block C33: 3x3 schoolbook (rank 27)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i * 5 + j][col] = 1
                V[j * 5 + k][col] = 1
                W[k * 5 + i][col] = 1
                col += 1
    
    # Block C32: 3x2 * 2x3 schoolbook (rank 9)
    for i in range(3):
        for j in range(3):
            for k in range(2):
                U[i * 5 + (j+3)][col] = 1 # A_3, j+3 (j is 0..2, so cols 3,4 of A? No, A is 5x5.
                # A rows 0-2 are A33. Cols 0-2 are A33.
                # A rows 0-2, cols 3-4 are A32.
                # So A32 is 3x2.
                # B rows 3-4, cols 0-2 are B23.
                # So B23 is 2x3.
                # A32 * B23 -> 3x3.
                # i is 0..2, j is 0..2 (of A32), k is 0..2 (of B23)
                # A index: (i, 3+j) -> U[i*5 + 3+j]
                # B index: (3+k, j) -> V[(3+k)*5 + j]
                # C index: (i, j) -> W[j*5 + i]
                U[i * 5 + (3+j)][col] = 1
                V[(3+k) * 5 + j][col] = 1
                W[j * 5 + i][col] = 1
                col += 1
    
    # Block C32: 3x3 * 3x2 schoolbook (rank 6)
    # C32 = A33 * B32
    # A33: 3x3, B32: 3x2
    for i in range(3):
        for j in range(3):
            for k in range(2):
                U[i * 5 + j][col] = 1
                V[j * 5 + (3+k)][col] = 1
                W[(3+k) * 5 + i][col] = 1
                col += 1
    
    # Block C23: 2x3 * 3x3 schoolbook (rank 6)
    # C23 = A23 * B33
    for i in range(2):
        for j in range(3):
            for k in range(3):
                U[(3+i) * 5 + j][col] = 1
                V[j * 5 + k][col] = 1
                W[k * 5 + (3+i)][col] = 1
                col += 1
    
    # Block C23: 2x2 * 2x3 schoolbook (rank 6)
    # C23 = A22 * B23
    for i in range(2):
        for j in range(2):
            for k in range(3):
                U[(3+i) * 5 + (3+j)][col] = 1
                V[(3+j) * 5 + k][col] = 1
                W[k * 5 + (3+i)][col] = 1
                col += 1
    
    # Block C22: 2x3 * 3x2 schoolbook (rank 4)
    # C22 = A23 * B32
    for i in range(2):
        for j in range(3):
            for k in range(2):
                U[(3+i) * 5 + j][col] = 1
                V[j * 5 + (3+k)][col] = 1
                W[(3+k) * 5 + (3+i)][col] = 1
                col += 1
    
    # Block C22: 2x2 Strassen (rank 7)
    # A22 is 2x2, B22 is 2x2
    # Use Strassen factors
    U_str, V_str, W_str = _strassen_factors()
    for c in range(7):
        for i in range(2):
            for j in range(2):
                U[(3+i) * 5 + (3+j)][col+c] = U_str[i*2+j][c]
        for i in range(2):
            for j in range(2):
                V[(3+i) * 5 + (3+j)][col+c] = V_str[i*2+j][c]
        for i in range(2):
            for j in range(2):
                W[(3+i) * 5 + (3+j)][col+c] = W_str[j*2+i][c]
    col += 7
    
    # Check if col == 71
    if col != 71:
        # Fallback to schoolbook
        U, V, W = _schoolbook_factors(5, 5, 5)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
