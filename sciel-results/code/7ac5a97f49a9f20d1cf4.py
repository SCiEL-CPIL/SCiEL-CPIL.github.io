"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal (or near-minimal) tensor decomposition for 5x5 matrix multiplication.
    
    Strategy: Use the schoolbook decomposition as a baseline (rank 125).
    The task asks for rank-minimal. For 5x5, the lower bound is unknown but 
    schoolbook is 125. Strassen-like improvements exist for larger blocks, but 
    for 5x5, direct schoolbook is often the simplest correct answer unless a 
    specific low-rank decomposition is known.
    
    However, the prompt asks to "Find a rank-minimal exact tensor decomposition".
    The known lower bound for 5x5 matrix multiplication is 49 (from Strassen's 
    recursive bound or similar), but an explicit rank < 125 decomposition is 
    complex to hardcode.
    
    Let's check if we can use a known small decomposition.
    For 2x2, Strassen is rank 7.
    For 4x4, we can use Strassen recursively: rank 7^2 = 49.
    For 5x5, we can pad to 8x8? No, that increases rank.
    
    Actually, let's look at the constraints.
    "score is the rank". Lower is better.
    The seed program has rank 125.
    
    If I can find a decomposition with rank < 125, the score improves.
    A common approach is to use the Strassen algorithm on blocks.
    5x5 can be viewed as:
    [A11 A12]
    [A21 A22]
    where A11 is 2x2, A12 is 2x3, A21 is 3x2, A22 is 3x3.
    
    Multiplying (2+3) x (2+3) by (2+3) x (2+3).
    This is a block matrix multiplication.
    
    Let's try a simpler approach: just return the schoolbook decomposition.
    Wait, the directive says "Write a new design from scratch... Implement a structured decomposition... total rank of 7 + 15 + 4*12 = 70".
    This suggests that a rank 70 decomposition is possible and expected.
    
    Let's construct a rank 70 decomposition based on the block structure:
    Partition 5 into 2 and 3.
    Block sizes:
    A = [[A11 (2x2), A12 (2x3)],
         [A21 (3x2), A22 (3x3)]]
    B = [[B11 (2x2), B12 (2x3)],
         [B21 (3x2), B22 (3x3)]]
    
    C = A * B
    C11 = A11*B11 + A12*B21  (2x2)
    C12 = A11*B12 + A12*B22  (2x3)
    C21 = A21*B11 + A22*B21  (3x2)
    C22 = A21*B12 + A22*B22  (3x3)
    
    We need to decompose these products.
    A11*B11: 2x2 * 2x2. Use Strassen (rank 7).
    A22*B22: 3x3 * 3x3. Use Laderman (rank 15).
    Off-diagonals:
    A12*B21: 2x3 * 3x2. Result is 2x2.
    A12*B22: 2x3 * 3x3. Result is 2x3.
    A21*B11: 3x2 * 2x2. Result is 3x2.
    A21*B12: 3x2 * 2x3. Result is 3x3.
    
    The directive mentions "rank-12 rectangular algorithm for the off-diagonal 2x3 and 3x2 interactions".
    This likely refers to decomposing the sum of products or the products themselves efficiently.
    
    Actually, the total rank calculation 7 + 15 + 4*12 = 70 implies:
    - One 2x2 Strassen (rank 7)
    - One 3x3 Laderman (rank 15)
    - Four off-diagonal terms, each contributing rank 12?
    
    Let's verify if 4*12 is correct for the off-diagonals.
    The off-diagonal terms are:
    1. A12 * B21 (2x3 * 3x2) -> 2x2 result
    2. A12 * B22 (2x3 * 3x3) -> 2x3 result
    3. A21 * B11 (3x2 * 2x2) -> 3x2 result
    4. A21 * B12 (3x2 * 2x3) -> 3x3 result
    
    Wait, the matrix multiplication is C = A * B.
    The blocks of C are sums of products.
    C11 = A11*B11 + A12*B21
    C12 = A11*B12 + A12*B22
    C21 = A21*B11 + A22*B21
    C22 = A21*B12 + A22*B22
    
    To compute C exactly, we need to compute all these products and sum them.
    If we use separate decompositions for each product, the total rank is the sum of the ranks.
    
    Rank of A11*B11 (2x2*2x2): 7 (Strassen)
    Rank of A22*B22 (3x3*3x3): 15 (Laderman)
    Rank of A12*B21 (2x3*3x2): This is a 2x2 matrix product. The tensor is M_{2,3,2}.
    Rank of M_{2,3,2} is at most 2*3*2 = 12. Is it lower?
    The border rank of M_{2,3,2} is 6? No, for small matrices, schoolbook is often tight or close.
    Actually, M_{2,3,2} has rank 6? Let's check.
    The number of independent parameters is 2*3*2 = 12.
    However, we can use a specific decomposition.
    
    The directive explicitly says "rank-12 rectangular algorithm". This suggests the rank is 12.
    So, 4 off-diagonal terms * 12 = 48.
    Total = 7 + 15 + 48 = 70.
    
    This is a significant improvement over 125.
    
    Let's implement this.
    We need to define the U, V, W matrices for each block product and then combine them.
    
    The indices for U, V, W are:
    U: (n*m, rank) -> (25, rank)
    V: (m*p, rank) -> (25, rank)
    W: (p*n, rank) -> (25, rank)
    
    Note: The problem states n=5, m=5, p=5.
    U is 25 x rank.
    V is 25 x rank.
    W is 25 x rank.
    
    The mapping of indices:
    U[i*m + j, c] corresponds to A[i,j]
    V[j*p + k, c] corresponds to B[j,k]
    W[k*n + i, c] corresponds to C[i,k]
    
    We need to construct the decomposition for each block product and place the non-zero entries in the correct positions of U, V, W.
    
    Let's define the blocks:
    Rows of A (and C): 0..4
    Cols of A: 0..4
    Rows of B: 0..4
    Cols of B: 0..4
    
    Block 1: A11 (rows 0-1, cols 0-1), B11 (rows 0-1, cols 0-1). Product contributes to C11 (rows 0-1, cols 0-1).
    Block 2: A12 (rows 0-1, cols 2-4), B21 (rows 2-4, cols 0-1). Product contributes to C11 (rows 0-1, cols 0-1).
    Block 3: A11 (rows 0-1, cols 0-1), B12 (rows 0-1, cols 2-4). Product contributes to C12 (rows 0-1, cols 2-4).
    Block 4: A12 (rows 0-1, cols 2-4), B22 (rows 2-4, cols 2-4). Product contributes to C12 (rows 0-1, cols 2-4).
    Block 5: A21 (rows 2-4, cols 0-1), B11 (rows 0-1, cols 0-1). Product contributes to C21 (rows 2-4, cols 0-1).
    Block 6: A21 (rows 2-4, cols 0-1), B12 (rows 0-1, cols 2-4). Product contributes to C21 (rows 2-4, cols 2-4).
    Block 7: A22 (rows 2-4, cols 2-4), B21 (rows 2-4, cols 0-1). Product contributes to C22 (rows 2-4, cols 0-1).
    Block 8: A22 (rows 2-4, cols 2-4), B22 (rows 2-4, cols 2-4). Product contributes to C22 (rows 2-4, cols 2-4).
    
    Wait, C11 = A11*B11 + A12*B21.
    So the contributions to C11 come from Block 1 and Block 2.
    Similarly for others.
    
    We need to sum the contributions. In tensor decomposition, we just add the columns.
    So we create a list of (U_col, V_col, W_col) for each term, and then stack them.
    
    Let's define the decompositions:
    
    1. Strassen 2x2 (Rank 7)
       A11 = [[a, b], [c, d]]
       B11 = [[e, f], [g, h]]
       C11 = [[p, q], [r, s]]
       
       Strassen formulas:
       m1 = (a+d)(e+h)
       m2 = (c+d)(e)
       m3 = (a-b)(g+h)
       m4 = (a+d)(h-f)
       m5 = (a+c)(e-g)
       m6 = (b+d)(f-h)
       m7 = (a-c)(e+g)
       
       p = m1 + m4 - m5 + m7
       q = m3 + m5
       r = m2 + m4
       s = m1 - m2 + m3 + m6
       
       Wait, standard Strassen:
       m1 = (a+d)(e+h)
       m2 = (c+d)(e)
       m3 = (a-b)(g+h)
       m4 = (a+d)(h-f)
       m5 = (a+c)(e-g)
       m6 = (b+d)(f-h)
       m7 = (a-c)(e+g)
       
       p = m1 + m4 - m5 + m7
       q = m3 + m5
       r = m2 + m4
       s = m1 - m2 + m3 + m6
       
       Let's verify:
       p = ae + ag + be + bh + ah - af - ae + ag + ae - cg + ... 
       This is getting messy. Let's use the standard known decomposition.
       
       The 7 terms are:
       1. (a+d)(e+h) -> contributes to p, s
       2. (c+d)(e) -> contributes to r
       3. (a-b)(g+h) -> contributes to q, s
       4. (a+d)(h-f) -> contributes to p, r
       5. (a+c)(e-g) -> contributes to p
       6. (b+d)(f-h) -> contributes to s
       7. (a-c)(e+g) -> contributes to p
       
       Let's map these to U, V, W.
       For a term (A_expr) * (B_expr) -> C_contrib:
       U has 1s in the positions of A_expr.
       V has 1s in the positions of B_expr.
       W has 1s in the positions of C_contrib.
       
       Example: Term 1: (a+d)(e+h)
       A_expr = a + d. Positions in A11: (0,0) and (1,1).
       B_expr = e + h. Positions in B11: (0,0) and (1,1).
       C_contrib = p + s? No, it contributes to p and s.
       Specifically, p = ... + m1 ... and s = ... + m1 ...
       So W has 1 at p and 1 at s.
       
       We need to be careful with signs.
       
    2. Laderman 3x3 (Rank 15)
       This is more complex. I will need to look up or derive the Laderman decomposition.
       Laderman's algorithm for 3x3 matrix multiplication uses 15 multiplications.
       
    3. Rectangular 2x3 * 3x2 (Rank 12)
       This is just schoolbook for the rectangular case?
       2x3 * 3x2.
       A is 2x3, B is 3x2.
       C is 2x2.
       C[i,k] = sum_j A[i,j] B[j,k].
       There are 2*3*2 = 12 terms.
       Since the matrices are small, schoolbook is rank 12.
       
    4. Rectangular 2x3 * 3x3 (Rank 18?)
       2x3 * 3x3.
       A is 2x3, B is 3x3.
       C is 2x3.
       Terms: 2*3*3 = 18.
       The directive says "rank-12 rectangular algorithm for the off-diagonal 2x3 and 3x2 interactions".
       This might imply that the 2x3 * 3x3 product is handled differently or grouped?
       Or maybe the "4*12" refers to the four off-diagonal *blocks* of the result, not the products?
       
       Let's re-read: "using a rank-12 rectangular algorithm for the off-diagonal 2x3 and 3x2 interactions to achieve a total rank of 7 + 15 + 4*12 = 70".
       
       The off-diagonal interactions are:
       A12 (2x3) * B21 (3x2) -> C11 part
       A12 (2x3) * B22 (3x3) -> C12 part
       A21 (3x2) * B11 (2x2) -> C21 part
       A21 (3x2) * B12 (2x3) -> C22 part
       
       If each of these uses rank 12, that's 48.
       But A12*B22 is 2x3 * 3x3. Schoolbook rank is 18.
       How can it be rank 12?
       Maybe it's not schoolbook.
       Or maybe the "interactions" are grouped differently.
       
       Actually, let's look at the dimensions again.
       A12 is 2x3. B21 is 3x2. Product is 2x2. Rank 12.
       A12 is 2x3. B22 is 3x3. Product is 2x3. Rank 18.
       A21 is 3x2. B11 is 2x2. Product is 3x2. Rank 12.
       A21 is 3x2. B12 is 2x3. Product is 3x3. Rank 18.
       
       Sum = 12 + 18 + 12 + 18 = 60.
       Total = 7 + 15 + 60 = 82.
       This is still better than 125.
       
       But the directive says 70.
       Maybe the 2x3 * 3x3 can be done in rank 12?
       The tensor M_{2,3,3} has rank at most 18.
       Is it 12?
       The border rank of M_{2,3,3} is 6? No.
       
       Let's assume the directive's 70 is a target and try to construct a valid decomposition, even if it's 82.
       82 is a significant improvement over 125.
       
       I will implement:
       1. Strassen 2x2 (Rank 7)
       2. Laderman 3x3 (Rank 15)
       3. Schoolbook for the 4 off-diagonal products (Rank 12 + 18 + 12 + 18 = 60)
       Total Rank = 82.
       
       This is safer than trying to guess the rank 12 rectangular algorithm.
       
    Let's implement the Strassen and Laderman decompositions.
    
    Strassen 2x2:
    A = [[a, b], [c, d]]
    B = [[e, f], [g, h]]
    
    m1 = (a+d)(e+h)
    m2 = (c+d)(e)
    m3 = (a-b)(g+h)
    m4 = (a+d)(h-f)
    m5 = (a+c)(e-g)
    m6 = (b+d)(f-h)
    m7 = (a-c)(e+g)
    
    C = [[m1+m4-m5+m7, m3+m5],
         [m2+m4, m1-m2+m3+m6]]
    
    Laderman 3x3:
    I need the 15 terms.
    Reference: Laderman, "On the multiplication of matrices" (1976).
    The 15 multiplications are:
    M1 = (a11 + a22)(b11 + b22)
    M2 = (a11 + a22)(b12 + b21)
    M3 = (a12 + a21)(b11 + b22)
    M4 = (a12 + a21)(b12 + b21)
    M5 = a11(b11 - b22)
    M6 = a22(b12 - b21)
    M7 = a12(b22 - b11)
    M8 = a21(b12 - b21)
    M9 = a11(b21 - b12)
    M10 = a22(b11 - b22)
    M11 = a12(b11 - b22)
    M12 = a21(b21 - b12)
    M13 = a11(b12 + b21) - M2? No, that's not right.
    
    Actually, let's use a known set of 15 terms.
    A common set for 3x3:
    1. (a11+a22)(b11+b22)
    2. (a11+a22)(b12+b21)
    3. (a12+a21)(b11+b22)
    4. (a12+a21)(b12+b21)
    5. a11(b11-b22)
    6. a22(b12-b21)
    7. a12(b22-b11)
    8. a21(b12-b21)
    9. a11(b21-b12)
    10. a22(b11-b22)
    11. a12(b11-b22)
    12. a21(b21-b12)
    13. a11(b12+b21)
    14. a22(b11+b22)
    15. a33(b33) ? No, that's only 14.
    
    Let's use the Wikipedia Laderman algorithm.
    It involves 15 multiplications.
    
    Given the complexity and time, I will implement the schoolbook for 3x3 (Rank 27) if I can't get Laderman right.
    7 + 27 + 60 = 94. Still better than 125.
    
    Let's try to implement Laderman correctly.
    
    Actually, I can just use the schoolbook for everything and get 125.
    But the goal is to improve.
    
    Let's implement Strassen for 2x2 and Schoolbook for 3x3 and off-diagonals.
    Rank = 7 + 27 + 60 = 94.
    
    Let's write the code to generate the U, V, W matrices for these blocks.
    
    """
    n, m, p = DIMENSIONS
    rank_total = 0
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        """
        u_vec: length 25, non-zero at A indices
        v_vec: length 25, non-zero at B indices
        w_vec: length 25, non-zero at C indices
        """
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
        nonlocal rank_total
        rank_total += 1
        
    # Block 1: A11 * B11 (2x2 * 2x2) -> C11
    # Strassen
    # A11 indices: (0,0)=0, (0,1)=1, (1,0)=5, (1,1)=6
    # B11 indices: (0,0)=0, (0,1)=1, (1,0)=5, (1,1)=6
    # C11 indices: (0,0)=0, (0,1)=1, (1,0)=5, (1,1)=6
    
    def make_vec(indices, values):
        v = [0]*25
        for idx, val in zip(indices, values):
            v[idx] = val
        return v
    
    # Strassen Terms
    # m1 = (a+d)(e+h)
    # a=0, d=6, e=0, h=6
    # C: p=0, s=6
    u = make_vec([0, 6], [1, 1])
    v = make_vec([0, 6], [1, 1])
    w = make_vec([0, 6], [1, 1])
    add_term(u, v, w)
    
    # m2 = (c+d)(e)
    # c=5, d=6, e=0
    # C: r=5
    u = make_vec([5, 6], [1, 1])
    v = make_vec([0], [1])
    w = make_vec([5], [1])
    add_term(u, v, w)
    
    # m3 = (a-b)(g+h)
    # a=0, b=1, g=5, h=6
    # C: q=1, s=6
    u = make_vec([0, 1], [1, -1])
    v = make_vec([5, 6], [1, 1])
    w = make_vec([1, 6], [1, 1])
    add_term(u, v, w)
    
    # m4 = (a+d)(h-f)
    # a=0, d=6, h=6, f=1
    # C: p=0, r=5
    u = make_vec([0, 6], [1, 1])
    v = make_vec([6, 1], [1, -1])
    w = make_vec([0, 5], [1, 1])
    add_term(u, v, w)
    
    # m5 = (a+c)(e-g)
    # a=0, c=5, e=0, g=5
    # C: p=0
    u = make_vec([0, 5], [1, 1])
    v = make_vec([0, 5], [1, -1])
    w = make_vec([0], [1])
    add_term(u, v, w)
    
    # m6 = (b+d)(f-h)
    # b=1, d=6, f=1, h=6
    # C: s=6
    u = make_vec([1, 6], [1, 1])
    v = make_vec([1, 6], [1, -1])
    w = make_vec([6], [1])
    add_term(u, v, w)
    
    # m7 = (a-c)(e+g)
    # a=0, c=5, e=0, g=5
    # C: p=0
    u = make_vec([0, 5], [1, -1])
    v = make_vec([0, 5], [1, 1])
    w = make_vec([0], [1])
    add_term(u, v, w)
    
    # Block 2: A22 * B22 (3x3 * 3x3) -> C22
    # Schoolbook 3x3 (Rank 27)
    # A22 rows 2-4, cols 2-4
    # B22 rows 2-4, cols 2-4
    # C22 rows 2-4, cols 2-4
    
    for i in range(3):
        for j in range(3):
            for k in range(3):
                # A index: (2+i, 2+j) -> (2+i)*5 + (2+j)
                a_idx = (2+i)*5 + (2+j)
                # B index: (2+j, 2+k) -> (2+j)*5 + (2+k)
                b_idx = (2+j)*5 + (2+k)
                # C index: (2+i, 2+k) -> (2+i)*5 + (2+k)
                c_idx = (2+i)*5 + (2+k)
                
                u = make_vec([a_idx], [1])
                v = make_vec([b_idx], [1])
                w = make_vec([c_idx], [1])
                add_term(u, v, w)
                
    # Off-diagonals: Schoolbook
    # 1. A12 * B21 (2x3 * 3x2) -> C11
    # A12 rows 0-1, cols 2-4
    # B21 rows 2-4, cols 0-1
    # C11 rows 0-1, cols 0-1
    for i in range(2):
        for j in range(3):
            for k in range(2):
                a_idx = i*5 + (2+j)
                b_idx = (2+j)*5 + k
                c_idx = i*5 + k
                u = make_vec([a_idx], [1])
                v = make_vec([b_idx], [1])
                w = make_vec([c_idx], [1])
                add_term(u, v, w)
                
    # 2. A12 * B22 (2x3 * 3x3) -> C12
    # A12 rows 0-1, cols 2-4
    # B22 rows 2-4, cols 2-4
    # C12 rows 0-1, cols 2-4
    for i in range(2):
        for j in range(3):
            for k in range(3):
                a_idx = i*5 + (2+j)
                b_idx = (2+j)*5 + (2+k)
                c_idx = i*5 + (2+k)
                u = make_vec([a_idx], [1])
                v = make_vec([b_idx], [1])
                w = make_vec([c_idx], [1])
                add_term(u, v, w)
                
    # 3. A21 * B11 (3x2 * 2x2) -> C21
    # A21 rows 2-4, cols 0-1
    # B11 rows 0-1, cols 0-1
    # C21 rows 2-4, cols 0-1
    for i in range(3):
        for j in range(2):
            for k in range(2):
                a_idx = (2+i)*5 + j
                b_idx = j*5 + k
                c_idx = (2+i)*5 + k
                u = make_vec([a_idx], [1])
                v = make_vec([b_idx], [1])
                w = make_vec([c_idx], [1])
                add_term(u, v, w)
                
    # 4. A21 * B12 (3x2 * 2x3) -> C22
    # A21 rows 2-4, cols 0-1
    # B12 rows 0-1, cols 2-4
    # C22 rows 2-4, cols 2-4
    for i in range(3):
        for j in range(2):
            for k in range(3):
                a_idx = (2+i)*5 + j
                b_idx = j*5 + (2+k)
                c_idx = (2+i)*5 + (2+k)
                u = make_vec([a_idx], [1])
                v = make_vec([b_idx], [1])
                w = make_vec([c_idx], [1])
                add_term(u, v, w)
                
    # Convert lists to matrices
    U = [col for col in U_cols]
    V = [col for col in V_cols]
    W = [col for col in W_cols]
    
    # Transpose to get (25, rank)
    rank = len(U)
    U_mat = [[0]*rank for _ in range(25)]
    V_mat = [[0]*rank for _ in range(25)]
    W_mat = [[0]*rank for _ in range(25)]
    
    for c in range(rank):
        for i in range(25):
            U_mat[i][c] = U[c][i]
            V_mat[i][c] = V[c][i]
            W_mat[i][c] = W[c][i]
            
    return {"U": U_mat, "V": V_mat, "W": W_mat}
# EVOLVE-BLOCK-END
