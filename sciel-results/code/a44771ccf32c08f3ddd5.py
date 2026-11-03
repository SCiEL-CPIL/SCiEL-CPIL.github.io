"""
Matrix multiplication 4x4 by 4x6: rank-minimal exact tensor decomposition.

Strategy:
  - Start with Strassen's algorithm for 4x4 * 4x4 (rank 84), then extend to 4x6.
  - The 4x6 multiplication can be decomposed as:
      C = A * B  where A is 4x4, B is 4x6
    Split B = [B1 | B2] where B1 is 4x4, B2 is 4x2.
    Then C = A*B1 + A*B2.
    
    However, a better approach: use the known rank-84 decomposition for 4x4*4x4
    (Strassen's 2-level algorithm) and handle the extra 2 columns of B.
    
    Actually, let's think more carefully. The tensor is T_{i,j,k} = delta_{i,k} for
    (A_{i,j} * B_{j,k}) = C_{i,k} where A is 4x4, B is 4x6, C is 4x6.
    
    The schoolbook rank is 4*4*6 = 96.
    
    For 4x4 * 4x4, Strassen gives rank 84 (2-level Strassen: 7^2 = 49 for 2x2 blocks,
    but 4x4 = 2x(2x2), so 2-level gives 7^2 = 49... wait no).
    
    Let me reconsider. Standard Strassen for 2x2: rank 7.
    For 4x4: use 2-level Strassen: split into 2x2 blocks, each block multiplication
    uses 7 scalar mults, so 4x4 uses 7*7 = 49? No, that's not right either.
    
    Actually for 4x4 * 4x4:
    - Level 1: split into 2x2 blocks (4 block multiplications of 2x2 matrices)
    - Each 2x2 multiplication uses 7 scalar multiplications (Strassen)
    - Total: 4 * 7 = 28? No, that's not how it works.
    
    Let me be precise. The matrix multiplication tensor M_{n} has rank:
    - M_2: rank 7 (Strassen)
    - M_4: can be done with 2-level Strassen. Split 4x4 into four 2x2 blocks.
      The product of two 4x4 matrices involves 8 block multiplications of 2x2 matrices
      (since C = A*B has 4 blocks, each block is a sum of 2 products of 2x2 blocks).
      Wait no: C_{11} = A_{11}B_{11} + A_{12}B_{21}, etc. So 8 products of 2x2 matrices.
      Each 2x2 product uses 7 scalar mults, so total 8*7 = 56.
      
    Hmm, but the knowledge says Strassen reduces 96 to 84 for 4x4*4x4... wait, 4x4*4x4
    schoolbook is 64, not 96. The task is 4x4 * 4x6, schoolbook = 96.
    
    The knowledge says: "Strassen's algorithm reduces the rank from 96 to 84 by exploiting
    block structure and 7 multiplications instead of 12 for 2x2 blocks."
    
    So for 4x4 * 4x6:
    - Split into 2x2 blocks. A is 4x4 = 2x2 blocks of 2x2. B is 4x6 = 2x3 blocks of 2x2.
    - C is 4x6 = 2x3 blocks of 2x2.
    - Each output 2x2 block C_{ab} = sum over c of A_{ac} * B_{cb}.
    - There are 2*3 = 6 output blocks, each requiring 2 products of 2x2 matrices.
    - Total: 6 * 2 = 12 products of 2x2 matrices.
    - Schoolbook: 12 * 4 = 48... no wait, each 2x2 product is 4 scalar mults, so 12*4 = 48?
      But schoolbook total is 96 = 4*4*6. Let me recount.
    
    Actually the "12 multiplications" refers to the 12 block-level 2x2 matrix multiplications.
    Each 2x2 matrix multiplication uses 4 scalar multiplications (schoolbook) or 7 (Strassen).
    So schoolbook: 12 * 4 = 48? That doesn't match 96.
    
    I think the "12" and "7" refer to something else. Let me re-read: "7 multiplications instead
    of 12 for 2x2 blocks". So for each 2x2 block multiplication, schoolbook uses 4 scalar mults,
    Strassen uses 7. But 12 vs 7... 
    
    Oh I see - maybe they're counting the total number of scalar multiplications in a different way.
    96/12 = 8, and 84/7 = 12. That doesn't work either.
    
    Let me just try a different approach. The rank-84 decomposition for 4x4*4x6:
    
    Actually, I recall that the known upper bound for rank of 4x4 * 4x6 matrix multiplication
    is not well-established in simple terms. Let me try a practical approach:
    
    Use a 2-level Strassen-like decomposition:
    - Split A (4x4) into 4 blocks of 2x2: A = [[A11, A12], [A21, A22]]
    - Split B (4x6) into 6 blocks of 2x2: B = [[B11, B12, B13], [B21, B22, B23]]
    - C (4x6) has 6 blocks: C = [[C11, C12, C13], [C21, C22, C23]]
    
    C11 = A11*B11 + A12*B21
    C12 = A11*B12 + A12*B22
    C13 = A11*B13 + A12*B23
    C21 = A21*B11 + A22*B21
    C22 = A21*B12 + A22*B22
    C23 = A21*B13 + A22*B23
    
    That's 12 products of 2x2 matrices. With Strassen (7 scalar mults each): 12*7 = 84.
    With schoolbook (4 scalar mults each): 12*4 = 48... but total schoolbook is 96.
    
    Wait, I'm confusing myself. Each 2x2 * 2x2 product produces a 2x2 result, requiring
    4 scalar multiplications in schoolbook. So 12 * 4 = 48 scalar multiplications total.
    But the full schoolbook for 4x4*4x6 is 4*4*6 = 96 scalar multiplications.
    
    The discrepancy is because when we split into 2x2 blocks, each block product
    (2x2 * 2x2) actually involves 4*2 = 8 scalar multiplications? No...
    
    Let me just count directly. A is 4x4, B is 4x6. C[i][k] = sum_j A[i][j]*B[j][k].
    Total scalar multiplications = 4*4*6 = 96.
    
    When we group into 2x2 blocks:
    A has blocks A[0][0], A[0][1], A[1][0], A[1][1] (each 2x2)
    B has blocks B[0][0], B[0][1], B[0][2], B[1][0], B[1][1], B[1][2] (each 2x2)
    
    C[0][0] = A[0][0]*B[0][0] + A[0][1]*B[1][0]  (two 2x2 matrix products)
    Each 2x2 matrix product = 4 scalar mults (schoolbook)
    So C[0][0] needs 8 scalar mults.
    There are 6 such C blocks, so 6*8 = 48 scalar mults.
    
    But that's only 48, not 96! The issue is that a 2x2 * 2x2 product is NOT just 4 scalar
    multiplications in the tensor decomposition sense. In the tensor decomposition, each
    "multiplication" corresponds to one rank-1 term. A 2x2*2x2->2x2 multiplication tensor
    has rank 4 (schoolbook) or 7 (Strassen). 
    
    So if we use 12 block-level 2x2*2x2->2x2 multiplications:
    - Schoolbook: 12 * 4 = 48 rank-1 terms? But that gives rank 48, not 96.
    
    I think the confusion is that the "rank" in the tensor decomposition counts each
    scalar multiplication as one rank-1 term. So:
    - Full schoolbook for 4x4*4x6: 96 rank-1 terms.
    - Using 2x2 block structure: 12 block multiplications, each of which is a 2x2*2x2->2x2
      tensor. The rank of this sub-tensor is 4 (schoolbook) or 7 (Strassen).
    - If we use schoolbook for each block: 12 * 4 = 48. But this should equal 96?
    
    No wait - I think the issue is that the 2x2*2x2->2x2 tensor has rank 4 in the sense
    that it requires 4 scalar multiplications, but each scalar multiplication in the
    original problem corresponds to one entry. Let me just verify: 4x4*4x6 has 4*6 = 24
    output entries, each requiring 4 multiplications (sum over j=0..3), so 24*4 = 96.
    
    With 2x2 blocks: 6 output blocks, each 2x2 = 4 entries, each entry requires 2 block-level
    multiplications (since sum over 2 block-rows). Each block-level multiplication of two
    2x2 matrices to get one entry of the result requires 2 scalar multiplications.
    So per output block: 4 entries * 2 block-mults * 2 scalar-mults per block-mult = 16.
    Total: 6 * 16 = 96. ✓
    
    So each 2x2*2x2->2x2 block multiplication, when expanded, contributes 4 scalar multiplications
    to the rank count (one for each of the 4 entries of the 2x2 result, each needing... no).
    
    I'm overcomplicating this. Let me just use the known result: Strassen gives rank 84 for
    4x4*4x6. The mechanism is: 12 block-level 2x2*2x2->2x2 multiplications, each replaced by
    Strassen's 7-term decomposition instead of 4-term, giving 12*7 = 84.
    
    But wait, 12*4 = 48 ≠ 96. So this counting is wrong.
    
    Let me look at it differently. The 2x2*2x2->2x2 multiplication tensor T^{(2)} has:
    - Indices: (i,j) for input A (2x2), (j,k) for input B (2x2), (i,k) for output C (2x2)
    - T[i,j,k,l,m,n] = delta[j,l] * delta[i,m]... 
    
    Actually the standard formulation: for multiplying 2x2 matrices, the tensor is
    T_{(i,j),(j,k),(i,k)} where i,j,k ∈ {0,1}. This has 2*2*2 = 8 free indices (i,j,k for
    each matrix, with the constraint that B's row index matches A's column index).
    Total number of (i,j,k) triples: 2*2*2 = 8.
    Rank of this tensor: 7 (Strassen) or 4 (schoolbook... no, schoolbook is also 4? No).
    
    Schoolbook for 2x2: C[0][0] = A[0][0]*B[0][0] + A[0][1]*B[1][0], etc.
    That's 8 scalar multiplications, not 4! I was wrong earlier.
    
    So 2x2*2x2->2x2 schoolbook rank = 8, Strassen rank = 7.
    
    Now for 4x4*4x6:
    - 12 block-level 2x2*2x2->2x2 multiplications
    - Schoolbook: 12 * 8 = 96 ✓
    - Strassen: 12 * 7 = 84 ✓
    
    So the rank-84 decomposition uses Strassen for each of the 12 block multiplications.
    
    Let me implement this. I'll use the standard Strassen decomposition for 2x2 matrices
    and apply it to each of the 12 block multiplications.
    """
import time

DIMENSIONS = [4, 4, 6]

def _strassen_2x2():
    """
    Return the Strassen decomposition of the 2x2 * 2x2 -> 2x2 multiplication tensor.
    
    The tensor T has indices (i, j, k) where i = output row, j = A column / B row, k = B column.
    T[i,j,k] = 1 if this term appears in C[i][k] = sum_j A[i][j] * B[j][k].
    
    Strassen's 7 products:
    M1 = (A00 + A11)(B00 + B11)
    M2 = (A10 + A11)B00
    M3 = A00(B10 - B11)
    M4 = A11(B01 - B00)
    M5 = (A00 + A01)B11
    M6 = (A10 - A00)(B00 + B01)
    M7 = (A01 - A11)(B10 + B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    
    For the tensor decomposition, we need to express each of the 8 terms
    (i,j,k) as a sum of products of 7 rank-1 components.
    
    The 8 terms of the 2x2 multiplication tensor are:
    (i=0,j=0,k=0): A[0][0]*B[0][0]
    (i=0,j=1,k=0): A[0][1]*B[1][0]
    (i=0,j=0,k=1): A[0][0]*B[0][1]
    (i=0,j=1,k=1): A[0][1]*B[1][1]
    (i=1,j=0,k=0): A[1][0]*B[0][0]
    (i=1,j=1,k=0): A[1][1]*B[1][0]
    (i=1,j=0,k=1): A[1][0]*B[0][1]
    (i=1,j=1,k=1): A[1][1]*B[1][1]
    
    Each Strassen product Mr = (sum of A terms) * (sum of B terms) expands to
    a sum of A[i][j]*B[j][k] terms. We need to find coefficients such that:
    sum_r coeff_r(i,j,k) * U_r(i,j) * V_r(j,k) = delta for the tensor.
    
    Actually, the standard tensor decomposition approach:
    - U has shape (4, 7) where U[i*2+j, r] = coefficient of A[i][j] in the r-th Strassen product
    - V has shape (4, 7) where V[j*2+k, r] = coefficient of B[j][k] in the r-th Strassen product
    - W has shape (4, 7) where W[k*2+i, r] = coefficient such that C[i][k] = sum_r W[k*2+i, r] * (U·A_r) * (V·B_r)
    
    Wait, the trace convention is W[k*n+i, r]. For 2x2, n=2, so W[k*2+i, r].
    
    Let me define the Strassen products:
    P1 = (A00+A11)*(B00+B11)
    P2 = (A10+A11)*B00
    P3 = A00*(B10-B11)
    P4 = A11*(B01-B00)
    P5 = (A00+A01)*B11
    P6 = (A10-A00)*(B00+B01)
    P7 = (A01-A11)*(B10+B11)
    
    U coefficients (for A):
    P1: A00=1, A11=1 -> U[0*2+0]=1, U[1*2+1]=1
    P2: A10=1, A11=1 -> U[1*2+0]=1, U[1*2+1]=1
    P3: A00=1 -> U[0*2+0]=1
    P4: A11=1 -> U[1*2+1]=1
    P5: A00=1, A01=1 -> U[0*2+0]=1, U[0*2+1]=1
    P6: A10=1, A00=-1 -> U[1*2+0]=1, U[0*2+0]=-1
    P7: A01=1, A11=-1 -> U[0*2+1]=1, U[1*2+1]=-1
    
    V coefficients (for B):
    P1: B00=1, B11=1 -> V[0*2+0]=1, V[1*2+1]=1
    P2: B00=1 -> V[0*2+0]=1
    P3: B10=1, B11=-1 -> V[1*2+0]=1, V[1*2+1]=-1
    P4: B01=1, B00=-1 -> V[0*2+1]=1, V[0*2+0]=-1
    P5: B11=1 -> V[1*2+1]=1
    P6: B00=1, B01=1 -> V[0*2+0]=1, V[0*2+1]=1
    P7: B10=1, B11=1 -> V[1*2+0]=1, V[1*2+1]=1
    
    W coefficients (for C, trace convention W[k*2+i]):
    C00 = P1 + P4 - P5 + P7
    C01 = P3 + P5
    C10 = P2 + P4
    C11 = P1 - P2 + P3 + P6
    
    W[k*2+i, r]:
    C00: W[0*2+0] = [1, 0, 0, 1, -1, 0, 1]
    C01: W[0*2+1] = [0, 0, 1, 0, 1, 0, 0]
    C10: W[1*2+0] = [0, 1, 0, 1, 0, 0, 0]
    C11: W[1*2+1] = [1, -1, 1, 0, 0, 1, 0]
    
    Let me verify: C[0][0] = sum_r W[0*2+0, r] * P_r = 1*P1 + 0*P2 + 0*P3 + 1*P4 + (-1)*P5 + 0*P6 + 1*P7
    = P1 + P4 - P5 + P7 ✓
    """
    # U: 4x7, V: 4x7, W: 4x7
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # U[r] for each Strassen product r (0-indexed: P0..P6)
    # P0 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    # P1 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    # P2 = A00(B10-B11)
    U[0][2] = 1
    # P3 = A11(B01-B00)
    U[3][3] = 1
    # P4 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    # P5 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    # P6 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    
    # V
    # P0
    V[0][0] = 1; V[3][0] = 1
    # P1
    V[0][1] = 1
    # P2
    V[2][2] = 1; V[3][2] = -1
    # P3
    V[1][3] = 1; V[0][3] = -1
    # P4
    V[3][4] = 1
    # P5
    V[0][5] = 1; V[1][5] = 1
    # P6
    V[2][6] = 1; V[3][6] = 1
    
    # W: W[k*2+i, r]
    # C[0][0] = P0 + P3 - P4 + P6
    W[0][0] = 1; W[0][3] = 1; W[0][4] = -1; W[0][6] = 1
    # C[0][1] = P2 + P4
    W[1][2] = 1; W[1][4] = 1
    # C[1][0] = P1 + P3
    W[2][1] = 1; W[2][3] = 1
    # C[1][1] = P0 - P1 + P2 + P5
    W[3][0] = 1; W[3][1] = -1; W[3][2] = 1; W[3][5] = 1
    
    return U, V, W


def construct():
    """
    Construct rank-84 decomposition for 4x4 * 4x6 matrix multiplication.
    
    Split into 2x2 blocks:
    A (4x4) = [[A00, A01], [A10, A11]]  (each 2x2)
    B (4x6) = [[B00, B01, B02], [B10, B11, B12]]  (each 2x2)
    C (4x6) = [[C00, C01, C02], [C10, C11, C12]]  (each 2x2)
    
    C00 = A00*B00 + A01*B10
    C01 = A00*B01 + A01*B11
    C02 = A00*B02 + A01*B12
    C10 = A10*B00 + A11*B10
    C11 = A10*B01 + A11*B11
    C12 = A10*B02 + A11*B12
    
    12 block-level 2x2*2x2->2x2 multiplications.
    Each uses Strassen (rank 7), total rank = 12*7 = 84.
    
    The 12 block multiplications are:
    0: A00*B00 -> contributes to C00
    1: A01*B10 -> contributes to C00
    2: A00*B01 -> contributes to C01
    3: A01*B11 -> contributes to C01
    4: A00*B02 -> contributes to C02
    5: A01*B12 -> contributes to C02
    6: A10*B00 -> contributes to C10
    7: A11*B10 -> contributes to C10
    8: A10*B01 -> contributes to C11
    9: A11*B11 -> contributes to C11
    10: A10*B02 -> contributes to C12
    11: A11*B12 -> contributes to C12
    """
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Get the Strassen decomposition for 2x2*2x2->2x2
    sU, sV, sW = _strassen_2x2()
    
    rank = 84
    U = [[0] * rank for _ in range(n * m)]  # 16 x 84
    V = [[0] * rank for _ in range(m * p)]  # 24 x 84
    W = [[0] * rank for _ in range(p * n)]  # 24 x 84
    
    # Block indices:
    # A blocks: A[r][c] where r,c in {0,1}, block is 2x2
    #   A[0][0] = rows 0-1, cols 0-1
    #   A[0][1] = rows 0-1, cols 2-3
    #   A[1][0] = rows 2-3, cols 0-1
    #   A[1][1] = rows 2-3, cols 2-3
    #   Index in U: (block_row * 2 + local_row) * m + (block_col * 2 + local_col)
    #   = (br*2+lr)*4 + (bc*2+lc)
    
    # B blocks: B[r][c] where r in {0,1}, c in {0,1,2}
    #   B[0][0] = rows 0-1, cols 0-1
    #   B[0][1] = rows 0-1, cols 2-3
    #   B[0][2] = rows 0-1, cols 4-5
    #   B[1][0] = rows 2-3, cols 0-1
    #   B[1][1] = rows 2-3, cols 2-3
    #   B[1][2] = rows 2-3, cols 4-5
    #   Index in V: (block_row * 2 + local_row) * p + (block_col * 2 + local_col)
    #   = (br*2+lr)*6 + (bc*2+lc)
    
    # C blocks: C[r][c] where r in {0,1}, c in {0,1,2}
    #   Index in W (trace convention k*n+i): (block_col * 2 + local_col) * n + (block_row * 2 + local_row)
    #   = (bc*2+lc)*4 + (br*2+lr)
    
    # The 12 block multiplications:
    # (a_block, b_block, c_block)
    # a_block = (ar, ac), b_block = (br, bc), c_block = (cr, cc)
    # where cr=ar, cc=bc
    block_mults = [
        # (A_block_row, A_block_col, B_block_row, B_block_col, C_block_row, C_block_col)
        (0, 0, 0, 0, 0, 0),  # A00*B00 -> C00
        (0, 1, 1, 0, 0, 0),  # A01*B10 -> C00
        (0, 0, 0, 1, 0, 1),  # A00*B01 -> C01
        (0, 1, 1, 1, 0, 1),  # A01*B11 -> C01
        (0, 0, 0, 2, 0, 2),  # A00*B02 -> C02
        (0, 1, 1, 2, 0, 2),  # A01*B12 -> C02
        (1, 0, 0, 0, 1, 0),  # A10*B00 -> C10
        (1, 1, 1, 0, 1, 0),  # A11*B10 -> C10
        (1, 0, 0, 1, 1, 1),  # A10*B01 -> C11
        (1, 1, 1, 1, 1, 1),  # A11*B11 -> C11
        (1, 0, 0, 2, 1, 2),  # A10*B02 -> C12
        (1, 1, 1, 2, 1, 2),  # A11*B12 -> C12
    ]
    
    for mult_idx, (ar, ac, br, bc, cr, cc) in enumerate(block_mults):
        base = mult_idx * 7  # start column for this block's Strassen decomposition
        
        for r in range(7):
            col = base + r
            
            # U: for each (local_row, local_col) in the A block
            for lr in range(2):
                for lc in range(2):
                    u_idx = (ar * 2 + lr) * m + (ac * 2 + lc)
                    U[u_idx][col] = sU[lr * 2 + lc][r]
            
            # V: for each (local_row, local_col) in the B block
            for lr in range(2):
                for lc in range(2):
                    v_idx = (br * 2 + lr) * p + (bc * 2 + lc)
                    V[v_idx][col] = sV[lr * 2 + lc][r]
            
            # W: for each (local_row, local_col) in the C block
            # Trace convention: W[k*n + i] where k = col index, i = row index
            for lr in range(2):
                for lc in range(2):
                    # k = cc*2+lc, i = cr*2+lr
                    w_idx = (cc * 2 + lc) * n + (cr * 2 + lr)
                    W[w_idx][col] = sW[lc * 2 + lr][r]  # sW[k*2+i, r] where k=lc, i=lr
    
    return {"U": U, "V": V, "W": W}
