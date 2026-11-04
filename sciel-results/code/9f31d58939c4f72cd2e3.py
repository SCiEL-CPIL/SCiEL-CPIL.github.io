"""Matrix multiplication 4x4x6 - rank-reduced decomposition."""
import time

DIMENSIONS = [4, 4, 6]
DEADLINE = time.time() + 130.0

# EVOLVE-BLOCK-START
def _strassen_2x2x2():
    """Return (U, V, W) for 2x2x2 matrix mult, rank 7.
    U: shape (4, 7) - input A flattened (row-major: a00,a01,a10,a11)
    V: shape (4, 7) - input B flattened (row-major: b00,b01,b10,b11)
    W: shape (4, 7) - output C flattened (row-major: c00,c01,c10,c11)
    """
    # Strassen's 7 multiplications:
    # m1 = (a00+a11)*(b00+b11)
    # m2 = (a10+a11)*b00
    # m3 = a00*(b01-b11)
    # m4 = a11*(b10-b00)
    # m5 = (a00+a01)*b11
    # m6 = (a10-a00)*(b00+b10)
    # m7 = (a01-a11)*(b01+b11)
    # c00 = m1 + m4 - m5 + m7
    # c01 = m3 + m5
    # c10 = m2 + m4
    # c11 = m1 - m2 + m3 + m6
    U = [
        [1, 1, 1, 0, 1, 0, -1],  # a00
        [0, 0, 0, 0, 1, 0, 0],   # a01
        [0, 1, 0, 0, 0, 1, 0],   # a10
        [1, 1, 0, 1, 0, -1, -1], # a11
    ]
    V = [
        [1, 1, -1, 1, 0, 1, 0],  # b00
        [0, 0, 1, 0, 0, 0, 1],   # b01
        [1, 0, 0, -1, 1, 1, 0],  # b10
        [1, 0, -1, 0, 1, 0, 1],  # b11
    ]
    W = [
        [1, 0, 0, 1, -1, 0, 1],  # c00
        [0, 0, 1, 0, 1, 0, 0],   # c01
        [0, 1, 0, 1, 0, 0, 0],   # c10
        [1, -1, 0, 0, 0, 1, 0],  # c11
    ]
    return U, V, W

def _decompose_4x4x4():
    """Decompose 4x4x4 matrix multiplication.
    Uses block structure with 2x2 Strassen.
    A(4x4) x B(4x4) -> C(4x4)
    Split into 2x2 blocks: A = [[A00,A01],[A10,A11]], B = [[B00,B01],[B10,B11]], C = [[C00,C01],[C10,C11]]
    C00 = A00*B00 + A01*B10
    C01 = A00*B01 + A01*B11
    C10 = A10*B00 + A11*B10
    C11 = A10*B01 + A11*B11
    Each 2x2*2x2->2x2 uses rank 7 (Strassen).
    Total: 16 * 7 = 112... that's worse than schoolbook 64.
    
    Better: use schoolbook for 4x4x4: rank 64.
    Or use a smarter approach.
    
    Actually, let me use the known rank 49 construction.
    For simplicity and correctness, use schoolbook: rank 64.
    """
    n = 4
    rank = n * n * n
    U = [[0] * rank for _ in range(n * n)]
    V = [[0] * rank for _ in range(n * n)]
    W = [[0] * rank for _ in range(n * n)]
    col = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                U[i * n + j][col] = 1
                V[j * n + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _decompose_4x4x2():
    """Decompose 4x4x2 matrix multiplication.
    A(4x4) x B(4x2) -> C(4x2)
    Schoolbook: rank 4*4*2 = 32
    """
    n, m, p = 4, 4, 2
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

def construct():
    """Construct rank-minimal decomposition for 4x4x6 matrix multiplication.
    
    Strategy: Split 4x4x6 into 4x4x4 (rank 64) + 4x4x2 (rank 32) = rank 96.
    But we can do better using Strassen for the 2x2 sub-problems.
    
    Alternative: Direct approach with 2x2 blocks and Strassen.
    
    Let me try: Use the 2x2x2 Strassen (rank 7) applied to sub-blocks.
    
    For 4x4x6:
    Split A(4x4) into 2x2 blocks: A = [[A00, A01], [A10, A11]]
    Split B(4x6) into 2x2 blocks: B = [[B00, B01, B02], [B10, B11, B12]] (each 2x2)
    C(4x6) = [[C00, C01, C02], [C10, C11, C12]] (each 2x2)
    
    C00 = A00*B00 + A01*B10
    C01 = A00*B01 + A01*B11
    C02 = A00*B02 + A01*B12
    C10 = A10*B00 + A11*B10
    C11 = A10*B01 + A11*B11
    C12 = A10*B02 + A11*B12
    
    Each Ci is sum of two 2x2*2x2 products. Using Strassen for each 2x2*2x2,
    we get 12 * 7 = 84 rank terms. But there's sharing possible.
    
    Actually, let me just use the split approach: 4x4x4 + 4x4x2.
    For 4x4x4, use rank 64 (schoolbook). For 4x4x2, use rank 32 (schoolbook).
    Total: 96. Same as original. Not an improvement.
    
    Let me try harder. Use Strassen for 4x4x4:
    4x4x4: Split into 2x2 blocks. C_ij = A_i0*B_0j + A_i1*B_1j for i,j in {0,1}.
    That's 4 C-blocks, each a sum of two 2x2*2x2 products.
    Using Strassen (rank 7) for each 2x2*2x2: 8 * 7 = 56 rank.
    
    So 4x4x4 in rank 56, 4x4x2 in rank 32: total 88. Better than 96!
    
    But can I share between the 4x4x4 and 4x4x2 parts? The 4x4x2 uses B columns 4,5 (the last 2), while 4x4x4 uses B columns 0-3. No sharing in B.
    
    Actually wait, for 4x4x6, I should think of it as:
    B is 4x6. Split into B_left(4x4) and B_right(4x2).
    C = A * [B_left | B_right] = [A*B_left | A*B_right]
    
    So rank(4x4x6) = rank(4x4x4) + rank(4x4x2) if no sharing.
    
    With Strassen for 4x4x4: rank 56. With schoolbook for 4x4x2: rank 32. Total: 88.
    
    Can I do 4x4x2 in less than 32? 
    4x4x2: A(4x4)*B(4x2)->C(4x2). 
    Split A into 2x2 blocks, B into two 2x2 blocks (top and bottom).
    C_top = A00*B_top + A01*B_bot
    C_bot = A10*B_top + A11*B_bot
    Each 2x2*2x2->2x2 in rank 7. 4 such products: 4*7=28.
    
    So 4x4x2 in rank 28! Total: 56 + 28 = 84.
    
    Can I do even better? Let me verify the 4x4x2 in rank 28.
    
    Actually, I realize I should construct these properly. Let me implement the block-based approach.
    """
    # Use block decomposition with Strassen
    # 4x4x6 = 4x4x4 (rank 56 via Strassen blocks) + 4x4x2 (rank 28 via Strassen blocks)
    # Total rank: 84
    
    # Actually, let me just use a simpler but correct approach.
    # Schoolbook for 4x4x6 gives rank 96.
    # Let me try to get rank 84.
    
    n, m, p = 4, 4, 6
    
    # Build the decomposition using 2x2 Strassen blocks
    # For 4x4x4 part (columns 0-3 of B):
    # A is 4x4, split into 2x2 blocks A00,A01,A10,A11 (each 2x2)
    # B_left is 4x4, split into 2x2 blocks B00,B01,B10,B11 (each 2x2)
    # C_left is 4x4, split into 2x2 blocks C00,C01,C10,C11 (each 2x2)
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # For 4x4x2 part (columns 4-5 of B):
    # B_right is 4x2, split into B_top(2x2) and B_bot(2x2)
    # C_right is 4x2, split into C_top_right(2x2) and C_bot_right(2x2)
    # C_top_right = A00*B_top + A01*B_bot
    # C_bot_right = A10*B_top + A11*B_bot
    
    # Each 2x2*2x2->2x2 uses Strassen rank 7.
    # 4x4x4: 8 such products (2 per C-block, 4 C-blocks) -> 8*7=56
    # 4x4x2: 4 such products -> 4*7=28
    # Total: 84
    
    sU, sV, sW = _strassen_2x2x2()
    
    # Helper to apply Strassen to a 2x2*2x2 product
    # a_base: index of a00 in the 2x2 block (row-major within block)
    # b_base: index of b00
    # c_base: index of c00
    # U_block: 4x7, V_block: 4x7, W_block: 4x7
    # Returns indices and values for U, V, W
    
    all_U = []
    all_V = []
    all_W = []
    
    # For each 2x2*2x2->2x2 Strassen instance, add 7 columns
    # A index: the 4 elements of the 2x2 block in A (flattened row-major)
    # B index: the 4 elements of the 2x2 block in B
    # C index: the 4 elements of the 2x2 block in C
    
    def add_strassen_instance(a_indices, b_indices, c_indices):
        """a_indices: list of 4 indices (row-major) for the 2x2 block in A (size 4x4 -> 16 elements)
        b_indices: list of 4 indices for the 2x2 block in B
        c_indices: list of 4 indices for the 2x2 block in C
        """
        for t in range(7):
            u_col = [0] * (n * m)
            v_col = [0] * (m * p)
            w_col = [0] * (p * n)
            for r in range(4):
                u_col[a_indices[r]] = sU[r][t]
                v_col[b_indices[r]] = sV[r][t]
                w_col[c_indices[r]] = sW[r][t]
            all_U.append(u_col)
            all_V.append(v_col)
            all_W.append(w_col)
    
    # A is 4x4, flattened row-major: index = i*4 + j
    # Block A00: rows 0-1, cols 0-1: indices [0,1,4,5]
    # Block A01: rows 0-1, cols 2-3: indices [2,3,6,7]
    # Block A10: rows 2-3, cols 0-1: indices [8,9,12,13]
    # Block A11: rows 2-3, cols 2-3: indices [10,11,14,15]
    
    A00 = [0, 1, 4, 5]
    A01 = [2, 3, 6, 7]
    A10 = [8, 9, 12, 13]
    A11 = [10, 11, 14, 15]
    
    # B is 4x6, flattened row-major: index = i*6 + j
    # For 4x4x4 part (cols 0-3):
    # B00: rows 0-1, cols 0-1: [0,1,6,7]
    # B01: rows 0-1, cols 2-3: [2,3,8,9]
    # B10: rows 2-3, cols 0-1: [12,13,18,19]
    # B11: rows 2-3, cols 2-3: [14,15,20,21]
    
    B00 = [0, 1, 6, 7]
    B01 = [2, 3, 8, 9]
    B10 = [12, 13, 18, 19]
    B11 = [14, 15, 20, 21]
    
    # For 4x4x2 part (cols 4-5):
    # B_top: rows 0-1, cols 4-5: [4,5,10,11]
    # B_bot: rows 2-3, cols 4-5: [16,17,22,23]
    
    B_top = [4, 5, 10, 11]
    B_bot = [16, 17, 22, 23]
    
    # C is 4x6, output. Trace convention: W[k*n+i, r] where n=4.
    # So C index = k*4 + i where C[i,k] = sum_j A[i,j]*B[j,k]
    # C is 4x6: C[i,k] for i in 0..3, k in 0..5.
    # Flattened: index = i*6 + k? No, the W matrix has shape (p*n, rank) = (6*4, rank) = (24, rank).
    # W[k*n+i, r] = W[k*4+i, r]. So the row index in W is k*4+i.
    # C[i,k] corresponds to W[k*4+i].
    
    # For 4x4x4 part (k=0..3):
    # C00: i=0,1; k=0,1: W indices: k*4+i -> (0*4+0, 0*4+1, 1*4+0, 1*4+1) = [0,1,4,5]
    # C01: i=0,1; k=2,3: [2,3,6,7]
    # C10: i=2,3; k=0,1: [8,9,12,13]
    # C11: i=2,3; k=2,3: [10,11,14,15]
    
    C00 = [0, 1, 4, 5]
    C01 = [2, 3, 6, 7]
    C10 = [8, 9, 12, 13]
    C11 = [10, 11, 14, 15]
    
    # For 4x4x2 part (k=4,5):
    # C_top_right: i=0,1; k=4,5: [16,17,20,21]
    # C_bot_right: i=2,3; k=4,5: [18,19,22,23]
    
    C_top_right = [16, 17, 20, 21]
    C_bot_right = [18, 19, 22, 23]
    
    # 4x4x4 part:
    # C00 = A00*B00 + A01*B10
    add_strassen_instance(A00, B00, C00)
    add_strassen_instance(A01, B10, C00)
    # C01 = A00*B01 + A01*B11
    add_strassen_instance(A00, B01, C01)
    add_strassen_instance(A01, B11, C01)
    # C10 = A10*B00 + A11*B10
    add_strassen_instance(A10, B00, C10)
    add_strassen_instance(A11, B10, C10)
    # C11 = A10*B01 + A11*B11
    add_strassen_instance(A10, B01, C11)
    add_strassen_instance(A11, B11, C11)
    
    # 4x4x2 part:
    # C_top_right = A00*B_top + A01*B_bot
    add_strassen_instance(A00, B_top, C_top_right)
    add_strassen_instance(A01, B_bot, C_top_right)
    # C_bot_right = A10*B_top + A11*B_bot
    add_strassen_instance(A10, B_top, C_bot_right)
    add_strassen_instance(A11, B_bot, C_bot_right)
    
    rank = len(all_U)
    U = all_U
    V = all_V
    W = all_W
    
    # Verify
    assert len(U) == n * m
    assert len(V) == m * p
    assert len(W) == p * n
    for col in range(rank):
        assert len(U[col]) == n * m
        assert len(V[col]) == m * p
        assert len(W[col]) == p * n
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

def _verify_decomposition(U, V, W):
    """Verify that the decomposition correctly computes matrix multiplication."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    
    # Check: sum_t U[a,t] * V[b,t] * W[c,t] = delta condition
    # For matrix mult: T[i,j,k] = 1 iff j is the "middle" index
    # A is n x m, B is m x p, C is n x p
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor: T[i, j, k] = 1 (scalar coefficient)
    # Decomposition: sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = T[i,j,k]
    
    # Test with random values
    import random
    random.seed(42)
    A = [[random.randint(-5, 5) for _ in range(m)] for _ in range(n)]
    B = [[random.randint(-5, 5) for _ in range(p)] for _ in range(m)]
    
    # Compute C = A * B
    C = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
    
    # Compute using decomposition
    C_dec = [[0] * p for _ in range(n)]
    for t in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    C_dec[i][k] += U[i * m + j][t] * V[j * p + k][t] * W[k * n + i][t]
    
    for i in range(n):
        for k in range(p):
            if C_dec[i][k] != C[i][k]:
                print(f"Mismatch at [{i},{k}]: {C_dec[i][k]} vs {C[i][k]}")
                return False
    return True

if __name__ == "__main__":
    result = construct()
    rank = len(result["U"][0])
    print(f"Rank: {rank}")
    valid = _verify_decomposition(result["U"], result["V"], result["W"])
    print(f"Valid: {valid}")
