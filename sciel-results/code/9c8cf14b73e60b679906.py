"""Rewritten program for the suite task; EVOLVE-BLOCK replaced with a stronger algorithm."""
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def _build_strassen_2x2x2():
    """Return (U, V, W) lists for 2x2x2 Strassen decomposition.
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
    U[i*2+j, k] is the coefficient of A[i,j] in the k-th multiplication.
    V[j*2+k, k] is the coefficient of B[j,k] in the k-th multiplication.
    W[k*2+i, k] is the coefficient of C[i,k] in the k-th addition.
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    # M1 = (A00+A11)*(B00+B11) -> contributes to C00 (+), C11 (+), C01 (-), C10 (+)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1; W[1][0] = -1; W[2][0] = 1
    # M2 = (A10+A11)*B00 -> contributes to C10 (+), C11 (-)
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    # M3 = A00*(B01-B11) -> contributes to C01 (+), C00 (-)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[0][2] = -1
    # M4 = A11*(B10-B00) -> contributes to C10 (+), C11 (-)  [wait, let me re-check]
    # Actually standard Strassen:
    # M4 = A11*(B10 - B00) -> contributes to C10? Let me use the standard set.
    # Let me use the most common formulation:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 - M4 + M6
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M7
    
    # Let's rebuild properly.
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M0: (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    # This contributes to C00, C11, C01(-), C10(+) in the combination below, but let's use the standard combination.
    # Actually, let's use the exact standard formulas for W:
    # C00 = M1 - M4 + M6
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M7
    
    # M1 (index 0): (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1  # Will be combined
    
    # M2 (index 1): (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # M3 (index 2): A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1
    
    # M4 (index 3): A11(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[2][3] = 1; W[0][3] = -1  # Wait, C10 = M2 + M4, C00 = M1 - M4 + M6
    # So M4 contributes +1 to C10, -1 to C00.
    W[2][3] = 1; W[0][3] = -1
    
    # M5 (index 4): (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[1][4] = 1
    
    # M6 (index 5): (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[0][5] = 1
    
    # M7 (index 6): (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[3][6] = 1
    
    # Now fix W to match the combination:
    # C00 = M1 - M4 + M6
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M7
    
    W = [[0]*7 for _ in range(4)]
    W[0][0] = 1  # C00 += M1
    W[0][3] = -1 # C00 -= M4
    W[0][5] = 1  # C00 += M6
    
    W[1][2] = 1  # C01 += M3
    W[1][4] = 1  # C01 += M5
    
    W[2][1] = 1  # C10 += M2
    W[2][3] = 1  # C10 += M4
    
    W[3][0] = 1  # C11 += M1
    W[3][2] = -1 # C11 -= M3
    W[3][6] = 1  # C11 += M7
    
    return U, V, W

def _build_schoolbook_2x2x2x1():
    """Return (U, V, W) for 2x2 matrix times 2x1 vector, schoolbook. Rank 4.
    U: (4, 4), V: (2, 4), W: (2, 4)
    """
    U = [[0]*4 for _ in range(4)]
    V = [[0]*4 for _ in range(2)]
    W = [[0]*4 for _ in range(2)]
    U[0][0] = 1; V[0][0] = 1; W[0][0] = 1  # A00*B0
    U[1][1] = 1; V[1][1] = 1; W[0][1] = 1  # A01*B1
    U[2][2] = 1; V[0][2] = 1; W[1][2] = 1  # A10*B0
    U[3][3] = 1; V[1][3] = 1; W[1][3] = 1  # A11*B1
    return U, V, W

def _build_block(n_block, m_block, p_block):
    """Build decomposition for n_block x m_block matrix times m_block x p_block matrix.
    For 2x2x2: Strassen (rank 7)
    For 2x2x1: Schoolbook (rank 4)
    Returns (U, V, W) where U is (n_block*m_block, rank), V is (m_block*p_block, rank), W is (p_block*n_block, rank)
    """
    if p_block == 2:
        return _build_strassen_2x2x2()
    elif p_block == 1:
        return _build_schoolbook_2x2x2x1()
    else:
        # General schoolbook for now
        rank = n_block * m_block * p_block
        U = [[0]*rank for _ in range(n_block*m_block)]
        V = [[0]*rank for _ in range(m_block*p_block)]
        W = [[0]*rank for _ in range(p_block*n_block)]
        col = 0
        for i in range(n_block):
            for j in range(m_block):
                for k in range(p_block):
                    U[i*m_block+j][col] = 1
                    V[j*p_block+k][col] = 1
                    W[k*n_block+i][col] = 1
                    col += 1
        return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Decompose 4x4 times 4x7 into 4 blocks of 2x2 times 2x(sub-p)
    # Row blocks: A[0:2, 0:4] and A[2:4, 0:4] -> actually A is 4x4, so 2x2 blocks in rows
    # A = [A11 A12; A21 A22] where each is 2x2
    # B = 4x7. Split columns into 4 blocks? No, m=4 is the shared dimension.
    # A is 4x4, B is 4x7.
    # Split A into 2x2 blocks: A = [[A00, A01], [A10, A11]] (each 2x2)
    # Split B into 2x7 blocks: B = [B_top; B_bottom] where B_top is 2x7, B_bottom is 2x7
    # Then C = A*B = [A00*B_top + A01*B_bottom, A10*B_top + A11*B_bottom]
    # Each term is 2x2 times 2x7 -> 2x7 result.
    # But we want to use Strassen/schoolbook which are for 2x2 x 2x2 or 2x2 x 2x1.
    # So split B's columns: B_top = [B0, B1] where B0, B1 are 2x2? No, B is 2x7, so 2x(2+2+2+1) or similar.
    # Let's split the 7 columns into blocks of size 2,2,2,1.
    # So B_top = [B00, B01, B02, B03] (2x2, 2x2, 2x2, 2x1)
    # B_bottom = [B10, B11, B12, B13] (2x2, 2x2, 2x2, 2x1)
    
    # For A00 (2x2) times B_top (2x7):
    # A00 * [B00, B01, B02, B03] = [A00*B00, A00*B01, A00*B02, A00*B03]
    # A00*B00: 2x2 x 2x2 -> Strassen rank 7
    # A00*B01: 2x2 x 2x2 -> Strassen rank 7
    # A00*B02: 2x2 x 2x2 -> Strassen rank 7
    # A00*B03: 2x2 x 2x1 -> Schoolbook rank 4
    # Total for A00*B_top: 7+7+7+4 = 25
    
    # Similarly for A01*B_top, A10*B_bottom, A11*B_bottom: each 25
    # Total rank: 4 * 25 = 100
    
    # Wait, can we share? The Strassen multiplications for different blocks are independent.
    # Rank 100. Schoolbook was 112. So 100 is better.
    
    # Let's construct this.
    # U has shape (16, R), V has shape (28, R), W has shape (28, R)
    # n=4, m=4, p=7
    # U[i*4+j, col] where i in 0..3, j in 0..3
    # V[j*7+k, col] where j in 0..3, k in 0..6
    # W[k*4+i, col] where k in 0..6, i in 0..3
    
    # Let's define the 4 sub-problems:
    # Sub-problem (ai, bj) where ai in {0,1} (row block of A), bj in {0,1} (col block of B, i.e., row block of B)
    # A block: rows 2*ai to 2*ai+1, cols 0 to 3 (all cols, but we split into 2x2 blocks? No)
    
    # Actually, let me re-think the blocking.
    # A is 4x4. B is 4x7.
    # A = [A_row0; A_row1] where A_row0 is 2x4, A_row1 is 2x4
    # B = [B_col0; B_col1] where B_col0 is 2x7, B_col1 is 2x7
    # C = [A_row0 * B_col0 + A_row0 * B_col1; A_row1 * B_col0 + A_row1 * B_col1]
    # Each product is 2x2 * 2x7 -> 2x7.
    # A_row0 * B_col0: 2x4 * 2x7. Split the 4 cols of A_row0 into two 2x2 blocks.
    # A_row0 = [A00, A01] (2x2 each)
    # B_col0 = 2x7, split cols into 2,2,2,1.
    # A00 * B_col0 = A00 * [B0, B1, B2, B3] = [A00*B0, A00*B1, A00*B2, A00*B3]
    # A01 * B_col0 = A01 * [B0, B1, B2, B3] = [A01*B0, A01*B1, A01*B2, A01*B3]
    # Each Axx*Bx is 2x2 * 2x2 (rank 7) or 2x2 * 2x1 (rank 4).
    
    # So for each of the 4 (A_row, B_col) pairs, we have 4 sub-products.
    # Total 16 sub-products. 12 are 2x2x2 (rank 7 each), 4 are 2x2x1 (rank 4 each).
    # Total rank = 12*7 + 4*4 = 84 + 16 = 100.
    
    # Let's build it.
    R = 100
    U = [[0]*R for _ in range(16)]
    V = [[0]*R for _ in range(28)]
    W = [[0]*R for _ in range(28)]
    
    col = 0
    for ai in range(2):  # A row block
        for aj in range(2):  # A col sub-block (2x2)
            for bi in range(2):  # B row block
                for bk in range(4):  # B col sub-block index (0,1,2,3)
                    if bk < 3:
                        # 2x2 x 2x2, rank 7
                        U_sub, V_sub, W_sub = _build_strassen_2x2x2()
                        p_sub = 2
                    else:
                        # 2x2 x 2x1, rank 4
                        U_sub, V_sub, W_sub = _build_schoolbook_2x2x2x1()
                        p_sub = 1
                    
                    rank_sub = len(U_sub[0])
                    # Map sub-indices to global
                    # A block: rows 2*ai to 2*ai+1, cols 2*aj to 2*aj+1
                    # B block: rows 2*bi to 2*bi+1, cols 2*bk to 2*bk+p_sub-1
                    # C block: rows 2*bi to 2*bi+1 (from B), cols 2*ai to 2*ai+1 (from A)
                    
                    for r in range(2):
                        for c in range(2):
                            U_idx = (2*ai + r) * 4 + (2*aj + c)
                            for s in range(rank_sub):
                                U[U_idx][col+s] += U_sub[r*2+c][s]
                    
                    for r in range(2):
                        for c in range(p_sub):
                            V_idx = (2*bi + r) * 7 + (2*bk + c)
                            for s in range(rank_sub):
                                V[V_idx][col+s] += V_sub[r*p_sub+c][s]
                    
                    for r in range(p_sub):
                        for c in range(2):
                            W_idx = (2*bk + r) * 4 + (2*ai + c)
                            for s in range(rank_sub):
                                W[W_idx][col+s] += W_sub[r*2+c][s]
                    
                    col += rank_sub
    
    # Verify we used all columns
    assert col == R, f"Expected {R} columns, got {col}"
    
    # Verify the decomposition
    # C[i,k] = sum_j A[i,j] * B[j,k] = sum_c sum_{i',j'} U[i*4+j', c] * V[j'*7+k', c] * W[k'*4+i, c]
    # Let's do a quick check with a random matrix (if time permits)
    if time.time() < deadline - 10.0:
        try:
            import random
            random.seed(42)
            A_test = [[random.randint(-1, 1) for _ in range(4)] for _ in range(4)]
            B_test = [[random.randint(-1, 1) for _ in range(7)] for _ in range(4)]
            
            # Direct product
            C_direct = [[0]*7 for _ in range(4)]
            for i in range(4):
                for k in range(7):
                    s = 0
                    for j in range(4):
                        s += A_test[i][j] * B_test[j][k]
                    C_direct[i][k] = s
            
            # Decomposed product
            C_decomp = [[0]*7 for _ in range(4)]
            for c in range(R):
                for i in range(4):
                    for j in range(4):
                        if U[i*4+j][c] != 0:
                            u_val = U[i*4+j][c]
                            for kk in range(7):
                                for jj in range(4):
                                    if V[jj*7+kk][c] != 0:
                                        v_val = V[jj*7+kk][c]
                                        # This is getting complex, let's do it differently
                                        pass
            
            # Better: for each term c, compute the outer product contribution
            # C_decomp[i,k] += U[i*4+j, c] * V[j*7+k, c] * W[k*4+i, c] ... no
            # C[i,k] = sum_c (sum_j U[i*4+j, c] * V[j*7+k, c]) * ... no
            # The tensor is sum_c U_c (tensor) V_c (tensor) W_c
            # C[i,k] = sum_c sum_j U[i*4+j, c] * V[j*7+k, c] * ... wait
            # W has indices W[k*4+i, c]. The output C[i,k] gets contribution W[k*4+i, c] * (U[i*4+j,c]*V[j*7+k,c] summed over j)
            # No, that's not right. The multiplication tensor is T[i,j,k] = delta[i,i]delta[j,j]... 
            # T[i,j,k] is 1 if the multiplication A[i,j]*B[j,k] contributes to C[i,k].
            # In standard matrix mult, C[i,k] = sum_j A[i,j]*B[j,k].
            # So T[i,j,k] = 1 for all i,j,k? No, that's the identity.
            # The tensor for matrix multiplication M_{n,m,p} has T[i,j,k] = 1 if we're computing C[i,k] from A[i,j] and B[j,k].
            # Actually, the rank decomposition is: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            # Where T[i,j,k] = 1 (since each product A[i,j]*B[j,k] contributes to C[i,k]).
            # Wait, no. In matrix multiplication, C[i,k] = sum_j A[i,j]*B[j,k].
            # So the tensor T is such that T[i,j,k] = 1 for all i,j,k? That would mean the tensor is all ones.
            # No, that's not right. The tensor is defined such that the bilinear map (A, B) -> C has:
            # C[i,k] = sum_{j} A[i,j] * B[j,k]
            # In tensor form, if we vectorize A, B, C, then C = T * (A ⊗ B)
            # The tensor T has entries T[i,k, j] ... this is getting confusing.
            
            # Let's use the given convention: W[k*n+i, r]
            # The decomposition is: for each r, we have U_r (size n*m), V_r (size m*p), W_r (size p*n)
            # C[i,k] = sum_r W_r[k*n+i] * (sum_j U_r[i*m+j] * V_r[j*p+k])
            # No, that's not bilinear in A and B.
            
            # Correct form: C[i,k] = sum_r (sum_j U_r[i*m+j] * A[i,j]) * (sum_j V_r[j*p+k] * B[j,k]) * W_r[k*n+i]
            # No, that's cubic.
            
            # The standard form is: C[i,k] = sum_r U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i] * A[i,j] * B[j,k]
            # where the sum is over r and j? No.
            
            # Let's think of it as: for each r, we compute a scalar s_r = sum_j U_r[i*m+j] * A[i,j] * V_r[j*p+k] * B[j,k]
            # Then C[i,k] = sum_r W_r[k*n+i] * s_r
            # This is correct! It's bilinear in A and B for each r.
            
            # So let's verify this way.
            C_decomp = [[0]*7 for _ in range(4)]
            for i in range(4):
                for k in range(7):
                    s = 0
                    for r in range(R):
                        w_val = W[k*4+i][r]
                        if w_val == 0:
                            continue
                        inner = 0
                        for j in range(4):
                            u_val = U[i*4+j][r]
                            v_val = V[j*7+k][r]
                            if u_val != 0 and v_val != 0:
                                inner += u_val * A_test[i][j] * v_val * B_test[j][k]
                        s += w_val * inner
                    C_decomp[i][k] = s
            
            # Compare
            max_diff = 0
            for i in range(4):
                for k in range(7):
                    diff = abs(C_direct[i][k] - C_decomp[i][k])
                    if diff > max_diff:
                        max_diff = diff
            
            if max_diff > 0:
                print(f"Verification failed! Max diff: {max_diff}")
                # Fall back to schoolbook
                n, m, p = DIMENSIONS
                rank = n * m * p
                U = [[0] * rank for _ in range(n * m)]
                V = [[0] * rank for _ in range(m * p)]
                W = [[0] * rank for _ in range(p * n)]
                column = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            U[i * m + j][column] = 1
                            V[j * p + k][column] = 1
                            W[k * n + i][column] = 1
                            column += 1
            else:
                print(f"Verification passed! Rank {R}")
        except Exception as e:
            print(f"Verification error: {e}")
            # Fall back to schoolbook
            n, m, p = DIMENSIONS
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            column = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j][column] = 1
                        V[j * p + k][column] = 1
                        W[k * n + i][column] = 1
                        column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
