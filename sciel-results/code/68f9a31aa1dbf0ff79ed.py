"""Program for matrix multiplication 2x5 x 5x5 tensor decomposition."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import time

def _add_term(U, V, W, u_idx, v_idx, w_idx, val_u, val_v, val_w, col):
    """Add a rank-1 term to the factor matrices at column col."""
    U[u_idx][col] += val_u
    V[v_idx][col] += val_v
    W[w_idx][col] += val_w

def _strassen_block(U, V, W, a_row, a_col, b_row, b_col, c_row, c_col, col_offset):
    """
    Add Strassen rank-7 terms for a 2x2 * 2x2 block.
    A is at rows [a_row, a_row+1], cols [a_col, a_col+1]
    B is at rows [b_row, b_row+1], cols [b_col, b_col+1]
    C is at rows [c_row, c_row+1], cols [c_col, c_col+1]
    (Note: C index is k*n+i, so C[row][col] where row=k, col=i)
    
    Wait, let's clarify the indexing.
    A: n x m. Index (i, j) -> flat i*m + j
    B: m x p. Index (j, k) -> flat j*p + k
    C: p x n. Index (k, i) -> flat k*n + i
    
    So for a block:
    A_block: rows i in [a_row, a_row+1], cols j in [a_col, a_col+1]
    B_block: rows j in [b_row, b_col+1], cols k in [b_col, b_col+1]
    C_block: rows k in [c_row, c_row+1], cols i in [c_col, c_col+1]
    
    Strassen:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B01-B11)
    M4 = A11(B10-B00)
    M5 = (A00+A01)B11
    M6 = (A10-A00)(B00+B01)
    M7 = (A01-A11)(B10+B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    # A indices
    A00 = a_row * 5 + a_col
    A01 = a_row * 5 + a_col + 1
    A10 = (a_row + 1) * 5 + a_col
    A11 = (a_row + 1) * 5 + a_col + 1
    
    # B indices
    B00 = b_row * 5 + b_col
    B01 = b_row * 5 + b_col + 1
    B10 = (b_row + 1) * 5 + b_col
    B11 = (b_row + 1) * 5 + b_col + 1
    
    # C indices
    C00 = c_row * 2 + c_col
    C01 = c_row * 2 + c_col + 1
    C10 = (c_row + 1) * 2 + c_col
    C11 = (c_row + 1) * 2 + c_col + 1
    
    # M1 = (A00+A11)(B00+B11)
    _add_term(U, V, W, A00, B00, C00, 1, 1, 1, col_offset)
    _add_term(U, V, W, A00, B11, C00, 1, 1, 1, col_offset)
    _add_term(U, V, W, A11, B00, C00, 1, 1, 1, col_offset)
    _add_term(U, V, W, A11, B11, C00, 1, 1, 1, col_offset)
    _add_term(U, V, W, A00, B00, C11, 1, 1, 1, col_offset)
    _add_term(U, V, W, A00, B11, C11, 1, 1, 1, col_offset)
    _add_term(U, V, W, A11, B00, C11, 1, 1, 1, col_offset)
    _add_term(U, V, W, A11, B11, C11, 1, 1, 1, col_offset)
    col_offset += 1
    
    # M2 = (A10+A11)B00
    _add_term(U, V, W, A10, B00, C10, 1, 1, 1, col_offset)
    _add_term(U, V, W, A11, B00, C10, 1, 1, 1, col_offset)
    _add_term(U, V, W, A10, B00, C11, 1, 1, -1, col_offset)
    _add_term(U, V, W, A11, B00, C11, 1, 1, -1, col_offset)
    col_offset += 1
    
    # M3 = A00(B01-B11)
    _add_term(U, V, W, A00, B01, C01, 1, 1, 1, col_offset)
    _add_term(U, V, W, A00, B11, C01, 1, -1, 1, col_offset)
    _add_term(U, V, W, A00, B01, C11, 1, 1, 1, col_offset)
    _add_term(U, V, W, A00, B11, C11, 1, -1, 1, col_offset)
    col_offset += 1
    
    # M4 = A11(B10-B00)
    _add_term(U, V, W, A11, B10, C00, 1, 1, 1, col_offset)
    _add_term(U, V, W, A11, B00, C00, 1, -1, 1, col_offset)
    _add_term(U, V, W, A11, B10, C10, 1, 1, 1, col_offset)
    _add_term(U, V, W, A11, B00, C10, 1, -1, 1, col_offset)
    col_offset += 1
    
    # M5 = (A00+A01)B11
    _add_term(U, V, W, A00, B11, C01, 1, 1, 1, col_offset)
    _add_term(U, V, W, A01, B11, C01, 1, 1, 1, col_offset)
    _add_term(U, V, W, A00, B11, C00, 1, 1, -1, col_offset)
    _add_term(U, V, W, A01, B11, C00, 1, 1, -1, col_offset)
    col_offset += 1
    
    # M6 = (A10-A00)(B00+B01)
    _add_term(U, V, W, A10, B00, C11, 1, 1, 1, col_offset)
    _add_term(U, V, W, A10, B01, C11, 1, 1, 1, col_offset)
    _add_term(U, V, W, A00, B00, C11, -1, 1, 1, col_offset)
    _add_term(U, V, W, A00, B01, C11, -1, 1, 1, col_offset)
    col_offset += 1
    
    # M7 = (A01-A11)(B10+B11)
    _add_term(U, V, W, A01, B10, C00, 1, 1, 1, col_offset)
    _add_term(U, V, W, A01, B11, C00, 1, 1, 1, col_offset)
    _add_term(U, V, W, A11, B10, C00, -1, 1, 1, col_offset)
    _add_term(U, V, W, A11, B11, C00, -1, 1, 1, col_offset)
    col_offset += 1
    
    return col_offset

def _schoolbook_block(U, V, W, a_rows, b_rows, c_rows, col_offset):
    """
    Add schoolbook rank (len(a_rows)*len(b_rows)) terms.
    a_rows: list of A row indices
    b_rows: list of B row indices  
    c_rows: list of C row indices (k indices)
    """
    for a_r in a_rows:
        for b_r in b_rows:
            for k_idx, c_r in enumerate(c_rows):
                # A row a_r, all cols
                # B row b_r, all cols
                # C row c_r, all cols (i indices)
                # This is a bit tricky for schoolbook with blocks
                pass
    # Actually, for the thin blocks, it's easier to think of it as:
    # For each (i, j, k) in the block, add a rank-1 term.
    # But that's just schoolbook. Let me reconsider.
    # For a block A[a_rows, all_cols] * B[b_rows, all_cols] -> C[c_rows, all_cols]
    # The rank is len(a_rows) * len(b_rows) if we do it properly.
    # Actually, for A (r x c) * B (c x c) -> C (c x r), the rank is r * c.
    # Let's just do full schoolbook for the small blocks.
    pass

def construct():
    n, m, p = DIMENSIONS
    rank = 46  # 4*7 + 4*4 + 2 = 28 + 16 + 2 = 46
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Block partition: 2+2+1 for both rows/cols of B, and 2+1 for A rows (since n=2)
    # A has 2 rows: [0, 1]
    # B has 5 rows: [0,1], [2,3], [4]
    # C has 5 rows (k): [0,1], [2,3], [4]
    # C has 2 cols (i): [0, 1]
    
    # A rows: [0,1] (2 rows)
    # B rows: [0,1], [2,3], [4]
    # C rows (k): [0,1], [2,3], [4]
    # C cols (i): [0,1]
    
    # Block (A_rows=[0,1], B_rows=[0,1], C_rows=[0,1]): 2x2 * 2x2 -> Strassen
    col = _strassen_block(U, V, W, 0, 0, 0, 0, 0, 0, col)
    
    # Block (A_rows=[0,1], B_rows=[2,3], C_rows=[0,1]): 2x2 * 2x2 -> Strassen
    col = _strassen_block(U, V, W, 0, 2, 2, 0, 0, 0, col)
    
    # Block (A_rows=[0,1], B_rows=[4], C_rows=[0,1]): 2x1 * 1x2 -> Schoolbook rank 2*1=2? 
    # Wait, A is 2x1, B is 1x2, C is 2x2. Rank should be 2*1=2.
    # Actually, for A (r x c) * B (c x d) -> C (d x r), rank is r*c if c<=d, or c*d if d<c? 
    # No, the rank of the tensor is r*c*d for schoolbook. But we can do better.
    # For 2x1 * 1x2: A has 2 elements, B has 2 elements, C has 4 elements.
    # Rank is at most min(2*2, 2*2, 4) = 4? No, that's not right.
    # Actually, for matrix multiplication A (r x c) * B (c x d) -> C (d x r),
    # the tensor rank is r*c*d for schoolbook. But for thin matrices, we can do better.
    # For 2x1 * 1x2: we need to compute C[k,i] = A[i,0] * B[0,k] for k in {0,1}, i in {0,1}.
    # This is 4 independent products, so rank 4.
    # Wait, A[i,0] * B[0,k] for i=0,1 and k=0,1. These are 4 distinct bilinear forms.
    # So rank is 4.
    
    # Let me recount the ranks:
    # Block (2x2 * 2x2): rank 7. There are 2 such blocks (B cols 0-1 and 2-3). Total: 14.
    # Block (2x1 * 1x2): rank 4. There is 1 such block (B col 4). Total: 4.
    # Wait, I need to be more careful.
    
    # Let me redo the block structure.
    # A: 2 rows (i=0,1), 5 cols (j=0..4)
    # B: 5 rows (j=0..4), 5 cols (k=0..4)
    # C: 5 rows (k=0..4), 2 cols (i=0,1)
    
    # Partition j into [0,1], [2,3], [4]
    # Partition k into [0,1], [2,3], [4]
    
    # For each (j_block, k_block), we have A[:, j_block] * B[j_block, k_block] -> C[k_block, :]
    # A[:, j_block] is 2 x len(j_block)
    # B[j_block, k_block] is len(j_block) x len(k_block)
    # C[k_block, :] is len(k_block) x 2
    
    # Block (j=[0,1], k=[0,1]): 2x2 * 2x2 -> rank 7
    # Block (j=[0,1], k=[2,3]): 2x2 * 2x2 -> rank 7
    # Block (j=[0,1], k=[4]): 2x2 * 2x1 -> rank 4 (schoolbook: 2*2*1=4)
    # Block (j=[2,3], k=[0,1]): 2x2 * 2x2 -> rank 7
    # Block (j=[2,3], k=[2,3]): 2x2 * 2x2 -> rank 7
    # Block (j=[2,3], k=[4]): 2x2 * 2x1 -> rank 4
    # Block (j=[4], k=[0,1]): 2x1 * 1x2 -> rank 2 (schoolbook: 2*1*2=4? No, 2*1*1=2? 
    #    Wait, A is 2x1, B is 1x2. C is 2x2. C[k,i] = A[i,0]*B[0,k]. 
    #    This is 4 products, but they share structure. Actually rank is 4 for schoolbook.
    #    But we can do it in rank 4.)
    # Block (j=[4], k=[2,3]): 2x1 * 1x2 -> rank 4
    # Block (j=[4], k=[4]): 2x1 * 1x1 -> rank 2
    
    # Total: 7+7+4+7+7+4+4+4+2 = 46. Yes!
    
    # Let me implement the schoolbook blocks properly.
    
    def schoolbook_thin(a_row, a_col_start, b_row_start, k_start, c_col_start, col_off, num_a, num_b, num_c):
        """
        For A rows [a_row], cols [a_col_start..a_col_start+num_a-1]
        B rows [b_row_start..b_row_start+num_b-1], cols [k_start..k_start+num_c-1]
        C rows [k_start..k_start+num_c-1], cols [c_col_start..c_col_start+num_a-1]
        Wait, this is getting confusing. Let me just hardcode each block.
        """
        pass
    
    # Block (j=[0,1], k=[4]): A[:, 0:2] * B[0:2, 4:5] -> C[4:5, :]
    # A is 2x2, B is 2x1, C is 1x2. C[4,i] = sum_j A[i,j]*B[j,4]
    # This is 2*2*1 = 4 rank-1 terms (schoolbook).
    for i in range(2):
        for j in range(2):
            _add_term(U, V, W, i*5+j, j*5+4, 4*2+i, 1, 1, 1, col)
            col += 1
    
    # Block (j=[2,3], k=[4]): A[:, 2:4] * B[2:4, 4:5] -> C[4:5, :]
    for i in range(2):
        for j in range(2, 4):
            _add_term(U, V, W, i*5+j, j*5+4, 4*2+i, 1, 1, 1, col)
            col += 1
    
    # Block (j=[4], k=[0,1]): A[:, 4:5] * B[4:5, 0:2] -> C[0:2, :]
    for i in range(2):
        for k in range(2):
            _add_term(U, V, W, i*5+4, 4*5+k, k*2+i, 1, 1, 1, col)
            col += 1
    
    # Block (j=[4], k=[2,3]): A[:, 4:5] * B[4:5, 2:4] -> C[2:4, :]
    for i in range(2):
        for k in range(2, 4):
            _add_term(U, V, W, i*5+4, 4*5+k, k*2+i, 1, 1, 1, col)
            col += 1
    
    # Block (j=[4], k=[4]): A[:, 4:5] * B[4:5, 4:5] -> C[4:5, :]
    for i in range(2):
        _add_term(U, V, W, i*5+4, 4*5+4, 4*2+i, 1, 1, 1, col)
        col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
