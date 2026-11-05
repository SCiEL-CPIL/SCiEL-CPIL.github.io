"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W

def _strassen_2x2():
    # Strassen 7-term decomposition for 2x2x2
    # U rows indexed by (i,j) for i,j in {0,1}, V rows by (j,k), W rows by (k,i)
    # Convention: W[k*n+i, r] where n=2
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # Strassen terms:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A10)B11
    # M6 = (A01-A10)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # U[i*m+j, r] where m=2, so row = i*2+j
    # V[j*p+k, r] where p=2, so row = j*2+k
    # W[k*n+i, r] where n=2, so row = k*2+i
    
    # M1: U: A00=1, A11=1; V: B00=1, B11=1; W: C00+=1, C01+=0, C10+=0, C11+=-1... 
    # Actually C = M1 - M3 - M4 + M5 for C00? Let me use the standard form.
    # C00 = M1 - M3 - M4 + M5
    # C01 = M3 + M7
    # C10 = M2 + M4
    # C11 = M1 - M2 + M6 - M5
    
    # Term 0 (M1): A00=1, A11=1 | B00=1, B11=1 | C00=1, C11=-1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = -1  # W[k*n+i]: C00 is k=0,i=0 -> row 0; C11 is k=1,i=1 -> row 3
    
    # Term 1 (M2): A10=1, A11=1 | B00=1 | C10=1
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1  # C10: k=1, i=0 -> row 2
    
    # Term 2 (M3): A00=1 | B01=1, B11=-1 | C00=-1, C01=1
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[0, 2] = -1; W[1, 2] = 1  # C00: row 0, C01: k=0,i=1 -> row 1
    
    # Term 3 (M4): A11=1 | B10=1, B00=-1 | C00=-1, C10=1
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = -1; W[2, 3] = 1  # C00: row 0, C10: row 2
    
    # Term 4 (M5): A00=1, A10=1 | B11=1 | C00=1, C11=-1
    U[0, 4] = 1; U[2, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[3, 4] = -1
    
    # Term 5 (M6): A01=1, A10=-1 | B00=1, B10=1 | C11=1
    U[1, 5] = 1; U[2, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[3, 5] = 1  # C11: row 3
    
    # Term 6 (M7): A01=1, A11=-1 | B01=1, B11=1 | C01=1
    U[1, 6] = 1; U[3, 6] = -1
    V[1, 6] = 1; V[3, 6] = 1
    W[1, 6] = 1  # C01: row 1
    
    return U, V, W

def _combine_blocks(n, m, p, blocks):
    """
    Combine block decompositions.
    blocks: list of (m_block, U_block, V_block, W_block)
    where U_block has shape (n*m_block, rank_block)
    V_block has shape (m_block*p, rank_block)
    W_block has shape (p*n, rank_block)
    """
    total_rank = sum(b[1].shape[1] for b in blocks)
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)
    
    col_offset = 0
    for m_b, U_b, V_b, W_b in blocks:
        r_b = U_b.shape[1]
        U[:, col_offset:col_offset+r_b] = U_b
        V[:, col_offset:col_offset+r_b] = V_b
        W[:, col_offset:col_offset+r_b] = W_b
        col_offset += r_b
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    n, m, p = DIMENSIONS  # 3, 5, 7
    
    # Strategy: Split the 5 columns into 2+2+1
    # For each column block of width w, we need to multiply 3xw by w x 7
    # For w=2: use Strassen on the 2x2 part, but we need 3x2x7
    # Actually, let's think differently.
    
    # Block decomposition: split m=5 into m1=2, m2=2, m3=1
    # For each block, we compute A_block (3 x m_i) * B_block (m_i x 7)
    # The tensor for each block is 3 x m_i x 7
    
    # For m_i=2: 3x2x7 tensor. We can use Strassen-like approach.
    # Actually, for 3x2x7, the schoolbook rank is 42.
    # Can we do better? 
    # 
    # Let's use a different approach: 
    # Split n=3 into 2+1, m=5 into 2+2+1, p=7 into 4+3 or other splits.
    # 
    # Actually, let's try: n=3=2+1, m=5=2+2+1, p=7=4+3
    # This gives 2*3*2 = 12 sub-blocks, each of size (n_i x m_j x p_k)
    # 
    # Sub-block sizes:
    # (2,2,4), (2,2,3), (2,2,4), (2,2,3), (2,2,4), (2,2,3),
    # (1,2,4), (1,2,3), (1,2,4), (1,2,3), (1,2,4), (1,2,3)
    #
    # This is getting complex. Let me use a simpler approach.
    #
    # For the 3x2x7 block:
    # We can view this as: for each of the 3 rows of A and 7 cols of B,
    # we have 2 inner products of length 2.
    # Schoolbook: 3*2*7 = 42 terms.
    # 
    # Using Strassen on the 2x2 part:
    # A is 3x2, B is 2x7. We can pad A to 2x2 and B to 2x2? No, that doesn't work directly.
    #
    # Alternative: Use the fact that for 3x2x7, we can apply Strassen to pairs of rows.
    # 
    # Let me try a direct approach: for m=2, the 3x2x7 tensor has rank at most 3*7*2=42.
    # But we can do better by using the structure.
    #
    # Actually, let's use a known result: the 2x2x2 tensor has rank 7 (Strassen).
    # For 3x2x7, we can think of it as 7 copies of 3x2x1, but that's not helpful.
    #
    # Let me try a different decomposition. Split p=7 into 4+3.
    # For 3x2x4: schoolbook rank = 24. 
    # For 3x2x3: schoolbook rank = 18.
    #
    # Hmm, let me just use the schoolbook for now and see if I can improve specific blocks.
    #
    # Actually, the key insight from the inspiration is that splitting into 3x2x7 blocks
    # and using Strassen for 2x2x2 sub-blocks gives rank 99.
    #
    # Let me implement: split m=5 into 2+2+1.
    # For the 3x2x7 blocks, use a Strassen-based decomposition.
    #
    # For 3x2x7: We can use the following approach.
    # View the 2x2 part (m=2, and take 2 of the 3 rows) as Strassen.
    # But we have 3 rows, so we can do:
    # - Apply Strassen to rows 0,1 of A with all of B (2x2x7)
    # - Add row 2 separately (1x2x7)
    #
    # For 2x2x7: schoolbook rank = 28. With Strassen on 2x2x2:
    # We can split p=7 into 4+3 or just use 7 separate 2x2x1.
    # Actually, for 2x2x7, we can use Strassen for each column pair? No.
    #
    # Let me just use schoolbook for the 3x2x7 blocks and see.
    # 3x2x7 schoolbook = 42, so two of them = 84, plus 3x1x7 = 21, total = 105.
    # That's the same as the full schoolbook.
    #
    # To get below 105, I need to use Strassen somewhere.
    #
    # Let me try: split n=3 into 2+1, m=5 into 2+2+1, p=7 into 4+3.
    # Sub-blocks:
    # (2,2,4): rank 16 schoolbook, but with Strassen on 2x2x2 part... 
    # Actually, for 2x2x4, I can use Strassen for 2x2x2 twice? No, 4 is not 2.
    #
    # Let me try a simpler improvement: for the 3x2x7 block, use Strassen on the 2x2x2 sub-tensor.
    # Split p=7 into 2+2+2+1.
    # For each 3x2x2 sub-block: schoolbook rank = 12. With Strassen: 7 terms for the 2x2x2 part,
    # but we have 3 rows. 
    #
    # For 3x2x2: We can use the following.
    # A is 3x2, B is 2x2, C is 3x2.
    # We can write C = A * B.
    # Using Strassen on the 2x2 part:
    # Take rows 0,1 of A: A_top is 2x2. Apply Strassen to get C_top (2x2) in 7 terms.
    # Row 2 of A: A_bottom is 1x2. C_bottom = A_bottom * B (1x2) in 2 terms.
    # Total: 9 terms for 3x2x2.
    #
    # So for 3x2x7 = 3x2x(2+2+2+1):
    # 3 * 9 (for the three 3x2x2 blocks) + 3*2*1 (for the 3x2x1 block) = 27 + 6 = 33.
    # Two such blocks: 66.
    # Plus 3x1x7: schoolbook = 21.
    # Total: 66 + 21 = 87.
    #
    # That's better than 105! Let me implement this.
    
    def _strassen_3x2x2():
        """Decompose 3x2x2 tensor. A is 3x2, B is 2x2, C is 3x2.
        U rows: i*2+j for i in 0..2, j in 0..1 (6 rows)
        V rows: j*2+k for j in 0..1, k in 0..1 (4 rows)  
        W rows: k*3+i for k in 0..1, i in 0..2 (6 rows)
        """
        U = np.zeros((6, 9), dtype=int)
        V = np.zeros((4, 9), dtype=int)
        W = np.zeros((6, 9), dtype=int)
        
        # Strassen on rows 0,1 of A (2x2x2 part), 7 terms
        # Then row 2 separately, 2 terms
        # Total 9 terms
        
        # Strassen terms for A_top (rows 0,1) x B:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A10)B11
        # M6 = (A01-A10)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        
        # U row indices: i*2+j
        # i=0,j=0 -> 0; i=0,j=1 -> 1; i=1,j=0 -> 2; i=1,j=1 -> 3
        # V row indices: j*2+k
        # j=0,k=0 -> 0; j=0,k=1 -> 1; j=1,k=0 -> 2; j=1,k=1 -> 3
        # W row indices: k*3+i
        # k=0,i=0 -> 0; k=0,i=1 -> 1; k=1,i=0 -> 3; k=1,i=1 -> 4
        
        # M1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[4, 0] = -1
        
        # M2
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[3, 1] = 1
        
        # M3
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[0, 2] = -1; W[1, 2] = 1
        
        # M4
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = -1; W[3, 3] = 1
        
        # M5
        U[0, 4] = 1; U[2, 4] = 1
        V[3, 4] = 1
        W[0, 4] = 1; W[4, 4] = -1
        
        # M6
        U[1, 5] = 1; U[2, 5] = -1
        V[0, 5] = 1; V[2, 5] = 1
        W[4, 5] = 1
        
        # M7
        U[1, 6] = 1; U[3, 6] = -1
        V[1, 6] = 1; V[3, 6] = 1
        W[1, 6] = 1
        
        # Row 2 of A: A[2,0]*B[0,0] + A[2,1]*B[1,0] for C[2,0]
        # A[2,0]*B[0,1] + A[2,1]*B[1,1] for C[2,1]
        # Term 7: C[2,0] = A[2,0]*B[0,0] + A[2,1]*B[1,0]
        U[4, 7] = 1  # i=2,j=0
        V[0, 7] = 1; V[2, 7] = 1  # j=0,k=0 and j=1,k=0
        W[3, 7] = 1  # k=0,i=2 -> row 3
        
        # Term 8: C[2,1] = A[2,0]*B[0,1] + A[2,1]*B[1,1]
        U[4, 8] = 1  # i=2,j=0 -- wait, this should be separate
        # Actually, for C[2,1]: A[2,0]*B[0,1] + A[2,1]*B[1,1]
        # This is two terms in schoolbook, but we can combine if we use a single rank-1 tensor?
        # No, C[2,1] = sum_j A[2,j]*B[j,1]. This is a dot product, needs 2 rank-1 terms in general.
        # But wait, we're building rank-1 tensors. Each term is U*V*W.
        # For C[2,0]: we need A[2,0]*B[0,0] and A[2,1]*B[1,0]. These are two separate contributions.
        # So we need two terms for row 2.
        
        # Let me redo: for row 2, we need:
        # C[2,0] = A[2,0]*B[0,0] + A[2,1]*B[1,0]  -> 2 terms
        # C[2,1] = A[2,0]*B[0,1] + A[2,1]*B[1,1]  -> 2 terms
        # Total 4 terms for row 2, not 2.
        
        # So total is 7 + 4 = 11 terms for 3x2x2.
        
        # Let me rebuild with 11 terms.
        U = np.zeros((6, 11), dtype=int)
        V = np.zeros((4, 11), dtype=int)
        W = np.zeros((6, 11), dtype=int)
        
        # Terms 0-6: Strassen for rows 0,1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[4, 0] = -1
        
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[3, 1] = 1
        
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[0, 2] = -1; W[1, 2] = 1
        
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = -1; W[3, 3] = 1
        
        U[0, 4] = 1; U[2, 4] = 1
        V[3, 4] = 1
        W[0, 4] = 1; W[4, 4] = -1
        
        U[1, 5] = 1; U[2, 5] = -1
        V[0, 5] = 1; V[2, 5] = 1
        W[4, 5] = 1
        
        U[1, 6] = 1; U[3, 6] = -1
        V[1, 6] = 1; V[3, 6] = 1
        W[1, 6] = 1
        
        # Term 7: A[2,0]*B[0,0] -> C[2,0]
        U[4, 7] = 1  # i=2,j=0
        V[0, 7] = 1  # j=0,k=0
        W[3, 7] = 1  # k=0,i=2 -> row 3
        
        # Term 8: A[2,1]*B[1,0] -> C[2,0]
        U[5, 8] = 1  # i=2,j=1
        V[2, 8] = 1  # j=1,k=0
        W[3, 8] = 1  # k=0,i=2 -> row 3
        
        # Term 9: A[2,0]*B[0,1] -> C[2,1]
        U[4, 9] = 1  # i=2,j=0
        V[1, 9] = 1  # j=0,k=1
        W[4, 9] = 1  # k=1,i=2 -> row 4
        
        # Term 10: A[2,1]*B[1,1] -> C[2,1]
        U[5, 10] = 1  # i=2,j=1
        V[3, 10] = 1  # j=1,k=1
        W[4, 10] = 1  # k=1,i=2 -> row 4
        
        return U, V, W
    
    def _schoolbook_3x2x1():
        """3x2x1 tensor. A is 3x2, B is 2x1, C is 3x1.
        U rows: i*2+j (6 rows)
        V rows: j*1+k = j (2 rows)
        W rows: k*3+i = i (3 rows)
        """
        U = np.zeros((6, 6), dtype=int)
        V = np.zeros((2, 6), dtype=int)
        W = np.zeros((3, 6), dtype=int)
        c = 0
        for i in range(3):
            for j in range(2):
                U[i*2+j, c] = 1
                V[j, c] = 1
                W[i, c] = 1
                c += 1
        return U, V, W
    
    def _schoolbook_3x1x7():
        """3x1x7 tensor. A is 3x1, B is 1x7, C is 3x7.
        U rows: i*1+j = i (3 rows)
        V rows: j*7+k = k (7 rows)
        W rows: k*3+i (21 rows)
        """
        U = np.zeros((3, 21), dtype=int)
        V = np.zeros((7, 21), dtype=int)
        W = np.zeros((21, 21), dtype=int)
        c = 0
        for i in range(3):
            for k in range(7):
                U[i, c] = 1
                V[k, c] = 1
                W[k*3+i, c] = 1
                c += 1
        return U, V, W
    
    # Build the full decomposition
    # Split m=5 into 2+2+1
    # For each m_block=2: split p=7 into 2+2+2+1
    # For each m_block=1: use schoolbook 3x1x7
    
    # 3x2x2 block: 11 terms
    # 3x2x1 block: 6 terms
    # So 3x2x7 = 3*11 + 6 = 39 terms
    # Two 3x2x7 blocks: 78 terms
    # 3x1x7 block: 21 terms
    # Total: 78 + 21 = 99 terms
    
    # Let's build this
    U_3x2x2, V_3x2x2, W_3x2x2 = _strassen_3x2x2()
    U_3x2x1, V_3x2x1, W_3x2x1 = _schoolbook_3x2x1()
    U_3x1x7, V_3x1x7, W_3x1x7 = _schoolbook_3x1x7()
    
    # Now assemble the full 3x5x7 decomposition
    # m indices: 0,1 (first block), 2,3 (second block), 4 (third block)
    # p indices: for each m_block=2, p splits into 0,1 | 2,3 | 4,5 | 6
    
    rank = 99
    U = np.zeros((15, rank), dtype=int)  # 3*5=15
    V = np.zeros((35, rank), dtype=int)  # 5*7=35
    W = np.zeros((21, rank), dtype=int)  # 7*3=21
    
    col = 0
    for block_idx in range(2):
        m_offset = block_idx * 2
        for p_sub in range(4):
            if p_sub < 3:
                # 3x2x2 block
                U_b = U_3x2x2
                V_b = V_3x2x2
                W_b = W_3x2x2
                r_b = 11
            else:
                # 3x2x1 block
                U_b = U_3x2x1
                V_b = V_3x2x1
                W_b = W_3x2x1
                r_b = 6
            
            # Map U: row i*5 + (m_offset + j)
            for i in range(3):
                for j in range(2):
                    src_row = i*2 + j
                    dst_row = i*5 + (m_offset + j)
                    U[dst_row, col:col+r_b] = U_b[src_row, :]
            
            # Map V: row (m_offset + j)*7 + (p_sub*2 + k) for p_sub<3, or p_sub*2+0 for p_sub=3
            for j in range(2):
                for k in range(2 if p_sub < 3 else 1):
                    src_row = j*2 + k
                    if p_sub < 3:
                        dst_row = (m_offset + j)*7 + (p_sub*2 + k)
                    else:
                        dst_row = (m_offset + j)*7 + 6
                    V[dst_row, col:col+r_b] = V_b[src_row, :]
            
            # Map W: row (p_sub*2 + k)*3 + i for p_sub<3, or 6*3+i for p_sub=3
            for k in range(2 if p_sub < 3 else 1):
                for i in range(3):
                    src_row = k*3 + i
                    if p_sub < 3:
                        dst_row = (p_sub*2 + k)*3 + i
                    else:
                        dst_row = 6*3 + i
                    W[dst_row, col:col+r_b] = W_b[src_row, :]
            
            col += r_b
    
    # Third block: m=4 (single column), 3x1x7
    U[12:15, col:col+21] = U_3x1x7
    V[28:35, col:col+21] = V_3x1x7
    W[:, col:col+21] = W_3x1x7
    col += 21
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
