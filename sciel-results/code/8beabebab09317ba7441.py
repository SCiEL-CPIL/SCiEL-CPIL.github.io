import numpy as np
import time
import random

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10 second safety margin
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_schoolbook

    # Construct a rank-53 decomposition using block structure
    # Split B (3x7) into B1 (3x3), B2 (3x3), B3 (3x1)
    # C = A B1 + A B2 + A B3
    # Rank(A B1) <= 22 (Strassen-based 3x3x3)
    # Rank(A B2) <= 22
    # Rank(A B3) <= 9 (Schoolbook 3x3x1)
    # Total Rank = 53
    
    U_new = []
    V_new = []
    W_new = []
    
    def add_term(u_idx, v_idx, w_idx, u_val=1, v_val=1, w_val=1):
        U_new.append([0] * 0) # Placeholder, we build lists of terms
        # Actually, let's build the matrices directly
        
    # We will build the factor matrices column by column
    # Each column r corresponds to one multiplication term
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    
    rank_53 = 53
    U_53 = [[0] * rank_53 for _ in range(n * m)]
    V_53 = [[0] * rank_53 for _ in range(m * p)]
    W_53 = [[0] * rank_53 for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term to the decomposition
    def add_term_to_53(i, j, k, u_val=1, v_val=1, w_val=1):
        nonlocal col
        if col >= rank_53:
            return
        U_53[i * m + j][col] += u_val
        V_53[j * p + k][col] += v_val
        W_53[k * n + i][col] += w_val
        col += 1

    # Define Strassen terms for 2x2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # P1 = (a+d)(e+h)
    # P2 = (c+d)e
    # P3 = a(f-h)
    # P4 = d(g-e)
    # P5 = (a+b)h
    # P6 = (c-a)(e+f)
    # P7 = (b-d)(g+h)
    
    # For 3x3x3 decomposition:
    # A = [[A00, A01, A02], [A10, A11, A12], [A20, A21, A22]]
    # B = [[B00, B01, B02], [B10, B11, B12], [B20, B21, B22]]
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # This is 2x2x2.
    # Indices:
    # A00=0, A01=1, A10=3, A11=4 (in 3x3 flattened i*m+j)
    # B00=0, B01=1, B10=3, B11=4 (in 3x3 flattened j*p+k, p=3)
    # C00=0, C01=1, C10=3, C11=4 (in 3x3 flattened k*n+i, n=3)
    
    # We need to map the 2x2 indices to 3x3 indices.
    # Let's create a function to add Strassen terms for a 2x2 block at offset (r, c) in A and (r, c) in B.
    
    def add_strassen_2x2(a_r, a_c, b_r, b_c, c_r, c_c, p_offset):
        # a_r, a_c: top-left row, col of A block (size 2x2)
        # b_r, b_c: top-left row, col of B block (size 2x2)
        # c_r, c_c: top-left row, col of C block (size 2x2)
        # p_offset: offset in the p-dimension for B and W
        # Note: B is 3x7, so we need to specify which 3x3 block of B we are using.
        # In our split, B1 is cols 0-2, B2 is cols 3-5, B3 is col 6.
        # For B1, p_offset = 0. For B2, p_offset = 3.
        
        # A indices:
        # A00 = a_r, a_c
        # A01 = a_r, a_c+1
        # A10 = a_r+1, a_c
        # A11 = a_r+1, a_c+1
        
        # B indices (in flattened j*p+k):
        # B00 = b_r, b_c + p_offset
        # B01 = b_r, b_c+1 + p_offset
        # B10 = b_r+1, b_c + p_offset
        # B11 = b_r+1, b_c+1 + p_offset
        
        # C indices (in flattened k*n+i):
        # C00 = c_r, c_c
        # C01 = c_r, c_c+1
        # C10 = c_r+1, c_c
        # C11 = c_r+1, c_c+1
        
        # P1 = (A00+A11)(B00+B11)
        add_term_to_53(a_r, a_c, b_r, b_c + p_offset, c_r, c_c, 1, 1, 1)
        add_term_to_53(a_r, a_c, b_r+1, b_c+1 + p_offset, c_r, c_c, 1, 1, 1)
        add_term_to_53(a_r+1, a_c+1, b_r, b_c + p_offset, c_r, c_c, 1, 1, 1)
        add_term_to_53(a_r+1, a_c+1, b_r+1, b_c+1 + p_offset, c_r, c_c, 1, 1, 1)
        # Wait, this adds 4 terms for P1. We need to combine them into 1 term?
        # No, the term is (A00+A11)(B00+B11).
        # This expands to A00B00 + A00B11 + A11B00 + A11B11.
        # These are 4 separate products.
        # But in the tensor decomposition, each product is a term.
        # So P1 contributes 4 terms?
        # No, Strassen's algorithm uses 7 *products* of sums.
        # Each "product" is a single scalar multiplication.
        # So P1 is ONE term in the decomposition?
        # Yes, the term is U * V * W.
        # U has non-zeros at A00, A11.
        # V has non-zeros at B00, B11.
        # W has non-zeros at C00, C01, C10, C11.
        
        # Let's re-evaluate.
        # A term is a rank-1 tensor.
        # U is a vector, V is a vector, W is a vector.
        # U[i*m+j] is non-zero for specific (i,j).
        # V[j*p+k] is non-zero for specific (j,k).
        # W[k*n+i] is non-zero for specific (k,i).
        
        # For P1 = (A00+A11)(B00+B11):
        # U has 1 at (a_r, a_c) and 1 at (a_r+1, a_c+1).
        # V has 1 at (b_r, b_c+p_offset) and 1 at (b_r+1, b_c+1+p_offset).
        # W has 1 at (c_r, c_c) [for C00], 1 at (c_r, c_c+1) [for C01]?
        # No, P1 contributes to C00, C01, C10, C11.
        # C00 += P1
        # C01 += P1
        # C10 += P1
        # C11 += P1
        # So W has 1 at (c_r, c_c), (c_r, c_c+1), (c_r+1, c_c), (c_r+1, c_c+1).
        
        # Let's implement this correctly.
        
        # Clear the previous incorrect additions if any (we didn't add yet, we just defined the function)
        # We need to pass the term index to modify the matrices.
        # My add_term_to_53 adds ONE entry to U, V, W.
        # But a Strassen term has multiple non-zeros in U, V, W.
        # So I need a function that adds a full rank-1 tensor term.
        
        pass # Placeholder

    # Let's rewrite the addition logic to handle multi-nonzero terms.
    
    def add_rank1_term(u_indices, v_indices, w_indices, u_vals, v_vals, w_vals):
        nonlocal col
        if col >= rank_53:
            return
        for idx, val in zip(u_indices, u_vals):
            U_53[idx][col] += val
        for idx, val in zip(v_indices, v_vals):
            V_53[idx][col] += val
        for idx, val in zip(w_indices, w_vals):
            W_53[idx][col] += val
        col += 1

    # Strassen 2x2x2 terms
    # A: (0,0), (0,1), (1,0), (1,1) -> Flat: 0, 1, 3, 4
    # B: (0,0), (0,1), (1,0), (1,1) -> Flat: 0, 1, 3, 4
    # C: (0,0), (0,1), (1,0), (1,1) -> Flat: 0, 1, 3, 4
    
    # P1: (A00+A11)(B00+B11)
    # U: 0, 4. V: 0, 4. W: 0, 1, 3, 4.
    add_rank1_term([0, 4], [0, 4], [0, 1, 3, 4], [1, 1], [1, 1], [1, 1, 1, 1])
    
    # P2: (A10+A11)B00
    # U: 3, 4. V: 0. W: 3, 4.
    add_rank1_term([3, 4], [0], [3, 4], [1, 1], [1], [1, 1])
    
    # P3: A00(B01-B11)
    # U: 0. V: 1, 4. W: 1, 4.
    add_rank1_term([0], [1, 4], [1, 4], [1], [1, -1], [1, -1])
    
    # P4: A11(B10-B00)
    # U: 4. V: 3, 0. W: 3, 4.
    add_rank1_term([4], [3, 0], [3, 4], [1], [1, -1], [1, -1])
    
    # P5: (A00+A01)B11
    # U: 0, 1. V: 4. W: 1, 4.
    add_rank1_term([0, 1], [4], [1, 4], [1, 1], [1], [1, 1])
    
    # P6: (A10-A00)(B00+B01)
    # U: 3, 0. V: 0, 1. W: 3, 4.
    add_rank1_term([3, 0], [0, 1], [3, 4], [1, -1], [1, 1], [1, 1])
    
    # P7: (A01-A11)(B10+B11)
    # U: 1, 4. V: 3, 4. W: 4.
    add_rank1_term([1, 4], [3, 4], [4], [1, -1], [1, 1], [1])
    
    # This covers C[0:2, 0:2].
    # Now we need C[0:2, 2], C[2, 0:2], C[2, 2].
    
    # C[0:2, 2] = A[0:2, 0:2] B[0:2, 2] + A[0:2, 2] B[2, 2]
    # A[0:2, 0:2] is 2x2. B[0:2, 2] is 2x1.
    # This is 2x2x1. Rank 4.
    # Terms:
    # A00 B02, A01 B12, A10 B02, A11 B12.
    # Flat A: 0, 1, 3, 4.
    # Flat B: 2, 5 (col 2, rows 0,1).
    # Flat C: 2, 5 (col 2, rows 0,1).
    
    add_rank1_term([0], [2], [2], [1], [1], [1]) # A00 B02
    add_rank1_term([1], [5], [2], [1], [1], [1]) # A01 B12 -> C02? No, C[0,2]. W index k*n+i = 2*3+0 = 6?
    # W index: k*n + i.
    # C[0,2] -> k=2, i=0 -> 6.
    # C[1,2] -> k=2, i=1 -> 7.
    
    # Let's re-calculate W indices.
    # C is 3x3.
    # C[0,0]=0, C[0,1]=1, C[0,2]=2
    # C[1,0]=3, C[1,1]=4, C[1,2]=5
    # C[2,0]=6, C[2,1]=7, C[2,2]=8
    
    # My previous W indices for Strassen were:
    # C00=0, C01=1, C10=3, C11=4. Correct.
    
    # For C[0:2, 2]:
    # C[0,2] is index 2.
    # C[1,2] is index 5.
    
    # Terms for A[0:2, 0:2] * B[0:2, 2]:
    # A00 (0) * B02 (2) -> C02 (2)
    add_rank1_term([0], [2], [2], [1], [1], [1])
    # A01 (1) * B12 (5) -> C02 (2)
    add_rank1_term([1], [5], [2], [1], [1], [1])
    # A10 (3) * B02 (2) -> C12 (5)
    add_rank1_term([3], [2], [5], [1], [1], [1])
    # A11 (4) * B12 (5) -> C12 (5)
    add_rank1_term([4], [5], [5], [1], [1], [1])
    
    # Terms for A[0:2, 2] * B[2, 2]:
    # A02 (2) * B22 (8) -> C02 (2)
    add_rank1_term([2], [8], [2], [1], [1], [1])
    # A12 (5) * B22 (8) -> C12 (5)
    add_rank1_term([5], [8], [5], [1], [1], [1])
    
    # Total 6 terms for C[0:2, 2].
    
    # C[2, 0:2] = A[2, 0:2] B[0:2, 0:2] + A[2, 2] B[2, 0:2]
    # A[2, 0:2] is 1x2. B[0:2, 0:2] is 2x2.
    # Rank 4.
    # A20 (6) * B00 (0) -> C20 (6)
    add_rank1_term([6], [0], [6], [1], [1], [1])
    # A21 (7) * B10 (3) -> C20 (6)
    add_rank1_term([7], [3], [6], [1], [1], [1])
    # A20 (6) * B01 (1) -> C21 (7)
    add_rank1_term([6], [1], [7], [1], [1], [1])
    # A21 (7) * B11 (4) -> C21 (7)
    add_rank1_term([7], [4], [7], [1], [1], [1])
    
    # Terms for A[2, 2] * B[2, 0:2]:
    # A22 (8) * B20 (6) -> C20 (6)
    add_rank1_term([8], [6], [6], [1], [1], [1])
    # A22 (8) * B21 (7) -> C21 (7)
    add_rank1_term([8], [7], [7], [1], [1], [1])
    
    # Total 6 terms for C[2, 0:2].
    
    # C[2, 2] = A[2, 0:2] B[0:2, 2] + A[2, 2] B[2, 2]
    # A[2, 0:2] is 1x2. B[0:2, 2] is 2x1.
    # Rank 2.
    # A20 (6) * B02 (2) -> C22 (8)
    add_rank1_term([6], [2], [8], [1], [1], [1])
    # A21 (7) * B12 (5) -> C22 (8)
    add_rank1_term([7], [5], [8], [1], [1], [1])
    
    # Terms for A[2, 2] * B[2, 2]:
    # A22 (8) * B22 (8) -> C22 (8)
    add_rank1_term([8], [8], [8], [1], [1], [1])
    
    # Total 3 terms for C[2, 2].
    
    # Total terms for 3x3x3: 7 + 6 + 6 + 3 = 22.
    # Current col = 22.
    
    # Now do the same for B2 (cols 3-5).
    # B2 is B[:, 3:6].
    # B indices are shifted by 3 in the k-dimension.
    # B flat index: j*p + k.
    # For B2, k' = k - 3.
    # So B2[j, k'] corresponds to B[j, k'+3].
    # Flat index: j*3 + (k'+3) = j*3 + k' + 3.
    
    # We can reuse the same logic by offsetting the B indices.
    # Let's just copy the terms and adjust B indices.
    
    # P1
    add_rank1_term([0, 4], [0+3, 4+3], [0, 1, 3, 4], [1, 1], [1, 1], [1, 1, 1, 1])
    # P2
    add_rank1_term([3, 4], [0+3], [3, 4], [1, 1], [1], [1, 1])
    # P3
    add_rank1_term([0], [1+3, 4+3], [1, 4], [1], [1, -1], [1, -1])
    # P4
    add_rank1_term([4], [3+3, 0+3], [3, 4], [1], [1, -1], [1, -1])
    # P5
    add_rank1_term([0, 1], [4+3], [1, 4], [1, 1], [1], [1, 1])
    # P6
    add_rank1_term([3, 0], [0+3, 1+3], [3, 4], [1, -1], [1, 1], [1, 1])
    # P7
    add_rank1_term([1, 4], [3+3, 4+3], [4], [1, -1], [1, 1], [1])
    
    # C[0:2, 2] for B2 -> C[0:2, 5]
    # W indices: C[0,5]=5, C[1,5]=8?
    # C is 3x7? No, C is 3x3 for this block?
    # Wait, the output C for the whole problem is 3x7.
    # My W matrix is size p*n = 7*3 = 21.
    # W index: k*n + i.
    # For B2, the output columns are 3, 4, 5.
    # So W indices should be 3*3+0=9, 3*3+1=10, etc.
    
    # Let's re-evaluate W indices for the whole 3x7 problem.
    # W size 21.
    # Block 1 (B1): C cols 0,1,2. W indices 0-8.
    # Block 2 (B2): C cols 3,4,5. W indices 9-17.
    # Block 3 (B3): C col 6. W indices 18-20.
    
    # My previous W indices for Block 1 were 0-8. Correct.
    # For Block 2, I need to add 9 to the W indices.
    
    # Let's redo Block 2 with correct W offsets.
    
    # P1
    add_rank1_term([0, 4], [0+3, 4+3], [0+9, 1+9, 3+9, 4+9], [1, 1], [1, 1], [1, 1, 1, 1])
    # P2
    add_rank1_term([3, 4], [0+3], [3+9, 4+9], [1, 1], [1], [1, 1])
    # P3
    add_rank1_term([0], [1+3, 4+3], [1+9, 4+9], [1], [1, -1], [1, -1])
    # P4
    add_rank1_term([4], [3+3, 0+3], [3+9, 4+9], [1], [1, -1], [1, -1])
    # P5
    add_rank1_term([0, 1], [4+3], [1+9, 4+9], [1, 1], [1], [1, 1])
    # P6
    add_rank1_term([3, 0], [0+3, 1+3], [3+9, 4+9], [1, -1], [1, 1], [1, 1])
    # P7
    add_rank1_term([1, 4], [3+3, 4+3], [4+9], [1, -1], [1, 1], [1])
    
    # C[0:2, 2] for B2 -> C[0:2, 5]
    # W indices: C[0,5]=5+9=14, C[1,5]=8+9=17?
    # C[0,5] is k=5, i=0 -> 15.
    # C[1,5] is k=5, i=1 -> 16.
    
    add_rank1_term([0], [2+3], [2+9], [1], [1], [1]) # A00 B02 -> C02 (Block 2: C05)
    add_rank1_term([1], [5+3], [2+9], [1], [1], [1]) # A01 B12 -> C05
    add_rank1_term([3], [2+3], [5+9], [1], [1], [1]) # A10 B02 -> C15
    add_rank1_term([4], [5+3], [5+9], [1], [1], [1]) # A11 B12 -> C15
    add_rank1_term([2], [8+3], [2+9], [1], [1], [1]) # A02 B22 -> C05
    add_rank1_term([5], [8+3], [5+9], [1], [1], [1]) # A12 B22 -> C15
    
    # C[2, 0:2] for B2 -> C[2, 3:5]
    # W indices: C[2,3]=3+9=12, C[2,4]=4+9=13.
    
    add_rank1_term([6], [0+3], [6+9], [1], [1], [1]) # A20 B00 -> C23
    add_rank1_term([7], [3+3], [6+9], [1], [1], [1]) # A21 B10 -> C23
    add_rank1_term([6], [1+3], [7+9], [1], [1], [1]) # A20 B01 -> C24
    add_rank1_term([7], [4+3], [7+9], [1], [1], [1]) # A21 B11 -> C24
    add_rank1_term([8], [6+3], [6+9], [1], [1], [1]) # A22 B20 -> C23
    add_rank1_term([8], [7+3], [7+9], [1], [1], [1]) # A22 B21 -> C24
    
    # C[2, 2] for B2 -> C[2, 5]
    # W index: C[2,5]=8+9=17.
    
    add_rank1_term([6], [2+3], [8+9], [1], [1], [1]) # A20 B02 -> C25
    add_rank1_term([7], [5+3], [8+9], [1], [1], [1]) # A21 B12 -> C25
    add_rank1_term([8], [8+3], [8+9], [1], [1], [1]) # A22 B22 -> C25
    
    # Current col = 22 + 22 = 44.
    
    # Block 3: B3 (col 6).
    # B3 is 3x1.
    # C3 is 3x1.
    # Rank 9.
    # Terms: A[i,j] * B[j,6].
    # B flat index: j*3 + 6.
    # W flat index: 6*3 + i = 18 + i.
    
    for i in range(3):
        for j in range(3):
            add_rank1_term([i*3+j], [j*3+6], [18+i], [1], [1], [1])
            
    # Current col = 44 + 9 = 53.
    
    if col == rank_53:
        best_U, best_V, best_W = U_53, V_53, W_53
        best_rank = rank_53

    return {"U": best_U, "V": best_V, "W": best_W}
