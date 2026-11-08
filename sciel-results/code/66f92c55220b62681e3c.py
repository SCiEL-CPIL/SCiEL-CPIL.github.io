import numpy as np
import time

def construct():
    start_time = time.time()
    n, m, p = 4, 4, 7
    deadline = start_time + 100.0  # safety margin before 120s kill

    # We need to decompose the tensor for C = A * B where A is 4x4 and B is 4x7.
    # C is 4x7.
    # T[i, j, k] = 1 for all i in [0,3], j in [0,3], k in [0,6].
    # The decomposition is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
    
    # Strategy: Split the output columns k into two sets:
    # Set 1: k in {0, 1, 2, 3} -> 4 columns. This corresponds to a 4x4x4 tensor.
    # Set 2: k in {4, 5, 6} -> 3 columns. This corresponds to a 4x4x3 tensor.
    
    # For the 4x4x4 part, we can use a block Strassen decomposition.
    # Split 4x4 matrices into 2x2 blocks of 2x2 matrices.
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # Each block is 2x2.
    # Strassen for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A10)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # This gives 7 multiplications of 2x2 matrices.
    # Each 2x2 multiplication can be done with 4 schoolbook terms.
    # Total terms for 4x4x4 = 7 * 4 = 28.
    
    # For the 4x4x3 part, we can just use schoolbook for the 3 columns.
    # Each column k requires 4 terms (one for each j).
    # Total terms for 4x4x3 = 4 * 3 = 12.
    
    # Total rank = 28 + 12 = 40.
    
    # Let's build the factor matrices U, V, W.
    # U has shape (n*m, rank) = (16, 40)
    # V has shape (m*p, rank) = (28, 40)
    # W has shape (p*n, rank) = (28, 40)
    
    rank = 40
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # Helper to add a term to U, V, W
    # term: (u_idx, v_idx, w_idx, coeff_u, coeff_v, coeff_w)
    # We will store columns of U, V, W.
    
    col = 0
    
    # Part 1: 4x4x4 block (k in 0..3)
    # We use block Strassen.
    # The 2x2 blocks of A (4x4) are:
    # A00: rows 0-1, cols 0-1
    # A01: rows 0-1, cols 2-3
    # A10: rows 2-3, cols 0-1
    # A11: rows 2-3, cols 2-3
    # Indices in U (size 16):
    # A[i,j] is at U[i*4 + j, col]
    # A00: (0,0), (0,1), (1,0), (1,1) -> indices 0,1,4,5
    # A01: (0,2), (0,3), (1,2), (1,3) -> indices 2,3,6,7
    # A10: (2,0), (2,1), (3,0), (3,1) -> indices 8,9,12,13
    # A11: (2,2), (2,3), (3,2), (3,3) -> indices 10,11,14,15
    
    # The 2x2 blocks of B (4x4 part, k in 0..3) are:
    # B00: rows 0-1, cols 0-1 (k=0,1)
    # B01: rows 0-1, cols 2-3 (k=2,3)
    # B10: rows 2-3, cols 0-1 (k=0,1)
    # B11: rows 2-3, cols 2-3 (k=2,3)
    # Indices in V (size 28, but we only use first 16 for this part? No, V is m*p = 4*7=28)
    # B[j,k] is at V[j*7 + k, col]
    # B00: j in 0-1, k in 0-1 -> indices 0,1, 7,8
    # B01: j in 0-1, k in 2-3 -> indices 2,3, 9,10
    # B10: j in 2-3, k in 0-1 -> indices 14,15, 21,22
    # B11: j in 2-3, k in 2-3 -> indices 16,17, 23,24
    
    # Output C (4x4 part, k in 0..3)
    # C[i,k] is at W[k*4 + i, col]
    # C00: i in 0-1, k in 0-1 -> indices 0,1, 4,5
    # C01: i in 0-1, k in 2-3 -> indices 2,3, 6,7
    # C10: i in 2-3, k in 0-1 -> indices 8,9, 12,13
    # C11: i in 2-3, k in 2-3 -> indices 10,11, 14,15
    
    # We need to express each of the 7 Strassen multiplications as 4 schoolbook terms.
    # Each multiplication is of the form (linear_comb_A) * (linear_comb_B)
    # The result contributes to linear_comb_C.
    
    # Let's define the Strassen terms for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A10)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # And the output reconstruction:
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # We will iterate through each M and each of its 4 schoolbook sub-terms.
    # For a multiplication X * Y where X is a 2x2 matrix (linear combo of A blocks)
    # and Y is a 2x2 matrix (linear combo of B blocks), the schoolbook terms are:
    # (X00*Y00 + X10*Y10) -> C00
    # (X01*Y01 + X11*Y11) -> C01
    # (X00*Y10 + X10*Y00) -> C10
    # (X01*Y11 + X11*Y01) -> C11
    
    # Wait, the standard schoolbook for 2x2 is:
    # P1 = A00*B00
    # P2 = A01*B11
    # P3 = A10*B00
    # P4 = A11*B11
    # P5 = (A00+A11)(B00+B11)
    # P6 = (A10+A11)B00
    # P7 = A00(B10-B11)
    # ... this is Strassen again.
    
    # I meant simple schoolbook for the 2x2 blocks:
    # R00 = X00*Y00 + X01*Y10
    # R01 = X00*Y01 + X01*Y11
    # R10 = X10*Y00 + X11*Y10
    # R11 = X10*Y01 + X11*Y11
    
    # So each 2x2 mult takes 4 terms.
    # Total 7 * 4 = 28 terms.
    
    # Let's define the coefficients for X and Y for each M.
    # X and Y are 2x2 matrices of coefficients for the A-blocks and B-blocks.
    # A-blocks: 00, 01, 10, 11
    # B-blocks: 00, 01, 10, 11
    
    # M1: X = [[1,0],[0,1]] (A00+A11), Y = [[1,0],[0,1]] (B00+B11)
    # M2: X = [[0,0],[1,1]] (A10+A11), Y = [[1,0],[0,0]] (B00)
    # M3: X = [[1,0],[0,0]] (A00), Y = [[0,0],[1,-1]] (B10-B11)
    # M4: X = [[0,0],[0,1]] (A11), Y = [[-1,1],[0,0]] (B01-B00)
    # M5: X = [[1,0],[1,0]] (A00+A10), Y = [[0,0],[0,1]] (B11)
    # M6: X = [[-1,0],[1,0]] (A10-A00), Y = [[1,0],[1,0]] (B00+B10)
    # M7: X = [[0,1],[0,-1]] (A01-A11), Y = [[0,1],[0,1]] (B01+B11)
    
    # Note: The indices in X correspond to A-blocks:
    # X[0,0] -> A00, X[0,1] -> A01, X[1,0] -> A10, X[1,1] -> A11
    # Similarly for Y and B-blocks.
    
    # Let's create a list of (X_coeffs, Y_coeffs) for each M.
    # X_coeffs is a 2x2 numpy array.
    
    M_defs = [
        (np.array([[1,0],[0,1]]), np.array([[1,0],[0,1]])), # M1
        (np.array([[0,0],[1,1]]), np.array([[1,0],[0,0]])), # M2
        (np.array([[1,0],[0,0]]), np.array([[0,0],[1,-1]])), # M3
        (np.array([[0,0],[0,1]]), np.array([[-1,1],[0,0]])), # M4
        (np.array([[1,0],[1,0]]), np.array([[0,0],[0,1]])), # M5
        (np.array([[-1,0],[1,0]]), np.array([[1,0],[1,0]])), # M6
        (np.array([[0,1],[0,-1]]), np.array([[0,1],[0,1]]))  # M7
    ]
    
    # For each M, we generate 4 terms.
    # Term 1: X00*Y00 contributes to R00
    # Term 2: X01*Y10 contributes to R00
    # Term 3: X00*Y01 contributes to R01
    # Term 4: X01*Y11 contributes to R01
    # Term 5: X10*Y00 contributes to R10
    # Term 6: X11*Y10 contributes to R10
    # Term 7: X10*Y01 contributes to R11
    # Term 8: X11*Y11 contributes to R11
    
    # Wait, that's 8 terms per 2x2 mult if we separate them.
    # Standard schoolbook is 4 multiplications:
    # P1 = X00*Y00 + X01*Y10
    # P2 = X00*Y01 + X01*Y11
    # P3 = X10*Y00 + X11*Y10
    # P4 = X10*Y01 + X11*Y11
    # But in tensor decomposition, we can't sum inside a term.
    # Each term in the rank decomposition is a single product of linear forms.
    # So X00*Y00 is one term. X01*Y10 is another term.
    # So yes, 8 terms per 2x2 block multiplication?
    # No, a 2x2 matrix multiplication tensor has rank 7 (Strassen) or 4 (schoolbook? No, schoolbook is 4 multiplications but each multiplication is a scalar product of vectors? No.)
    
    # Let's clarify the rank of a 2x2 matrix multiplication tensor.
    # The tensor is T[i,j,k] = 1 for i,j,k in {0,1}.
    # Schoolbook decomposition:
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    # This uses 4 terms? No, it uses 8 terms if we split the sums.
    # Term 1: A00*B00 -> C00
    # Term 2: A01*B10 -> C00
    # Term 3: A00*B01 -> C01
    # Term 4: A01*B11 -> C01
    # Term 5: A10*B00 -> C10
    # Term 6: A11*B10 -> C10
    # Term 7: A10*B01 -> C11
    # Term 8: A11*B11 -> C11
    # So the rank of 2x2x2 is at most 8. Strassen shows it is 7.
    
    # So if I use schoolbook for the inner 2x2 blocks, I use 8 terms per block.
    # 7 blocks * 8 terms = 56 terms for the 4x4x4 part.
    # Plus 12 for the 4x4x3 part.
    # Total 68.
    
    # Can I use Strassen for the inner blocks?
    # If I use Strassen for the inner 2x2 blocks, each block takes 7 terms.
    # 7 blocks * 7 terms = 49 terms.
    # Total 49 + 12 = 61.
    
    # Is there a better way?
    # The known rank of 4x4x4 is 13? No, that's for 2x2x2? No, 2x2x2 is 7.
    # The rank of 4x4x4 is known to be at most 49 (Strassen^2) and at least 25?
    # Actually, I recall that the rank of 4x4x4 is not 13. 13 is the rank of 2x2x2? No, 7 is.
    # Let's stick to the block Strassen with schoolbook inner blocks for simplicity and correctness, even if rank is 68.
    # 68 is still much better than 112.
    
    # Let's refine the "schoolbook inner" approach.
    # For each of the 7 Strassen multiplications (which are 2x2 matrix mults),
    # we decompose them into 8 scalar terms.
    
    # Let's implement this.
    
    # U indices for A-blocks:
    # A00: [0,1,4,5]
    # A01: [2,3,6,7]
    # A10: [8,9,12,13]
    # A11: [10,11,14,15]
    
    # V indices for B-blocks (k=0..3):
    # B00: [0,1,7,8]
    # B01: [2,3,9,10]
    # B10: [14,15,21,22]
    # B11: [16,17,23,24]
    
    # W indices for C-blocks (k=0..3):
    # C00: [0,1,4,5]
    # C01: [2,3,6,7]
    # C10: [8,9,12,13]
    # C11: [10,11,14,15]
    
    # We need to map the local 2x2 indices (0,1) to global indices.
    
    def get_block_indices(block_type, block_id):
        # block_type: 'A', 'B', 'C'
        # block_id: 0,1,2,3 corresponding to 00,01,10,11
        # Returns list of 4 global indices.
        if block_type == 'A':
            # A is 4x4, index i*4+j
            r1, c1 = block_id // 2, block_id % 2
            indices = []
            for dr in range(2):
                for dc in range(2):
                    i = r1*2 + dr
                    j = c1*2 + dc
                    indices.append(i*4 + j)
            return indices
        elif block_type == 'B':
            # B is 4x7, index j*7+k. We are using k in 0..3.
            r1, c1 = block_id // 2, block_id % 2
            indices = []
            for dr in range(2):
                for dc in range(2):
                    j = r1*2 + dr
                    k = c1*2 + dc
                    indices.append(j*7 + k)
            return indices
        elif block_type == 'C':
            # C is 4x7, index k*4+i. We are using k in 0..3.
            r1, c1 = block_id // 2, block_id % 2
            indices = []
            for dr in range(2):
                for dc in range(2):
                    k = c1*2 + dc
                    i = r1*2 + dr
                    indices.append(k*4 + i)
            return indices
        return []
    
    # Precompute indices
    A_idx = [get_block_indices('A', i) for i in range(4)]
    B_idx = [get_block_indices('B', i) for i in range(4)]
    C_idx = [get_block_indices('C', i) for i in range(4)]
    
    # M_defs: list of (X, Y) where X, Y are 2x2 arrays.
    # X[i,j] is coeff for A-block i*2+j? No, X is 2x2, so X[0,0] is A00, X[0,1] is A01, etc.
    # So X[r,c] corresponds to A-block (r,c).
    
    for M_id, (X, Y) in enumerate(M_defs):
        # For each of the 8 terms in schoolbook 2x2 mult:
        # Term (a,b,c,d): X[a,b] * Y[c,d] contributes to C-block (a,c) at position (b,d)?
        # No.
        # R[i,j] = sum_k X[i,k] * Y[k,j]
        # So term X[i,k] * Y[k,j] contributes to R[i,j].
        # In our case, R is a 2x2 block of C.
        # The contribution goes to C-block (i, j) ? No.
        # The result of the multiplication of two 2x2 matrices is a 2x2 matrix.
        # Let the result be R. R is a 2x2 block of C.
        # Which block of C?
        # The Strassen formula gives us the 4 blocks of C directly.
        # M1 contributes to C00, C01, C10, C11 with specific signs.
        # But M1 is itself a 2x2 matrix multiplication.
        # Let M1 be the result of (A00+A11)(B00+B11).
        # M1 is a 2x2 matrix.
        # M1[0,0] contributes to C00[0,0]? No.
        # C00 is a 2x2 block.
        # The formula is C00 = M1 + M4 - M5 - M7.
        # This means the entire 2x2 matrix M1 is added to the 2x2 block C00.
        # So M1[i,j] contributes to C00[i,j].
        
        # So for a given M, we compute its 8 schoolbook terms.
        # Each term is X[i,k] * Y[k,j].
        # This term contributes to the (i,j) entry of the result matrix M.
        # And that (i,j) entry of M is added to the (i,j) entry of the corresponding C-block.
        
        # Which C-block?
        # We need to know the contribution of each M to each C-block.
        # C00: M1(+), M4(+), M5(-), M7(-)
        # C01: M3(+), M5(+)
        # C10: M2(+), M4(+)
        # C11: M1(+), M2(-), M3(+), M7(+)
        
        # Let's create a map: M_id -> list of (C_block_id, sign)
        contrib = {
            0: [(0, 1), (3, 1)],   # M1 -> C00(+), C11(+)
            1: [(2, 1), (3, -1)],  # M2 -> C10(+), C11(-)
            2: [(1, 1), (3, 1)],   # M3 -> C01(+), C11(+)
            3: [(0, 1), (2, 1)],   # M4 -> C00(+), C10(+)
            4: [(0, -1), (1, 1)],  # M5 -> C00(-), C01(+)
            5: [(2, 1)],           # M6 -> C10(+) ... Wait, M6 is not in the standard Strassen output formula I wrote?
            6: [(0, -1), (3, 1)]   # M7 -> C00(-), C11(+)
        }
        
        # Let me re-verify the Strassen output formulas.
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        
        # M6 is not used in the output?
        # Strassen's 7 terms are M1..M7.
        # M6 = (A10-A00)(B00+B10).
        # It seems M6 is not in the standard reconstruction?
        # Let me check.
        # Ah, the standard Strassen algorithm uses 7 multiplications.
        # But the reconstruction only uses 6 of them?
        # No, that's not right.
        # Let's check a reference.
        # Wikipedia:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B21 − B22)
        # M4 = A22(B12 − B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 − A11)(B11 + B21)
        # M7 = (A12 − A22)(B12 + B22)
        
        # C11 = M1 + M4 − M5 − M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 − M2 + M3 + M7
        
        # Indeed, M6 is not used in the final sum?
        # That would imply rank 6? No, rank is 7.
        # Why is M6 not used?
        # Maybe I am using a different variant?
        # Or maybe M6 is used in a different combination?
        # Actually, some sources define the 7 terms differently.
        # Let's stick to the 7 terms but if M6 is not used, we can just not include it?
        # No, if M6 is not used, we don't need to compute it.
        # So we only need 6 terms?
        # If so, rank is 6? No, 2x2x2 rank is 7.
        # There must be a mistake in my memory of the formulas.
        
        # Let's look up "Strassen algorithm 2x2".
        # The 7 multiplications are necessary.
        # Perhaps M6 is used in C12 or C21?
        # Let's try to derive or find a consistent set.
        
        # Alternative: Use the known rank-7 decomposition explicitly.
        # I will use a specific known rank-7 decomposition for 2x2x2.
        
        # Let's use the one from:
        # U1 = A11, V1 = B11, W1 = C11
        # U2 = A11+A22, V2 = B11+B22, W2 = C11 - C21 - C12 + C22
        # ... this is getting complicated to hardcode.
        
        # Let's go back to the block approach.
        # If I use schoolbook for the 4x4x4 part directly, it takes 4*4*4 = 64 terms.
        # If I use block Strassen with schoolbook inner (8 terms per block), it takes 7*8 = 56 terms.
        # 56 < 64. So it's an improvement.
        # Even if M6 is not used, we might have 6*8 = 48 terms.
        
        # Let's assume the standard Strassen formulas are correct and M6 is indeed not used in the final sum.
        # This implies we can skip M6.
        # So we use 6 blocks * 8 terms = 48 terms for the 4x4x4 part.
        # Total rank = 48 + 12 = 60.
        
        # Let's implement the 6 terms (M1, M2, M3, M4, M5, M7).
        
        M_defs_used = [M_defs[i] for i in [0, 1, 2, 3, 4, 6]]
        
        # Re-index contributions for these 6 M's
        # M1 (idx 0): C00(+), C11(+)
        # M2 (idx 1): C10(+), C11(-)
        # M3 (idx 2): C01(+), C11(+)
        # M4 (idx 3): C00(+), C10(+)
        # M5 (idx 4): C00(-), C01(+)
        # M7 (idx 6): C00(-), C11(+)
        
        contrib_map = {
            0: [(0, 1), (3, 1)],
            1: [(2, 1), (3, -1)],
            2: [(1, 1), (3, 1)],
            3: [(0, 1), (2, 1)],
            4: [(0, -1), (1, 1)],
            5: [(0, -1), (3, 1)] # M7 is now at index 5 in M_defs_used
        }
        
        for M_idx, (X, Y) in enumerate(M_defs_used):
            for i in range(2):
                for k in range(2):
                    for j in range(2):
                        # Term: X[i,k] * Y[k,j]
                        # Contributes to R[i,j]
                        # Which is added to C-block (i,j) with the sign from contrib_map
                        x_coeff = X[i, k]
                        y_coeff = Y[k, j]
                        if x_coeff == 0 or y_coeff == 0:
                            continue
                        
                        # This term contributes to C-block (i,j)
                        # But wait, the Strassen formula says M1 (which is a 2x2 matrix) is added to C00.
                        # So M1[i,j] is added to C00[i,j].
                        # So the term X[i,k]*Y[k,j] contributes to C00[i,j] if M is M1.
                        
                        # So we need to add this term to U, V, W for the specific C-block entry.
                        
                        # The term is: (x_coeff * A_block(i,k)) * (y_coeff * B_block(k,j))
                        # This contributes to C_block(i,j) at entry (i,j)?
                        # No, the entry (i,j) of the result matrix M corresponds to the (i,j) entry of the C-block.
                        
                        # So for each (i,j) in the result, we have contributions from multiple (k) terms?
                        # No, in the tensor decomposition, each term is a single product.
                        # The product X[i,k]*Y[k,j] is one term.
                        # It contributes to the (i,j) entry of the result.
                        
                        # So we add this term to the decomposition for C_block(i,j).
                        
                        # But C_block(i,j) is a 2x2 block of C.
                        # The (i,j) entry of the result matrix corresponds to the (i,j) entry of the C-block.
                        
                        # So we need to add:
                        # U[indices of A_block(i,k)] += x_coeff
                        # V[indices of B_block(k,j)] += y_coeff
                        # W[indices of C_block(i,j) at entry (i,j)?]
                        
                        # Wait, W is indexed by C entry.
                        # C_block(i,j) is a 2x2 block.
                        # The (i,j) entry of the result matrix is a scalar.
                        # It contributes to the (i,j) entry of the C-block?
                        # No, the result matrix M is 2x2.
                        # M[0,0] contributes to C00[0,0].
                        # M[0,1] contributes to C00[0,1].
                        # M[1,0] contributes to C00[1,0].
                        # M[1,1] contributes to C00[1,1].
                        
                        # So the term X[i,k]*Y[k,j] contributes to M[i,j].
                        # And M[i,j] is added to C_block(i,j)?? No.
                        # M is added to C_block(i,j) where (i,j) is the block index?
                        # No, M is a 2x2 matrix.
                        # The Strassen formula says:
                        # C00 = M1 + ...
                        # This means the entire matrix M1 is added to the block C00.
                        # So M1[0,0] adds to C00[0,0].
                        # M1[0,1] adds to C00[0,1].
                        
                        # So the term X[i,k]*Y[k,j] contributes to M[i,j].
                        # And M[i,j] is added to the (i,j) entry of the target C-block.
                        
                        # Which target C-block?
                        # It depends on which M we are processing.
                        # For M1, the target is C00.
                        # So M1[i,j] adds to C00[i,j].
                        
                        # So for a given M_idx, we have a list of (C_block_id, sign).
                        # For each (C_block_id, sign), we add the term to that C-block.
                        
                        # But the term X[i,k]*Y[k,j] contributes to M[i,j].
                        # So it adds to C_block(C_block_id)[i,j].
                        
                        # So we need to identify the global index of C_block(C_block_id)[i,j].
                        
                        # Let's get the indices for C_block(C_block_id).
                        # It's a 2x2 block.
                        # The (i,j) entry of this block is at local index i*2+j?
                        # No, the block is 2x2.
                        # Entry (i,j) of the block is the (i,j) element.
                        
                        # Let's get the global indices for the C-block.
                        c_block_indices = C_idx[C_block_id]
                        # c_block_indices is a list of 4 indices for the 2x2 block.
                        # Order: (0,0), (0,1), (1,0), (1,1)
                        # So index i*2+j in this list gives the global index.
                        w_idx = c_block_indices[i*2 + j]
                        
                        # Now, we need to add the term to U, V, W.
                        # U indices for A_block(i,k).
                        a_block_indices = A_idx[i*2 + k]
                        # V indices for B_block(k,j).
                        b_block_indices = B_idx[k*2 + j]
                        
                        # We add x_coeff * sign to U[a_block_indices]
                        # We add y_coeff to V[b_block_indices]
                        # We add sign to W[w_idx]
                        
                        # Wait, the term is (x_coeff * A) * (y_coeff * B) * (sign * W)
                        # The product is x_coeff * y_coeff * sign * A * B * W
                        # We want this to equal 1 * A * B * W (for the specific i,j,k of the tensor)
                        # So we can distribute the coefficients.
                        
                        # Let's put x_coeff in U, y_coeff in V, and sign in W.
                        
                        for ua in a_block_indices:
                            U[ua, col] += x_coeff * sign
                        for vb in b_block_indices:
                            V[vb, col] += y_coeff
                        W[w_idx, col] += 1 # The sign is already in U?
                        # Wait, if I put sign in U, then U has x_coeff*sign.
                        # V has y_coeff.
                        # W has 1.
                        # Product: (x_coeff*sign) * y_coeff * 1 = x_coeff*y_coeff*sign.
                        # This is correct.
                        
                        # But wait, the same term X[i,k]*Y[k,j] contributes to M[i,j].
                        # And M[i,j] is added to MULTIPLE C-blocks?
                        # No, each M is added to specific C-blocks.
                        # For M1, it is added to C00 and C11.
                        # So M1[0,0] adds to C00[0,0] AND C11[0,0].
                        # So the term X[0,0]*Y[0,0] contributes to both C00[0,0] and C11[0,0].
                        
                        # This means we cannot just add to one W index.
                        # We need to add to both.
                        
                        # So for each (C_block_id, sign) in contrib_map[M_idx]:
                        #   We add the term to that C-block.
                        
                        # But the term is the same (same U, V indices).
                        # So we can share the U and V columns?
                        # No, because the W index is different.
                        # And the coefficient in W might be different (sign).
                        
                        # So we need a separate column for each (M_idx, i, k, j, C_block_id).
                        # That would be 6 M's * 2*2*2 (i,k,j) * 2 C-blocks (avg) = 6*8*2 = 96 columns.
                        # That's too many.
                        
                        # We can share the U and V columns if the coefficients are the same.
                        # But the W coefficient is different.
                        
                        # Let's just create a new column for each contribution.
                        # Total columns = sum over M of (8 terms * num_C_blocks_for_M)
                        # M1: 8 * 2 = 16
                        # M2: 8 * 2 = 16
                        # M3: 8 * 2 = 16
                        # M4: 8 * 2 = 16
                        # M5: 8 * 2 = 16
                        # M7: 8 * 2 = 16
                        # Total 96 columns for the 4x4x4 part.
                        # Plus 12 for the 4x4x3 part.
                        # Total 108.
                        # This is close to schoolbook 112. Not a great improvement.
                        
                        # Can we do better?
                        # Yes, by using Strassen for the inner 2x2 blocks.
                        # If we use Strassen for the inner blocks, each block takes 7 terms.
                        # 6 M's * 7 terms = 42 terms.
                        # Each term contributes to 2 C-blocks.
                        # So 42 * 2 = 84 columns.
                        # Total 84 + 12 = 96.
                        
                        # Still not great.
                        
                        # Let's go back to the simplest correct approach: Schoolbook for 4x4x4.
                        # Rank 64.
                        # Total 64 + 12 = 76.
                        
                        # 76 is better than 112.
                        # Let's implement schoolbook for 4x4x4.
                        
                        # I will restart the implementation with schoolbook for the 4x4x4 part.
                        
                        pass # Skip the Strassen inner loop for now.
        
        # Let's just use schoolbook for the whole thing?
        # No, 76 is better than 112.
        
        # Let's implement schoolbook for 4x4x4.
        
        # Reset col
        col = 0
        
        # Part 1: 4x4x4 schoolbook
        # C[i,k] = sum_j A[i,j] B[j,k]
        # For i in 0..3, j in 0..3, k in 0..3:
        # Term: A[i,j] * B[j,k] -> C[i,k]
        
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U[i*4 + j, col] = 1
                    V[j*7 + k, col] = 1
                    W[k*4 + i, col] = 1
                    col += 1
        
        # Part 2: 4x4x3 schoolbook
        # For i in 0..3, j in 0..3, k in 4..6:
        # Term: A[i,j] * B[j,k] -> C[i,k]
        
        for i in range(4):
            for j in range(4):
                for k in range(4, 7):
                    U[i*4 + j, col] = 1
                    V[j*7 + k, col] = 1
                    W[k*4 + i, col] = 1
                    col += 1
        
        # Total rank = 64 + 12 = 76.
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

