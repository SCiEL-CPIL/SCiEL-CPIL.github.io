import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130.0  # safety margin before 150s kill
    
    # Strategy: Construct a rank-38 decomposition using a hybrid of Strassen and Schoolbook.
    # 
    # We split the inner dimension m=3 into m1=2 and m2=1.
    # A = [A1 | A2] where A1 is 2x2, A2 is 2x1.
    # B = [B1; B2] where B1 is 2x7, B2 is 1x7.
    # C = A1 B1 + A2 B2.
    #
    # Part 1: A2 B2 (2x1 * 1x7)
    # This is an outer product. It can be done with rank 1 if we allow the factors to be sparse?
    # T[i, 2, k] = A2[i, 0] B2[0, k].
    # We can represent this with 1 term:
    # U[i*3+2, r] = A2[i, 0] (for i=0,1)
    # V[2*7+k, r] = B2[0, k] (for k=0..6)
    # W[k*2+i, r] = 1 (for all k, i)
    # Wait, the term is U * V * W.
    # If we set:
    # r0:
    # U[2, r0] = 1, U[5, r0] = 1 (corresponding to A2[0,0], A2[1,0])
    # V[14+k, r0] = 1 for all k (corresponding to B2[0,k])
    # W[k*2+i, r0] = 1 for all k, i
    # Then sum_r U[i*3+2, r] V[14+k, r] W[k*2+i, r] = 1 * 1 * 1 = 1.
    # This matches the tensor entry T[i, 2, k] = 1 (since the tensor for matrix mult has 1s where indices match).
    # So A2 B2 takes 1 term.
    
    # Part 2: A1 B1 (2x2 * 2x7)
    # We split the output dimension p=7 into three 2x2 blocks and one 2x1 block?
    # Or just use Strassen for the whole 2x2 * 2x7?
    # Strassen works for 2x2 * 2x2.
    # We can process the 7 columns in chunks.
    # Let's use 3 chunks of 2 columns (6 cols) and 1 chunk of 1 column.
    # For the 3 chunks of 2x2 * 2x2, we use Strassen (7 terms each). Total 21 terms.
    # For the last chunk of 2x2 * 2x1, we use Schoolbook (4 terms).
    # Total for A1 B1: 21 + 4 = 25 terms.
    
    # Total Rank = 1 + 25 = 26?
    # Wait, is this valid?
    # The tensor decomposition is a sum of rank-1 tensors.
    # The terms from different parts of the block decomposition act on disjoint sets of indices in U, V, W?
    # No, they share the same U, V, W matrices but different columns (terms).
    # Since the terms are additive, we can just concatenate the terms.
    # The indices in U, V, W for the A2 B2 term are distinct from the indices used in A1 B1 terms?
    # A2 B2 uses U indices 2, 5. V indices 14..20. W indices 0..13.
    # A1 B1 uses U indices 0,1,3,4. V indices 0..13. W indices 0..13.
    # They share W indices! But that's fine, because each term is a separate rank-1 tensor.
    # The sum is over r. Each r corresponds to one term.
    # So yes, we can just sum the terms.
    
    # Let's verify the Strassen decomposition for 2x2 * 2x2.
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    #
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # We need to map this to the global indices.
    # For A1 B1, the sub-matrix A1 is at rows 0,1 and cols 0,1 of A.
    # So A1[0,0] is A[0,0] -> U index 0*3+0 = 0.
    # A1[0,1] is A[0,1] -> U index 0*3+1 = 1.
    # A1[1,0] is A[1,0] -> U index 1*3+0 = 3.
    # A1[1,1] is A[1,1] -> U index 1*3+1 = 4.
    
    # B1 is at rows 0,1 and cols 0..6 of B.
    # B1[0,0] is B[0,0] -> V index 0*7+0 = 0.
    # ...
    # B1[1,6] is B[1,6] -> V index 1*7+6 = 13.
    
    # W indices for C (2x7).
    # C[0,0] -> W index 0*2+0 = 0.
    # C[0,1] -> W index 1*2+0 = 2.
    # ...
    # C[1,6] -> W index 6*2+1 = 13.
    
    # Let's construct the terms for one Strassen block (2x2 * 2x2) for columns k0, k0+1.
    # Let's write a helper to add a Strassen term.
    
    terms = []
    
    # Helper to add a term (u_indices, u_vals, v_indices, v_vals, w_indices, w_vals)
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        terms.append({
            'u': {u_idx: u_val},
            'v': {v_idx: v_val},
            'w': {w_idx: w_val}
        })
    
    # --- Part 1: A2 B2 (1 term) ---
    # U indices: 2 (A[0,2]), 5 (A[1,2])
    # V indices: 14..20 (B[2,0]..B[2,6])
    # W indices: 0..13 (C[0,0]..C[1,6])
    # We need U[i*3+2, r] = 1, V[14+k, r] = 1, W[k*2+i, r] = 1.
    # This is a single term where multiple indices in U, V, W are non-zero.
    # My `add_term` above assumes single indices. I need to generalize.
    
    # Let's restart the term representation to allow sparse vectors.
    terms = []
    
    def add_sparse_term(u_dict, v_dict, w_dict):
        terms.append((u_dict, v_dict, w_dict))
    
    # Term 0: A2 B2
    u_dict = {2: 1, 5: 1}
    v_dict = {14+k: 1 for k in range(7)}
    w_dict = {k*2+i: 1 for k in range(7) for i in range(2)}
    add_sparse_term(u_dict, v_dict, w_dict)
    
    # --- Part 2: A1 B1 ---
    # We have 3 blocks of 2x2 * 2x2 and 1 block of 2x2 * 2x1.
    
    # Strassen Block Helper
    def add_strassen_block(row_offset, col_offset):
        # row_offset is the row in A1 (0 or 1) -> global A row is row_offset
        # col_offset is the col in B1 (0..6) -> global B col is col_offset
        
        # Local indices in A1:
        # A00 -> U[row_offset*3 + 0]
        # A01 -> U[row_offset*3 + 1]
        # A10 -> U[(row_offset+1)*3 + 0]
        # A11 -> U[(row_offset+1)*3 + 1]
        
        # Wait, A1 is 2x2. The rows of A1 are 0,1. The cols of A1 are 0,1.
        # So A1[r, c] corresponds to A[r, c] in the full matrix?
        # Yes, because A = [A1 | A2]. A1 is the first 2 columns.
        # So A1[0,0] = A[0,0] -> U[0]
        # A1[0,1] = A[0,1] -> U[1]
        # A1[1,0] = A[1,0] -> U[3]
        # A1[1,1] = A[1,1] -> U[4]
        
        # Local indices in B1:
        # B1 is 2x7. Rows 0,1. Cols 0..6.
        # B1[r, c] corresponds to B[r, c] in the full matrix?
        # Yes, because B = [B1; B2]. B1 is the first 2 rows.
        # So B1[0, col_offset] = B[0, col_offset] -> V[0*7 + col_offset]
        # B1[1, col_offset] = B[1, col_offset] -> V[1*7 + col_offset]
        
        # Local indices in C (for this block):
        # The output C is 2x7.
        # The block covers C[0, col_offset], C[0, col_offset+1], C[1, col_offset], C[1, col_offset+1].
        # C[i, k] -> W[k*2 + i]
        
        # Let's define the local A, B indices for the 2x2 block.
        # a00 = A1[0,0] -> U[0]
        # a01 = A1[0,1] -> U[1]
        # a10 = A1[1,0] -> U[3]
        # a11 = A1[1,1] -> U[4]
        
        # b00 = B1[0, col_offset] -> V[col_offset]
        # b01 = B1[0, col_offset+1] -> V[col_offset+1]
        # b10 = B1[1, col_offset] -> V[7+col_offset]
        # b11 = B1[1, col_offset+1] -> V[7+col_offset+1]
        
        # c00 = C[0, col_offset] -> W[col_offset*2 + 0]
        # c01 = C[0, col_offset+1] -> W[(col_offset+1)*2 + 0]
        # c10 = C[1, col_offset] -> W[col_offset*2 + 1]
        # c11 = C[1, col_offset+1] -> W[(col_offset+1)*2 + 1]
        
        # M1 = (a00 + a11) * (b00 + b11)
        u1 = {0: 1, 4: 1}
        v1 = {col_offset: 1, 7+col_offset+1: 1}
        w1 = {col_offset*2: 1, (col_offset+1)*2: 1, col_offset*2+1: 1, (col_offset+1)*2+1: 1}
        # Wait, the term is U*V*W.
        # The contribution to C is sum_r U V W.
        # For M1, it contributes to all 4 C entries with coefficient 1.
        # So W should have 1s at all 4 positions.
        add_sparse_term(u1, v1, w1)
        
        # M2 = (a10 + a11) * b00
        u2 = {3: 1, 4: 1}
        v2 = {col_offset: 1}
        w2 = {col_offset*2+1: 1, (col_offset+1)*2+1: 1} # C10, C11
        add_sparse_term(u2, v2, w2)
        
        # M3 = a00 * (b01 - b11)
        u3 = {0: 1}
        v3 = {col_offset+1: 1, 7+col_offset+1: -1}
        w3 = {(col_offset+1)*2: 1, (col_offset+1)*2+1: 1} # C01, C11
        add_sparse_term(u3, v3, w3)
        
        # M4 = a11 * (b10 - b00)
        u4 = {4: 1}
        v4 = {7+col_offset: 1, col_offset: -1}
        w4 = {col_offset*2: 1, col_offset*2+1: 1} # C00, C10
        add_sparse_term(u4, v4, w4)
        
        # M5 = (a00 + a01) * b11
        u5 = {0: 1, 1: 1}
        v5 = {7+col_offset+1: 1}
        w5 = {col_offset*2: 1, (col_offset+1)*2: 1} # C00, C01? No.
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        # M5 contributes to C00 (-1) and C01 (+1).
        # So W should have -1 at C00 and +1 at C01?
        # No, the term is U*V*W. If W has -1, it contributes -1.
        # So for M5:
        # Contribution to C00 is -1. Contribution to C01 is +1.
        # So w5 = {col_offset*2: -1, (col_offset+1)*2: 1}
        w5 = {col_offset*2: -1, (col_offset+1)*2: 1}
        add_sparse_term(u5, v5, w5)
        
        # M6 = (a10 - a00) * (b00 + b01)
        u6 = {3: 1, 0: -1}
        v6 = {col_offset: 1, col_offset+1: 1}
        # M6 contributes to C10 (+1) and C11 (+1)?
        # Wait, standard Strassen:
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        # Where is M6?
        # Ah, the standard Strassen formulas I listed earlier:
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B01 - B11)
        # M4 = A11 * (B10 - B00)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        #
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        #
        # I missed M6 in the reconstruction?
        # Let's check a reference.
        # C00 = M1 - M5 + M4 - M7
        # C01 = M5 + M3
        # C10 = M2 + M4
        # C11 = M1 - M2 + M7 + M3
        #
        # It seems M6 is not used in the final sum? That can't be right.
        # Ah, the formulas for C depend on the specific Strassen variant.
        # Let's use the standard one:
        # p1 = (a00+a11)(b00+b11)
        # p2 = (a10+a11)b00
        # p3 = a00(b01-b11)
        # p4 = a11(b10-b00)
        # p5 = (a00+a01)b11
        # p6 = (a10-a00)(b00+b01)
        # p7 = (a01-a11)(b10+b11)
        #
        # c00 = p1 - p5 + p4 - p7
        # c01 = p3 + p5
        # c10 = p2 + p4
        # c11 = p1 - p2 + p7 + p3
        #
        # Wait, p6 is missing in the c's?
        # Let's re-verify.
        # c00 = (a00+a11)(b00+b11) - (a00+a01)b11 + a11(b10-b00) - (a01-a11)(b10+b11)
        # = a00b00 + a00b11 + a11b00 + a11b11 - a00b11 - a01b11 + a11b10 - a11b00 - a01b10 - a01b11 + a11b10 + a11b11
        # = a00b00 + a11b11 - a01b11 - a01b10 - a01b11 + a11b10 + a11b11
        # = a00b00 + a11b11 - 2a01b11 - a01b10 + a11b10
        # This is not a00b00 + a01b10.
        #
        # I am mixing up formulas. Let's use a reliable source or derive.
        # Actually, for this task, I can just use the known rank 7 decomposition for 2x2*2x2.
        # I will hardcode the coefficients for the 7 terms based on a known valid decomposition.
        # From Wikipedia "Strassen algorithm":
        # M1 = (A00 + A11) (B00 + B11)
        # M2 = (A10 + A11) B00
        # M3 = A00 (B01 - B11)
        # M4 = A11 (B10 - B00)
        # M5 = (A00 + A01) B11
        # M6 = (A10 - A00) (B00 + B01)
        # M7 = (A01 - A11) (B10 + B11)
        #
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        #
        # Let's check C10 = M2 + M4 = (A10+A11)B00 + A11(B10-B00) = A10B00 + A11B00 + A11B10 - A11B00 = A10B00 + A11B10. Correct.
        # Let's check C11 = M1 - M2 + M3 + M7
        # = (A00+A11)(B00+B11) - (A10+A11)B00 + A00(B01-B11) + (A01-A11)(B10+B11)
        # = A00B00 + A00B11 + A11B00 + A11B11 - A10B00 - A11B00 + A00B01 - A00B11 + A01B10 + A01B11 - A11B10 - A11B11
        # = A00B00 + A00B01 + A01B10 + A01B11 - A10B00 - A11B10
        # This is not A10B01 + A11B11.
        #
        # There is a mistake in my memory of the formulas.
        # Let's look up the correct ones.
        # C00 = M1 - M5 + M4 - M7
        # C01 = M5 + M3
        # C10 = M2 + M4
        # C11 = M1 - M2 + M7 + M3
        #
        # Let's check C11 again with these.
        # It's the same as above.
        #
        # Okay, I will use a different known decomposition or just use Schoolbook for the 2x2 blocks to be safe,
        # and rely on the A2 B2 part to reduce rank.
        # Schoolbook for 2x2 * 2x2 is rank 4.
        # 3 blocks * 4 + 1 block * 4 (for 2x2*2x1) = 12 + 4 = 16.
        # Plus 1 for A2 B2 = 17.
        # This is much better than 42.
        
        # Let's implement Schoolbook for the 2x2 * 2x2 blocks.
        # Terms:
        # 1. A00 * B00 -> C00
        # 2. A01 * B10 -> C00
        # 3. A00 * B01 -> C01
        # 4. A01 * B11 -> C01
        # 5. A10 * B00 -> C10
        # 6. A11 * B10 -> C10
        # 7. A10 * B01 -> C11
        # 8. A11 * B11 -> C11
        
        # This is 8 terms? No, 4 terms per 2x2*2x2.
        # C00 = A00B00 + A01B10
        # C01 = A00B01 + A01B11
        # C10 = A10B00 + A11B10
        # C11 = A10B01 + A11B11
        
        # So 4 terms.
        # Term 1: A00 B00. U={0:1}, V={col_offset:1}, W={col_offset*2:1}
        # Term 2: A01 B10. U={1:1}, V={7+col_offset:1}, W={col_offset*2:1}
        # Term 3: A00 B01. U={0:1}, V={col_offset+1:1}, W={(col_offset+1)*2:1}
        # Term 4: A01 B11. U={1:1}, V={7+col_offset+1:1}, W={(col_offset+1)*2:1}
        # Term 5: A10 B00. U={3:1}, V={col_offset:1}, W={col_offset*2+1:1}
        # Term 6: A11 B10. U={4:1}, V={7+col_offset:1}, W={col_offset*2+1:1}
        # Term 7: A10 B01. U={3:1}, V={col_offset+1:1}, W={(col_offset+1)*2+1:1}
        # Term 8: A11 B11. U={4:1}, V={7+col_offset+1:1}, W={(col_offset+1)*2+1:1}
        
        # Wait, schoolbook for 2x2*2x2 is 4 multiplications, but 8 terms in the tensor sum?
        # No, each multiplication is one term.
        # C00 = A00B00 + A01B10. This is 2 terms.
        # So 8 terms total for 2x2*2x2?
        # Yes, because each entry of C is a sum of 2 products.
        # Total terms = 4 entries * 2 products/entry = 8 terms.
        
        # So for 3 blocks of 2x2*2x2: 3 * 8 = 24 terms.
        # For 1 block of 2x2*2x1:
        # C00 = A00B0 + A01B1
        # C10 = A10B0 + A11B1
        # 4 terms.
        # Total for A1 B1: 24 + 4 = 28 terms.
        # Plus 1 for A2 B2 = 29 terms.
        
        # This is a valid decomposition of rank 29.
        # Let's implement this.
        
        # Term 1: A00 B00
        add_sparse_term({0: 1}, {col_offset: 1}, {col_offset*2: 1})
        # Term 2: A01 B10
        add_sparse_term({1: 1}, {7+col_offset: 1}, {col_offset*2: 1})
        # Term 3: A00 B01
        add_sparse_term({0: 1}, {col_offset+1: 1}, {(col_offset+1)*2: 1})
        # Term 4: A01 B11
        add_sparse_term({1: 1}, {7+col_offset+1: 1}, {(col_offset+1)*2: 1})
        # Term 5: A10 B00
        add_sparse_term({3: 1}, {col_offset: 1}, {col_offset*2+1: 1})
        # Term 6: A11 B10
        add_sparse_term({4: 1}, {7+col_offset: 1}, {col_offset*2+1: 1})
        # Term 7: A10 B01
        add_sparse_term({3: 1}, {col_offset+1: 1}, {(col_offset+1)*2+1: 1})
        # Term 8: A11 B11
        add_sparse_term({4: 1}, {7+col_offset+1: 1}, {(col_offset+1)*2+1: 1})
        
    # Add 3 blocks for cols 0, 2, 4
    add_strassen_block(0, 0)
    add_strassen_block(0, 2)
    add_strassen_block(0, 4)
    
    # Add 1 block for col 6 (2x2 * 2x1)
    # C00 = A00 B0 + A01 B1
    # C10 = A10 B0 + A11 B1
    # B0 is B[0, 6], B1 is B[1, 6]
    # C00 is C[0, 6] -> W[12]
    # C10 is C[1, 6] -> W[13]
    
    add_sparse_term({0: 1}, {6: 1}, {12: 1})
    add_sparse_term({1: 1}, {13: 1}, {12: 1})
    add_sparse_term({3: 1}, {6: 1}, {13: 1})
    add_sparse_term({4: 1}, {13: 1}, {13: 1})
    
    # Total terms: 1 + 8*3 + 4 = 29.
    
    # Now construct U, V, W matrices.
    R = len(terms)
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    for r, (u_dict, v_dict, w_dict) in enumerate(terms):
        for idx, val in u_dict.items():
            U[idx, r] = val
        for idx, val in v_dict.items():
            V[idx, r] = val
        for idx, val in w_dict.items():
            W[idx, r] = val
            
    # Verify the decomposition
    # T[i, j, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # T should be 1 if the term A[i,j] B[j,k] contributes to C[i,k].
    # For matrix multiplication, T[i, j, k] = 1 for all i, j, k.
    # Wait, no. The tensor T has T[i, j, k] = 1 if the product A[i,j] B[j,k] is part of the sum for C[i,k].
    # In standard matrix multiplication, C[i,k] = sum_j A[i,j] B[j,k].
    # So for each i, k, the sum is over j.
    # The tensor T is defined such that C[i,k] = sum_{j} A[i,j] B[j,k].
    # In the decomposition, we want sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 for all i, j, k?
    # No, the tensor T[i, j, k] is 1 if the term exists.
    # For matrix multiplication, the term A[i,j] B[j,k] always exists.
    # So T[i, j, k] = 1 for all i, j, k.
    
    # Let's verify this.
    T_reconstructed = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(R):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                T_reconstructed[i, j, k] = s
                
    T_expected = np.ones((n, m, p))
    
    if not np.allclose(T_reconstructed, T_expected):
        # Fallback to schoolbook
        U = np.zeros((n * m, n * m * p))
        V = np.zeros((m * p, n * m * p))
        W = np.zeros((p * n, n * m * p))
        terms_schoolbook = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
        for c, (i, j, k) in enumerate(terms_schoolbook):
            U[i * m + j, c] = 1.0
            V[j * p + k, c] = 1.0
            W[k * n + i, c] = 1.0
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
        
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
