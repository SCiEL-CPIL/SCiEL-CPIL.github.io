"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a block decomposition based on Strassen's algorithm.
    # A is 2x5, B is 5x6.
    # Split A into A1 (2x2), A2 (2x2), A3 (2x1).
    # Split B into B1 (2x6), B2 (2x6), B3 (1x6).
    
    # For A1*B1 (2x2 * 2x6):
    # Split B1 into 3 blocks of 2x2: B1_1, B1_2, B1_3.
    # Each 2x2 * 2x2 multiplication uses Strassen (rank 7).
    # Total rank for A1*B1 is 3 * 7 = 21.
    
    # For A2*B2 (2x2 * 2x6):
    # Same as above. Total rank 21.
    
    # For A3*B3 (2x1 * 1x6):
    # This is just a dot product for each row.
    # C_i,k = A3_i * B3_k.
    # This requires 2 * 6 = 12 multiplications? No.
    # Wait, A3 is 2x1, B3 is 1x6.
    # C is 2x6.
    # C_i,k = sum_j A3_i,j B3_j,k. Since j only has 1 value (index 4),
    # C_i,k = A3_i,4 * B3_4,k.
    # This is a rank-1 tensor for each (i,k) pair? No, it's just element-wise product.
    # The tensor for A3*B3 has rank 12?
    # Let's check: T[i,k] = A[i,4] B[4,k].
    # This is a sum of 12 terms?
    # Actually, the rank of the tensor for multiplying (n x 1) by (1 x p) is n*p.
    # Here n=2, p=6, so rank is 12.
    
    # Total rank = 21 + 21 + 12 = 54.
    # This is better than 60.
    
    # Let's verify if we can do better.
    # Strassen for 2x2x2 is rank 7.
    # Schoolbook for 2x2x2 is rank 4.
    # So Strassen is better.
    
    # Can we combine blocks?
    # A1 and A2 are separate columns. B1 and B2 are separate rows.
    # There is no overlap in the indices j.
    # So the decompositions are independent.
    
    # Is there a better way to handle the 2x5 x 5x6?
    # Maybe use a different block size?
    # If we split A into 2x1, 2x1, 2x1, 2x1, 2x1.
    # And B into 1x6, 1x6, 1x6, 1x6, 1x6.
    # Then we have 5 dot products.
    # Each dot product is (2x1)*(1x6), rank 12.
    # Total rank 5 * 12 = 60.
    # This is the schoolbook rank.
    
    # What if we split A into 2x2, 2x2, 2x1?
    # And B into 2x6, 2x6, 1x6?
    # This is what we did. Rank 54.
    
    # What if we split A into 2x3, 2x2?
    # And B into 3x6, 2x6?
    # A1 (2x3) * B1 (3x6).
    # Split B1 into 3x2, 3x2, 3x2.
    # A1 * B1_1 (2x3 * 3x2).
    # This is not a standard block.
    # We can use the general Strassen-like algorithm?
    # Or just schoolbook?
    # Schoolbook for 2x3 * 3x2: 2*3*2 = 12 mults.
    # 3 such blocks: 36 mults.
    # A2 (2x2) * B2 (2x6).
    # Split B2 into 2x2, 2x2, 2x2.
    # 3 * 7 = 21 mults.
    # Total 36 + 21 = 57.
    # Worse than 54.
    
    # What if we split A into 2x2, 2x3?
    # And B into 2x6, 3x6?
    # A1 (2x2) * B1 (2x6): 21 mults.
    # A2 (2x3) * B2 (3x6):
    # Split B2 into 3x2, 3x2, 3x2.
    # A2 * B2_1 (2x3 * 3x2): 12 mults.
    # 3 such blocks: 36 mults.
    # Total 21 + 36 = 57.
    # Worse.
    
    # So 54 seems to be a good candidate.
    
    # Let's construct the matrices for rank 54.
    
    rank = 54
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    col = 0
    
    # Helper to add a 2x2x2 Strassen block
    def add_strassen_block(A_row_offset, A_col_offset, B_row_offset, B_col_offset):
        nonlocal col
        # A is 2x2, B is 2x2, C is 2x2.
        # Indices:
        # A: i in {0,1}, j in {0,1} -> global i*m + (A_col_offset+j)
        # B: j in {0,1}, k in {0,1} -> global (B_row_offset+j)*p + (B_col_offset+k)
        # C: i in {0,1}, k in {0,1} -> global (B_col_offset+k)*n + i
        
        # M1 = A00 * (B10 + B11)
        # M2 = (A00 + A10) * B11
        # M3 = (A01 + A11) * B00
        # M4 = A11 * (B00 - B01)
        # M5 = (A00 + A11) * (B00 + B11)
        # M6 = (A01 - A11) * (B10 + B11)
        # M7 = (A00 - A10) * (B01 + B11)
        
        # C00 = M1 - M2 + M4 + M5
        # C01 = M3 + M2 - M5 + M7
        # C10 = M1 + M6 - M4 - M5
        # C11 = M3 + M5 + M7 - M6
        
        # We add 7 terms to the decomposition.
        
        # Term 1: M1
        # A00 * B10, A00 * B11
        # U[A00, col] = 1, V[B10, col] = 1, W[C00, col] = 1
        # U[A00, col] = 1, V[B11, col] = 1, W[C00, col] = -1
        # Wait, M1 is a single multiplication.
        # The tensor term is U * V * W.
        # For M1: A00 * (B10 + B11).
        # This is not a single rank-1 tensor.
        # Strassen's algorithm is a bilinear algorithm.
        # It expresses the product as a sum of 7 rank-1 tensors.
        
        # Let's write out the 7 rank-1 tensors for 2x2x2 Strassen.
        # Let A = [[a, b], [c, d]], B = [[e, f], [g, h]].
        # C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
        
        # m1 = a * (g+h)
        # m2 = (a+c) * h
        # m3 = (b+d) * e
        # m4 = d * (e-f)
        # m5 = (a+d) * (e+h)
        # m6 = (b-d) * (g+h)
        # m7 = (a-c) * (f+h)
        
        # C00 = m1 - m2 + m4 + m5
        # C01 = m3 + m2 - m5 + m7
        # C10 = m1 + m6 - m4 - m5
        # C11 = m3 + m5 + m7 - m6
        
        # Each mj is a product of a linear form in A and a linear form in B.
        # So each mj corresponds to a rank-1 tensor.
        
        # We need to map the local indices to global indices.
        
        # Local A indices:
        # a: (0,0), b: (0,1), c: (1,0), d: (1,1)
        # Global A index: i*m + (A_col_offset+j)
        
        # Local B indices:
        # e: (0,0), f: (0,1), g: (1,0), h: (1,1)
        # Global B index: (B_row_offset+j)*p + (B_col_offset+k)
        
        # Local C indices:
        # C00: (0,0), C01: (0,1), C10: (1,0), C11: (1,1)
        # Global C index: (B_col_offset+k)*n + i
        
        # Let's define the 7 terms.
        
        # Term 1: m1 = a * (g+h)
        # U: a -> 1
        # V: g -> 1, h -> 1
        # W: C00 -> 1, C10 -> 1
        
        # U[A00, col] += 1
        # V[B10, col] += 1
        # V[B11, col] += 1
        # W[C00, col] += 1
        # W[C10, col] += 1
        
        # But wait, this is not a rank-1 tensor.
        # A rank-1 tensor is U_i * V_j * W_k.
        # Here, the term is a * g + a * h.
        # This is the sum of two rank-1 tensors.
        # Strassen's algorithm reduces the number of multiplications,
        # but each "multiplication" in the algorithm is a product of two linear forms.
        # The tensor rank is the number of such products.
        # So each mj is one rank-1 tensor.
        
        # Let's re-express.
        # The tensor T is a trilinear form.
        # T(A, B, C) = sum_{i,j,k} A_{i,j} B_{j,k} C_{k,i}
        # We want to write T as a sum of rank-1 tensors.
        # A rank-1 tensor is U_i V_j W_k.
        
        # In Strassen, we compute:
        # m1 = (linear form in A) * (linear form in B)
        # The contribution to C is a linear combination of the C entries.
        # So the term is:
        # (sum_a u_a A_a) * (sum_b v_b B_b) * (sum_c w_c C_c)
        # This is a rank-1 tensor.
        
        # So for m1 = a * (g+h):
        # U: a -> 1
        # V: g -> 1, h -> 1
        # W: C00 -> 1, C10 -> 1
        
        # This is correct.
        
        # Let's implement this.
        
        def add_term(u_indices, v_indices, w_indices):
            nonlocal col
            for idx in u_indices:
                U[idx, col] += 1
            for idx in v_indices:
                V[idx, col] += 1
            for idx in w_indices:
                W[idx, col] += 1
            col += 1
        
        # Define local indices
        a = A_row_offset * m + (A_col_offset + 0)
        b = A_row_offset * m + (A_col_offset + 1)
        c = (A_row_offset + 1) * m + (A_col_offset + 0)
        d = (A_row_offset + 1) * m + (A_col_offset + 1)
        
        e = (B_row_offset + 0) * p + (B_col_offset + 0)
        f = (B_row_offset + 0) * p + (B_col_offset + 1)
        g = (B_row_offset + 1) * p + (B_col_offset + 0)
        h = (B_row_offset + 1) * p + (B_col_offset + 1)
        
        C00 = (B_col_offset + 0) * n + 0
        C01 = (B_col_offset + 1) * n + 0
        C10 = (B_col_offset + 0) * n + 1
        C11 = (B_col_offset + 1) * n + 1
        
        # m1 = a * (g+h)
        add_term([a], [g, h], [C00, C10])
        # m2 = (a+c) * h
        add_term([a, c], [h], [C00, C01])
        # m3 = (b+d) * e
        add_term([b, d], [e], [C01, C11])
        # m4 = d * (e-f)
        add_term([d], [e], [C00, C10]) # W[C00] += 1, W[C10] -= 1? No, add_term adds 1.
        # We need to handle signs.
        
        # Let's modify add_term to take signs.
        
    # This is getting complicated. Let's just write out the 21 terms for A1*B1 manually.
    
    # A1: rows 0,1; cols 0,1.
    # B1: rows 0,1; cols 0,1,2,3,4,5.
    
    # Block B1_1: cols 0,1.
    # Strassen for A1 * B1_1.
    
    # Let's just use the known rank-7 decomposition for 2x2x2.
    
    # I will implement a helper to add a 2x2x2 Strassen block.
    
    def add_2x2x2_strassen(A_i, A_j, B_i, B_k):
        nonlocal col
        # A is 2x2 at (A_i, A_j)
        # B is 2x2 at (B_i, B_k)
        # C is 2x2 at (B_k, A_i)
        
        a = A_i * m + A_j
        b = A_i * m + (A_j + 1)
        c = (A_i + 1) * m + A_j
        d = (A_i + 1) * m + (A_j + 1)
        
        e = B_i * p + B_k
        f = B_i * p + (B_k + 1)
        g = (B_i + 1) * p + B_k
        h = (B_i + 1) * p + (B_k + 1)
        
        C00 = B_k * n + A_i
        C01 = (B_k + 1) * n + A_i
        C10 = B_k * n + (A_i + 1)
        C11 = (B_k + 1) * n + (A_i + 1)
        
        # m1 = a * (g+h)
        U[a, col] = 1
        V[g, col] = 1
        V[h, col] = 1
        W[C00, col] = 1
        W[C10, col] = 1
        col += 1
        
        # m2 = (a+c) * h
        U[a, col] = 1
        U[c, col] = 1
        V[h, col] = 1
        W[C00, col] = -1
        W[C01, col] = 1
        col += 1
        
        # m3 = (b+d) * e
        U[b, col] = 1
        U[d, col] = 1
        V[e, col] = 1
        W[C01, col] = 1
        W[C11, col] = 1
        col += 1
        
        # m4 = d * (e-f)
        U[d, col] = 1
        V[e, col] = 1
        V[f, col] = -1
        W[C00, col] = 1
        W[C10, col] = -1
        col += 1
        
        # m5 = (a+d) * (e+h)
        U[a, col] = 1
        U[d, col] = 1
        V[e, col] = 1
        V[h, col] = 1
        W[C00, col] = 1
        W[C01, col] = -1
        W[C10, col] = -1
        W[C11, col] = 1
        col += 1
        
        # m6 = (b-d) * (g+h)
        U[b, col] = 1
        U[d, col] = -1
        V[g, col] = 1
        V[h, col] = 1
        W[C10, col] = 1
        W[C11, col] = -1
        col += 1
        
        # m7 = (a-c) * (f+h)
        U[a, col] = 1
        U[c, col] = -1
        V[f, col] = 1
        V[h, col] = 1
        W[C01, col] = 1
        W[C11, col] = 1
        col += 1
    
    # A1 * B1
    add_2x2x2_strassen(0, 0, 0, 0) # B1_1: cols 0,1
    add_2x2x2_strassen(0, 0, 0, 2) # B1_2: cols 2,3
    add_2x2x2_strassen(0, 0, 0, 4) # B1_3: cols 4,5
    
    # A2 * B2
    add_2x2x2_strassen(0, 2, 2, 0) # B2_1: cols 0,1
    add_2x2x2_strassen(0, 2, 2, 2) # B2_2: cols 2,3
    add_2x2x2_strassen(0, 2, 2, 4) # B2_3: cols 4,5
    
    # A3 * B3
    # A3 is 2x1 (col 4), B3 is 1x6 (row 4).
    # C_i,k = A_i,4 * B_4,k
    # This is a sum of 12 rank-1 tensors.
    for i in range(n):
        for k in range(p):
            U[i * m + 4, col] = 1
            V[4 * p + k, col] = 1
            W[k * n + i, col] = 1
            col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
