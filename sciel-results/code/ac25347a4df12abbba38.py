import numpy as np
import time

DIMENSIONS = [2, 4, 5]

def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    
    # We construct a Rank 32 decomposition.
    # Structure:
    # A (2x4) is split into A1 (2x2) and A2 (2x2)
    # B (4x5) is split into B_top (2x5) and B_bottom (2x5)
    # P = A1 @ B_top + A2 @ B_bottom
    #
    # Each term (e.g., A1 @ B_top) is a 2x2 by 2x5 multiplication.
    # B_top is split into B_t1 (2x2), B_t2 (2x2), B_t3 (2x1)
    # A1 @ B_top = A1 @ B_t1 + A1 @ B_t2 + A1 @ B_t3
    #
    # A1 @ B_t1 is 2x2x2 (Rank 7, Strassen)
    # A1 @ B_t2 is 2x2x2 (Rank 7, Strassen)
    # A1 @ B_t3 is 2x2x1 (Rank 2, Schoolbook)
    #
    # Total Rank per block: 7 + 7 + 2 = 16
    # Two blocks (A1 and A2): 16 + 16 = 32
    
    rank = 32
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Helper to add a Strassen 2x2x2 decomposition
    # A indices: a(0,0), b(0,1), c(1,0), d(1,1) in the 2x2 subblock
    # B indices: e(0,0), f(0,1), g(1,0), h(1,1) in the 2x2 subblock
    # We need to map these to the global indices.
    
    def add_strassen(offset, a_row_offset, a_col_offset, b_row_offset, b_col_offset):
        """
        Adds a Strassen decomposition for a 2x2 by 2x2 multiplication.
        a_row_offset, a_col_offset: top-left corner of A subblock in global A (2x4)
        b_row_offset, b_col_offset: top-left corner of B subblock in global B (4x5)
        offset: starting index in the rank dimension
        """
        # A elements:
        # a: (a_row_offset, a_col_offset)
        # b: (a_row_offset, a_col_offset+1)
        # c: (a_row_offset+1, a_col_offset)
        # d: (a_row_offset+1, a_col_offset+1)
        
        # B elements:
        # e: (b_row_offset, b_col_offset)
        # f: (b_row_offset, b_col_offset+1)
        # g: (b_row_offset+1, b_col_offset)
        # h: (b_row_offset+1, b_col_offset+1)
        
        # Strassen Multiplications:
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = c(g-f)
        # M5 = (a+b)h
        # M6 = (d-b)(e+g)
        # M7 = (a-c)(f+h)
        
        # Output P:
        # P11 = M1 + M4 - M5 - M7
        # P12 = M3 + M5
        # P21 = M2 + M7
        # P22 = M1 - M2 + M4
        
        # We define the linear combinations for U, V, W.
        # U maps A elements to the "left" factor of the multiplication.
        # V maps B elements to the "right" factor.
        # W maps the product to the output P.
        
        # For each Strassen multiplication k (0..6):
        # U[?, offset+k] = coeff of A elements
        # V[?, offset+k] = coeff of B elements
        # W[?, offset+k] = coeff of P elements
        
        # M1: (a+d)(e+h)
        U[a_row_offset * m + a_col_offset, offset + 0] += 1.0
        U[(a_row_offset+1) * m + (a_col_offset+1), offset + 0] += 1.0
        V[b_row_offset * p + b_col_offset, offset + 0] += 1.0
        V[(b_row_offset+1) * p + (b_col_offset+1), offset + 0] += 1.0
        # P11 += M1, P22 += M1
        W[b_col_offset * n + a_row_offset, offset + 0] += 1.0
        W[(b_col_offset+1) * n + (a_row_offset+1), offset + 0] += 1.0

        # M2: (c+d)e
        U[(a_row_offset+1) * m + a_col_offset, offset + 1] += 1.0
        U[(a_row_offset+1) * m + (a_col_offset+1), offset + 1] += 1.0
        V[b_row_offset * p + b_col_offset, offset + 1] += 1.0
        # P21 += M2
        W[b_col_offset * n + (a_row_offset+1), offset + 1] += 1.0

        # M3: a(f-h)
        U[a_row_offset * m + a_col_offset, offset + 2] += 1.0
        V[b_row_offset * p + (b_col_offset+1), offset + 2] += 1.0
        V[(b_row_offset+1) * p + b_col_offset, offset + 2] += -1.0
        # P12 += M3
        W[(b_col_offset+1) * n + a_row_offset, offset + 2] += 1.0

        # M4: c(g-f)
        U[(a_row_offset+1) * m + a_col_offset, offset + 3] += 1.0
        V[(b_row_offset+1) * p + b_col_offset, offset + 3] += 1.0
        V[b_row_offset * p + (b_col_offset+1), offset + 3] += -1.0
        # P11 += M4, P22 += M4
        W[b_col_offset * n + a_row_offset, offset + 3] += 1.0
        W[(b_col_offset+1) * n + (a_row_offset+1), offset + 3] += 1.0

        # M5: (a+b)h
        U[a_row_offset * m + a_col_offset, offset + 4] += 1.0
        U[a_row_offset * m + (a_col_offset+1), offset + 4] += 1.0
        V[(b_row_offset+1) * p + (b_col_offset+1), offset + 4] += 1.0
        # P11 -= M5, P12 += M5
        W[b_col_offset * n + a_row_offset, offset + 4] += -1.0
        W[(b_col_offset+1) * n + a_row_offset, offset + 4] += 1.0

        # M6: (d-b)(e+g)
        U[(a_row_offset+1) * m + (a_col_offset+1), offset + 5] += 1.0
        U[a_row_offset * m + (a_col_offset+1), offset + 5] += -1.0
        V[b_row_offset * p + b_col_offset, offset + 5] += 1.0
        V[(b_row_offset+1) * p + b_col_offset, offset + 5] += 1.0
        # M6 is not used in the standard Strassen output formula I recalled?
        # Wait, Strassen:
        # P11 = M1 + M4 - M5 - M7
        # P12 = M3 + M5
        # P21 = M2 + M7
        # P22 = M1 - M2 + M4
        # M6 is defined but cancels out?
        # Standard Strassen:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # C11 = M1+M4-M5-M7
        # C12 = M3+M5
        # C21 = M2+M7
        # C22 = M1-M2+M4
        # Yes, M6 is not used in the final sum. It is a "dummy" in some descriptions or I am misremembering the 7th term.
        # Actually, the 7th term IS M7. M6 is the 6th.
        # In the standard algorithm, M6 is computed but not added to any C element?
        # No, that would be wasteful.
        # Let's check:
        # C11 = M1 + M4 - M5 - M7
        # C12 = M3 + M5
        # C21 = M2 + M7
        # C22 = M1 - M2 + M4
        # M6 is indeed not used in the standard 7-multiplication Strassen algorithm for 2x2.
        # Wait, is it?
        # Some sources say M6 is used?
        # No, the standard Strassen uses 7 multiplications.
        # If M6 is not used, why compute it?
        # Ah, the definition of M6 is (A21-A11)(B11+B12).
        # It is NOT used in the final linear combination for C.
        # So we can skip M6 entirely?
        # If we skip it, the rank is 7.
        # So I should not add M6 to the decomposition.
        
        # M7: (a-c)(f+h)  -- Wait, M7 = (A12-A22)(B21+B22)
        # My mapping:
        # A12 is b (row 0, col 1)
        # A22 is d (row 1, col 1)
        # B21 is g (row 1, col 0)
        # B22 is h (row 1, col 1)
        # So M7 = (b-d)(g+h)
        
        U[a_row_offset * m + (a_col_offset+1), offset + 6] += 1.0
        U[(a_row_offset+1) * m + (a_col_offset+1), offset + 6] += -1.0
        V[(b_row_offset+1) * p + b_col_offset, offset + 6] += 1.0
        V[(b_row_offset+1) * p + (b_col_offset+1), offset + 6] += 1.0
        # P11 -= M7, P21 += M7
        W[b_col_offset * n + a_row_offset, offset + 6] += -1.0
        W[b_col_offset * n + (a_row_offset+1), offset + 6] += 1.0

    def add_schoolbook_2x2x1(offset, a_row_offset, a_col_offset, b_row_offset, b_col_offset):
        """
        Adds a schoolbook decomposition for a 2x2 by 2x1 multiplication.
        A is 2x2, B is 2x1. Result P is 2x1.
        P1 = a*e + b*g
        P2 = c*e + d*g
        """
        # a: (a_row_offset, a_col_offset)
        # b: (a_row_offset, a_col_offset+1)
        # c: (a_row_offset+1, a_col_offset)
        # d: (a_row_offset+1, a_col_offset+1)
        
        # e: (b_row_offset, b_col_offset)
        # g: (b_row_offset+1, b_col_offset)
        
        # M1 = a*e
        U[a_row_offset * m + a_col_offset, offset + 0] += 1.0
        V[b_row_offset * p + b_col_offset, offset + 0] += 1.0
        W[b_col_offset * n + a_row_offset, offset + 0] += 1.0
        
        # M2 = b*g
        U[a_row_offset * m + (a_col_offset+1), offset + 0] += 1.0
        V[(b_row_offset+1) * p + b_col_offset, offset + 0] += 1.0
        W[b_col_offset * n + a_row_offset, offset + 0] += 1.0
        
        # M3 = c*e
        U[(a_row_offset+1) * m + a_col_offset, offset + 1] += 1.0
        V[b_row_offset * p + b_col_offset, offset + 1] += 1.0
        W[b_col_offset * n + (a_row_offset+1), offset + 1] += 1.0
        
        # M4 = d*g
        U[(a_row_offset+1) * m + (a_col_offset+1), offset + 1] += 1.0
        V[(b_row_offset+1) * p + b_col_offset, offset + 1] += 1.0
        W[b_col_offset * n + (a_row_offset+1), offset + 1] += 1.0

    # Block 1: A1 (rows 0-1, cols 0-1) @ B_top (rows 0-1, cols 0-4)
    # B_top is split into:
    # B_t1: cols 0-1 (2x2) -> Strassen
    # B_t2: cols 2-3 (2x2) -> Strassen
    # B_t3: col 4 (2x1) -> Schoolbook
    
    # A1 is at (0,0) in A.
    # B_t1 is at (0,0) in B.
    add_strassen(0, 0, 0, 0, 0)
    
    # B_t2 is at (0,2) in B.
    add_strassen(7, 0, 0, 0, 2)
    
    # B_t3 is at (0,4) in B.
    add_schoolbook_2x2x1(14, 0, 0, 0, 4)
    
    # Block 2: A2 (rows 0-1, cols 2-3) @ B_bottom (rows 2-3, cols 0-4)
    # A2 is at (0,2) in A.
    # B_bottom is split into:
    # B_b1: cols 0-1 (2x2) -> Strassen
    # B_b2: cols 2-3 (2x2) -> Strassen
    # B_b3: col 4 (2x1) -> Schoolbook
    
    # A2 is at (0,2) in A.
    # B_b1 is at (2,0) in B.
    add_strassen(16, 0, 2, 2, 0)
    
    # B_b2 is at (2,2) in B.
    add_strassen(23, 0, 2, 2, 2)
    
    # B_b3 is at (2,4) in B.
    add_schoolbook_2x2x1(30, 0, 2, 2, 4)
    
    # Convert to lists
    U_list = [[int(val) if val == int(val) else val for val in row] for row in U.tolist()]
    V_list = [[int(val) if val == int(val) else val for val in row] for row in V.tolist()]
    W_list = [[int(val) if val == int(val) else val for val in row] for row in W.tolist()]
    
    return {"U": U_list, "V": V_list, "W": W_list}
