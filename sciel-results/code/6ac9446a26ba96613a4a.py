import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank 65 decomposition
    # 49 for 4x4x4 part (cols 0-3)
    # 16 for 4x4x1 part (col 4)
    rank = 65
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 term
    # u_idx, v_idx, w_idx, coeff
    def add_term(u_idx, v_idx, w_idx, coeff):
        nonlocal col
        U[u_idx][col] = coeff
        V[v_idx][col] = coeff
        W[w_idx][col] = coeff
        col += 1

    # --- Part 1: 4x4x4 using Block Strassen ---
    # Block indices:
    # A blocks: A00, A01, A10, A11 (each 2x2)
    # B blocks: B00, B01, B10, B11 (each 2x2)
    # C blocks: C00, C01, C10, C11 (each 2x2)
    
    # The 7 Strassen multiplications for the 4x4 case
    # Each is a 2x2 matrix multiplication
    # We need to define the linear combinations for A and B blocks
    
    # M1 = A00 * (B00 + B11)
    # M2 = (A00 + A10) * B11
    # M3 = (A01 + A11) * B00
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B10
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # For each block multiplication X * Y = Z, we use inner Strassen
    # Inner Strassen produces 7 terms for 2x2 matrices
    
    # We will iterate through the 7 block multiplications
    # For each, we define the coefficients for the 4 A-blocks and 4 B-blocks
    # Then we apply the inner Strassen logic
    
    # Inner Strassen Logic:
    # Given X = [[x00, x01], [x10, x11]] and Y = [[y00, y01], [y10, y11]]
    # The 7 products are:
    # P1 = x00 * (y00 + y11)
    # P2 = (x00 + x10) * y11
    # P3 = (x01 + x11) * y00
    # P4 = x11 * (y10 - y00)
    # P5 = (x00 + x01) * y10
    # P6 = (x10 - x00) * (y00 + y01)
    # P7 = (x01 - x11) * (y10 + y11)
    
    # Reconstruction:
    # z00 = P1 + P4 - P2 + P6
    # z01 = P3 + P7
    # z10 = P5 + P7
    # z11 = P1 - P3 + P5 + P4
    
    # We need to map x00, x01, x10, x11 to global A indices
    # and y00, y01, y10, y11 to global B indices
    # and z00... to global C indices
    
    # Global indices for A (4x4):
    # A[0,0] A[0,1]
    # A[1,0] A[1,1]  <- Block A00
    # A[2,0] A[2,1]  <- Block A10 (row 2,3 are block row 1)
    # A[3,0] A[3,1]
    
    # Wait, standard row-major:
    # A[0,0] is (0,0). A[0,1] is (0,1).
    # Block A00: rows 0,1; cols 0,1.
    # Block A01: rows 0,1; cols 2,3.
    # Block A10: rows 2,3; cols 0,1.
    # Block A11: rows 2,3; cols 2,3.
    
    # Helper to get global index for A[i,j]
    def a_idx(i, j):
        return i * m + j
        
    def b_idx(i, j):
        return i * p + j
        
    def c_idx(i, j):
        return j * n + i # W[k*n+i] where k is col of C, i is row of C
        
    # Define the 7 block multiplications
    # Each defined by coeffs for A00, A01, A10, A11 and B00, B01, B10, B11
    # And which C block it contributes to
    
    block_mults = [
        # (coeffs_A, coeffs_B, C_block_row, C_block_col, sign_in_recon)
        # We need to handle the reconstruction signs carefully.
        # It is easier to just generate the 49 terms directly by expanding the inner Strassen
        # for each of the 7 outer multiplications, and accumulating into the C blocks.
        
        # Let's define the 7 outer multiplications as pairs of linear combinations
        # L_A = sum c_a * A_block
        # L_B = sum c_b * B_block
        # Product P = L_A * L_B
        
        # Outer 1: A00 * (B00 + B11)
        # Outer 2: (A00 + A10) * B11
        # Outer 3: (A01 + A11) * B00
        # Outer 4: A11 * (B10 - B00)
        # Outer 5: (A00 + A01) * B10
        # Outer 6: (A10 - A00) * (B00 + B01)
        # Outer 7: (A01 - A11) * (B10 + B11)
    ]
    
    # We will process each outer multiplication
    # For each, we determine the 2x2 matrices X and Y (conceptually)
    # X is the linear combo of A blocks
    # Y is the linear combo of B blocks
    
    # Then we apply inner Strassen to X and Y
    # The inner Strassen produces 7 products P1..P7 of 2x2 matrices?
    # No, X and Y are 2x2 matrices themselves (linear combos of 2x2 blocks are 2x2 matrices).
    # So the inner Strassen computes X*Y using 7 scalar multiplications?
    # No, X and Y are 2x2 matrices. The "scalar" multiplications in inner Strassen
    # are actually multiplications of the entries of X and Y.
    # The entries of X are linear combos of A entries.
    # The entries of Y are linear combos of B entries.
    # So each "inner" product is a product of a linear combo of A entries and a linear combo of B entries.
    # This results in a rank-1 term in the global tensor?
    # Yes, because the product is (sum a_i A_i) * (sum b_j B_j).
    # This is a single rank-1 term in the decomposition if we consider the linearity?
    # No, the rank-1 term in the tensor decomposition is $u \otimes v \otimes w$.
    # $u$ is a vector of coeffs for A, $v$ for B, $w$ for C.
    # If we have a term like $(A_{00} + A_{10}) B_{11}$, this is not a single rank-1 term in the $4 \times 4 \times 4$ tensor.
    # It is a sum of rank-1 terms?
    # No, Strassen's algorithm works by replacing 8 multiplications with 7.
    # The 7 "multiplications" are of the form $X_{ij} Y_{kl}$.
    # In the context of the full tensor, each such product corresponds to a rank-1 term
    # IF X and Y are single entries.
    # But here X and Y are 2x2 matrices.
    # So we must decompose the 2x2 matrix multiplication $X Y = Z$ into 7 rank-1 terms
    # where the "entries" of X and Y are themselves linear combinations of the original A and B entries.
    # This is valid because the rank of the tensor is sub-multiplicative?
    # Yes. The rank of $M_{4,4,4}$ is $\le 7 \times \text{rank}(M_{2,2,2}) = 7 \times 7 = 49$.
    
    # So, for each of the 7 outer multiplications (which are 2x2 matrix products),
    # we decompose that 2x2 product into 7 inner rank-1 terms.
    # Total 49 terms.
    
    # Let's define the 7 outer multiplications explicitly with their A and B linear combos.
    # We represent a 2x2 block as a dict of (local_row, local_col) -> global_index
    # But it's easier to just define the linear combination coefficients for the 4 A-blocks and 4 B-blocks.
    
    # A00: A[0:2, 0:2]
    # A01: A[0:2, 2:4]
    # A10: A[2:4, 0:2]
    # A11: A[2:4, 2:4]
    
    # For a given outer multiplication, we have:
    # X = c00*A00 + c01*A01 + c10*A10 + c11*A11  (This is a 2x2 matrix)
    # Y = d00*B00 + d01*B01 + d10*B10 + d11*B11  (This is a 2x2 matrix)
    
    # Wait, the Strassen formulas are:
    # M1 = A00 * (B00 + B11)
    # Here X = A00 (so coeff 1 for A00, 0 others).
    # Y = B00 + B11 (so coeff 1 for B00, 1 for B11, 0 others).
    
    # So for each outer mult, we have a set of coeffs for A blocks and B blocks.
    # Then we need to compute the product of two 2x2 matrices X and Y,
    # where X and Y are defined by these coeffs.
    # And we decompose this 2x2 product into 7 rank-1 terms using inner Strassen.
    
    # Inner Strassen on X and Y:
    # X = [[X00, X01], [X10, X11]]
    # Y = [[Y00, Y01], [Y10, Y11]]
    # The 7 products are:
    # P1 = X00 * (Y00 + Y11)
    # P2 = (X00 + X10) * Y11
    # ...
    
    # Each Xij is a linear combination of A entries.
    # Each Yij is a linear combination of B entries.
    # So P1 is (sum a A) * (sum b B).
    # This corresponds to a rank-1 term in the global tensor,
    # with U vector being the coeffs of A, V vector being coeffs of B,
    # and W vector being the coeffs of C (from reconstruction).
    
    # We will generate these 49 terms.
    
    # Define outer multiplications
    # (coeffs_A, coeffs_B)
    # coeffs_A: [c_A00, c_A01, c_A10, c_A11]
    # coeffs_B: [c_B00, c_B01, c_B10, c_B11]
    
    outer_mults = [
        ([1, 0, 0, 0], [1, 0, 0, 1]), # M1: A00 * (B00 + B11)
        ([1, 0, 1, 0], [0, 0, 0, 1]), # M2: (A00 + A10) * B11
        ([0, 1, 0, 1], [1, 0, 0, 0]), # M3: (A01 + A11) * B00
        ([0, 0, 0, 1], [0, 1, -1, 0]),# M4: A11 * (B10 - B00)
        ([1, 1, 0, 0], [0, 0, 1, 0]), # M5: (A00 + A01) * B10
        ([1, 0, -1, 0], [1, 1, 0, 0]),# M6: (A10 - A00) * (B00 + B01)
        ([0, 1, 0, -1], [0, 0, 1, 1]) # M7: (A01 - A11) * (B10 + B11)
    ]
    
    # Reconstruction coefficients for C blocks
    # C00 = M1 + M4 - M2 + M6
    # C01 = M3 + M7
    # C10 = M5 + M7
    # C11 = M1 - M3 + M5 + M4
    
    # Map: C_block -> list of (M_index, coeff)
    recon = {
        (0,0): [(0, 1), (3, 1), (1, -1), (5, 1)],
        (0,1): [(2, 1), (6, 1)],
        (1,0): [(4, 1), (6, 1)],
        (1,1): [(0, 1), (2, -1), (4, 1), (3, 1)]
    }
    
    # Process each outer multiplication
    for m_idx in range(7):
        ca = outer_mults[m_idx][0]
        cb = outer_mults[m_idx][1]
        
        # Construct X and Y 2x2 matrices (conceptually)
        # X00 = sum ca[k] * A_block_k[0,0]
        # We need the global indices for each entry of the blocks.
        
        # Block A00: A[0,0], A[0,1], A[1,0], A[1,1]
        # Block A01: A[0,2], A[0,3], A[1,2], A[1,3]
        # Block A10: A[2,0], A[2,1], A[3,0], A[3,1]
        # Block A11: A[2,2], A[2,3], A[3,2], A[3,3]
        
        # Get global index for A[i,j]
        def get_a_idx(block, lr, lc):
            if block == 0: # A00
                return (lr) * 4 + (lc)
            elif block == 1: # A01
                return (lr) * 4 + (2 + lc)
            elif block == 2: # A10
                return (2 + lr) * 4 + (lc)
            elif block == 3: # A11
                return (2 + lr) * 4 + (2 + lc)
                
        def get_b_idx(block, lr, lc):
            if block == 0: # B00
                return (lr) * 5 + (lc)
            elif block == 1: # B01
                return (lr) * 5 + (2 + lc)
            elif block == 2: # B10
                return (2 + lr) * 5 + (lc)
            elif block == 3: # B11
                return (2 + lr) * 5 + (2 + lc)
                
        def get_c_idx(block, lr, lc):
            # C block (br, bc)
            # C[br*2+lr, bc*2+lc]
            # W index: k*n + i where k=col, i=row
            i = br * 2 + lr
            k = bc * 2 + lc
            return k * 4 + i

        # Inner Strassen 7 products
        # We need to define the 7 products P1..P7
        # Each P is a product of a linear combo of X entries and Y entries.
        # X entries are linear combos of A entries.
        # Y entries are linear combos of B entries.
        
        # X00, X01, X10, X11
        # Y00, Y01, Y10, Y11
        
        # P1 = X00 * (Y00 + Y11)
        # P2 = (X00 + X10) * Y11
        # P3 = (X01 + X11) * Y00
        # P4 = X11 * (Y10 - Y00)
        # P5 = (X00 + X01) * Y10
        # P6 = (X10 - X00) * (Y00 + Y01)
        # P7 = (X01 - X11) * (Y10 + Y11)
        
        # We will create 7 rank-1 terms for this outer multiplication.
        # Each term has U, V, W vectors.
        # U vector: coeffs for A
        # V vector: coeffs for B
        # W vector: coeffs for C (from reconstruction)
        
        # For each inner product Pq:
        # We determine the A-coeffs and B-coeffs.
        # Then we determine the C-coeffs by applying the reconstruction formula
        # for the specific C block that this Pq contributes to?
        # No, Pq contributes to ALL C blocks with specific signs.
        # So the W vector will have non-zero entries in multiple C blocks.
        
        # Let's define the A and B coeffs for each of the 7 inner products.
        
        # Helper to get A coeffs for a specific X entry (e.g., X00)
        # X00 = sum_{b} ca[b] * A_block_b[0,0]
        
        def get_x_coeffs(lr, lc):
            # Returns list of (a_idx, coeff)
            coeffs = []
            for b in range(4):
                if ca[b] != 0:
                    idx = get_a_idx(b, lr, lc)
                    coeffs.append((idx, ca[b]))
            return coeffs
            
        def get_y_coeffs(lr, lc):
            coeffs = []
            for b in range(4):
                if cb[b] != 0:
                    idx = get_b_idx(b, lr, lc)
                    coeffs.append((idx, cb[b]))
            return coeffs
            
        # Define the 7 inner products as (x_coeffs_list, y_coeffs_list)
        # x_coeffs_list is a list of (a_idx, coeff)
        # y_coeffs_list is a list of (b_idx, coeff)
        
        # P1: X00 * (Y00 + Y11)
        x1 = get_x_coeffs(0, 0)
        y1 = get_y_coeffs(0, 0) + get_y_coeffs(1, 1)
        
        # P2: (X00 + X10) * Y11
        x2 = get_x_coeffs(0, 0) + get_x_coeffs(1, 0)
        y2 = get_y_coeffs(1, 1)
        
        # P3: (X01 + X11) * Y00
        x3 = get_x_coeffs(0, 1) + get_x_coeffs(1, 1)
        y3 = get_y_coeffs(0, 0)
        
        # P4: X11 * (Y10 - Y00)
        x4 = get_x_coeffs(1, 1)
        y4 = get_y_coeffs(1, 0) + get_y_coeffs(0, 0)
        # Note: Y00 coeff is -1
        y4 = [(idx, c if i < len(get_y_coeffs(1,0)) else -c) for i, (idx, c) in enumerate(y4)] 
        # This is getting messy. Let's just sum them up properly.
        
        # Better approach:
        # For each of the 7 inner products, we build the U and V vectors directly.
        
        inner_products = [
            # (x_entry_indices, y_entry_indices)
            # x_entry_indices: list of (lr, lc, coeff)
            # y_entry_indices: list of (lr, lc, coeff)
            [([(0,0,1)], [(0,0,1), (1,1,1)]), # P1
             ([(0,0,1), (1,0,1)], [(1,1,1)]), # P2
             ([(0,1,1), (1,1,1)], [(0,0,1)]), # P3
             ([(1,1,1)], [(1,0,1), (0,0,-1)]), # P4
             ([(0,0,1), (0,1,1)], [(1,0,1)]), # P5
             ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)]), # P6
             ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)])  # P7
            ]
        ]
        
        for p_idx in range(7):
            x_defs, y_defs = inner_products[0][p_idx]
            
            # Build U vector (coeffs for A)
            u_vec = {}
            for lr, lc, c in x_defs:
                for b in range(4):
                    if ca[b] != 0:
                        idx = get_a_idx(b, lr, lc)
                        u_vec[idx] = u_vec.get(idx, 0) + ca[b] * c
                        
            # Build V vector (coeffs for B)
            v_vec = {}
            for lr, lc, c in y_defs:
                for b in range(4):
                    if cb[b] != 0:
                        idx = get_b_idx(b, lr, lc)
                        v_vec[idx] = v_vec.get(idx, 0) + cb[b] * c
                        
            # Build W vector (coeffs for C)
            # This P_idx contributes to C blocks with specific signs.
            # We need to map p_idx to the reconstruction signs.
            # The reconstruction is:
            # C00 = P1 + P4 - P2 + P6
            # C01 = P3 + P7
            # C10 = P5 + P7
            # C11 = P1 - P3 + P5 + P4
            
            # So for a given p_idx (0-based, so P1 is 0), we check which C blocks it appears in.
            w_vec = {}
            
            # Map p_idx to signs in each C block
            # C00: P1(1), P2(-1), P3(0), P4(1), P5(0), P6(1), P7(0)
            # C01: P1(0), P2(0), P3(1), P4(0), P5(0), P6(0), P7(1)
            # C10: P1(0), P2(0), P3(0), P4(0), P5(1), P6(0), P7(1)
            # C11: P1(1), P2(0), P3(-1), P4(1), P5(1), P6(0), P7(0)
            
            signs = {
                0: [1, 0, 0, 1], # C00, C01, C10, C11
                1: [-1, 0, 0, 0],
                2: [0, 1, 0, -1],
                3: [1, 0, 0, 1],
                4: [0, 0, 1, 1],
                5: [1, 0, 0, 0],
                6: [0, 1, 1, 0]
            }
            
            s = signs[p_idx]
            for c_block_idx in range(4):
                if s[c_block_idx] != 0:
                    # c_block_idx 0: C00, 1: C01, 2: C10, 3: C11
                    # Map to (br, bc)
                    br, bc = divmod(c_block_idx, 2) # 0->00, 1->01, 2->10, 3->11
                    # Add to W vector for all entries in this C block
                    for lr in range(2):
                        for lc in range(2):
                            idx = get_c_idx(br, bc, lr, lc)
                            w_vec[idx] = w_vec.get(idx, 0) + s[c_block_idx]
                            
            # Add the term
            for idx, val in u_vec.items():
                if val != 0:
                    U[idx][col] = val
            for idx, val in v_vec.items():
                if val != 0:
                    V[idx][col] = val
            for idx, val in w_vec.items():
                if val != 0:
                    W[idx][col] = val
            col += 1
            
    # --- Part 2: 4x4x1 (Col 4) ---
    # Rank 16
    for i in range(4):
        for j in range(4):
            u_idx = i * 4 + j
            v_idx = j * 5 + 4
            w_idx = 4 * 4 + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}

