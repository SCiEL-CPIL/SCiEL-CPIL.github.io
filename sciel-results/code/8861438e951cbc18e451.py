"""Matrix multiplication 4x5x5: Strassen on 4x4x4 sub-block + schoolbook for remainder."""
import time
import random
import numpy as np

DIMENSIONS = [4, 5, 5]

# Deadline guard
T_START = time.time()
TIME_LIMIT = 135  # seconds, leaving margin before 150s kill

def construct():
    """Construct a rank-minimal (or low-rank) exact tensor decomposition for 4x5x5 matmul."""
    n, m, p = 4, 5, 5
    
    # We'll build U (n*m x rank), V (m*p x rank), W (p*n x rank)
    # using a list of (u_vec, v_vec, w_vec) triples, then convert to matrices.
    
    terms = []  # list of (u, v, w) where u is length n*m, v length m*p, w length p*n
    
    def add_term(u_indices_vals, v_indices_vals, w_indices_vals):
        """Add a rank-1 term. Each arg is a dict {row_idx: val}."""
        u = [0.0] * (n * m)
        v = [0.0] * (m * p)
        w = [0.0] * (p * n)
        for idx, val in u_indices_vals.items():
            u[idx] = val
        for idx, val in v_indices_vals.items():
            v[idx] = val
        for idx, val in w_indices_vals.items():
            w[idx] = val
        terms.append((u, v, w))
    
    # --- Part 1: Strassen decomposition for the 4x4x4 sub-tensor ---
    # We decompose C[0:4, 0:4] = A[0:4, 0:4] * B[0:4, 0:4]
    # Using Strassen's 2x2 block algorithm applied to 4x4 matrices split into 2x2 blocks.
    
    # For 2x2 Strassen:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For 4x4, split into 2x2 blocks of size 2x2.
    # Block indices: A11 = A[0:2,0:2], A12 = A[0:2,2:4], A21 = A[2:4,0:2], A22 = A[2:4,2:4]
    # Similarly for B.
    
    # In the tensor decomposition, each "linear form" like (A11+A22) means:
    # for each element position in the 2x2 block, add the corresponding entries from both blocks.
    
    # Let's define helper to get U coefficients for a sum/difference of 2x2 blocks of A.
    # U row index for A[i,j] is i*m + j = i*5 + j (since m=5)
    
    def a_block(r1, r2, c1, c2):
        """Return dict of U indices for A[r1:r2, c1:c2]."""
        d = {}
        for i in range(r1, r2):
            for j in range(c1, c2):
                d[i * m + j] = 1.0
        return d
    
    def a_linear_form(blocks):
        """blocks is list of (sign, r1, r2, c1, c2). Returns dict of U indices."""
        d = {}
        for sign, r1, r2, c1, c2 in blocks:
            for i in range(r1, r2):
                for j in range(c1, c2):
                    idx = i * m + j
                    d[idx] = d.get(idx, 0.0) + sign
        return d
    
    def b_block(r1, r2, c1, c2):
        """Return dict of V indices for B[r1:r2, c1:c2]. V row index is j*p + k."""
        d = {}
        for j in range(r1, r2):
            for k in range(c1, c2):
                d[j * p + k] = 1.0
        return d
    
    def b_linear_form(blocks):
        """blocks is list of (sign, r1, r2, c1, c2). Returns dict of V indices."""
        d = {}
        for sign, r1, r2, c1, c2 in blocks:
            for j in range(r1, r2):
                for k in range(c1, c2):
                    idx = j * p + k
                    d[idx] = d.get(idx, 0.0) + sign
        return d
    
    def c_block(r1, r2, c1, c2):
        """Return dict of W indices for C[r1:r2, c1:c2]. W row index is k*n + i."""
        d = {}
        for k in range(r1, r2):
            for i in range(c1, c2):
                d[k * n + i] = 1.0
        return d
    
    def c_linear_form(blocks):
        """blocks is list of (sign, r1, r2, c1, c2). Returns dict of W indices."""
        d = {}
        for sign, r1, r2, c1, c2 in blocks:
            for k in range(r1, r2):
                for i in range(c1, c2):
                    idx = k * n + i
                    d[idx] = d.get(idx, 0.0) + sign
        return d
    
    # Strassen 2x2 formulas, each M is a product of a linear form in A and a linear form in B.
    # We need to express each M as: for each element in the resulting 2x2 block,
    # the contribution to C is via the W coefficients.
    
    # Actually, the Strassen algorithm computes:
    # M1 = (A11+A22) * (B11+B22)  -> this is a 2x2 matrix product
    # In tensor terms, M1 contributes to C11, C22 (with coefficients from the C formulas)
    
    # But wait - in a tensor decomposition, each rank-1 term corresponds to ONE scalar product
    # of one linear form in A times one linear form in B, contributing to multiple C entries.
    
    # The standard Strassen for 2x2 has 7 scalar multiplications. For 2x2 blocks that are themselves
    # 2x2 matrices, each "scalar multiplication" is actually a 2x2 matrix multiplication,
    # which itself has 4 scalar multiplications in schoolbook.
    
    # So for 4x4 using Strassen with 2x2 sub-blocks:
    # Each of the 7 M terms is a 2x2*2x2 product = 4 scalar mults
    # Total = 7*4 = 28 scalar multiplications for the 4x4x4 part.
    
    # Let me construct these 28 terms.
    
    # Block definitions (0-indexed):
    # A11: rows 0-1, cols 0-1; A12: rows 0-1, cols 2-3
    # A21: rows 2-3, cols 0-1; A22: rows 2-3, cols 2-3
    # Same for B.
    
    # Strassen M terms (each is a product of two 2x2 linear forms):
    # M1 = (A11+A22) * (B11+B22)
    # M2 = (A21+A22) * B11
    # M3 = A11 * (B12-B22)
    # M4 = A22 * (B21-B11)
    # M5 = (A11+A12) * B22
    # M6 = (A21-A11) * (B11+B12)
    # M7 = (A12-A22) * (B21+B22)
    
    # C11 = M1 + M4 - M5 + M7  (each is 2x2)
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For each M term, which is a 2x2 matrix = sum of 4 scalar products:
    # (L_A)_{a,b} * (L_B)_{c,d} contributes to (result)_{a,c}... 
    # Wait, let me think more carefully.
    
    # If L_A is a 2x2 linear form in A and L_B is a 2x2 linear form in B,
    # then M = L_A * L_B is a 2x2 matrix where M[r,c] = sum_k L_A[r,k]*L_B[k,c].
    
    # In tensor decomposition, each scalar multiplication L_A[r,k] * L_B[k,c]
    # contributes to the appropriate C entries.
    
    # So for each M term, we have 4 scalar multiplications (for a 2x2 result).
    
    # Let me define the 7 Strassen terms with their A and B linear forms:
    
    strassen_terms = [
        # (name, a_form_blocks, b_form_blocks, c_contributions)
        # c_contributions: list of (sign, C_block_r1, C_block_r2, C_block_c1, C_block_c2)
        # where the M term contributes sign * M to that C block.
        
        # M1 = (A11+A22)*(B11+B22)
        # Contributes: +M1 to C11, +M1 to C22
        ("M1", 
         [(1, 0, 2, 0, 2), (1, 2, 4, 2, 4)],  # A11 + A22
         [(1, 0, 2, 0, 2), (1, 2, 4, 2, 4)],  # B11 + B22
         [(1, 0, 2, 0, 2), (1, 2, 4, 2, 4)]),  # +C11, +C22
        
        # M2 = (A21+A22)*B11
        # Contributes: +M2 to C21, -M2 to C22
        ("M2", 
         [(1, 2, 4, 0, 2), (1, 2, 4, 2, 4)],  # A21 + A22
         [(1, 0, 2, 0, 2)],                    # B11
         [(1, 2, 4, 0, 2), (-1, 2, 4, 2, 4)]), # +C21, -C22
        
        # M3 = A11*(B12-B22)
        # Contributes: +M3 to C12, +M3 to C22
        ("M3", 
         [(1, 0, 2, 0, 2)],                    # A11
         [(1, 0, 2, 2, 4), (-1, 2, 4, 2, 4)],  # B12 - B22
         [(1, 0, 2, 2, 4), (1, 2, 4, 2, 4)]),  # +C12, +C22
        
        # M4 = A22*(B21-B11)
        # Contributes: +M4 to C11, +M4 to C21
        ("M4", 
         [(1, 2, 4, 2, 4)],                    # A22
         [(1, 2, 4, 0, 2), (-1, 0, 2, 0, 2)],  # B21 - B11
         [(1, 0, 2, 0, 2), (1, 2, 4, 0, 2)]),  # +C11, +C21
        
        # M5 = (A11+A12)*B22
        # Contributes: -M5 to C11, +M5 to C12
        ("M5", 
         [(1, 0, 2, 0, 2), (1, 0, 2, 2, 4)],  # A11 + A12
         [(1, 2, 4, 2, 4)],                    # B22
         [(-1, 0, 2, 0, 2), (1, 0, 2, 2, 4)]), # -C11, +C12
        
        # M6 = (A21-A11)*(B11+B12)
        # Contributes: +M6 to C22
        ("M6", 
         [(1, 2, 4, 0, 2), (-1, 0, 2, 0, 2)],  # A21 - A11
         [(1, 0, 2, 0, 2), (1, 0, 2, 2, 4)],   # B11 + B12
         [(1, 2, 4, 2, 4)]),                    # +C22
        
        # M7 = (A12-A22)*(B21+B22)
        # Contributes: +M7 to C11
        ("M7", 
         [(1, 0, 2, 2, 4), (-1, 2, 4, 2, 4)],  # A12 - A22
         [(1, 2, 4, 0, 2), (1, 2, 4, 2, 4)],   # B21 + B22
         [(1, 0, 2, 0, 2)]),                    # +C11
    ]
    
    # Now for each Strassen M term, expand into 4 scalar multiplication terms.
    # Each M is a 2x2 matrix. M[r,c] = sum over k of L_A[r,k] * L_B[k,c].
    # For 2x2 blocks, r and k are in {0,1} within the block.
    
    # I need to track which global C entries each scalar product contributes to.
    
    for name, a_blocks, b_blocks, c_contribs in strassen_terms:
        # Get the full 2x2 linear forms
        L_A = a_linear_form(a_blocks)  # dict of U indices
        L_B = b_linear_form(b_blocks)  # dict of V indices
        
        # For each scalar product in the 2x2 matrix multiplication:
        # M[br, bc] = sum_{bk=0}^{1} L_A[br, bk] * L_B[bk, bc]
        # where br, bk, bc are local 2x2 block indices.
        
        # The local (br, bk) position in L_A maps to specific global A entries.
        # Similarly for L_B and the result.
        
        # This is getting complex. Let me use a different approach:
        # Directly compute the tensor entries by expanding all the linear forms.
        
        # For each (br, bk, bc) in the 2x2 matmul:
        # The scalar product is: (sum of A entries in L_A at local pos (br,bk)) * (sum of B entries in L_B at local pos (bk,bc))
        # This contributes to C entries according to c_contribs.
        
        # Actually, let me just build the U, V, W vectors for each scalar term directly.
        
        for br in range(2):
            for bk in range(2):
                for bc in range(2):
                    # Get the A coefficients for local position (br, bk) in the 2x2 block structure
                    # The 2x2 block structure: local row br means global rows depend on which block
                    # This is getting complicated. Let me use a simpler approach.
                    
                    pass  # We'll handle this differently below
    
    # Let me take a simpler, more direct approach.
    # I'll enumerate all scalar multiplications explicitly.
    
    # Clear and redo
    terms.clear()
    
    # For the 4x4x4 part, use Strassen.
    # Each Strassen M term involves multiplying two 2x2 matrices.
    # A 2x2 matrix product C = A*B has 4 entries:
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    
    # So each M term (which is a 2x2 matmul) gives 4 scalar multiplications.
    # 7 terms * 4 = 28 scalar mults for 4x4x4.
    
    # Let me define the global index mappings for each local position.
    
    # Block row/col to global row/col:
    # Block (0,0) = global (0,0) to (1,1)
    # Block (0,1) = global (0,2) to (1,3)
    # Block (1,0) = global (2,0) to (3,1)
    # Block (1,1) = global (2,2) to (3,3)
    
    # For a local 2x2 block at block-row br, block-col bc:
    # Global row = br*2 + local_row
    # Global col = bc*2 + local_col
    
    # A linear form like "A11 + A22" means:
    # For local position (lr, lc), the coefficient comes from:
    #   global A[0*2+lr, 0*2+lc] (from A11) + global A[1*2+lr, 1*2+lc] (from A22)
    # i.e., A[lr, lc] + A[2+lr, 2+lc]
    
    # So for a linear form defined by blocks [(sign, b1r, b1c), (sign2, b2r, b2c), ...]:
    # At local position (lr, lc), the U coefficient is:
    #   sum over each block (s, br, bc): s * 1 at global index (br*2+lr)*m + (bc*2+lc)
    
    # Let me restructure. Define linear forms as lists of (sign, block_row, block_col).
    
    def make_a_linear_form(block_specs):
        """
        block_specs: list of (sign, block_row, block_col)
        Returns a function that given (local_row, local_col) returns dict of {global_U_idx: coeff}
        """
        def get_coeff(local_row, local_col):
            d = {}
            for sign, b_row, b_col in block_specs:
                global_i = b_row * 2 + local_row
                global_j = b_col * 2 + local_col
                idx = global_i * m + global_j
                d[idx] = d.get(idx, 0.0) + sign
            return d
        return get_coeff
    
    def make_b_linear_form(block_specs):
        """
        block_specs: list of (sign, block_row, block_col)
        Returns a function that given (local_row, local_col) returns dict of {global_V_idx: coeff}
        """
        def get_coeff(local_row, local_col):
            d = {}
            for sign, b_row, b_col in block_specs:
                global_j = b_row * 2 + local_row
                global_k = b_col * 2 + local_col
                idx = global_j * p + global_k
                d[idx] = d.get(idx, 0.0) + sign
            return d
        return get_coeff
    
    def make_c_contribution(block_specs):
        """
        block_specs: list of (sign, block_row, block_col)
        Returns a function that given (local_row, local_col) returns dict of {global_W_idx: coeff}
        """
        def get_coeff(local_row, local_col):
            d = {}
            for sign, b_row, b_col in block_specs:
                global_k = b_row * 2 + local_row
                global_i = b_col * 2 + local_col
                idx = global_k * n + global_i
                d[idx] = d.get(idx, 0.0) + sign
            return d
        return get_coeff
    
    # Define the 7 Strassen terms with block-level specifications
    # Block indices: (0,0)=top-left, (0,1)=top-right, (1,0)=bottom-left, (1,1)=bottom-right
    
    S = [
        # (a_blocks, b_blocks, c_blocks)
        # M1 = (A11+A22)*(B11+B22), contributes +M1 to C11 and C22
        ([(1, 0, 0), (1, 1, 1)], [(1, 0, 0), (1, 1, 1)], [(1, 0, 0), (1, 1, 1)]),
        
        # M2 = (A21+A22)*B11, contributes +M2 to C21, -M2 to C22
        ([(1, 1, 0), (1, 1, 1)], [(1, 0, 0)], [(1, 1, 0), (-1, 1, 1)]),
        
        # M3 = A11*(B12-B22), contributes +M3 to C12 and C22
        ([(1, 0, 0)], [(1, 0, 1), (-1, 1, 1)], [(1, 0, 1), (1, 1, 1)]),
        
        # M4 = A22*(B21-B11), contributes +M4 to C11 and C21
        ([(1, 1, 1)], [(1, 1, 0), (-1, 0, 0)], [(1, 0, 0), (1, 1, 0)]),
        
        # M5 = (A11+A12)*B22, contributes -M5 to C11, +M5 to C12
        ([(1, 0, 0), (1, 0, 1)], [(1, 1, 1)], [(-1, 0, 0), (1, 0, 1)]),
        
        # M6 = (A21-A11)*(B11+B12), contributes +M6 to C22
        ([(1, 1, 0), (-1, 0, 0)], [(1, 0, 0), (1, 0, 1)], [(1, 1, 1)]),
        
        # M7 = (A12-A22)*(B21+B22), contributes +M7 to C11
        ([(1, 0, 1), (-1, 1, 1)], [(1, 1, 0), (1, 1, 1)], [(1, 0, 0)]),
    ]
    
    for a_blocks, b_blocks, c_blocks in S:
        a_func = make_a_linear_form(a_blocks)
        b_func = make_b_linear_form(b_blocks)
        c_func = make_c_contribution(c_blocks)
        
        # For each scalar multiplication in the 2x2 matmul:
        for lr in range(2):      # result row (local)
            for lk in range(2):  # inner index
                for lc in range(2):  # result col (local)
                    # This scalar product contributes to result[lr, lc]
                    # A part: local position (lr, lk)
                    # B part: local position (lk, lc)
                    u_dict = a_func(lr, lk)
                    v_dict = b_func(lk, lc)
                    w_dict = c_func(lr, lc)
                    
                    add_term(u_dict, v_dict, w_dict)
    
    # --- Part 2: Schoolbook for remaining parts ---
    # C[i, 4] = sum_{j=0}^{4} A[i, j] * B[j, 4]  for i=0..3
    # This is A[0:4, 0:5] * B[0:5, 4:5], a 4x5 * 5x1 product = 20 scalar mults
    
    for i in range(4):
        for j in range(5):
            u_dict = {i * m + j: 1.0}
            v_dict = {j * p + 4: 1.0}
            w_dict = {4 * n + i: 1.0}
            add_term(u_dict, v_dict, w_dict)
    
    # C[4, k] = sum_{j=0}^{4} A[4, j] * B[j, k]  for k=0..4
    # This is A[4:5, 0:5] * B[0:5, 0:5], a 1x5 * 5x5 product = 25 scalar mults
    
    for j in range(5):
        for k in range(5):
            u_dict = {4 * m + j: 1.0}
            v_dict = {j * p + k: 1.0}
            w_dict = {k * n + 4: 1.0}
            add_term(u_dict, v_dict, w_dict)
    
    # Total rank = 28 + 20 + 25 = 73
    rank = len(terms)
    
    # Build the matrices
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for c, (u, v, w) in enumerate(terms):
        U[:, c] = u
        V[:, c] = v
        W[:, c] = w
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

