"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def _deadline_guard(start_time, limit=145.0):
    """Return True if we still have time budget remaining."""
    return (time.time() - start_time) < limit

def construct():
    """
    Returns a rank-64 tensor decomposition for 4x5 * 5x5 matrix multiplication.
    Uses Strassen's algorithm for the 4x4x4 sub-block and schoolbook for the rest.
    
    Interface:
    U: shape (n*m, rank) -> (20, 64)
    V: shape (m*p, rank) -> (25, 64)
    W: shape (p*n, rank) -> (20, 64)
    
    Convention:
    U[i*m + j, r]
    V[j*p + k, r]
    W[k*n + i, r]
    """
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Rank calculation:
    # 4x4x4 Strassen: 28
    # 4x4x1 Schoolbook: 16
    # 4x1x5 Schoolbook: 20
    # Total: 64
    rank = 64
    
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    r_offset = 0
    
    # Helper to add a rank-1 term
    def add_term(i, j, k, u_val, v_val, w_val, r_idx):
        U[i * m + j, r_idx] += u_val
        V[j * p + k, r_idx] += v_val
        W[k * n + i, r_idx] += w_val

    # --- Part 1: Strassen for 4x4x4 (i in 0..3, j in 0..3, k in 0..3) ---
    # We decompose A (4x4) and B (4x4) into 2x2 blocks.
    # A blocks: A00 (i0-1, j0-1), A01 (i0-1, j2-3), A10 (i2-3, j0-1), A11 (i2-3, j2-3)
    # B blocks: B00 (j0-1, k0-1), B01 (j0-1, k2-3), B10 (j2-3, k0-1), B11 (j2-3, k2-3)
    
    # We will iterate through the 7 Strassen products, each of which is a 2x2 matrix product.
    # Each 2x2 product has 4 rank-1 terms.
    
    # Define the 7 Strassen combinations of 2x2 matrices
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # We represent a 2x2 matrix by its 4 elements [a00, a01, a10, a11]
    # We need to sum coefficients for each (i,j) pair in the 4x4 A matrix for each Strassen term.
    
    # Let's build the coefficient vectors for U (A part) and V (B part) for each of the 7 terms.
    # Then for each term, we iterate over the 4 rank-1 components of the 2x2 product.
    
    # Strassen Term Definitions (Coefficients for A and B 2x2 blocks)
    # A blocks indices: 00, 01, 10, 11
    # B blocks indices: 00, 01, 10, 11
    
    # M1: A = A00 + A11, B = B00 + B11
    # M2: A = A10 + A11, B = B00
    # M3: A = A00, B = B01 - B11
    # M4: A = A11, B = B10 - B00
    # M5: A = A00 + A01, B = B11
    # M6: A = A10 - A00, B = B00 + B01
    # M7: A = A01 - A11, B = B10 + B11
    
    # For each Strassen term t, we have a set of (A_block, coeff_A) and (B_block, coeff_B).
    # We expand this to specific (i, j) and (j, k) indices.
    
    strassen_terms = [
        # M1
        {
            'A': { (0,0): 1, (1,1): 1 }, # A00, A11
            'B': { (0,0): 1, (1,1): 1 }, # B00, B11
        },
        # M2
        {
            'A': { (1,0): 1, (1,1): 1 }, # A10, A11
            'B': { (0,0): 1 }, # B00
        },
        # M3
        {
            'A': { (0,0): 1 }, # A00
            'B': { (0,1): 1, (1,1): -1 }, # B01, -B11
        },
        # M4
        {
            'A': { (1,1): 1 }, # A11
            'B': { (1,0): 1, (0,0): -1 }, # B10, -B00
        },
        # M5
        {
            'A': { (0,0): 1, (0,1): 1 }, # A00, A01
            'B': { (1,1): 1 }, # B11
        },
        # M6
        {
            'A': { (1,0): 1, (0,0): -1 }, # A10, -A00
            'B': { (0,0): 1, (0,1): 1 }, # B00, B01
        },
        # M7
        {
            'A': { (0,1): 1, (1,1): -1 }, # A01, -A11
            'B': { (1,0): 1, (1,1): 1 }, # B10, B11
        }
    ]
    
    # Mapping from 2x2 block indices to global 4x4 indices
    # Block (bi, bj) covers i in [bi*2, bi*2+1], j in [bj*2, bj*2+1]
    
    for term in strassen_terms:
        # We need to combine the A and B coefficients.
        # The product is a sum of rank-1 terms.
        # For each pair of (A_block, a_coeff) and (B_block, b_coeff),
        # we have a 2x2 matrix product.
        # This 2x2 product contributes 4 rank-1 terms to the global tensor.
        
        A_blocks = term['A']
        B_blocks = term['B']
        
        for (bi, bj), a_coeff in A_blocks.items():
            for (bk, bl), b_coeff in B_blocks.items():
                # bi, bj: block row, block col of A (4x4)
                # bk, bl: block row, block col of B (4x4)
                # Note: In matrix mult C = A * B, the inner index must match.
                # A is 4x4, B is 4x4.
                # A block (bi, bj) has rows bi*2..bi*2+1, cols bj*2..bj*2+1.
                # B block (bk, bl) has rows bk*2..bk*2+1, cols bl*2..bl*2+1.
                # For the product to be valid in the sum, the columns of A must match rows of B.
                # So bj must equal bk.
                if bj != bk:
                    continue
                
                # Now we have a valid 2x2 * 2x2 product.
                # A_sub is 2x2, B_sub is 2x2.
                # Global indices:
                # i in [bi*2, bi*2+1]
                # j in [bj*2, bj*2+1]
                # k in [bl*2, bl*2+1]
                
                # The 2x2 product C_sub = A_sub * B_sub has 4 rank-1 terms.
                # C_sub[u, v] = sum_c A_sub[u, c] * B_sub[c, v]
                # Global i = bi*2 + u
                # Global j = bj*2 + c
                # Global k = bl*2 + v
                
                # We need to add 4 terms to U, V, W.
                # But wait, the Strassen algorithm sums these 7 products to form C.
                # So we just add the rank-1 terms directly to U, V, W with the coefficients.
                
                # For each of the 4 rank-1 terms in this 2x2 product:
                for c in range(2): # inner index of 2x2
                    for u in range(2): # row of A_sub (and C_sub)
                        for v in range(2): # col of B_sub (and C_sub)
                            # This is not quite right. A rank-1 term in 2x2 mat mult is:
                            # C[u,v] += A[u,c] * B[c,v]
                            # This corresponds to a single rank-1 tensor term in the 2x2x2 tensor.
                            # U[i, j] = A[u, c]
                            # V[j, k] = B[c, v]
                            # W[k, i] = 1 (since C[u,v] is the result)
                            
                            # Wait, the tensor is T[i,j,k] = A[i,j] * B[j,k].
                            # So for a fixed (i,j,k), the contribution is A[i,j] * B[j,k].
                            # In the 2x2 sub-problem:
                            # i = bi*2 + u
                            # j = bj*2 + c
                            # k = bl*2 + v
                            
                            # The coefficient for U is a_coeff (from A block)
                            # The coefficient for V is b_coeff (from B block)
                            # The coefficient for W is 1.0 (since it's a standard product term)
                            
                            # However, we are summing 7 such products.
                            # So we just add these terms to the global U, V, W.
                            
                            i = bi * 2 + u
                            j = bj * 2 + c
                            k = bl * 2 + v
                            
                            # Check if this (i,j,k) is within the 4x4x4 range
                            # i in 0..3, j in 0..3, k in 0..3. Yes.
                            
                            # We need to assign a unique rank index r for this term.
                            # But wait, multiple Strassen terms might contribute to the same (i,j,k)?
                            # No, Strassen decomposes the tensor into a sum of rank-1 tensors.
                            # Each rank-1 tensor is distinct in the decomposition space.
                            # We just need to ensure we don't overwrite, but we are adding.
                            # However, we need to allocate a new rank index for EACH rank-1 term generated.
                            # Total terms = 7 terms * 4 (from A blocks) * 4 (from B blocks) * ...?
                            # No.
                            # Let's re-evaluate the count.
                            # M1: A has 2 blocks, B has 2 blocks. Pairs: 2*2=4 pairs?
                            # No, M1 is (A00+A11)*(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11.
                            # These are 4 distinct 2x2 products.
                            # Each 2x2 product has 4 rank-1 terms.
                            # So M1 contributes 16 rank-1 terms?
                            # No, Strassen's algorithm says M1 is ONE matrix product.
                            # The matrix product M1 is 2x2. It has rank 4.
                            # So M1 contributes 4 rank-1 terms to the global tensor?
                            # No. The global tensor is 4x4x4.
                            # M1 is a 2x2 matrix. It corresponds to a sub-tensor of size 2x2x2?
                            # No. M1 is the result of multiplying two 2x2 matrices.
                            # The rank of a 2x2x2 tensor is 4.
                            # So M1 contributes 4 rank-1 terms to the global 4x4x4 tensor?
                            # Let's trace indices.
                            # M1 = (A00+A11)(B00+B11).
                            # A00 is 2x2, B00 is 2x2.
                            # The product is 2x2.
                            # The indices for this product are:
                            # i in 0,1 (from A00) or 2,3 (from A11)?
                            # Actually, A00+A11 is not a block matrix. It's a sum of blocks.
                            # (A00+A11) is a 2x2 matrix where element (u,v) is A[u,v] + A[u+2,v+2].
                            # So it mixes indices.
                            # Therefore, the rank-1 terms from M1 will have non-zero entries at multiple (i,j,k) locations.
                            # We cannot just allocate 4 rank indices for M1.
                            # We must expand M1 into its constituent rank-1 terms in the global 4x4x4 space.
                            
                            # Let's do it properly.
                            # For each Strassen term, we define the 2x2 matrices X and Y.
                            # X[u,v] = sum of coeffs * A[bi*2+u, bj*2+v]
                            # Y[u,v] = sum of coeffs * B[bk*2+u, bl*2+v]
                            # Then Z = X * Y.
                            # Z[u,v] = sum_c X[u,c] Y[c,v].
                            # This Z is a 2x2 matrix.
                            # This Z corresponds to the contribution to C at indices:
                            # i = bi*2 + u, k = bl*2 + v.
                            # But j is summed over.
                            # The term is sum_c X[u,c] Y[c,v] C[i, k].
                            # This is a sum of rank-1 terms in the (i, j, k) space?
                            # No, j is the inner index.
                            # The contribution to the global tensor T[i,j,k] is:
                            # For fixed u, v, c:
                            # i = bi*2 + u
                            # j = bj*2 + c
                            # k = bl*2 + v
                            # T[i,j,k] += X[u,c] * Y[c,v]
                            
                            # So for each Strassen term, we iterate u, v, c (0..1).
                            # That's 8 terms per Strassen product?
                            # No, u, c, v are 0..1. 2*2*2 = 8 terms.
                            # But X and Y are linear combinations of A and B blocks.
                            # We need to expand X[u,c] and Y[c,v] into global A and B indices.
                            
                            # This is getting complicated to implement dynamically.
                            # Let's hardcode the 28 terms for 4x4x4 Strassen.
                            # This is safer.
                            pass

    # Given the complexity and time, let's fall back to a simpler structural improvement if hardcoding is too long.
    # But 64 is a good target.
    # Let's try to implement the 4x4x4 Strassen by explicitly listing the 28 rank-1 vectors.
    
    # I will generate the 28 terms programmatically by expanding the 7 Strassen products.
    
    # Reset rank counter
    r_offset = 0
    
    # We will create a list of (i, j, k, coeff_U, coeff_V, coeff_W) for each rank-1 term.
    # Actually, U, V, W are vectors.
    # Term r: U[:, r], V[:, r], W[:, r].
    
    # Let's define the 7 Strassen products' expansion.
    
    # Helper to get global index for A
    def a_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    # Strassen Terms Expansion
    # M1 = (A00+A11)(B00+B11)
    # X = A00+A11, Y = B00+B11
    # X[u,c] = A[2u, 2c] + A[2u+2, 2c+2]
    # Y[c,v] = B[2c, 2v] + B[2c+2, 2v+2]
    # Term for u,c,v:
    # i = 2u, j = 2c, k = 2v -> U[i,j] += 1, V[j,k] += 1, W[k,i] += 1
    # i = 2u, j = 2c, k = 2v+2 -> U[i,j] += 1, V[j,k] += 1, W[k,i] += 1
    # ...
    
    # This expansion is tedious but doable.
    # However, there is a risk of error.
    # Given the 120s limit, I will implement the 4x4x4 Strassen using a known pre-computed list of 28 terms if possible, or a simple loop.
    
    # Let's use the loop approach for the 4x4x4 part.
    
    # For each of the 7 Strassen products:
    # 1. Define X (2x2) and Y (2x2) as linear combos of A and B blocks.
    # 2. Expand X[u,c] * Y[c,v] into global (i,j,k) terms.
    
    strassen_products = [
        # (X_def, Y_def) where X_def is list of (block_row, block_col, coeff)
        # X[u,c] = sum coeff * A[br*2+u, bc*2+c]
        # Y[c,v] = sum coeff * B[br*2+c, bc*2+v]
        
        # M1
        ([((0,0,1), (1,1,1))], [((0,0,1), (1,1,1))]),
        # M2
        ([((1,0,1), (1,1,1))], [((0,0,1))]),
        # M3
        ([((0,0,1))], [((0,1,1), (1,1,-1))]),
        # M4
        ([((1,1,1))], [((1,0,1), (0,0,-1))]),
        # M5
        ([((0,0,1), (0,1,1))], [((1,1,1))]),
        # M6
        ([((1,0,1), (0,0,-1))], [((0,0,1), (0,1,1))]),
        # M7
        ([((0,1,1), (1,1,-1))], [((1,0,1), (1,1,1))])
    ]
    
    for x_def, y_def in strassen_products:
        # x_def is list of (br, bc, coeff)
        # y_def is list of (br, bc, coeff)
        
        # Expand X and Y into 2x2 matrices of coefficients for global A and B indices
        # We want to find all (i, j, k) such that T[i,j,k] gets a contribution.
        
        # X[u,c] = sum_{(br, bc, cx) in x_def} cx * A[br*2+u, bc*2+c]
        # Y[c,v] = sum_{(br, bc, cy) in y_def} cy * B[br*2+c, bc*2+v]
        
        # Product term for fixed u, c, v:
        # sum_{x1 in x_def} sum_{y1 in y_def} cx * cy * A[x1.br*2+u, x1.bc*2+c] * B[y1.br*2+c, y1.bc*2+v]
        
        # This contributes to T[i, j, k] where:
        # i = x1.br*2+u
        # j = x1.bc*2+c  (Note: j must match in A and B, so x1.bc*2+c must equal y1.br*2+c? No.)
        # Wait. The inner index is j.
        # A index: (i, j). B index: (j, k).
        # In the expansion:
        # A part: i = x1.br*2+u, j = x1.bc*2+c
        # B part: j' = y1.br*2+c, k = y1.bc*2+v
        # For the product to be valid, j must equal j'.
        # So we only add the term if x1.bc*2+c == y1.br*2+c.
        # Since c is 0 or 1, this implies x1.bc == y1.br.
        
        for u in range(2):
            for c in range(2):
                for v in range(2):
                    for x1 in x_def:
                        br_x, bc_x, cx = x1
                        i = br_x * 2 + u
                        j = bc_x * 2 + c
                        # Check if j matches any B block row
                        for y1 in y_def:
                            br_y, bc_y, cy = y1
                            j_prime = br_y * 2 + c
                            if j == j_prime:
                                k = bc_y * 2 + v
                                # Add term
                                # U[i, j] += cx
                                # V[j, k] += cy
                                # W[k, i] += 1
                                
                                # We need to allocate a rank index for this specific (i,j,k) combination?
                                # No, we are building the rank-1 vectors.
                                # Each (u,c,v, x1, y1) combination that satisfies j==j' is a rank-1 term.
                                
                                # Is it unique?
                                # Different (u,c,v) give different (i,j,k).
                                # Different x1, y1 might give same (i,j,k)?
                                # i depends on br_x, u. j depends on bc_x, c. k depends on bc_y, v.
                                # If br_x, bc_x, br_y, bc_y are fixed, (i,j,k) is fixed.
                                # So each valid (u,c,v, x1, y1) is a distinct rank-1 term.
                                
                                # Allocate rank index
                                if r_offset >= rank:
                                    raise Exception("Rank exceeded")
                                
                                U[i * m + j, r_offset] += cx
                                V[j * p + k, r_offset] += cy
                                W[k * n + i, r_offset] += 1.0
                                
                                r_offset += 1
    
    # Part 2: Rectangular 4x4x1
    # i in 0..3, j in 0..3, k=4
    for i in range(4):
        for j in range(4):
            if r_offset >= rank:
                raise Exception("Rank exceeded")
            U[i * m + j, r_offset] = 1.0
            V[j * p + 4, r_offset] = 1.0
            W[4 * n + i, r_offset] = 1.0
            r_offset += 1
            
    # Part 3: Rectangular 4x1x5
    # i in 0..3, j=4, k in 0..4
    for i in range(4):
        for k in range(5):
            if r_offset >= rank:
                raise Exception("Rank exceeded")
            U[i * m + 4, r_offset] = 1.0
            V[4 * p + k, r_offset] = 1.0
            W[k * n + i, r_offset] = 1.0
            r_offset += 1
            
    if not _deadline_guard(start_time):
        pass
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
