import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We will generate the decomposition for 4x4 * 4x4 using Strassen (rank 49)
    # and apply it twice for the two 4x4 blocks of the 4x8 matrix.
    
    # Helper to get flat index for A (n x m)
    def idx_A(i, j):
        return i * m + j
        
    # Helper to get flat index for B (m x p)
    # B is split into B_L (m x p/2) and B_R (m x p/2)
    # We will handle B_L and B_R separately or just map indices.
    # Let's create a generic function to get terms for N x N * N x N
    # where N is a power of 2.
    
    terms = [] # List of (u_vec, v_vec, w_vec) for one 4x4*4x4 multiplication
    
    def strassen_terms(N, offset_A=0, offset_B=0, offset_C=0):
        """
        Generates terms for N x N * N x N multiplication.
        offset_A, offset_B, offset_C are offsets into the global vectors.
        u_vec, v_vec, w_vec will be dictionaries {flat_index: coeff}
        """
        if N == 1:
            # 1x1 * 1x1
            # A[0,0] * B[0,0] -> C[0,0]
            u = {offset_A: 1}
            v = {offset_B: 1}
            w = {offset_C: 1}
            return [(u, v, w)]
        
        half = N // 2
        # Blocks:
        # A00, A01
        # A10, A11
        # Offsets:
        # A00: offset_A
        # A01: offset_A + half*N
        # A10: offset_A + half*half
        # A11: offset_A + half*half + half*N
        
        # B00, B01
        # B10, B11
        # Offsets similar for B
        
        # C00, C01
        # C10, C11
        
        # Strassen Multiplications:
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B01 - B11)
        # M4 = A11 * (B10 - B00)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        
        # We need to combine terms from recursive calls.
        # Each "multiplication" like (A00+A11)*(B00+B11) is not a single rank-1 term
        # unless we expand it. But Strassen's trick is that these are the scalar multiplications.
        # Wait, the recursion should return the rank-1 terms for the sub-problems.
        
        # Actually, the standard way to generate the tensor decomposition from Strassen
        # is that each of the 7 "products" in the Strassen formula is a matrix product
        # of size N/2. So we recurse for each of the 7 products.
        
        # But the linear forms in A and B are sums of blocks.
        # So we need to add the linear forms.
        
        # Let's define a function to get the linear form (dict) for a block sum.
        # e.g. A00 + A11 means for each element in the N/2 x N/2 block, coeff is 1.
        
        def get_block_lin_form(base_offset, block_row, block_col, size, coeff=1):
            """
            Returns a dict {global_index: coeff} for a block of size size x size
            starting at base_offset, located at block_row, block_col in the N x N matrix.
            """
            d = {}
            for i in range(size):
                for j in range(size):
                    # Global row = block_row * size + i
                    # Global col = block_col * size + j
                    # Flat index = (block_row * size + i) * N + (block_col * size + j)
                    # But wait, the offset is into the vector of size N*N.
                    # The vector index is simply (global_row * N + global_col).
                    # The offset shifts this.
                    gr = block_row * size + i
                    gc = block_col * size + j
                    idx = base_offset + gr * N + gc
                    d[idx] = d.get(idx, 0) + coeff
            return d

        # We need to be careful: the "N" in the index calculation above is the current N.
        # The offsets are relative to the start of the current N x N block in the global vector.
        
        # Let's refine get_block_lin_form to use current N for indexing.
        
        def combine_lin_forms(d1, d2, sign1=1, sign2=1):
            d = {}
            for k, v in d1.items():
                d[k] = d.get(k, 0) + sign1 * v
            for k, v in d2.items():
                d[k] = d.get(k, 0) + sign2 * v
            return d
        
        # Recurse for the 7 products.
        # Each product is a matrix multiplication of size half.
        # We need to pass the linear forms of the factors to the recursive call?
        # No, the recursive call generates terms for the *product* of two matrices.
        # But here the factors are *linear combinations* of blocks.
        
        # Alternative approach:
        # The Strassen decomposition for N x N can be built by taking the
        # Strassen decomposition for N/2 x N/2 and "lifting" it.
        
        # Let's just generate the 49 terms explicitly for N=4.
        # It's easier to hardcode the structure for N=4 based on N=2 Strassen.
        
        pass # Placeholder, will implement explicit N=4 generation
        
    # Explicit generation for N=4
    # A is 4x4. B is 4x4. C is 4x4.
    # We use the 7 Strassen products for the 2x2 block structure.
    # Each product is a 2x2 * 2x2 multiplication.
    # We use Strassen for the 2x2 * 2x2 as well (rank 7).
    
    # Let's define the 2x2 Strassen terms.
    # Input: vectors of length 4 (A2, B2). Output: vector of length 4 (C2).
    # Terms: list of (u2, v2, w2) where u2, v2, w2 are dicts {idx: coeff}
    
    def strassen_2x2_terms():
        # Indices: 0:00, 1:01, 2:10, 3:11
        terms = []
        
        # M1 = (A00+A11)(B00+B11)
        u1 = {0:1, 3:1}
        v1 = {0:1, 3:1}
        # C00 += M1, C11 += M1
        # We need to distribute M1 to C.
        # Actually, the term is u1 * v1. The result is added to C.
        # In tensor decomposition, each rank-1 term contributes to C via w.
        # So we need to determine w for each term.
        
        # The Strassen algorithm computes:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # So for each M_k, we have a linear form in A (u_k) and B (v_k).
        # And we know which C elements it contributes to and with what coefficient.
        
        # M1: u={0:1, 3:1}, v={0:1, 3:1}
        # Contributes to C00 (+1), C11 (+1)
        # So we need TWO rank-1 terms for M1?
        # No, the tensor term is u \otimes v \otimes w.
        # If M1 contributes to C00 and C11, we can have one term with w having non-zero entries at C00 and C11.
        # Yes, w is a vector in C-space.
        
        # So for M1:
        # u = {0:1, 3:1}
        # v = {0:1, 3:1}
        # w = {0:1, 3:1}  # C00 is idx 0, C11 is idx 3
        
        terms.append(({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}))
        
        # M2 = (A10+A11)B00
        # u = {2:1, 3:1}, v = {0:1}
        # Contributes to C10 (+1)
        # C10 is idx 2.
        terms.append(({2:1, 3:1}, {0:1}, {2:1}))
        
        # M3 = A00(B01-B11)
        # u = {0:1}, v = {1:1, 3:-1}
        # Contributes to C01 (+1), C11 (+1)
        # C01 is idx 1, C11 is idx 3.
        terms.append(({0:1}, {1:1, 3:-1}, {1:1, 3:1}))
        
        # M4 = A11(B10-B00)
        # u = {3:1}, v = {2:1, 0:-1}
        # Contributes to C00 (+1), C10 (+1)
        # C00 is idx 0, C10 is idx 2.
        terms.append(({3:1}, {2:1, 0:-1}, {0:1, 2:1}))
        
        # M5 = (A00+A01)B11
        # u = {0:1, 1:1}, v = {3:1}
        # Contributes to C00 (-1), C01 (+1)
        # C00 is idx 0, C01 is idx 1.
        terms.append(({0:1, 1:1}, {3:1}, {0:-1, 1:1}))
        
        # M6 = (A10-A00)(B00+B01)
        # u = {2:1, 0:-1}, v = {0:1, 1:1}
        # Contributes to C11 (+1)
        # C11 is idx 3.
        terms.append(({2:1, 0:-1}, {0:1, 1:1}, {3:1}))
        
        # M7 = (A01-A11)(B10+B11)
        # u = {1:1, 3:-1}, v = {2:1, 3:1}
        # Contributes to C00 (+1)
        # C00 is idx 0.
        terms.append(({1:1, 3:-1}, {2:1, 3:1}, {0:1}))
        
        return terms

    # Now, for 4x4, we have 7 block products.
    # Each block product is a 2x2 * 2x2 multiplication.
    # We use the 7 terms from strassen_2x2_terms() for each block product.
    # Total 49 terms.
    
    # We need to map the 2x2 indices to 4x4 indices.
    # Block A00 (top-left 2x2):
    #   0->0, 1->1, 2->4, 3->5
    # Block A01 (top-right 2x2):
    #   0->2, 1->3, 2->6, 3->7
    # Block A10 (bottom-left 2x2):
    #   0->8, 1->9, 2->12, 3->13
    # Block A11 (bottom-right 2x2):
    #   0->10, 1->11, 2->14, 3->15
    
    # Similar for B and C.
    
    def map_2x2_to_4x4(idx_2x2, block_row, block_col):
        # idx_2x2 is 0..3
        # block_row, block_col are 0..1
        r = (idx_2x2 // 2) + block_row * 2
        c = (idx_2x2 % 2) + block_col * 2
        return r * 4 + c

    def expand_block_term(u_2x2, v_2x2, w_2x2, block_A, block_B, block_C):
        # block_A, block_B, block_C are (row, col) in the 2x2 block structure
        u_4x4 = {}
        for idx, coeff in u_2x2.items():
            global_idx = map_2x2_to_4x4(idx, block_A[0], block_A[1])
            u_4x4[global_idx] = u_4x4.get(global_idx, 0) + coeff
            
        v_4x4 = {}
        for idx, coeff in v_2x2.items():
            global_idx = map_2x2_to_4x4(idx, block_B[0], block_B[1])
            v_4x4[global_idx] = v_4x4.get(global_idx, 0) + coeff
            
        w_4x4 = {}
        for idx, coeff in w_2x2.items():
            global_idx = map_2x2_to_4x4(idx, block_C[0], block_C[1])
            w_4x4[global_idx] = w_4x4.get(global_idx, 0) + coeff
            
        return (u_4x4, v_4x4, w_4x4)

    base_terms = strassen_2x2_terms()
    
    # Strassen block definitions for 4x4:
    # M1: (A00+A11) * (B00+B11) -> C00, C11
    # This is NOT a single block multiplication.
    # Wait, my previous logic was flawed.
    # Strassen for 4x4:
    # We define 7 products of 2x2 matrices.
    # P1 = (A00+A11) * (B00+B11)
    # P2 = (A10+A11) * B00
    # ...
    # Each Pk is a 2x2 matrix product.
    # So for P1, the "A" part is A00+A11, "B" part is B00+B11.
    # This is not a standard block. It's a sum of blocks.
    # So we cannot just map a single 2x2 block.
    
    # We need to handle the linear combinations of blocks.
    # For P1:
    # u is linear form on A00 and A11.
    # v is linear form on B00 and B11.
    # w is linear form on C00 and C11.
    
    # Let's construct the 49 terms by iterating over the 7 Strassen products
    # and for each, iterating over the 7 terms of the inner 2x2 Strassen.
    
    # For a given Strassen product Pk:
    # It involves specific blocks of A and B.
    # e.g. P1 involves A00, A11 and B00, B11.
    # The inner Strassen terms operate on "elements" which are 2x2 blocks.
    # So term 1 of P1 (which is (A_elem00 + A_elem11) * (B_elem00 + B_elem11))
    # corresponds to:
    # u_inner = {0:1, 3:1} (indices in the 2x2 "super-matrix")
    # v_inner = {0:1, 3:1}
    # w_inner = {0:1, 3:1}
    
    # We need to expand these "super-indices" to actual 4x4 indices.
    # Super-index 0 corresponds to A00.
    # Super-index 3 corresponds to A11.
    # So u_global will have contributions from all elements in A00 and A11.
    
    def expand_super_term(u_super, v_super, w_super, 
                          A_blocks, B_blocks, C_blocks):
        """
        A_blocks: list of (row, col) for the A blocks involved in this super-term.
                  Must correspond to super-indices 0,1,2,3.
        """
        u_global = {}
        v_global = {}
        w_global = {}
        
        # Map super-index to block
        # super 0 -> A_blocks[0], 1 -> A_blocks[1], 2 -> A_blocks[2], 3 -> A_blocks[3]
        
        for s_idx, coeff in u_super.items():
            if s_idx < 0 or s_idx > 3: continue
            block_row, block_col = A_blocks[s_idx]
            # Expand 2x2 block
            for i in range(2):
                for j in range(2):
                    local_idx = i * 2 + j
                    global_idx = map_2x2_to_4x4(local_idx, block_row, block_col)
                    u_global[global_idx] = u_global.get(global_idx, 0) + coeff
                    
        for s_idx, coeff in v_super.items():
            if s_idx < 0 or s_idx > 3: continue
            block_row, block_col = B_blocks[s_idx]
            for i in range(2):
                for j in range(2):
                    local_idx = i * 2 + j
                    global_idx = map_2x2_to_4x4(local_idx, block_row, block_col)
                    v_global[global_idx] = v_global.get(global_idx, 0) + coeff
                    
        for s_idx, coeff in w_super.items():
            if s_idx < 0 or s_idx > 3: continue
            block_row, block_col = C_blocks[s_idx]
            for i in range(2):
                for j in range(2):
                    local_idx = i * 2 + j
                    global_idx = map_2x2_to_4x4(local_idx, block_row, block_col)
                    w_global[global_idx] = w_global.get(global_idx, 0) + coeff
                    
        return (u_global, v_global, w_global)

    # Define the 7 Strassen products for 4x4
    # A blocks: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B blocks: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # C blocks: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # P1: (A0+A3) * (B0+B3) -> C0, C3
    # Super u: {0:1, 3:1} (A0, A3)
    # Super v: {0:1, 3:1} (B0, B3)
    # Super w: {0:1, 3:1} (C0, C3)
    # A_blocks for this product: [ (0,0), None, None, (1,1) ] ?
    # No, the super-index 0 refers to the first block in the list.
    # We need to define the mapping from super-index to block for each product.
    
    # P1: A uses blocks 0, 3. B uses blocks 0, 3. C uses blocks 0, 3.
    # Super 0 -> A0, Super 3 -> A3.
    # So A_blocks = [ (0,0), (0,0), (0,0), (1,1) ] ? 
    # It's easier to just pass the specific blocks.
    
    # Let's define a helper to get the blocks for a specific Strassen product.
    
    products = [
        # (u_super, v_super, w_super, A_blocks, B_blocks, C_blocks)
        # P1
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}, 
         [(0,0), (0,0), (0,0), (1,1)], 
         [(0,0), (0,0), (0,0), (1,1)], 
         [(0,0), (0,0), (0,0), (1,1)]),
        # P2: (A2+A3) * B0 -> C2
        ({2:1, 3:1}, {0:1}, {2:1},
         [(0,0), (0,0), (1,0), (1,1)],
         [(0,0), (0,0), (0,0), (0,0)],
         [(0,0), (0,0), (1,0), (0,0)]),
        # P3: A0 * (B1-B3) -> C1, C3
        ({0:1}, {1:1, 3:-1}, {1:1, 3:1},
         [(0,0), (0,0), (0,0), (0,0)],
         [(0,0), (0,1), (0,0), (1,1)],
         [(0,0), (0,1), (0,0), (1,1)]),
        # P4: A3 * (B2-B0) -> C0, C2
        ({3:1}, {2:1, 0:-1}, {0:1, 2:1},
         [(0,0), (0,0), (0,0), (1,1)],
         [(0,0), (0,0), (1,0), (0,0)],
         [(0,0), (0,0), (1,0), (0,0)]),
        # P5: (A0+A1) * B3 -> C0(-1), C1(1)
        ({0:1, 1:1}, {3:1}, {0:-1, 1:1},
         [(0,0), (0,1), (0,0), (0,0)],
         [(0,0), (0,0), (0,0), (1,1)],
         [(0,0), (0,1), (0,0), (0,0)]),
        # P6: (A2-A0) * (B0+B1) -> C3
        ({2:1, 0:-1}, {0:1, 1:1}, {3:1},
         [(0,0), (0,0), (1,0), (0,0)],
         [(0,0), (0,1), (0,0), (0,0)],
         [(0,0), (0,0), (0,0), (1,1)]),
        # P7: (A1-A3) * (B2+B3) -> C0
        ({1:1, 3:-1}, {2:1, 3:1}, {0:1},
         [(0,0), (0,1), (0,0), (1,1)],
         [(0,0), (0,0), (1,0), (1,1)],
         [(0,0), (0,0), (0,0), (0,0)])
    ]
    
    all_terms = []
    for u_s, v_s, w_s, Ab, Bb, Cb in products:
        for u_2, v_2, w_2 in base_terms:
            term = expand_super_term(u_2, v_2, w_2, Ab, Bb, Cb)
            all_terms.append(term)
            
    # Now we have 49 terms for 4x4 * 4x4.
    # We need to apply this to B_L and B_R.
    
    U = [0] * (n * m * 2) # Wait, U is for A (4x4). Size 16.
    # U shape: (n*m, rank) = (16, 98)
    # V shape: (m*p, rank) = (32, 98)
    # W shape: (p*n, rank) = (32, 98)
    
    rank = 98
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill U, V, W for B_L (cols 0..3 of B)
    for r in range(49):
        u, v, w = all_terms[r]
        for idx, coeff in u.items():
            U[idx][r] += coeff
        for idx, coeff in v.items():
            # v idx is 0..15 for B_L
            # B_L is the left half of B (4x8).
            # B_L[j, k] where k in 0..3.
            # Flat index in B_L: j*4 + k.
            # Flat index in B (4x8): j*8 + k.
            # So V index is j*8 + k.
            # Our v idx is j*4 + k.
            # We need to map j*4+k to j*8+k.
            j = idx // 4
            k = idx % 4
            V_idx = j * 8 + k
            V[V_idx][r] += coeff
        for idx, coeff in w.items():
            # w idx is 0..15 for C_L (4x4).
            # C_L[i, k] where k in 0..3.
            # W index is k*n + i = k*4 + i.
            # Our w idx is i*4 + k.
            # Map i*4+k to k*4+i.
            i = idx // 4
            k = idx % 4
            W_idx = k * 4 + i
            W[W_idx][r] += coeff
            
    # Fill U, V, W for B_R (cols 4..7 of B)
    for r in range(49):
        u, v, w = all_terms[r]
        col_offset = 49
        for idx, coeff in u.items():
            U[idx][r + col_offset] += coeff
        for idx, coeff in v.items():
            j = idx // 4
            k = idx % 4
            # B_R is right half. k in 4..7.
            # B[j, k_global] where k_global = k + 4.
            V_idx = j * 8 + (k + 4)
            V[V_idx][r + col_offset] += coeff
        for idx, coeff in w.items():
            i = idx // 4
            k = idx % 4
            # C_R[i, k_global] where k_global = k + 4.
            # W index is k_global*4 + i.
            W_idx = (k + 4) * 4 + i
            W[W_idx][r + col_offset] += coeff
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
