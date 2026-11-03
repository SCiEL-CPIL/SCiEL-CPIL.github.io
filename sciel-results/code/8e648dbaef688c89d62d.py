"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin
    
    n, m, p = DIMENSIONS  # 5, 5, 5
    
    U = []
    V = []
    W = []
    
    def add_term(u_val, v_val, w_val):
        U.append(u_val)
        V.append(v_val)
        W.append(w_val)
    
    # Strategy: Use Strassen for 4x4 block (indices 0-3) and schoolbook for the rest
    # The 5x5 matrix multiplication can be decomposed into blocks:
    # A = [[A00, A01], [A10, A11]] where A00 is 4x4, A01 is 4x1, A10 is 1x4, A11 is 1x1
    # But that's not square blocks. Let's use a different approach.
    
    # Better: Treat the 5x5 multiplication as consisting of:
    # 1. A 4x4 Strassen decomposition for the top-left 4x4 block of A and B
    # 2. Schoolbook for all other terms
    
    # Actually, let's use a cleaner block decomposition:
    # Partition A into blocks A_00 (4x4), A_01 (4x1), A_10 (1x4), A_11 (1x1)
    # Partition B into blocks B_00 (4x4), B_01 (4x1), B_10 (1x4), B_11 (1x1)
    
    # The product C = A*B has blocks:
    # C_00 = A_00*B_00 + A_01*B_10  (4x4)
    # C_01 = A_00*B_01 + A_01*B_11  (4x1)
    # C_10 = A_10*B_00 + A_11*B_10  (1x4)
    # C_11 = A_10*B_01 + A_11*B_11  (1x1)
    
    # For C_00: Use Strassen on A_00*B_00 (7 terms) + schoolbook on A_01*B_10 (4 terms)
    # For C_01: Schoolbook (4*1*1 = 4 terms)
    # For C_10: Schoolbook (1*4*1 = 4 terms)
    # For C_11: Schoolbook (1*1*1 = 1 term)
    # For cross terms between blocks:
    # Actually, we need to be more careful. Let me enumerate all (i,j,k) triples.
    
    # Let me use a simpler approach: 
    # - For (i,j,k) where i,j,k all in {0,1,2,3}: use Strassen 4x4 (which uses 49 rank-1 terms)
    # - For all other (i,j,k): use schoolbook (1 rank-1 term each)
    
    # Total terms: 49 + (125 - 64) = 49 + 61 = 110
    
    # First, implement Strassen 4x4 for indices 0-3
    # Strassen 2x2 base case for 2x2 matrices
    def strassen_2x2_term_generator(offset_i, offset_j, offset_k):
        """
        Generate the 7 rank-1 terms for a 2x2 matrix multiplication.
        A[0..1][0..1] * B[0..1][0..1] = C[0..1][0..1]
        where global indices are offset_i + local_i, etc.
        
        Yields (i, j, k, coeff) for each rank-1 term contribution.
        Actually, we need to yield the full (u, v, w) vectors for each term.
        """
        terms = []
        
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11)B00
        # M3 = A00(B01 - B11)
        # M4 = A11(B10 - B00)
        # M5 = (A00 + A10)B11
        # M6 = (A01 - A10)(B10 + B11)
        # M7 = (A01 - A11)(B00 + B10)
        
        # Each Mi is a 2x2 product, decomposed into 7 rank-1 terms
        
        def gen_2x2_product(u_offsets, v_offsets, w_offsets, sign=1):
            """
            Generate 7 rank-1 terms for a 2x2 product.
            u_offsets: list of 4 indices for A matrix (row-major)
            v_offsets: list of 4 indices for B matrix (row-major)
            w_offsets: list of 4 indices for C matrix (row-major)
            sign: multiplier for all terms
            """
            a00, a01, a10, a11 = u_offsets
            b00, b01, b10, b11 = v_offsets
            c00, c01, c10, c11 = w_offsets
            
            results = []
            
            # Standard 2x2 Strassen:
            # P1 = A00 * (B00 + B11)
            # P2 = (A10 + A11) * B00
            # P3 = (A00 + A01) * B11
            # P4 = A11 * (B10 - B00)
            # P5 = (A00 + A11) * (B00 + B11)
            # P6 = (A10 - A00) * B10
            # P7 = (A01 - A11) * B01
            #
            # C00 = P5 + P4 - P3 - P1
            # C01 = P1 + P3
            # C10 = P2 + P6
            # C11 = P5 - P2 + P4 + P7
            
            # P1: A00 * (B00 + B11)
            results.append((a00, [b00, b11], [c00, c01, c10, c11], sign))
            
            # P2: (A10 + A11) * B00
            results.append(([a10, a11], b00, [c10, c11], sign))
            
            # P3: (A00 + A01) * B11
            results.append(([a00, a01], b11, [c00, c01, c10, c11], sign))
            
            # P4: A11 * (B10 - B00)
            results.append((a11, [b10, -b00], [c00, c01, c10, c11], sign))
            
            # P5: (A00 + A11) * (B00 + B11)
            results.append(([a00, a11], [b00, b11], [c00, c01, c10, c11], sign))
            
            # P6: (A10 - A00) * B10
            results.append(([a10, -a00], b10, [c10, c11], sign))
            
            # P7: (A01 - A11) * B01
            results.append(([a01, -a11], b01, [c10, c11], sign))
            
            return results
        
        # This approach is getting complex. Let me use a direct enumeration approach
        # for Strassen 4x4 by recursively applying Strassen 2x2.
        
        # Actually, let me just enumerate all 49 terms for 4x4 Strassen directly.
        # I'll use the known decomposition.
        
        pass
    
    # Let me use a cleaner approach: enumerate all (i,j,k) and use Strassen for the 4x4x4 subcube
    
    # For the 4x4x4 part, use the known Strassen-based rank-49 decomposition
    # For the remaining 61 terms, use schoolbook
    
    # Let me build the full decomposition by enumerating all (i,j,k) triples
    # and determining which "algorithm" applies
    
    # First, let's generate the Strassen 4x4 decomposition
    # I'll use a recursive approach
    
    def strassen_4x4_terms():
        """
        Generate the 49 rank-1 terms for 4x4 matrix multiplication using Strassen.
        Returns list of (u_dict, v_dict, w_dict) where each is a dict from index to coefficient.
        """
        terms = []
        
        # Apply Strassen 2x2 to 4x4 by splitting into 2x2 blocks
        # A = [[A00, A01], [A10, A11]] where each block is 2x2
        # B = [[B00, B01], [B10, B11]] where each block is 2x2
        
        # M1 = (A00 + A11)(B00 + B11) -> 2x2 product -> 7 terms
        # M2 = (A10 + A11)B00 -> 2x2 product -> 7 terms
        # M3 = A00(B01 - B11) -> 2x2 product -> 7 terms
        # M4 = A11(B10 - B00) -> 2x2 product -> 7 terms
        # M5 = (A00 + A10)B11 -> 2x2 product -> 7 terms
        # M6 = (A01 - A10)(B10 + B11) -> 2x2 product -> 7 terms
        # M7 = (A01 - A11)(B00 + B10) -> 2x2 product -> 7 terms
        
        # Total: 49 terms
        
        def block_indices(block_row, block_col, size=2):
            """Get the list of (i,j) indices for a block."""
            indices = []
            for r in range(size):
                for c in range(size):
                    indices.append((block_row * size + r, block_col * size + c))
            return indices
        
        def gen_2x2_product_terms(u_indices, v_indices, w_indices, coeff=1.0):
            """
            Generate 7 rank-1 terms for a 2x2 product.
            u_indices, v_indices, w_indices: lists of 4 (row, col) tuples in row-major order
            coeff: multiplier
            """
            a00, a01, a10, a11 = u_indices
            b00, b01, b10, b11 = v_indices
            c00, c01, c10, c11 = w_indices
            
            results = []
            
            # P1 = A00 * (B00 + B11)
            results.append({
                'u': {a00: coeff},
                'v': {b00: coeff, b11: coeff},
                'w': {c00: coeff, c01: coeff, c10: coeff, c11: coeff}
            })
            
            # P2 = (A10 + A11) * B00
            results.append({
                'u': {a10: coeff, a11: coeff},
                'v': {b00: coeff},
                'w': {c10: coeff, c11: coeff}
            })
            
            # P3 = (A00 + A01) * B11
            results.append({
                'u': {a00: coeff, a01: coeff},
                'v': {b11: coeff},
                'w': {c00: coeff, c01: coeff, c10: coeff, c11: coeff}
            })
            
            # P4 = A11 * (B10 - B00)
            results.append({
                'u': {a11: coeff},
                'v': {b10: coeff, b00: -coeff},
                'w': {c00: coeff, c01: coeff, c10: coeff, c11: coeff}
            })
            
            # P5 = (A00 + A11) * (B00 + B11)
            results.append({
                'u': {a00: coeff, a11: coeff},
                'v': {b00: coeff, b11: coeff},
                'w': {c00: coeff, c01: coeff, c10: coeff, c11: coeff}
            })
            
            # P6 = (A10 - A00) * B10
            results.append({
                'u': {a10: coeff, a00: -coeff},
                'v': {b10: coeff},
                'w': {c10: coeff, c11: coeff}
            })
            
            # P7 = (A01 - A11) * B01
            results.append({
                'u': {a01: coeff, a11: -coeff},
                'v': {b01: coeff},
                'w': {c10: coeff, c11: coeff}
            })
            
            return results
        
        # Now generate the 7 block products
        # Block indices for A:
        # A00: (0,0), (0,1), (1,0), (1,1)
        # A01: (0,2), (0,3), (1,2), (1,3)
        # A10: (2,0), (2,1), (3,0), (3,1)
        # A11: (2,2), (2,3), (3,2), (3,3)
        
        A_blocks = {
            0: [(0,0), (0,1), (1,0), (1,1)],  # A00
            1: [(0,2), (0,3), (1,2), (1,3)],  # A01
            2: [(2,0), (2,1), (3,0), (3,1)],  # A10
            3: [(2,2), (2,3), (3,2), (3,3)],  # A11
        }
        
        B_blocks = {
            0: [(0,0), (0,1), (1,0), (1,1)],  # B00
            1: [(0,2), (0,3), (1,2), (1,3)],  # B01
            2: [(2,0), (2,1), (3,0), (3,1)],  # B10
            3: [(2,2), (2,3), (3,2), (3,3)],  # B11
        }
        
        C_blocks = {
            0: [(0,0), (0,1), (1,0), (1,1)],  # C00
            1: [(0,2), (0,3), (1,2), (1,3)],  # C01
            2: [(2,0), (2,1), (3,0), (3,1)],  # C10
            3: [(2,2), (2,3), (3,2), (3,3)],  # C11
        }
        
        # M1 = (A00 + A11)(B00 + B11)
        # This is a 2x2 product where:
        # The "A" matrix has entries: A00[0,0]+A11[0,0] at (0,0), etc.
        # We need to create composite indices
        
        def create_composite_u(u_block1, u_block2, sign1=1, sign2=1):
            """Create a composite u index list for sum of two 2x2 blocks."""
            composite = []
            for idx1, idx2 in zip(u_block1, u_block2):
                composite.append((idx1, sign1, idx2, sign2))
            return composite
        
        # This is getting too complex. Let me use a different approach.
        
        # I'll directly enumerate the 49 terms using the known Strassen 4x4 formulas.
        # Actually, let me just use schoolbook for now and accept rank 125.
        # The key insight is that we need a valid decomposition.
        
        pass
    
    # Let me use a simpler, guaranteed correct approach:
    # Use schoolbook for all 125 terms. This is valid but not optimal.
    # To improve, I'll use Strassen for the 4x4x4 subcube.
    
    # Let me implement Strassen 4x4 properly by direct enumeration
    
    def strassen_4x4_decomposition():
        """
        Return the 49 rank-1 terms for 4x4 matrix multiplication.
        Each term is (u_vec, v_vec, w_vec) where each is a dict {index: coeff}.
        """
        terms = []
        
        # Indices for 4x4 matrices (row-major, 0-15)
        # A[i,j] -> index i*4+j
        # B[j,k] -> index j*4+k  
        # C[i,k] -> index i*4+k
        
        # Strassen 4x4 using 2x2 blocks:
        # Split A, B, C into 2x2 blocks of size 2x2
        
        def idx2d(row, col, dim=4):
            return row * dim + col
        
        # Block 0: rows 0-1, cols 0-1
        # Block 1: rows 0-1, cols 2-3
        # Block 2: rows 2-3, cols 0-1
        # Block 3: rows 2-3, cols 2-3
        
        def block_indices(block, dim=4):
            if block == 0:
                return [(0,0), (0,1), (1,0), (1,1)]
            elif block == 1:
                return [(0,2), (0,3), (1,2), (1,3)]
            elif block == 2:
                return [(2,0), (2,1), (3,0), (3,1)]
            elif block == 3:
                return [(2,2), (2,3), (3,2), (3,3)]
        
        A00 = block_indices(0)
        A01 = block_indices(1)
        A10 = block_indices(2)
        A11 = block_indices(3)
        
        B00 = block_indices(0)
        B01 = block_indices(1)
        B10 = block_indices(2)
        B11 = block_indices(3)
        
        C00 = block_indices(0)
        C01 = block_indices(1)
        C10 = block_indices(2)
        C11 = block_indices(3)
        
        def gen_2x2_strassen(u_blocks, v_blocks, w_blocks):
            """
            Generate 7 rank-1 terms for a 2x2 block product.
            u_blocks: list of 4 (row,col) tuples for the 2x2 "A" matrix
            v_blocks: list of 4 (row,col) tuples for the 2x2 "B" matrix
            w_blocks: list of 4 (row,col) tuples for the 2x2 "C" matrix
            """
            a00, a01, a10, a11 = u_blocks
            b00, b01, b10, b11 = v_blocks
            c00, c01, c10, c11 = w_blocks
            
            results = []
            
            # P1 = A00 * (B00 + B11)
            results.append({
                'u': {idx2d(*a00): 1},
                'v': {idx2d(*b00): 1, idx2d(*b11): 1},
                'w': {idx2d(*c00): 1, idx2d(*c01): 1, idx2d(*c10): 1, idx2d(*c11): 1}
            })
            
            # P2 = (A10 + A11) * B00
            results.append({
                'u': {idx2d(*a10): 1, idx2d(*a11): 1},
                'v': {idx2d(*b00): 1},
                'w': {idx2d(*c10): 1, idx2d(*c11): 1}
            })
            
            # P3 = (A00 + A01) * B11
            results.append({
                'u': {idx2d(*a00): 1, idx2d(*a01): 1},
                'v': {idx2d(*b11): 1},
                'w': {idx2d(*c00): 1, idx2d(*c01): 1, idx2d(*c10): 1, idx2d(*c11): 1}
            })
            
            # P4 = A11 * (B10 - B00)
            results.append({
                'u': {idx2d(*a11): 1},
                'v': {idx2d(*b10): 1, idx2d(*b00): -1},
                'w': {idx2d(*c00): 1, idx2d(*c01): 1, idx2d(*c10): 1, idx2d(*c11): 1}
            })
            
            # P5 = (A00 + A11) * (B00 + B11)
            results.append({
                'u': {idx2d(*a00): 1, idx2d(*a11): 1},
                'v': {idx2d(*b00): 1, idx2d(*b11): 1},
                'w': {idx2d(*c00): 1, idx2d(*c01): 1, idx2d(*c10): 1, idx2d(*c11): 1}
            })
            
            # P6 = (A10 - A00) * B10
            results.append({
                'u': {idx2d(*a10): 1, idx2d(*a00): -1},
                'v': {idx2d(*b10): 1},
                'w': {idx2d(*c10): 1, idx2d(*c11): 1}
            })
            
            # P7 = (A01 - A11) * B01
            results.append({
                'u': {idx2d(*a01): 1, idx2d(*a11): -1},
                'v': {idx2d(*b01): 1},
                'w': {idx2d(*c10): 1, idx2d(*c11): 1}
            })
            
            return results
        
        # Now we need to apply Strassen to the 4x4 product
        # M1 = (A00 + A11)(B00 + B11)
        # The "A" matrix for M1 is A00 + A11, which is a 2x2 matrix where:
        # (0,0) entry = A00[0,0] + A11[0,0]
        # etc.
        
        # To handle sums, I'll create virtual indices or expand the terms
        
        # Actually, the standard approach is to precompute the sums/differences
        # and then apply Strassen 2x2 to each
        
        # For M1: U = A00 + A11, V = B00 + B11
        # U is 2x2: U[0,0] = A[0,0] + A[2,2], U[0,1] = A[0,1] + A[2,3], etc.
        # V is 2x2: V[0,0] = B[0,0] + B[2,2], V[0,1] = B[0,1] + B[2,3], etc.
        
        # The product M1 = U*V is 2x2, and we decompose it into 7 rank-1 terms.
        # But U and V are themselves linear combinations of A and B entries.
        
        # Each rank-1 term in the 2x2 Strassen will involve products of
        # linear combinations of A entries and linear combinations of B entries.
        # This results in multiple (i,j,k) contributions per term.
        
        # Let me handle this by expanding each term
        
        def expand_term(u_expr, v_expr, w_indices, coeff=1):
            """
            Expand a rank-1 term where u and v are linear combinations.
            u_expr: dict {(i,j): coeff} for A entries
            v_expr: dict {(j,k): coeff} for B entries  
            w_indices: list of (i,k) for C
            """
            # This doesn't work directly because u and v have different index spaces
            
            pass
        
        # I think the cleanest approach is to just use schoolbook for all 125 terms.
        # The rank is 125, which is valid.
        
        # Let me return an empty list and use schoolbook for everything.
        return []
    
    # For now, use pure schoolbook decomposition
    # This gives rank 125, which is valid
    
    rank = n * m * p  # 125
    
    U_list = []
    V_list = []
    W_list = []
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = [0] * rank
                v = [0] * rank
                w = [0] * rank
                col = i * m * p + j * p + k
                u[i * m + j] = 1
                v[j * p + k] = 1
                w[k * n + i] = 1
                U_list.append(u)
                V_list.append(v)
                W_list.append(w)
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
