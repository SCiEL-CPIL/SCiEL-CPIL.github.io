import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 8]
n, m, p = DIMENSIONS

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin

    # We construct a rank 98 decomposition using Strassen's algorithm for 4x4x4
    # applied to two 4x4 blocks of the B matrix.
    # A is 4x4, B is 4x8 (two 4x4 blocks B1, B2).
    # C is 4x8 (two 4x4 blocks C1, C2).
    # C1 = A * B1, C2 = A * B2.
    # Rank of 4x4x4 Strassen is 7 * 7 = 49.
    # Total rank = 49 + 49 = 98.

    rank = 98
    U = np.zeros((n*m, rank), dtype=np.int8)
    V = np.zeros((m*p, rank), dtype=np.int8)
    W = np.zeros((p*n, rank), dtype=np.int8)

    # Strassen decomposition for 2x2 matrices
    # Returns 7 terms, each defined by (a_idx, b_idx, c_idx, sign)
    # where a_idx, b_idx are indices into the 2x2 matrices (flattened or nested)
    # and c_idx is the index into the result 2x2 matrix.
    # Actually, it's easier to just generate the 7 terms explicitly.
    
    # Strassen 2x2 terms:
    # M1 = (a00+a11)(b00+b11) -> c00 += 1, c11 += 1
    # M2 = (a10+a11)b00      -> c10 += 1, c11 -= 1
    # M3 = a00(b01-b11)      -> c01 += 1, c11 += 1
    # M4 = a11(b10-b00)      -> c00 += 1, c10 += 1
    # M5 = (a00+a01)b11      -> c00 -= 1, c01 += 1
    # M6 = (a10-a00)(b00+b01)-> c11 += 1
    # M7 = (a01-a11)(b10+b11)-> c00 -= 1

    # For 4x4 Strassen, we have 7 "meta-terms".
    # Each meta-term is a product of two 2x2 matrices (which are sums of 2x2 blocks).
    # Each 2x2 product has rank 7.
    # So we need to expand each meta-term into 7 scalar terms.
    # Total 7 * 7 = 49 terms per 4x4x4 multiplication.

    def add_2x2_product_to_tensor(a_blocks, b_blocks, c_blocks, base_col):
        """
        Adds the rank-7 Strassen decomposition of the product of two 2x2 matrices
        to the U, V, W matrices.
        
        a_blocks: list of (offset_i, offset_j, coeff) for the A 2x2 matrix
        b_blocks: list of (offset_j, offset_k, coeff) for the B 2x2 matrix
        c_blocks: list of (offset_i, offset_k, coeff) for the C 2x2 matrix
        base_col: starting column index in U, V, W
        """
        # We need to map the 2x2 indices (0,0), (0,1), (1,0), (1,1) to the global indices.
        # Since a_blocks, b_blocks, c_blocks can have multiple entries with different offsets,
        # we need to accumulate the coefficients for each 2x2 position.
        
        a_coeffs = np.zeros((2,2))
        for (oi, oj, c) in a_blocks:
            a_coeffs[oi % 2, oj % 2] += c
            
        b_coeffs = np.zeros((2,2))
        for (oj, ok, c) in b_blocks:
            b_coeffs[oj % 2, ok % 2] += c
            
        c_coeffs = np.zeros((2,2))
        for (oi, ok, c) in c_blocks:
            c_coeffs[oi % 2, ok % 2] += c

        # Now we have the effective 2x2 coefficients.
        # We also need the global offsets to place the terms.
        # This approach is tricky because different terms might use different global offsets.
        # However, in our specific 4x4 Strassen application, the blocks are aligned.
        # A00 and A11 are at (0,0) and (2,2).
        # If we treat (A00+A11) as a single virtual 2x2 matrix, the offsets are mixed.
        
        # Let's do it differently.
        # We will explicitly generate the 7 terms for the 2x2 Strassen algorithm
        # using the explicit linear combinations.
        
        # M1
        a1 = [(oi, oj, c) for (oi, oj, c) in a_blocks if (oi%2, oj%2) == (0,0) or (oi%2, oj%2) == (1,1)]
        b1 = [(oj, ok, c) for (oj, ok, c) in b_blocks if (oj%2, ok%2) == (0,0) or (oj%2, ok%2) == (1,1)]
        c1 = [(oi, ok, c) for (oi, ok, c) in c_blocks if (oi%2, ok%2) == (0,0) or (oi%2, ok%2) == (1,1)]
        # Wait, the Strassen formula uses specific entries.
        # M1 = (A00+A11)(B00+B11).
        # The term in U is (A00+A11)_{0,0} + (A00+A11)_{1,1}? No.
        # The U vector for M1 contains the coefficients of A in (A00+A11).
        # It is (A00+A11)_{0,0} + (A00+A11)_{1,1} for the "first" part? No.
        # Let's look at the 2x2 Strassen again.
        # M1 = (a00+a11)(b00+b11).
        # U contribution: a00 + a11.
        # V contribution: b00 + b11.
        # W contribution: c00 + c11? No, W contribution is the coefficient for the product.
        # The product is M1.
        # M1 contributes to C00 with coeff 1 and C11 with coeff 1.
        # So W has 1 at C00 and 1 at C11.
        
        # So for the 4x4 case, where A00+A11 is a sum of blocks:
        # U should contain the coefficients of A00 and A11 that form (a00+a11) of the virtual matrix.
        # But the virtual matrix entries are sums of global entries.
        # U[i*m+j] += coeff.
        # So if U needs to represent (A00+A11)_{0,0} + (A00+A11)_{1,1},
        # and (A00+A11)_{0,0} = A[0,0] + A[2,2], (A00+A11)_{1,1} = A[1,1] + A[3,3],
        # then U[0,0] += 1, U[2,2] += 1, U[1,1] += 1, U[3,3] += 1.
        
        # This is getting complicated to generalize with the block list.
        # Let's just hardcode the 49 terms for the 4x4x4 Strassen.
        # Or better, write a function that takes the 2x2 coefficients and the global offsets
        # and expands the 7 Strassen terms.
        
        pass

    # Let's restart with a cleaner approach.
    # We define the 49 terms for 4x4x4 Strassen explicitly.
    # We can generate them programmatically.
    
    # The 7 meta-terms of 4x4 Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Each meta-term is a 2x2 matrix product.
    # We can use the 2x2 Strassen decomposition for each.
    
    def add_2x2_strassen(a_entries, b_entries, c_entries, base_col):
        """
        Adds 7 terms to U, V, W starting at base_col.
        a_entries: dict (i,j) -> coeff
        b_entries: dict (j,k) -> coeff
        c_entries: dict (i,k) -> coeff
        """
        # M1
        a1 = {k: v for k, v in a_entries.items() if k in [(0,0), (1,1)]}
        b1 = {k: v for k, v in b_entries.items() if k in [(0,0), (1,1)]}
        c1 = {k: v for k, v in c_entries.items() if k in [(0,0), (1,1)]}
        # Wait, the filtering is wrong. We need to keep the original indices for U, V, W.
        # The filtering should be on the "local" 2x2 index, but we need to map to global.
        # My a_entries keys are global indices.
        # I need to know which global indices correspond to local (0,0), (1,1), etc.
        
        # This is too error-prone. I will hardcode the 49 terms.
        # Or, I will use the rank 128 schoolbook if I can't get 98 right quickly.
        # But 98 is a big improvement.
        
        pass

    # Let's try to be clever.
    # We can construct the U, V, W matrices by summing the contributions from the 7 meta-terms.
    # Each meta-term contributes 7 scalar terms.
    
    # Meta-term 1: (A00+A11)(B00+B11)
    # A part: A00 + A11.
    # B part: B00 + B11.
    # C part: C00 + C11? No, C part is the coefficient of the meta-term in the final C.
    # M1 contributes to C00 (1) and C11 (1).
    
    # So for Meta-term 1, we have:
    # A_mat = A00 + A11
    # B_mat = B00 + B11
    # C_mat = C00 + C11 (conceptually, but C00 and C11 are at different offsets)
    
    # We can call a function that adds the 2x2 Strassen decomposition for A_mat, B_mat, C_mat.
    # The function needs to know the global indices of the entries in A_mat, B_mat, C_mat.
    
    def add_strassen_2x2(a_map, b_map, c_map, base_col):
        """
        a_map: dict (global_i, global_j) -> coeff
        b_map: dict (global_j, global_k) -> coeff
        c_map: dict (global_i, global_k) -> coeff
        """
        # We need to identify the local 2x2 indices.
        # Assume the entries in a_map, b_map, c_map are aligned such that
        # local (0,0) corresponds to the min i, min j in a_map?
        # This is fragile.
        
        # Instead, let's just pass the 2x2 coefficient matrices and the base offsets.
        pass

    # Okay, I will implement the 49 terms explicitly by expanding the 7 meta-terms.
    # This is the safest way to ensure correctness.
    
    # Helper to add a single scalar term
    def add_term(a_dict, b_dict, c_dict, col):
        for (i,j), ua in a_dict.items():
            U[i*m+j, col] += ua
        for (j,k), vb in b_dict.items():
            V[j*p+k, col] += vb
        for (i,k), wc in c_dict.items():
            W[k*n+i, col] += wc

    # 2x2 Strassen terms for a generic 2x2 product
    # Inputs: A (2x2), B (2x2), C (2x2)
    # Outputs: 7 terms
    
    def get_2x2_strassen_terms(A, B, C):
        """
        A, B, C are 2x2 numpy arrays.
        Returns a list of 7 tuples (a_dict, b_dict, c_dict)
        where a_dict, b_dict, c_dict map (i,j) to coeff.
        """
        terms = []
        
        # M1 = (A00+A11)(B00+B11)
        a1 = {(0,0): A[0,0], (1,1): A[1,1]}
        b1 = {(0,0): B[0,0], (1,1): B[1,1]}
        c1 = {(0,0): C[0,0], (1,1): C[1,1]}
        terms.append((a1, b1, c1))
        
        # M2 = (A10+A11)B00
        a2 = {(1,0): A[1,0], (1,1): A[1,1]}
        b2 = {(0,0): B[0,0]}
        c2 = {(1,0): C[1,0], (1,1): -C[1,1]}
        terms.append((a2, b2, c2))
        
        # M3 = A00(B01-B11)
        a3 = {(0,0): A[0,0]}
        b3 = {(0,1): B[0,1], (1,1): -B[1,1]}
        c3 = {(0,1): C[0,1], (1,1): C[1,1]}
        terms.append((a3, b3, c3))
        
        # M4 = A11(B10-B00)
        a4 = {(1,1): A[1,1]}
        b4 = {(1,0): B[1,0], (0,0): -B[0,0]}
        c4 = {(0,0): C[0,0], (1,0): C[1,0]}
        terms.append((a4, b4, c4))
        
        # M5 = (A00+A01)B11
        a5 = {(0,0): A[0,0], (0,1): A[0,1]}
        b5 = {(1,1): B[1,1]}
        c5 = {(0,0): -C[0,0], (0,1): C[0,1]}
        terms.append((a5, b5, c5))
        
        # M6 = (A10-A00)(B00+B01)
        a6 = {(1,0): A[1,0], (0,0): -A[0,0]}
        b6 = {(0,0): B[0,0], (0,1): B[0,1]}
        c6 = {(1,1): C[1,1]}
        terms.append((a6, b6, c6))
        
        # M7 = (A01-A11)(B10+B11)
        a7 = {(0,1): A[0,1], (1,1): -A[1,1]}
        b7 = {(1,0): B[1,0], (1,1): B[1,1]}
        c7 = {(0,0): -C[0,0]}
        terms.append((a7, b7, c7))
        
        return terms

    # Now, for the 4x4 case.
    # We have 7 meta-terms.
    # For each meta-term, we define the 2x2 A, B, C matrices (coefficients)
    # and the global offsets.
    
    # Meta-term 1: (A00+A11)(B00+B11)
    # A_mat = [[1,0],[0,1]]? No, A00 and A11 are separate blocks.
    # We need to map the local 2x2 indices to global indices.
    # For A00+A11:
    # Local (0,0) -> Global (0,0) and (2,2)
    # Local (0,1) -> Global (0,1) and (2,3)
    # Local (1,0) -> Global (1,0) and (3,2)
    # Local (1,1) -> Global (1,1) and (3,3)
    
    # So when we get a_dict from get_2x2_strassen_terms, we need to expand it.
    # If a_dict has (0,0): 1, it means we need U[0,0] += 1 and U[2,2] += 1.
    
    def expand_term(a_dict, b_dict, c_dict, a_offsets, b_offsets, c_offsets, col):
        """
        a_offsets: dict local_idx -> list of global_idx
        """
        for (li, lj), coeff in a_dict.items():
            for gi, gj in a_offsets[(li, lj)]:
                U[gi*m+gj, col] += coeff
        for (lj, lk), coeff in b_dict.items():
            for gj, gk in b_offsets[(lj, lk)]:
                V[gj*p+gk, col] += coeff
        for (li, lk), coeff in c_dict.items():
            for gi, gk in c_offsets[(li, lk)]:
                W[gk*n+gi, col] += coeff

    # Define offsets for 4x4 blocks
    # A00: (0,0), A01: (0,2), A10: (2,0), A11: (2,2)
    
    # Meta-term 1: (A00+A11)(B00+B11)
    # A_offsets:
    # (0,0) -> [(0,0), (2,2)]
    # (0,1) -> [(0,1), (2,3)]
    # (1,0) -> [(1,0), (3,2)]
    # (1,1) -> [(1,1), (3,3)]
    # B_offsets: same
    # C_offsets: C00 and C11?
    # M1 contributes to C00 and C11.
    # So C_mat is not a single 2x2 block.
    # Wait, in the 4x4 Strassen, M1 is a 2x2 matrix.
    # M1 contributes to C00 (top-left 2x2) and C11 (bottom-right 2x2).
    # So the "C" in the 2x2 Strassen decomposition of M1 is actually a 2x2 matrix of blocks?
    # No.
    # Let's re-read Strassen for 4x4.
    # M1 = (A00+A11)(B00+B11). This is a 2x2 matrix.
    # C00 += M1. C11 += M1.
    # So M1 is added to C00 and C11.
    # So the "C" matrix in the 2x2 Strassen decomposition of M1 should be the identity matrix?
    # No, the 2x2 Strassen decomposition computes M1.
    # Then we add M1 to C00 and C11.
    # So the c_dict in the 2x2 Strassen terms should be the identity matrix?
    # No, the c_dict represents the contribution of the scalar product to the 2x2 result M1.
    # Then we map M1 to C.
    # So for Meta-term 1, the local C is M1 (a 2x2 matrix).
    # And M1 is added to C00 and C11.
    # So C_offsets for local (0,0) should be [(0,0), (2,2)]?
    # Yes!
    # Local (0,0) of M1 -> Global (0,0) of C00 and (2,2) of C11?
    # No, C00 is a 2x2 block. C11 is a 2x2 block.
    # M1 is a 2x2 matrix.
    # C00 += M1 means C[0:2, 0:2] += M1.
    # C11 += M1 means C[2:4, 2:4] += M1.
    # So local (0,0) of M1 -> Global (0,0) and (2,2).
    # Local (0,1) of M1 -> Global (0,1) and (2,3).
    # Local (1,0) of M1 -> Global (1,0) and (3,2).
    # Local (1,1) of M1 -> Global (1,1) and (3,3).
    
    # So C_offsets is the same as A_offsets and B_offsets for Meta-term 1.
    
    # Let's define the offsets for each meta-term.
    
    # A_offsets for (A00+A11):
    a_off_1 = {
        (0,0): [(0,0), (2,2)],
        (0,1): [(0,1), (2,3)],
        (1,0): [(1,0), (3,2)],
        (1,1): [(1,1), (3,3)]
    }
    # B_offsets for (B00+B11):
    b_off_1 = {
        (0,0): [(0,0), (2,2)],
        (0,1): [(0,1), (2,3)],
        (1,0): [(1,0), (3,2)],
        (1,1): [(1,1), (3,3)]
    }
    # C_offsets for M1 -> C00 + C11:
    c_off_1 = {
        (0,0): [(0,0), (2,2)],
        (0,1): [(0,1), (2,3)],
        (1,0): [(1,0), (3,2)],
        (1,1): [(1,1), (3,3)]
    }
    
    # A_mat for (A00+A11) is the identity matrix?
    # No, A00+A11 is a 2x2 matrix.
    # The coefficients are 1 for all entries.
    # So A_mat = [[1,1],[1,1]]? No.
    # A00 is a 2x2 matrix. A11 is a 2x2 matrix.
    # (A00+A11)_{ij} = A00_{ij} + A11_{ij}.
    # So the coefficient for each entry is 1.
    # So A_mat = np.ones((2,2)).
    
    A_mat_1 = np.ones((2,2))
    B_mat_1 = np.ones((2,2))
    C_mat_1 = np.ones((2,2)) # M1 contributes to C00 and C11 with coeff 1.
    
    terms_1 = get_2x2_strassen_terms(A_mat_1, B_mat_1, C_mat_1)
    
    col = 0
    for a_t, b_t, c_t in terms_1:
        expand_term(a_t, b_t, c_t, a_off_1, b_off_1, c_off_1, col)
        col += 1
        
    # Meta-term 2: (A10+A11)B00
    # A10: (2,0), A11: (2,2)
    # B00: (0,0)
    # M2 contributes to C10 and C11?
    # C10 += M2. C11 -= M2.
    # So C_offsets:
    # Local (0,0) -> Global (2,0) and (2,2)?
    # C10 is at (2,0). C11 is at (2,2).
    # M2 is a 2x2 matrix.
    # C10 += M2 means C[2:4, 0:2] += M2.
    # C11 -= M2 means C[2:4, 2:4] -= M2.
    # So local (0,0) of M2 -> Global (2,0) and (2,2) with coeff -1?
    # Yes.
    
    a_off_2 = {
        (0,0): [(2,0), (2,2)],
        (0,1): [(2,1), (2,3)],
        (1,0): [(3,0), (3,2)],
        (1,1): [(3,1), (3,3)]
    }
    b_off_2 = {
        (0,0): [(0,0)],
        (0,1): [(0,1)],
        (1,0): [(1,0)],
        (1,1): [(1,1)]
    }
    c_off_2 = {
        (0,0): [(2,0), (2,2)], # C10 (2,0) and C11 (2,2)
        (0,1): [(2,1), (2,3)],
        (1,0): [(3,0), (3,2)],
        (1,1): [(3,1), (3,3)]
    }
    # But C11 has coeff -1.
    # So we need to handle the sign in C_mat or C_offsets.
    # Let's put the sign in C_mat.
    # C_mat for M2:
    # M2 contributes to C10 with 1 and C11 with -1.
    # So C_mat = [[1, -1], [1, -1]]?
    # No, M2 is a 2x2 matrix.
    # C10 += M2. C11 -= M2.
    # So the "C" in the 2x2 Strassen decomposition of M2 should be such that
    # when we map it to global C, we get C10 += M2 and C11 -= M2.
    # So C_mat = [[1, -1], [1, -1]] is correct if C_offsets maps local (i,j) to global (2+i, j) and (2+i, 2+j).
    # Let's check:
    # Local (0,0) -> Global (2,0) with coeff 1, Global (2,2) with coeff -1.
    # Yes.
    
    A_mat_2 = np.ones((2,2))
    B_mat_2 = np.ones((2,2))
    C_mat_2 = np.array([[1, -1], [1, -1]])
    
    terms_2 = get_2x2_strassen_terms(A_mat_2, B_mat_2, C_mat_2)
    for a_t, b_t, c_t in terms_2:
        expand_term(a_t, b_t, c_t, a_off_2, b_off_2, c_off_2, col)
        col += 1
        
    # Meta-term 3: A00(B01-B11)
    # A00: (0,0)
    # B01: (0,2), B11: (2,2)
    # M3 contributes to C01 and C11?
    # C01 += M3. C11 += M3.
    # C01 is at (0,2). C11 is at (2,2).
    
    a_off_3 = {
        (0,0): [(0,0)],
        (0,1): [(0,1)],
        (1,0): [(1,0)],
        (1,1): [(1,1)]
    }
    b_off_3 = {
        (0,0): [(0,2), (2,2)],
        (0,1): [(0,3), (2,3)],
        (1,0): [(1,2), (3,2)],
        (1,1): [(1,3), (3,3)]
    }
    # B01-B11. So B_mat = [[1, 1], [1, 1]]?
    # B01 is at (0,2). B11 is at (2,2).
    # (B01-B11)_{ij} = B01_{ij} - B11_{ij}.
    # So B_mat = [[1, -1], [1, -1]]?
    # No, B01 and B11 are 2x2 matrices.
    # (B01-B11) is a 2x2 matrix.
    # The coefficients are 1 for B01 and -1 for B11.
    # So B_mat = [[1, -1], [1, -1]] is correct if b_off_3 maps local (i,j) to global (i, 2+j) and (2+i, 2+j).
    # Let's check:
    # Local (0,0) -> Global (0,2) with coeff 1, Global (2,2) with coeff -1.
    # Yes.
    
    c_off_3 = {
        (0,0): [(0,2), (2,2)], # C01 (0,2) and C11 (2,2)
        (0,1): [(0,3), (2,3)],
        (1,0): [(1,2), (3,2)],
        (1,1): [(1,3), (3,3)]
    }
    
    A_mat_3 = np.ones((2,2))
    B_mat_3 = np.array([[1, -1], [1, -1]])
    C_mat_3 = np.ones((2,2))
    
    terms_3 = get_2x2_strassen_terms(A_mat_3, B_mat_3, C_mat_3)
    for a_t, b_t, c_t in terms_3:
        expand_term(a_t, b_t, c_t, a_off_3, b_off_3, c_off_3, col)
        col += 1
        
    # Meta-term 4: A11(B10-B00)
    # A11: (2,2)
    # B10: (2,0), B00: (0,0)
    # M4 contributes to C00 and C10?
    # C00 += M4. C10 += M4.
    # C00 is at (0,0). C10 is at (2,0).
    
    a_off_4 = {
        (0,0): [(2,2)],
        (0,1): [(2,3)],
        (1,0): [(3,2)],
        (1,1): [(3,3)]
    }
    b_off_4 = {
        (0,0): [(2,0), (0,0)],
        (0,1): [(2,1), (0,1)],
        (1,0): [(3,0), (1,0)],
        (1,1): [(3,1), (1,1)]
    }
    # B10-B00.
    # B_mat = [[1, -1], [1, -1]]?
    # B10 is at (2,0). B00 is at (0,0).
    # (B10-B00)_{ij} = B10_{ij} - B00_{ij}.
    # So B_mat = [[1, -1], [1, -1]] is correct if b_off_4 maps local (i,j) to global (2+i, j) and (i, j).
    # Let's check:
    # Local (0,0) -> Global (2,0) with coeff 1, Global (0,0) with coeff -1.
    # Yes.
    
    c_off_4 = {
        (0,0): [(0,0), (2,0)], # C00 (0,0) and C10 (2,0)
        (0,1): [(0,1), (2,1)],
        (1,0): [(1,0), (3,0)],
        (1,1): [(1,1), (3,1)]
    }
    
    A_mat_4 = np.ones((2,2))
    B_mat_4 = np.array([[1, -1], [1, -1]])
    C_mat_4 = np.ones((2,2))
    
    terms_4 = get_2x2_strassen_terms(A_mat_4, B_mat_4, C_mat_4)
    for a_t, b_t, c_t in terms_4:
        expand_term(a_t, b_t, c_t, a_off_4, b_off_4, c_off_4, col)
        col += 1
        
    # Meta-term 5: (A00+A01)B11
    # A00: (0,0), A01: (0,2)
    # B11: (2,2)
    # M5 contributes to C00 and C01?
    # C00 -= M5. C01 += M5.
    # C00 is at (0,0). C01 is at (0,2).
    
    a_off_5 = {
        (0,0): [(0,0), (0,2)],
        (0,1): [(0,1), (0,3)],
        (1,0): [(1,0), (1,2)],
        (1,1): [(1,1), (1,3)]
    }
    # A00+A01.
    # A_mat = [[1, 1], [1, 1]]?
    # A00 is at (0,0). A01 is at (0,2).
    # (A00+A01)_{ij} = A00_{ij} + A01_{ij}.
    # So A_mat = [[1, 1], [1, 1]] is correct if a_off_5 maps local (i,j) to global (i, j) and (i, 2+j).
    # Let's check:
    # Local (0,0) -> Global (0,0) with coeff 1, Global (0,2) with coeff 1.
    # Yes.
    
    b_off_5 = {
        (0,0): [(2,2)],
        (0,1): [(2,3)],
        (1,0): [(3,2)],
        (1,1): [(3,3)]
    }
    
    c_off_5 = {
        (0,0): [(0,0), (0,2)], # C00 (0,0) and C01 (0,2)
        (0,1): [(0,1), (0,3)],
        (1,0): [(1,0), (1,2)],
        (1,1): [(1,1), (1,3)]
    }
    # C00 -= M5. C01 += M5.
    # So C_mat = [[-1, 1], [-1, 1]]?
    # Local (0,0) -> Global (0,0) with coeff -1, Global (0,2) with coeff 1.
    # Yes.
    
    A_mat_5 = np.ones((2,2))
    B_mat_5 = np.ones((2,2))
    C_mat_5 = np.array([[-1, 1], [-1, 1]])
    
    terms_5 = get_2x2_strassen_terms(A_mat_5, B_mat_5, C_mat_5)
    for a_t, b_t, c_t in terms_5:
        expand_term(a_t, b_t, c_t, a_off_5, b_off_5, c_off_5, col)
        col += 1
        
    # Meta-term 6: (A10-A00)(B00+B01)
    # A10: (2,0), A00: (0,0)
    # B00: (0,0), B01: (0,2)
    # M6 contributes to C11?
    # C11 += M6.
    # C11 is at (2,2).
    
    a_off_6 = {
        (0,0): [(2,0), (0,0)],
        (0,1): [(2,1), (0,1)],
        (1,0): [(3,0), (1,0)],
        (1,1): [(3,1), (1,1)]
    }
    # A10-A00.
    # A_mat = [[1, -1], [1, -1]]?
    # A10 is at (2,0). A00 is at (0,0).
    # (A10-A00)_{ij} = A10_{ij} - A00_{ij}.
    # So A_mat = [[1, -1], [1, -1]] is correct if a_off_6 maps local (i,j) to global (2+i, j) and (i, j).
    # Let's check:
    # Local (0,0) -> Global (2,0) with coeff 1, Global (0,0) with coeff -1.
    # Yes.
    
    b_off_6 = {
        (0,0): [(0,0), (0,2)],
        (0,1): [(0,1), (0,3)],
        (1,0): [(1,0), (1,2)],
        (1,1): [(1,1), (1,3)]
    }
    # B00+B01.
    # B_mat = [[1, 1], [1, 1]]?
    # B00 is at (0,0). B01 is at (0,2).
    # (B00+B01)_{ij} = B00_{ij} + B01_{ij}.
    # So B_mat = [[1, 1], [1, 1]] is correct if b_off_6 maps local (i,j) to global (i, j) and (i, 2+j).
    # Let's check:
    # Local (0,0) -> Global (0,0) with coeff 1, Global (0,2) with coeff 1.
    # Yes.
    
    c_off_6 = {
        (0,0): [(2,2)],
        (0,1): [(2,3)],
        (1,0): [(3,2)],
        (1,1): [(3,3)]
    }
    
    A_mat_6 = np.array([[1, -1], [1, -1]])
    B_mat_6 = np.ones((2,2))
    C_mat_6 = np.ones((2,2))
    
    terms_6 = get_2x2_strassen_terms(A_mat_6, B_mat_6, C_mat_6)
    for a_t, b_t, c_t in terms_6:
        expand_term(a_t, b_t, c_t, a_off_6, b_off_6, c_off_6, col)
        col += 1
        
    # Meta-term 7: (A01-A11)(B10+B11)
    # A01: (0,2), A11: (2,2)
    # B10: (2,0), B11: (2,2)
    # M7 contributes to C00?
    # C00 -= M7.
    # C00 is at (0,0).
    
    a_off_7 = {
        (0,0): [(0,2), (2,2)],
        (0,1): [(0,3), (2,3)],
        (1,0): [(1,2), (3,2)],
        (1,1): [(1,3), (3,3)]
    }
    # A01-A11.
    # A_mat = [[1, -1], [1, -1]]?
    # A01 is at (0,2). A11 is at (2,2).
    # (A01-A11)_{ij} = A01_{ij} - A11_{ij}.
    # So A_mat = [[1, -1], [1, -1]] is correct if a_off_7 maps local (i,j) to global (i, 2+j) and (2+i, 2+j).
    # Let's check:
    # Local (0,0) -> Global (0,2) with coeff 1, Global (2,2) with coeff -1.
    # Yes.
    
    b_off_7 = {
        (0,0): [(2,0), (2,2)],
        (0,1): [(2,1), (2,3)],
        (1,0): [(3,0), (3,2)],
        (1,1): [(3,1), (3,3)]
    }
    # B10+B11.
    # B_mat = [[1, 1], [1, 1]]?
    # B10 is at (2,0). B11 is at (2,2).
    # (B10+B11)_{ij} = B10_{ij} + B11_{ij}.
    # So B_mat = [[1, 1], [1, 1]] is correct if b_off_7 maps local (i,j) to global (2+i, j) and (2+i, 2+j).
    # Let's check:
    # Local (0,0) -> Global (2,0) with coeff 1, Global (2,2) with coeff 1.
    # Yes.
    
    c_off_7 = {
        (0,0): [(0,0)],
        (0,1): [(0,1)],
        (1,0): [(1,0)],
        (1,1): [(1,1)]
    }
    # C00 -= M7.
    # So C_mat = [[-1, -1], [-1, -1]]?
    # Local (0,0) -> Global (0,0) with coeff -1.
    # Yes.
    
    A_mat_7 = np.array([[1, -1], [1, -1]])
    B_mat_7 = np.ones((2,2))
    C_mat_7 = np.array([[-1, -1], [-1, -1]])
    
    terms_7 = get_2x2_strassen_terms(A_mat_7, B_mat_7, C_mat_7)
    for a_t, b_t, c_t in terms_7:
        expand_term(a_t, b_t, c_t, a_off_7, b_off_7, c_off_7, col)
        col += 1
        
    # Now we have 49 terms for the first 4x4 block (C1 = A * B1).
    # We need to do the same for the second 4x4 block (C2 = A * B2).
    # B2 is at cols 4-7.
    # So we just shift the B offsets by 4 in the k direction.
    
    # We can reuse the same logic, but with b_offsets shifted.
    
    # Let's create a function to add the 49 terms given a shift for B.
    
    def add_4x4_strassen(b_shift):
        nonlocal col
        # We can just copy the code above and add b_shift to all b_offsets.
        # Or better, refactor the offsets.
        
        # For simplicity, I will just duplicate the logic with b_shift.
        
        # Meta-term 1
        a_off = a_off_1
        b_off = {k: [(gj, gk+b_shift) for gj, gk in v] for k, v in b_off_1.items()}
        c_off = c_off_1
        A_mat = A_mat_1
        B_mat = B_mat_1
        C_mat = C_mat_1
        terms = get_2x2_strassen_terms(A_mat, B_mat, C_mat)
        for a_t, b_t, c_t in terms:
            expand_term(a_t, b_t, c_t, a_off, b_off, c_off, col)
            col += 1
            
        # Meta-term 2
        a_off = a_off_2
        b_off = {k: [(gj, gk+b_shift) for gj, gk in v] for k, v in b_off_2.items()}
        c_off = c_off_2
        A_mat = A_mat_2
        B_mat = B_mat_2
        C_mat = C_mat_2
        terms = get_2x2_strassen_terms(A_mat, B_mat, C_mat)
        for a_t, b_t, c_t in terms:
            expand_term(a_t, b_t, c_t, a_off, b_off, c_off, col)
            col += 1
            
        # Meta-term 3
        a_off = a_off_3
        b_off = {k: [(gj, gk+b_shift) for gj, gk in v] for k, v in b_off_3.items()}
        c_off = c_off_3
        A_mat = A_mat_3
        B_mat = B_mat_3
        C_mat = C_mat_3
        terms = get_2x2_strassen_terms(A_mat, B_mat, C_mat)
        for a_t, b_t, c_t in terms:
            expand_term(a_t, b_t, c_t, a_off, b_off, c_off, col)
            col += 1
            
        # Meta-term 4
        a_off = a_off_4
        b_off = {k: [(gj, gk+b_shift) for gj, gk in v] for k, v in b_off_4.items()}
        c_off = c_off_4
        A_mat = A_mat_4
        B_mat = B_mat_4
        C_mat = C_mat_4
        terms = get_2x2_strassen_terms(A_mat, B_mat, C_mat)
        for a_t, b_t, c_t in terms:
            expand_term(a_t, b_t, c_t, a_off, b_off, c_off, col)
            col += 1
            
        # Meta-term 5
        a_off = a_off_5
        b_off = {k: [(gj, gk+b_shift) for gj, gk in v] for k, v in b_off_5.items()}
        c_off = c_off_5
        A_mat = A_mat_5
        B_mat = B_mat_5
        C_mat = C_mat_5
        terms = get_2x2_strassen_terms(A_mat, B_mat, C_mat)
        for a_t, b_t, c_t in terms:
            expand_term(a_t, b_t, c_t, a_off, b_off, c_off, col)
            col += 1
            
        # Meta-term 6
        a_off = a_off_6
        b_off = {k: [(gj, gk+b_shift) for gj, gk in v] for k, v in b_off_6.items()}
        c_off = c_off_6
        A_mat = A_mat_6
        B_mat = B_mat_6
        C_mat = C_mat_6
        terms = get_2x2_strassen_terms(A_mat, B_mat, C_mat)
        for a_t, b_t, c_t in terms:
            expand_term(a_t, b_t, c_t, a_off, b_off, c_off, col)
            col += 1
            
        # Meta-term 7
        a_off = a_off_7
        b_off = {k: [(gj, gk+b_shift) for gj, gk in v] for k, v in b_off_7.items()}
        c_off = c_off_7
        A_mat = A_mat_7
        B_mat = B_mat_7
        C_mat = C_mat_7
        terms = get_2x2_strassen_terms(A_mat, B_mat, C_mat)
        for a_t, b_t, c_t in terms:
            expand_term(a_t, b_t, c_t, a_off, b_off, c_off, col)
            col += 1

    # We already added the first 49 terms (b_shift=0).
    # Now add the second 49 terms (b_shift=4).
    add_4x4_strassen(4)
    
    # Total rank is 98.
    # col should be 98.
    
    if col != 98:
        # Fallback to schoolbook if something went wrong
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    result = construct()
    r = len(result["U"][0])
    print(f"Rank: {r}")
