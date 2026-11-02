"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np
from fractions import Fraction

def construct():
    """
    Return a rank-98 tensor decomposition for 4x4 * 4x8 matrix multiplication.
    
    Uses Strassen's algorithm at the block level (2x2 blocks) applied to
    a 4x4 * 4x8 product, giving rank 7*4*4 = 112... wait, let me recalculate.
    
    Actually for 4x4 * 4x8:
    - Split A into 2x2 blocks of 2x2 matrices: A = [[A11, A12], [A21, A22]]
    - Split B into 2x2 blocks where each "block" is 2x4: B = [[B11, B12], [B21, B22]]
      where B11, B12 are 2x4 and B21, B22 are 2x4
    - C = A*B is 4x8, split into 2x2 blocks of 2x4: C = [[C11, C12], [C21, C22]]
    
    Strassen for 2x2 blocks:
    M1 = (A11+A22)*(B11+B22)
    M2 = (A21+A22)*B11
    M3 = A11*(B12-B22)
    M4 = A22*(B21-B11)
    M5 = (A11+A12)*B22
    M6 = (A21-A11)*(B11+B12)
    M7 = (A12-A22)*(B21+B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    Each Mi is a product of a 2x2 matrix and a 2x4 matrix, giving a 2x4 result.
    
    For each Strassen product, we need to decompose it into tensor rank terms.
    A 2x2 * 2x4 matrix multiplication has rank 16 (schoolbook) or 7 (Strassen).
    Using Strassen at the scalar level for each 2x2*2x4:
    
    For 2x2 * 2x4:
    Split the 2x4 B into two 2x2 blocks: B = [B_left, B_right]
    Then (2x2)*(2x4) = (2x2)*(2x2) || (2x2)*(2x2)
    Each 2x2*2x2 can use Strassen rank 7.
    So 2x2*2x4 has rank 14.
    
    Total rank = 7 (Strassen blocks) * 14 (each block product) = 98.
    """
    n, m, p = DIMENSIONS  # n=4, m=4, p=8
    
    # We'll build the decomposition using Strassen at two levels.
    # Level 1: 2x2 blocks of the 4x4 * 4x8 product
    # Level 2: For each 2x2*2x4 sub-product, use Strassen on 2x2*2x2 blocks
    
    # First, define Strassen for 2x2*2x2:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # For 2x2*2x4, split B into B=[B1|B2] where B1, B2 are 2x2:
    # Result = (A*B1, A*B2), each computed via Strassen rank 7.
    # Total rank per 2x2*2x4 = 14.
    
    # Now build the full 4x4*4x8 decomposition.
    # 
    # Let's index elements:
    # A[i,j] for i,j in 0..3 (4x4)
    # B[j,k] for j in 0..3, k in 0..7 (4x8)
    # C[i,k] for i in 0..3, k in 0..7 (4x8)
    #
    # W indexing: W[k*n+i, r] where n=4, so W[k*4+i, r] for k in 0..7, i in 0..3
    
    # Block structure:
    # A blocks: A11 = A[0:2,0:2], A12 = A[0:2,2:4], A21 = A[2:4,0:2], A22 = A[2:4,2:4]
    # B blocks: B11 = B[0:2,0:4], B12 = B[0:2,4:8], B21 = B[2:4,0:4], B22 = B[2:4,4:8]
    # C blocks: C11 = C[0:2,0:4], C12 = C[0:2,4:8], C21 = C[2:4,0:4], C22 = C[2:4,4:8]
    
    # Each Mi (2x2 * 2x4) will be decomposed into 14 scalar products.
    # We'll track which (i,j) and (j,k) each scalar product involves.
    
    # Let's build this step by step.
    
    # First, let's define a helper to get Strassen factors for 2x2*2x2.
    # Each factor c has U_part (2x2), V_part (2x2), W_part (2x2) entries.
    
    # For 2x2*2x2 Strassen:
    # Factor 1: (a+d)(e+h)
    # U: [[1,0],[1,1]], V: [[1,0],[1,1]], W: [[1,-1],[1,0]]... 
    # Wait, I need to be more careful. Let me use the standard formulation.
    
    # Actually, let me just directly enumerate all 98 tensor terms.
    
    # I'll use a systematic approach. For each of the 7 Strassen block-level factors,
    # and for each of the 14 scalar-level factors within each, I'll record the
    # (row_of_A, col_of_A), (row_of_B, col_of_B), (row_of_C, col_of_C) with signs.
    
    # Let me define the 2x2*2x2 Strassen factors explicitly.
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # C = [[c00, c01], [c10, c11]]
    
    # The 7 factors:
    # f1: (a00+a11)*(b00+b11) -> contributes to c00(+1), c11(-1)
    # f2: (a10+a11)*b00 -> contributes to c10(+1), c11(-1)  
    # f3: a00*(b01-b11) -> contributes to c01(+1), c11(+1)
    # f4: a11*(b10-b00) -> contributes to c10(+1), c00(+1)
    # f5: (a00+a01)*b11 -> contributes to c01(+1), c00(-1)
    # f6: (a10-a00)*(b00+b01) -> contributes to c11(+1)
    # f7: (a01-a11)*(b10+b11) -> contributes to c00(+1), c22... wait
    
    # Let me be very explicit. For 2x2*2x2:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Where:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # So the tensor factors for 2x2*2x2 (7 factors):
    # Each factor is a triple of (A_coeff_vector, B_coeff_vector, C_coeff_vector)
    # where each is a 4-vector (for the 4 entries of the 2x2 matrix).
    
    # Let me index 2x2 matrices as [00, 01, 10, 11] -> indices 0,1,2,3
    
    strassen_2x2 = [
        # (A_coeffs, B_coeffs, C_coeffs)
        # M1: (A00+A11)(B00+B11)
        # A: [1,0,0,1], B: [1,0,0,1]
        # C: contributes +1 to C00, -1 to C11 -> [1,0,0,-1]
        ([(1,0,0,1)], [(1,0,0,1)], [(1,0,0,-1)]),  # placeholder, need to handle sums
        
        # Actually, I need to think about this differently.
        # Each "factor" in the tensor decomposition is a single product of 
        # one A-entry-sum, one B-entry-sum, mapped to C-entries.
    ]
    
    # Let me take a different approach. I'll directly construct the rank-98 
    # decomposition by enumerating all 98 terms.
    
    # The 4x4 * 4x8 product can be viewed as:
    # - 7 Strassen block factors
    # - Each block factor is a product of a 2x2 "effective A" and 2x4 "effective B"
    # - Each such product uses 14 scalar Strassen factors (2 per 2x2*2x2 sub-block)
    
    # Let me define this more carefully.
    
    # For the block-level Strassen:
    # Block A indices: A11(0:2,0:2), A12(0:2,2:4), A21(2:4,0:2), A22(2:4,2:4)
    # Block B indices: B11(0:2,0:4), B12(0:2,4:8), B21(2:4,0:4), B22(2:4,4:8)
    # Block C indices: C11(0:2,0:4), C12(0:2,4:8), C21(2:4,0:4), C22(2:4,4:8)
    
    # Strassen block factors:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Now for each Mi, which is a 2x2 * 2x4 product, I need to decompose it.
    
    # Let's say Mi = Aeff * Beff where Aeff is 2x2 and Beff is 2x4.
    # Split Beff into Beff_left (2x2) and Beff_right (2x2).
    # Mi = Aeff * Beff_left, Aeff * Beff_right
    # Each 2x2*2x2 uses Strassen rank 7.
    
    # So total rank = 7 * 7 * 2 = 98.
    
    # Now I need to map each scalar product to specific (i,j), (j,k), (i,k) indices.
    
    # Let me build a list of 98 terms, each with:
    # - A_indices: list of (i, j, coeff) for the 2x2 effective A
    # - B_indices: list of (j, k, coeff) for the 2x4 effective B  
    # - C_indices: list of (i, k, coeff) for the 2x4 effective C
    
    # This is getting complex. Let me just hardcode the construction.
    
    # I'll use a dictionary approach. For each of the 98 tensor factors,
    # I'll determine which A[i,j], B[j,k], and C[i,k] entries it involves.
    
    # Let me define helper functions.
    
    # First, the 2x2*2x2 Strassen factors (7 factors):
    # Factor t has A_part[i] and B_part[j] and C_part[k] for i,j,k in {0,1,2,3}
    # where 0=00, 1=01, 2=10, 3=11 in 2x2 indexing.
    
    # I'll represent each 2x2*2x2 factor as (A_vec, B_vec, C_vec) where each is a dict.
    
    s22 = []
    # M1: (a00+a11)(b00+b11)
    # A: {0:1, 3:1}, B: {0:1, 3:1}, C: {0:1, 3:-1} (C00+=1, C11-=-1)
    s22.append(({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:-1}))
    # M2: (a10+a11)b00
    # A: {2:1, 3:1}, B: {0:1}, C: {2:1, 3:-1} (C10+=1, C11-=-1)
    s22.append(({2:1, 3:1}, {0:1}, {2:1, 3:-1}))
    # M3: a00(b01-b11)
    # A: {0:1}, B: {1:1, 3:-1}, C: {1:1, 3:1} (C01+=1, C11+=1)
    s22.append(({0:1}, {1:1, 3:-1}, {1:1, 3:1}))
    # M4: a11(b10-b00)
    # A: {3:1}, B: {2:1, 0:-1}, C: {2:1, 0:1} (C10+=1, C00+=1)
    s22.append(({3:1}, {2:1, 0:-1}, {2:1, 0:1}))
    # M5: (a00+a01)b11
    # A: {0:1, 1:1}, B: {3:1}, C: {1:1, 0:-1} (C01+=1, C00-=-1)
    s22.append(({0:1, 1:1}, {3:1}, {1:1, 0:-1}))
    # M6: (a10-a00)(b00+b01)
    # A: {2:1, 0:-1}, B: {0:1, 1:1}, C: {3:1} (C11+=1)
    s22.append(({2:1, 0:-1}, {0:1, 1:1}, {3:1}))
    # M7: (a01-a11)(b10+b11)
    # A: {1:1, 3:-1}, B: {2:1, 3:1}, C: {0:1} (C00+=1)
    s22.append(({1:1, 3:-1}, {2:1, 3:1}, {0:1}))
    
    # Now for the block level. Each block Mi is a 2x2 * 2x4 product.
    # I need to map the 2x2 indices within each block to global indices.
    
    # Block A11: rows 0,1; cols 0,1 -> local (0,0)->(0,0), (0,1)->(0,2), (1,0)->(2,0), (1,1)->(2,2)
    # Wait, A11 = A[0:2, 0:2], so local (r,c) -> global (r, c)
    # A12 = A[0:2, 2:4], local (r,c) -> global (r, c+2)
    # A21 = A[2:4, 0:2], local (r,c) -> global (r+2, c)
    # A22 = A[2:4, 2:4], local (r,c) -> global (r+2, c+2)
    
    # Block B11 = B[0:2, 0:4]: local (r, c) -> global (r, c) for c in 0..3
    # B12 = B[0:2, 4:8]: local (r, c) -> global (r, c+4)
    # B21 = B[2:4, 0:4]: local (r, c) -> global (r+2, c)
    # B22 = B[2:4, 4:8]: local (r, c) -> global (r+2, c+4)
    
    # Block C11 = C[0:2, 0:4]: local (r, c) -> global (r, c)
    # C12 = C[0:2, 4:8]: local (r, c) -> global (r, c+4)
    # C21 = C[2:4, 0:4]: local (r, c) -> global (r+2, c)
    # C22 = C[2:4, 4:8]: local (r, c) -> global (r+2, c+4)
    
    # For each block factor Mi, the effective A is a 2x2 and effective B is 2x4.
    # I'll split effective B into left 2x2 and right 2x2.
    
    # Let me define the block-level Strassen factors with their A-block and B-block definitions.
    
    # Each block factor has:
    # - A_blocks: list of (block_id, coeff) where block_id in {11,12,21,22}
    # - B_blocks: list of (block_id, coeff) where block_id in {11,12,21,22}
    # - C_blocks: list of (block_id, coeff) where block_id in {11,12,21,22}
    
    block_factors = []
    
    # M1 = (A11+A22)(B11+B22)
    # A: {11:1, 22:1}, B: {11:1, 22:1}, C: {11:1, 22:-1}
    block_factors.append(({11:1, 22:1}, {11:1, 22:1}, {11:1, 22:-1}))
    # M2 = (A21+A22)B11
    # A: {21:1, 22:1}, B: {11:1}, C: {21:1, 22:-1}
    block_factors.append(({21:1, 22:1}, {11:1}, {21:1, 22:-1}))
    # M3 = A11(B12-B22)
    # A: {11:1}, B: {12:1, 22:-1}, C: {12:1, 22:1}
    block_factors.append(({11:1}, {12:1, 22:-1}, {12:1, 22:1}))
    # M4 = A22(B21-B11)
    # A: {22:1}, B: {21:1, 11:-1}, C: {21:1, 11:1}
    block_factors.append(({22:1}, {21:1, 11:-1}, {21:1, 11:1}))
    # M5 = (A11+A12)B22
    # A: {11:1, 12:1}, B: {22:1}, C: {12:1, 11:-1}
    block_factors.append(({11:1, 12:1}, {22:1}, {12:1, 11:-1}))
    # M6 = (A21-A11)(B11+B12)
    # A: {21:1, 11:-1}, B: {11:1, 12:1}, C: {22:1}
    block_factors.append(({21:1, 11:-1}, {11:1, 12:1}, {22:1}))
    # M7 = (A12-A22)(B21+B22)
    # A: {12:1, 22:-1}, B: {21:1, 22:1}, C: {11:1}
    block_factors.append(({12:1, 22:-1}, {21:1, 22:1}, {11:1}))
    
    # Now, for each block factor, the effective A is a 2x2 formed by summing A-blocks,
    # and effective B is a 2x4 formed by summing B-blocks (each 2x4).
    
    # I need to decompose each such 2x2*2x4 product into 14 scalar Strassen factors.
    
    # To do this, I'll split the 2x4 B into two 2x2 halves.
    # For a given effective B (2x4), the left half is columns 0-1, right half is columns 2-3.
    
    # But the effective B is a sum of block B's, each 2x4. So I need to handle this carefully.
    
    # Let me take a different approach. I'll directly compute, for each of the 98 factors,
    # which global (i,j), (j,k), (i,k) it involves.
    
    # I'll iterate over:
    # - block_factor_idx (0..6)
    # - sub_block (left/right, 0/1)
    # - s22_factor_idx (0..6)
    
    # For each combination, I'll determine the A, B, C entries.
    
    # First, let me define the mapping from block_id and local 2x2 index to global index.
    
    # A block mappings (local (r,c) -> global (i,j)):
    A_block_map = {
        11: lambda r, c: (r, c),
        12: lambda r, c: (r, c+2),
        21: lambda r, c: (r+2, c),
        22: lambda r, c: (r+2, c+2),
    }
    
    # B block mappings (local (r,c) -> global (j,k)):
    B_block_map = {
        11: lambda r, c: (r, c),
        12: lambda r, c: (r, c+4),
        21: lambda r, c: (r+2, c),
        22: lambda r, c: (r+2, c+4),
    }
    
    # C block mappings (local (r,c) -> global (i,k)):
    C_block_map = {
        11: lambda r, c: (r, c),
        12: lambda r, c: (r, c+4),
        21: lambda r, c: (r+2, c),
        22: lambda r, c: (r+2, c+4),
    }
    
    # Now, for each block factor, the effective A is a sum over A-blocks.
    # The local 2x2 indices in the effective A correspond to the same local indices
    # as in the source blocks.
    
    # Similarly for B, but B is 2x4, so I split into left (cols 0-1) and right (cols 2-3).
    
    # For the left sub-block of effective B, I use local columns 0,1.
    # For the right sub-block, I use local columns 2,3.
    
    # Let me build the 98 factors.
    
    factors = []  # Each factor is (A_entries, B_entries, C_entries)
    # where each is a list of (index, coeff)
    # A index: i*4+j (since n=4, m=4, A is 4x4, index = i*4+j... wait)
    
    # U has shape (n*m, rank) = (16, rank). Index = i*m+j = i*4+j.
    # V has shape (m*p, rank) = (32, rank). Index = j*p+k = j*8+k.
    # W has shape (p*n, rank) = (32, rank). Index = k*n+i = k*4+i.
    
    for bf_idx in range(7):
        A_blocks, B_blocks, C_blocks = block_factors[bf_idx]
        
        # Effective A is 2x2, formed by summing A-blocks with coefficients.
        # Effective B is 2x4, formed by summing B-blocks with coefficients.
        
        # Split effective B into left (cols 0-1) and right (cols 2-3).
        for sub in range(2):  # 0=left, 1=right
            # For each s22 factor in the 2x2*2x2 decomposition:
            for s_idx in range(7):
                s22_A, s22_B, s22_C = s22[s_idx]
                
                A_entries = {}  # global (i,j) -> coeff
                B_entries = {}  # global (j,k) -> coeff
                C_entries = {}  # global (i,k) -> coeff
                
                # Process A part:
                for local_idx, a_coeff in s22_A.items():
                    local_r = local_idx // 2
                    local_c = local_idx % 2
                    for block_id, block_coeff in A_blocks.items():
                        global_i, global_j = A_block_map[block_id](local_r, local_c)
                        key = (global_i, global_j)
                        A_entries[key] = A_entries.get(key, 0) + a_coeff * block_coeff
                
                # Process B part:
                # sub=0: local cols 0,1; sub=1: local cols 2,3
                col_offset = sub * 2
                for local_idx, b_coeff in s22_B.items():
                    local_r = local_idx // 2
                    local_c = local_idx % 2
                    global_local_c = local_c + col_offset  # This is 0,1 for sub=0 or 2,3 for sub=1
                    # But wait, the local 2x2 indices are 0,1,2,3 corresponding to (0,0),(0,1),(1,0),(1,1)
                    # For the left sub-block, we use local cols 0,1 -> global_local_c = 0,1
                    # For the right sub-block, we use local cols 0,1 -> global_local_c = 2,3
                    # So global_local_c = local_c + col_offset
                    for block_id, block_coeff in B_blocks.items():
                        global_j, global_k = B_block_map[block_id](local_r, global_local_c)
                        key = (global_j, global_k)
                        B_entries[key] = B_entries.get(key, 0) + b_coeff * block_coeff
                
                # Process C part:
                # The C part from s22 gives local 2x2 C entries.
                # These map to the corresponding sub-block of the effective C.
                # For sub=0 (left), C local cols 0,1 -> global_local_c = 0,1
                # For sub=1 (right), C local cols 0,1 -> global_local_c = 2,3
                for local_idx, c_coeff in s22_C.items():
                    local_r = local_idx // 2
                    local_c = local_idx % 2
                    global_local_c = local_c + col_offset
                    for block_id, block_coeff in C_blocks.items():
                        global_i, global_k = C_block_map[block_id](local_r, global_local_c)
                        key = (global_i, global_k)
                        C_entries[key] = C_entries.get(key, 0) + c_coeff * block_coeff
                
                # Convert to lists
                A_list = [(i * 4 + j, coeff) for (i, j), coeff in A_entries.items() if coeff != 0]
                B_list = [(j * 8 + k, coeff) for (j, k), coeff in B_entries.items() if coeff != 0]
                C_list = [(k * 4 + i, coeff) for (i, k), coeff in C_entries.items() if coeff != 0]
                
                if A_list and B_list and C_list:
                    factors.append((A_list, B_list, C_list))
    
    # Now build U, V, W matrices.
    rank = len(factors)
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(32)]
    W = [[0] * rank for _ in range(32)]
    
    for r, (A_list, B_list, C_list) in enumerate(factors):
        for idx, coeff in A_list:
            U[idx][r] = coeff
        for idx, coeff in B_list:
            V[idx][r] = coeff
        for idx, coeff in C_list:
            W[idx][r] = coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
