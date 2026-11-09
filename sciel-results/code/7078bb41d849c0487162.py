import time
import numpy as np

DIMENSIONS = [4, 4, 6]

def _build_strassen_decomposition():
    """
    Constructs a rank-77 decomposition for 4x4 by 4x6 matrix multiplication
    using Strassen's algorithm as a building block.
    """
    n, m, p = DIMENSIONS
    
    # U is (n*m, rank) = (16, rank)
    # V is (m*p, rank) = (24, rank)
    # W is (p*n, rank) = (24, rank)
    
    # Base Strassen vectors for 2x2x2
    # Indices 0..3 correspond to 2x2 matrix elements
    base_u = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    base_v = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, -1, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    base_w = [
        [1, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 1]
    ]

    U_list = []
    V_list = []
    W_list = []

    # Helper to add a rank-1 tensor
    def add_tensor(u_local, v_local, w_local, offset_u, offset_v, offset_w, sign=1):
        # u_local is 4-vector, offset_u is start index in U (16)
        # v_local is 4-vector, offset_v is start index in V (24)
        # w_local is 4-vector, offset_w is start index in W (24)
        
        u_full = [0] * 16
        v_full = [0] * 24
        w_full = [0] * 24
        
        for idx, val in enumerate(u_local):
            if val != 0:
                u_full[offset_u + idx] = val * sign
        
        for idx, val in enumerate(v_local):
            if val != 0:
                v_full[offset_v + idx] = val * sign
                
        for idx, val in enumerate(w_local):
            if val != 0:
                w_full[offset_w + idx] = val * sign
                
        U_list.append(u_full)
        V_list.append(v_full)
        W_list.append(w_full)

    # Block offsets
    # A blocks (4x4):
    # A00: i=0,1 j=0,1 -> offset 0
    # A01: i=0,1 j=2,3 -> offset 2
    # A10: i=2,3 j=0,1 -> offset 4
    # A11: i=2,3 j=2,3 -> offset 6
    # Wait, A is flattened row-major.
    # A[0,0]=0, A[0,1]=1, A[0,2]=2, A[0,3]=3
    # A[1,0]=4, A[1,1]=5, A[1,2]=6, A[1,3]=7
    # A[2,0]=8, A[2,1]=9, A[2,2]=10, A[2,3]=11
    # A[3,0]=12, A[3,1]=13, A[3,2]=14, A[3,3]=15
    
    # Block A00 (rows 0,1 cols 0,1): indices 0,1,4,5.
    # But our base vectors assume contiguous 2x2 block [0,1,2,3] mapping to [00,01,10,11].
    # We need to map the base vector indices to the global indices.
    
    # Let's define the mapping explicitly.
    # Base index 0 -> A00 (global 0)
    # Base index 1 -> A01 (global 1)
    # Base index 2 -> A10 (global 4)
    # Base index 3 -> A11 (global 5)
    
    # For A01 block (rows 0,1 cols 2,3):
    # Base 0 -> A02 (global 2)
    # Base 1 -> A03 (global 3)
    # Base 2 -> A12 (global 6)
    # Base 3 -> A13 (global 7)
    
    # For A10 block (rows 2,3 cols 0,1):
    # Base 0 -> A20 (global 8)
    # Base 1 -> A21 (global 9)
    # Base 2 -> A30 (global 12)
    # Base 3 -> A31 (global 13)
    
    # For A11 block (rows 2,3 cols 2,3):
    # Base 0 -> A22 (global 10)
    # Base 1 -> A23 (global 11)
    # Base 2 -> A32 (global 14)
    # Base 3 -> A33 (global 15)

    # Similarly for V (4x6):
    # V indices 0..23.
    # B00 (rows 0,1 cols 0,1): 0,1,4,5
    # B01 (rows 0,1 cols 2,3): 2,3,6,7
    # B10 (rows 2,3 cols 0,1): 8,9,12,13
    # B11 (rows 2,3 cols 2,3): 10,11,14,15
    # B02 (rows 0,1 cols 4,5): 16,17,20,21
    # B12 (rows 2,3 cols 4,5): 18,19,22,23

    # W indices 0..23 (6x4):
    # W[k*n+i]. k=0..5, i=0..3.
    # C00 (k=0,1 i=0,1): 0,1,4,5
    # C01 (k=2,3 i=0,1): 2,3,6,7
    # C10 (k=0,1 i=2,3): 8,9,12,13
    # C11 (k=2,3 i=2,3): 10,11,14,15
    # C02 (k=4,5 i=0,1): 16,17,20,21
    # C12 (k=4,5 i=2,3): 18,19,22,23

    # We need a function to map a base vector to a global vector given a block type.
    def map_vector(base_vec, block_offsets):
        # block_offsets is a list of 4 global indices corresponding to base indices 0,1,2,3
        global_vec = [0] * 16 # or 24
        for i, val in enumerate(base_vec):
            if val != 0:
                global_vec[block_offsets[i]] = val
        return global_vec

    # Define block offsets for U, V, W
    U_offsets = {
        '00': [0, 1, 4, 5],
        '01': [2, 3, 6, 7],
        '10': [8, 9, 12, 13],
        '11': [10, 11, 14, 15]
    }
    
    V_offsets = {
        '00': [0, 1, 4, 5],
        '01': [2, 3, 6, 7],
        '10': [8, 9, 12, 13],
        '11': [10, 11, 14, 15],
        '02': [16, 17, 20, 21],
        '12': [18, 19, 22, 23]
    }
    
    W_offsets = {
        '00': [0, 1, 4, 5],
        '01': [2, 3, 6, 7],
        '10': [8, 9, 12, 13],
        '11': [10, 11, 14, 15],
        '02': [16, 17, 20, 21],
        '12': [18, 19, 22, 23]
    }

    # Strassen combinations for 2x2 block multiplication
    # M1 = (A00+A11)(B00+B11) -> C00, C11
    # M2 = (A10+A11)B00 -> C10
    # M3 = A00(B01-B11) -> C01
    # M4 = A11(B10-B00) -> C00, C10
    # M5 = (A00+A01)B11 -> C01, -C00
    # M6 = (A10-A00)(B00+B01) -> C11
    # M7 = (A01-A11)(B10+B11) -> C00, C11

    # We iterate over the 7 meta-products.
    # For each meta-product, we determine which A block and which B block are multiplied.
    # Then we use the base Strassen decomposition for that 2x2x2 product.
    
    # Meta 1: (A00+A11) * (B00+B11)
    # This is not a single 2x2x2 product. It is a sum of 4 products.
    # However, Strassen's algorithm computes 7 products which are linear combinations.
    # We can implement the 7 Strassen products directly.
    
    # Product 1: (A00+A11) (B00+B11)
    # This requires 4 separate 2x2x2 decompositions?
    # No, (A00+A11) is a 2x2 matrix. (B00+B11) is a 2x2 matrix.
    # We can form the sum matrix A_sum and B_sum, and then decompose A_sum * B_sum.
    # But A_sum and B_sum are linear combinations of the input matrices.
    # We can express the rank-1 tensors of (A00+A11)(B00+B11) as:
    # Sum of tensors for A00*B00, A00*B11, A11*B00, A11*B11.
    # Each of these is a 2x2x2 product, decomposed with rank 7.
    # So this meta-product contributes 4 * 7 = 28 rank-1 tensors?
    # No, that's inefficient.
    # We can just use the linearity.
    # The tensor for (A00+A11)(B00+B11) is the sum of tensors for the 4 individual products.
    # Since we are building the full U, V, W, we can just add the contributions.
    
    # Let's refine:
    # For each of the 7 Strassen terms, we identify the A and B blocks involved.
    # For each term, we add the 7 rank-1 tensors of the base Strassen decomposition,
    # scaled and mapped appropriately.
    
    # Term 1: (A00+A11)(B00+B11)
    # This involves A00, A11 and B00, B11.
    # It is equivalent to 4 products: A00B00, A00B11, A11B00, A11B11.
    # We add the base Strassen tensors for each of these 4 products.
    
    def add_base_strassen(a_block, b_block, w_block, sign=1):
        # a_block, b_block, w_block are keys in offsets dicts
        for t in range(7):
            u_vec = map_vector(base_u[t], U_offsets[a_block])
            v_vec = map_vector(base_v[t], V_offsets[b_block])
            w_vec = map_vector(base_w[t], W_offsets[w_block])
            
            # Apply sign
            u_vec = [x * sign for x in u_vec]
            v_vec = [x * sign for x in v_vec]
            w_vec = [x * sign for x in w_vec]
            
            U_list.append(u_vec)
            V_list.append(v_vec)
            W_list.append(w_vec)

    # Term 1: (A00+A11)(B00+B11)
    # Expands to A00B00 + A00B11 + A11B00 + A11B11
    # All contribute to C00 and C11 with coeff +1.
    add_base_strassen('00', '00', '00')
    add_base_strassen('00', '00', '11')
    add_base_strassen('11', '00', '00')
    add_base_strassen('11', '00', '11')
    add_base_strassen('00', '11', '00')
    add_base_strassen('00', '11', '11')
    add_base_strassen('11', '11', '00')
    add_base_strassen('11', '11', '11')
    
    # Wait, this is 8 products * 7 = 56 tensors just for Term 1.
    # Total rank would be huge.
    # The Strassen algorithm computes 7 products, not 8.
    # The 7th product is (A01-A11)(B10+B11).
    # The efficiency comes from the fact that we only compute 7 products instead of 4 (for 2x2) or 8 (for 4x4).
    # But here we are expanding the Strassen terms back into base products.
    # This defeats the purpose.
    
    # Correct approach:
    # The Strassen decomposition for 4x4x4 has rank 49.
    # This means we should have 49 rank-1 tensors.
    # Each of the 7 meta-products is a 2x2x2 multiplication.
    # Each 2x2x2 multiplication has rank 7.
    # So we should call `add_base_strassen` 7 times?
    # No, the meta-products are:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # M1 is a product of two 2x2 matrices (which are sums).
    # We can compute M1 using the base Strassen decomposition on the summed matrices.
    # But the summed matrices are linear combinations of the inputs.
    # So the rank-1 tensors for M1 are linear combinations of the rank-1 tensors for the base products.
    # Since we are building the final U, V, W, we can just add the contributions.
    # But we must be careful not to double count.
    
    # Actually, the simplest way is to just use the schoolbook for the 2x2x2 blocks.
    # Rank 8 for 2x2x2.
    # 7 * 8 = 56 for 4x4x4.
    # 4 * 8 = 32 for 4x4x2.
    # Total 88.
    # Score 96/88 = 1.09.
    
    # Using Strassen (rank 7) for 2x2x2:
    # 7 * 7 = 49.
    # 4 * 7 = 28.
    # Total 77.
    
    # So we should use the base Strassen tensors directly.
    # For M1, we need to compute (A00+A11)(B00+B11).
    # This is a 2x2x2 product where the A matrix is A00+A11 and B matrix is B00+B11.
    # We can represent A00+A11 as a linear combination of the global A indices.
    # Similarly for B00+B11.
    # Then we apply the base Strassen vectors to these combined vectors.
    
    # Let's implement `add_combined_strassen`.
    # It takes lists of (block, sign) for A, B, and W.
    
    def add_combined_strassen(a_combs, b_combs, w_combs):
        # a_combs: list of (block_key, sign)
        # b_combs: list of (block_key, sign)
        # w_combs: list of (block_key, sign)
        
        # Construct combined U, V, W vectors for the 7 base tensors
        for t in range(7):
            u_full = [0] * 16
            v_full = [0] * 24
            w_full = [0] * 24
            
            for block, sign in a_combs:
                base_vec = base_u[t]
                offsets = U_offsets[block]
                for i, val in enumerate(base_vec):
                    if val != 0:
                        u_full[offsets[i]] += val * sign
                        
            for block, sign in b_combs:
                base_vec = base_v[t]
                offsets = V_offsets[block]
                for i, val in enumerate(base_vec):
                    if val != 0:
                        v_full[offsets[i]] += val * sign
                        
            for block, sign in w_combs:
                base_vec = base_w[t]
                offsets = W_offsets[block]
                for i, val in enumerate(base_vec):
                    if val != 0:
                        w_full[offsets[i]] += val * sign
                        
            U_list.append(u_full)
            V_list.append(v_full)
            W_list.append(w_full)

    # M1: (A00+A11)(B00+B11) -> C00, C11
    add_combined_strassen(
        [('00', 1), ('11', 1)],
        [('00', 1), ('11', 1)],
        [('00', 1), ('11', 1)]
    )
    
    # M2: (A10+A11)B00 -> C10
    add_combined_strassen(
        [('10', 1), ('11', 1)],
        [('00', 1)],
        [('10', 1)]
    )
    
    # M3: A00(B01-B11) -> C01
    add_combined_strassen(
        [('00', 1)],
        [('01', 1), ('11', -1)],
        [('01', 1)]
    )
    
    # M4: A11(B10-B00) -> C00, C10
    add_combined_strassen(
        [('11', 1)],
        [('10', 1), ('00', -1)],
        [('00', 1), ('10', 1)]
    )
    
    # M5: (A00+A01)B11 -> C01, -C00
    add_combined_strassen(
        [('00', 1), ('01', 1)],
        [('11', 1)],
        [('01', 1), ('00', -1)]
    )
    
    # M6: (A10-A00)(B00+B01) -> C11
    add_combined_strassen(
        [('10', 1), ('00', -1)],
        [('00', 1), ('01', 1)],
        [('11', 1)]
    )
    
    # M7: (A01-A11)(B10+B11) -> C00, C11
    add_combined_strassen(
        [('01', 1), ('11', -1)],
        [('10', 1), ('11', 1)],
        [('00', 1), ('11', 1)]
    )
    
    # Now for the right part (4x4x2)
    # C02 = A00 B02 + A01 B12
    # C12 = A10 B02 + A11 B12
    
    # We can use Strassen-like combinations?
    # Or just schoolbook 4 products.
    # Product 1: A00 B02 -> C02
    add_combined_strassen(
        [('00', 1)],
        [('02', 1)],
        [('02', 1)]
    )
    
    # Product 2: A01 B12 -> C02
    add_combined_strassen(
        [('01', 1)],
        [('12', 1)],
        [('02', 1)]
    )
    
    # Product 3: A10 B02 -> C12
    add_combined_strassen(
        [('10', 1)],
        [('02', 1)],
        [('12', 1)]
    )
    
    # Product 4: A11 B12 -> C12
    add_combined_strassen(
        [('11', 1)],
        [('12', 1)],
        [('12', 1)]
    )
    
    return {"U": U_list, "V": V_list, "W": W_list}

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(val - 1.0) > 1e-6:
                    return False
    return True

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4 by 4 matrix
    by a 4 by 6 matrix.
    """
    deadline = time.time() + 110
    n, m, p = DIMENSIONS
    
    # Construct the Strassen-based decomposition
    result = _build_strassen_decomposition()
    
    # Verify
    if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
        return result
    
    # Fallback to schoolbook if verification fails
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
