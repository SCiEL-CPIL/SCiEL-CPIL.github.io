import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    
    # Base 2x2x2 Strassen decomposition
    # u: coeffs for a,b,c,d (indices 0,1,2,3)
    # v: coeffs for e,f,g,h (indices 0,1,2,3)
    # w: coeffs for p,q,r,s (indices 0,1,2,3) -> mapped to p=0, r=1, q=2, s=3?
    # Let's stick to local indices 0..3 for w, and map later.
    # Local C indices: 0:00, 1:10, 2:01, 3:11 (row, col)
    # But W index is k*n+i.
    # For 2x2x2, n=2.
    # C_00 (i=0,k=0) -> 0*2+0=0
    # C_10 (i=1,k=0) -> 0*2+1=1
    # C_01 (i=0,k=1) -> 1*2+0=2
    # C_11 (i=1,k=1) -> 1*2+1=3
    # So local w indices 0,1,2,3 correspond to C_00, C_10, C_01, C_11.
    
    base_terms = [
        # m1 = (a+d)(e+h)
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),
        # m2 = (c+d)e
        ({2:1, 3:1}, {0:1}, {1:1, 3:-1}),
        # m3 = a(f-h)
        ({0:1}, {1:1, 3:-1}, {2:1, 3:1}),
        # m4 = d(g-e)
        ({3:1}, {2:1, 0:-1}, {0:1, 1:1}),
        # m5 = (a+b)h
        ({0:1, 1:1}, {3:1}, {0:-1, 2:1}),
        # m6 = (c-a)(e+f)
        ({2:1, 0:-1}, {0:1, 1:1}, {3:1}),
        # m7 = (b-d)(g+h)
        ({1:1, 3:-1}, {2:1, 3:1}, {0:1}),
    ]
    
    # Strassen 4x4x4 structure
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
    
    # Blocks in A:
    # A11: rows 0-1, cols 0-1. Global indices: 0,1,4,5
    # A12: rows 0-1, cols 2-3. Global indices: 2,3,6,7
    # A21: rows 2-3, cols 0-1. Global indices: 8,9,12,13
    # A22: rows 2-3, cols 2-3. Global indices: 10,11,14,15
    
    # Blocks in B (for 4x4x4):
    # B11: rows 0-1, cols 0-1. Global indices (j*4+k): 0,1,4,5
    # B12: rows 0-1, cols 2-3. Global indices: 2,3,6,7
    # B21: rows 2-3, cols 0-1. Global indices: 8,9,12,13
    # B22: rows 2-3, cols 2-3. Global indices: 10,11,14,15
    
    # Blocks in C (for 4x4x4):
    # C11: rows 0-1, cols 0-1. Global indices (k*4+i): 
    #   C_00 (i=0,k=0) -> 0
    #   C_10 (i=1,k=0) -> 1
    #   C_01 (i=0,k=1) -> 4
    #   C_11 (i=1,k=1) -> 5
    #   So indices 0,1,4,5
    # C12: rows 0-1, cols 2-3.
    #   C_02 (i=0,k=2) -> 8
    #   C_12 (i=1,k=2) -> 9
    #   C_03 (i=0,k=3) -> 12
    #   C_13 (i=1,k=3) -> 13
    #   So indices 8,9,12,13
    # C21: rows 2-3, cols 0-1.
    #   C_20 (i=2,k=0) -> 2
    #   C_30 (i=3,k=0) -> 3
    #   C_21 (i=2,k=1) -> 6
    #   C_31 (i=3,k=1) -> 7
    #   So indices 2,3,6,7
    # C22: rows 2-3, cols 2-3.
    #   C_22 (i=2,k=2) -> 10
    #   C_32 (i=3,k=2) -> 11
    #   C_23 (i=2,k=3) -> 14
    #   C_33 (i=3,k=3) -> 15
    #   So indices 10,11,14,15

    def get_block_indices(block_id):
        # Returns list of 4 global indices for a 2x2 block
        # block_id: 0=A11/B11/C11, 1=A12/B12/C12, 2=A21/B21/C21, 3=A22/B22/C22
        if block_id == 0:
            return [0, 1, 4, 5]
        elif block_id == 1:
            return [2, 3, 6, 7]
        elif block_id == 2:
            return [8, 9, 12, 13]
        else:
            return [10, 11, 14, 15]
            
    # Define meta-terms
    # Each meta-term i has:
    # a_blocks: list of (block_id, coeff)
    # b_blocks: list of (block_id, coeff)
    # c_blocks: list of (block_id, coeff)
    
    meta_terms = [
        # M1
        ([(0,1), (3,1)], [(0,1), (3,1)], [(0,1), (3,1)]),
        # M2
        ([(2,1), (3,1)], [(0,1)], [(2,1), (3,-1)]),
        # M3
        ([(0,1)], [(1,1), (3,-1)], [(1,1), (3,1)]),
        # M4
        ([(3,1)], [(2,1), (0,-1)], [(0,1), (2,1)]),
        # M5
        ([(0,1), (1,1)], [(3,1)], [(0,-1), (1,1)]),
        # M6
        ([(2,1), (0,-1)], [(0,1), (1,1)], [(3,1)]),
        # M7
        ([(1,1), (3,-1)], [(2,1), (3,1)], [(0,1)]),
    ]
    
    # We need to generate 49 terms for 4x4x4
    terms_444 = []
    
    for i in range(7):
        a_defs, b_defs, c_defs = meta_terms[i]
        
        # We need to map the local 2x2 indices to global 4x4 indices
        # The local 2x2 multiplication uses a specific A' and B'.
        # A' is a 2x2 matrix formed by linear combinations of A blocks.
        # However, the base decomposition assumes A' is a single 2x2 matrix.
        # So we expand A' = sum(alpha * A_block).
        # The term in the decomposition is (sum alpha A_block) * (sum beta B_block).
        # This is not a single rank-1 term in the global space unless we distribute.
        # But wait, the base decomposition is for the bilinear form.
        # If A' is a sum, we can't just plug it in directly into the rank-7 terms?
        # No, the rank-7 decomposition is linear in A' and B'.
        # So if A' = A1 + A2, then the terms involving A1 and A2 are separate?
        # No, the rank-7 decomposition expresses the product of two 2x2 matrices.
        # If the inputs are sums, we must distribute.
        # Example: (A1+A2)B. This is A1B + A2B.
        # Each is a 2x2 multiplication.
        # So we need to generate terms for each combination of A-block and B-block?
        # No, that would increase the rank.
        # The Strassen algorithm works because it groups the sums.
        # The "multiplication" $M_i$ is a single operation in the recursive step.
        # In the tensor decomposition, a "multiplication" of size $2 \times 2 \times 2$ is rank 7.
        # If the inputs to this $2 \times 2$ multiplication are linear combinations of the original $4 \times 4$ blocks,
        # does the rank remain 7?
        # Yes, because the tensor $M_{2,2,2}$ is fixed. The inputs are just vectors in the input space.
        # The input space for $M_{2,2,2}$ is 4-dimensional (the entries of the 2x2 matrix).
        # Here, the "entries" of the 2x2 matrix $A'$ are themselves linear combinations of the entries of $A$.
        # So we are composing the tensor with linear maps.
        # The rank of the composed tensor is at most the rank of the original tensor.
        # So yes, each $M_i$ contributes at most 7 terms to the global rank.
        
        # So, for each base term t in the 7 terms of M_i:
        # The u vector is the sum of (alpha * u_t_local_mapped) for each A block.
        # The v vector is the sum of (beta * v_t_local_mapped) for each B block.
        # The w vector is the sum of (gamma * w_t_local_mapped) for each C block.
        
        for t_idx in range(7):
            u_local, v_local, w_local = base_terms[t_idx]
            
            u_global = {}
            v_global = {}
            w_global = {}
            
            # Expand u
            for block_id, coeff in a_defs:
                indices = get_block_indices(block_id)
                for local_idx, val in u_local.items():
                    global_idx = indices[local_idx]
                    u_global[global_idx] = u_global.get(global_idx, 0) + coeff * val
            
            # Expand v
            for block_id, coeff in b_defs:
                indices = get_block_indices(block_id)
                for local_idx, val in v_local.items():
                    global_idx = indices[local_idx]
                    v_global[global_idx] = v_global.get(global_idx, 0) + coeff * val
            
            # Expand w
            for block_id, coeff in c_defs:
                indices = get_block_indices(block_id)
                for local_idx, val in w_local.items():
                    global_idx = indices[local_idx]
                    w_global[global_idx] = w_global.get(global_idx, 0) + coeff * val
            
            terms_444.append((u_global, v_global, w_global))
            
    # Now we have 49 terms for 4x4x4.
    # We need to construct U, V, W for 4x4x8.
    # Rank will be 98.
    rank = 98
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(32)]
    W = [[0] * rank for _ in range(32)]
    
    # Block 1: B_left (cols 0-3), C_left (cols 0-3)
    # B indices in 4x4x4 context were 0..15.
    # In 4x4x8 context, B is 4x8.
    # B_left corresponds to k=0..3.
    # Index j*8 + k.
    # For j=0..1, k=0..3: 0..3, 8..11.
    # For j=2..3, k=0..3: 16..19, 24..27.
    # Wait, my previous mapping for B in 4x4x4 was j*4+k.
    # So local index 0 (j=0,k=0) -> global 0*8+0 = 0.
    # Local index 1 (j=0,k=1) -> global 0*8+1 = 1.
    # Local index 2 (j=0,k=2) -> global 0*8+2 = 2.
    # Local index 3 (j=0,k=3) -> global 0*8+3 = 3.
    # Local index 4 (j=1,k=0) -> global 1*8+0 = 8.
    # ...
    
    def map_b_index_444_to_448(idx):
        j = idx // 4
        k = idx % 4
        return j * 8 + k
        
    def map_c_index_444_to_448(idx, offset_k=0):
        # idx is k*4+i from 4x4x4 context (n=4)
        k = idx // 4
        i = idx % 4
        new_k = k + offset_k
        return new_k * 4 + i
        
    # Fill U, V, W for Block 1
    for col in range(49):
        u, v, w = terms_444[col]
        for idx, val in u.items():
            U[idx][col] = val
        for idx, val in v.items():
            new_idx = map_b_index_444_to_448(idx)
            V[new_idx][col] = val
        for idx, val in w.items():
            new_idx = map_c_index_444_to_448(idx, 0)
            W[new_idx][col] = val
            
    # Fill U, V, W for Block 2
    # Same U terms (A is shared)
    # V terms shifted to B_right (k=4..7)
    # W terms shifted to C_right (k=4..7)
    
    for col in range(49, 98):
        u, v, w = terms_444[col - 49]
        for idx, val in u.items():
            U[idx][col] = val
        for idx, val in v.items():
            # B_right: k in 4..7
            # Local idx in 444 context: j*4 + k_local (k_local 0..3)
            # Global idx in 448: j*8 + (k_local + 4)
            j = idx // 4
            k_local = idx % 4
            new_k = k_local + 4
            new_idx = j * 8 + new_k
            V[new_idx][col] = val
        for idx, val in w.items():
            new_idx = map_c_index_444_to_448(idx, 4)
            W[new_idx][col] = val
            
    return {"U": U, "V": V, "W": W}
