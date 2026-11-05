import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=7
    
    # Construct the target multiplication tensor T
    # T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
    # We represent T as a matrix M of shape (n*m*p, n*m*p) for rank-1 decomposition?
    # No, we need U (n*m x r), V (m*p x r), W (p*n x r)
    # such that T_{a,b,c} = sum_r U_{a,r} V_{b,r} W_{c,r}
    
    # The tensor T has indices (a, b, c) where:
    # a = i*m + j (0..15)
    # b = j*p + k (0..27)
    # c = k*n + i (0..27)
    
    # We use a "randomized" approach to find a low-rank decomposition.
    # The directive suggests fixing U and V to random generic configurations and solving for W.
    # However, for exact integer/half-integer solutions, random reals won't work directly.
    # We can try to find a structured decomposition using known algorithms (like Strassen) or 
    # verify if a lower rank is possible.
    
    # Given the constraints and the need for exactness, let's stick to the robust Schoolbook method 
    # unless we can implement a verified Strassen-like block decomposition.
    
    # Let's attempt a block decomposition for 4x4x7.
    # We can view this as 4x4x4 (Strassen, rank 49) + 4x4x3 (Schoolbook, rank 48) = 97?
    # Or maybe 4x4x4 (Strassen, rank 49) + 4x4x3 (Strassen-like? No, 4x4x3 is small).
    
    # Let's try to implement the 4x4x4 Strassen part correctly this time.
    # If it fails, fall back to Schoolbook (112).
    
    try:
        # --- 4x4x4 Strassen Decomposition ---
        # A is 4x4, B is 4x4, C is 4x4.
        # We partition A, B, C into 2x2 blocks of size 2x2.
        # A = [[A11, A12], [A21, A22]]
        # B = [[B11, B12], [B21, B22]]
        # C = [[C11, C12], [C21, C22]]
        
        # Strassen steps for 2x2 matrices:
        # M1 = (A11 + A22) * (B11 + B22)
        # M2 = (A21 + A22) * B11
        # M3 = A11 * (B12 - B22)
        # M4 = A22 * (B21 - B11)
        # M5 = (A11 + A12) * B22
        # M6 = (A21 - A11) * (B11 + B12)
        # M7 = (A12 - A22) * (B21 + B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Each M is a 2x2 matrix product. We decompose each 2x2 product using Strassen (rank 7).
        # So total rank for 4x4x4 is 7 * 7 = 49.
        
        # We need to map the local 2x2 indices to the global 4x4 indices.
        # Global A index: a = i*4 + j.
        # Block A11: i in {0,1}, j in {0,1} -> a in {0,1,4,5}
        # Block A12: i in {0,1}, j in {2,3} -> a in {2,3,6,7}
        # Block A21: i in {2,3}, j in {0,1} -> a in {8,9,12,13}
        # Block A22: i in {2,3}, j in {2,3} -> a in {10,11,14,15}
        
        # Similarly for B (4x4) and C (4x4).
        
        # Let's define the 2x2 Strassen steps as (u, v, w) vectors of length 4.
        # u: coeffs for A elements [A00, A01, A10, A11]
        # v: coeffs for B elements [B00, B01, B10, B11]
        # w: coeffs for C elements [C00, C01, C10, C11]
        
        strassen_2x2 = [
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # M1
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]), # M2
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),  # M3
            ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),  # M4
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),  # M5
            ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]),  # M6
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),  # M7
        ]
        
        # Top-level Strassen steps for 4x4x4.
        # Each step is defined by which blocks of A and B are summed/subtracted,
        # and how the resulting 2x2 product contributes to C blocks.
        
        # We will generate 49 columns.
        U_cols = []
        V_cols = []
        W_cols = []
        
        # Helper to get global indices for a block
        def get_block_indices(block_type, row_block, col_block):
            # block_type: 'A', 'B', or 'C'
            # row_block, col_block: 0 or 1
            # Returns list of 4 global indices in order [00, 01, 10, 11] local
            if block_type == 'A':
                # A is 4x4, index i*4+j
                base_i = row_block * 2
                base_j = col_block * 2
                return [
                    (base_i + 0) * 4 + (base_j + 0), # 00
                    (base_i + 0) * 4 + (base_j + 1), # 01
                    (base_i + 1) * 4 + (base_j + 0), # 10
                    (base_i + 1) * 4 + (base_j + 1)  # 11
                ]
            elif block_type == 'B':
                # B is 4x4, index i*4+j
                base_i = row_block * 2
                base_j = col_block * 2
                return [
                    (base_i + 0) * 4 + (base_j + 0), # 00
                    (base_i + 0) * 4 + (base_j + 1), # 01
                    (base_i + 1) * 4 + (base_j + 0), # 10
                    (base_i + 1) * 4 + (base_j + 1)  # 11
                ]
            elif block_type == 'C':
                # C is 4x4, index i*4+j
                base_i = row_block * 2
                base_j = col_block * 2
                return [
                    (base_i + 0) * 4 + (base_j + 0), # 00
                    (base_i + 0) * 4 + (base_j + 1), # 01
                    (base_i + 1) * 4 + (base_j + 0), # 10
                    (base_i + 1) * 4 + (base_j + 1)  # 11
                ]
        
        # Define the 7 top-level steps
        # Each step: (a_blocks, b_blocks, c_contrib)
        # a_blocks: dict {block_name: coeff} where block_name is (row, col)
        # b_blocks: dict {block_name: coeff}
        # c_contrib: dict {block_name: coeff} for the resulting 2x2 product
        
        top_steps = [
            # M1 = (A11 + A22) * (B11 + B22)
            (
                {(0,0): 1, (1,1): 1},
                {(0,0): 1, (1,1): 1},
                {(0,0): 1, (1,1): 1} # C11 and C22 get M1
            ),
            # M2 = (A21 + A22) * B11
            (
                {(1,0): 1, (1,1): 1},
                {(0,0): 1},
                {(1,0): 1, (1,1): -1} # C21 gets M2, C22 gets -M2
            ),
            # M3 = A11 * (B12 - B22)
            (
                {(0,0): 1},
                {(0,1): 1, (1,1): -1},
                {(0,1): 1, (1,1): 1} # C12 gets M3, C22 gets M3
            ),
            # M4 = A22 * (B21 - B11)
            (
                {(1,1): 1},
                {(1,0): 1, (0,0): -1},
                {(0,0): 1, (1,0): 1} # C11 gets M4, C21 gets M4
            ),
            # M5 = (A11 + A12) * B22
            (
                {(0,0): 1, (0,1): 1},
                {(1,1): 1},
                {(0,0): -1, (0,1): 1} # C11 gets -M5, C12 gets M5
            ),
            # M6 = (A21 - A11) * (B11 + B12)
            (
                {(1,0): 1, (0,0): -1},
                {(0,0): 1, (0,1): 1},
                {(1,1): 1} # C22 gets M6
            ),
            # M7 = (A12 - A22) * (B21 + B22)
            (
                {(0,1): 1, (1,1): -1},
                {(1,0): 1, (1,1): 1},
                {(0,0): 1} # C11 gets M7
            ),
        ]
        
        # Iterate through top-level steps
        for t_idx in range(7):
            a_blocks, b_blocks, c_contrib = top_steps[t_idx]
            
            # Iterate through 2x2 Strassen steps
            for s_idx in range(7):
                u_local, v_local, w_local = strassen_2x2[s_idx]
                
                u_vec = [0] * (n*m) # 16
                v_vec = [0] * (m*4) # 16 (for 4x4 part of B)
                w_vec = [0] * (4*n) # 16 (for 4x4 part of C)
                
                # Process U
                for local_idx, coeff in enumerate(u_local):
                    if coeff == 0: continue
                    # local_idx 0:00, 1:01, 2:10, 3:11
                    # This coefficient applies to the *sum* of the specified A blocks.
                    # So we need to add coeff to each global index in each specified block.
                    # Wait, u_local[0] is the coefficient for the (0,0) element of the *summed* matrix.
                    # The (0,0) element of the summed matrix is Sum_{block} coeff_block * A_block[0,0].
                    # So for each block in a_blocks, we add u_local[0] * coeff_block to the global index corresponding to [0,0] of that block.
                    
                    # Determine local row/col from local_idx
                    l_row = local_idx // 2
                    l_col = local_idx % 2
                    
                    for (br, bc), a_coeff in a_blocks.items():
                        global_indices = get_block_indices('A', br, bc)
                        # Find the index in global_indices corresponding to l_row, l_col
                        # global_indices is [00, 01, 10, 11]
                        # local_idx 0 -> 00, 1 -> 01, 2 -> 10, 3 -> 11
                        global_idx = global_indices[local_idx]
                        u_vec[global_idx] += coeff * a_coeff
                
                # Process V
                for local_idx, coeff in enumerate(v_local):
                    if coeff == 0: continue
                    for (br, bc), b_coeff in b_blocks.items():
                        global_indices = get_block_indices('B', br, bc)
                        global_idx = global_indices[local_idx]
                        v_vec[global_idx] += coeff * b_coeff
                
                # Process W
                # w_local is the coefficient for the C blocks of the *result* of the 2x2 product.
                # The 2x2 product is a 2x2 matrix. Its elements contribute to the C blocks specified in c_contrib.
                # c_contrib maps (row_block, col_block) to a coefficient.
                # The element w_local[local_idx] contributes to the C block at (br, bc) with weight c_contrib[(br,bc)].
                # So for each (br, bc) in c_contrib, we add w_local[local_idx] * c_contrib[(br,bc)] to the global index.
                
                for local_idx, coeff in enumerate(w_local):
                    if coeff == 0: continue
                    for (br, bc), c_coeff in c_contrib.items():
                        global_indices = get_block_indices('C', br, bc)
                        global_idx = global_indices[local_idx]
                        w_vec[global_idx] += coeff * c_coeff
                
                U_cols.append(u_vec)
                V_cols.append(v_vec)
                W_cols.append(w_vec)
        
        # Now we have 49 columns for the 4x4x4 part.
        # We need to handle the 4x4x3 part (last 3 columns of B).
        # B is 4x7. The first 4 columns are handled above.
        # The last 3 columns (indices 4,5,6) are not handled.
        # We can use Schoolbook for the 4x4x3 part.
        # Rank for 4x4x3 schoolbook is 4*4*3 = 48.
        
        # Total rank = 49 + 48 = 97.
        
        # Add Schoolbook columns for 4x4x3
        for i in range(n):
            for j in range(m):
                for k in range(4, 7): # k=4,5,6
                    u_vec = [0] * (n*m)
                    v_vec = [0] * (m*p)
                    w_vec = [0] * (p*n)
                    
                    u_vec[i*m + j] = 1
                    v_vec[j*p + k] = 1
                    w_vec[k*n + i] = 1
                    
                    U_cols.append(u_vec)
                    V_cols.append(v_vec)
                    W_cols.append(w_vec)
        
        # Convert to matrices
        rank = len(U_cols)
        U = np.array(U_cols).T
        V = np.array(V_cols).T
        W = np.array(W_cols).T
        
        # Verify the decomposition
        # T_{a,b,c} = sum_r U_{a,r} V_{b,r} W_{c,r}
        # We need to check if this equals the true multiplication tensor.
        
        # Construct true tensor T
        # T is of shape (16, 28, 28)
        # T[a, b, c] = 1 if a=i*4+j, b=j*7+k, c=k*4+i for some i,j,k
        # Note: In the 4x4x4 part, we assumed B was 4x4. But B is 4x7.
        # The indices for B in the 4x4x4 part are j*4+k where k=0..3.
        # In the global B index (4x7), this is j*7+k.
        # So my V vector indices for the 4x4x4 part are WRONG.
        # I used get_block_indices('B', ...) which returns indices for a 4x4 matrix (j*4+k).
        # But the global V vector has size m*p = 4*7 = 28.
        # The indices should be j*7+k.
        
        # I need to fix the index mapping for B and C in the 4x4x4 part.
        # B is 4x7. The first 4 columns are used in the 4x4x4 part.
        # So for B block (br, bc), the local column is bc*2 + l_col.
        # The global column is bc*2 + l_col.
        # The global row is br*2 + l_row.
        # The global index is (br*2 + l_row) * 7 + (bc*2 + l_col).
        
        # C is 7x4? No, C is p x n = 7 x 4.
        # Wait, the task says "multiplying a 4 by 4 matrix by a 4 by 7 matrix".
        # A (4x4) * B (4x7) = C (4x7).
        # So C is 4x7.
        # The interface says W has shape (p*n, rank) = (7*4, rank) = (28, rank).
        # And the convention is W[k*n+i, r].
        # So C index is k*n + i, where k is the column of C (0..6) and i is the row of C (0..3).
        # So C is 4x7, but indexed as k*4+i.
        
        # My previous logic for C was assuming C is 4x4 indexed i*4+j.
        # But C is 4x7 indexed k*4+i.
        # In the 4x4x4 part, we are computing the first 4 columns of C.
        # So k=0..3.
        # The local C block (br, bc) corresponds to rows i in {2*br, 2*br+1} and cols k in {2*bc, 2*bc+1}.
        # Global index for C element (i, k) is k*4 + i.
        
        # So I need to rewrite get_block_indices for B and C to handle the 7-column dimension.
        
        def get_block_indices_fixed(block_type, row_block, col_block):
            # row_block, col_block: 0 or 1
            # Returns list of 4 global indices in order [00, 01, 10, 11] local
            if block_type == 'A':
                # A is 4x4, index i*4+j
                base_i = row_block * 2
                base_j = col_block * 2
                return [
                    (base_i + 0) * 4 + (base_j + 0), # 00
                    (base_i + 0) * 4 + (base_j + 1), # 01
                    (base_i + 1) * 4 + (base_j + 0), # 10
                    (base_i + 1) * 4 + (base_j + 1)  # 11
                ]
            elif block_type == 'B':
                # B is 4x7, index i*7+j
                # We are using the first 4 columns (j=0..3)
                base_i = row_block * 2
                base_j = col_block * 2
                return [
                    (base_i + 0) * 7 + (base_j + 0), # 00
                    (base_i + 0) * 7 + (base_j + 1), # 01
                    (base_i + 1) * 7 + (base_j + 0), # 10
                    (base_i + 1) * 7 + (base_j + 1)  # 11
                ]
            elif block_type == 'C':
                # C is 4x7, index k*4+i
                # We are using the first 4 columns (k=0..3)
                # local (l_row, l_col) corresponds to i = 2*row_block + l_row, k = 2*col_block + l_col
                # global index = k*4 + i
                base_i = row_block * 2
                base_k = col_block * 2
                return [
                    (base_k + 0) * 4 + (base_i + 0), # 00 (l_row=0, l_col=0) -> i=base_i, k=base_k
                    (base_k + 1) * 4 + (base_i + 0), # 01 (l_row=0, l_col=1) -> i=base_i, k=base_k+1
                    (base_k + 0) * 4 + (base_i + 1), # 10 (l_row=1, l_col=0) -> i=base_i+1, k=base_k
                    (base_k + 1) * 4 + (base_i + 1)  # 11 (l_row=1, l_col=1) -> i=base_i+1, k=base_k+1
                ]
        
        # Re-generate U, V, W with fixed indices
        U_cols = []
        V_cols = []
        W_cols = []
        
        for t_idx in range(7):
            a_blocks, b_blocks, c_contrib = top_steps[t_idx]
            
            for s_idx in range(7):
                u_local, v_local, w_local = strassen_2x2[s_idx]
                
                u_vec = [0] * (n*m) # 16
                v_vec = [0] * (m*p) # 28
                w_vec = [0] * (p*n) # 28
                
                # Process U
                for local_idx, coeff in enumerate(u_local):
                    if coeff == 0: continue
                    for (br, bc), a_coeff in a_blocks.items():
                        global_indices = get_block_indices_fixed('A', br, bc)
                        global_idx = global_indices[local_idx]
                        u_vec[global_idx] += coeff * a_coeff
                
                # Process V
                for local_idx, coeff in enumerate(v_local):
                    if coeff == 0: continue
                    for (br, bc), b_coeff in b_blocks.items():
                        global_indices = get_block_indices_fixed('B', br, bc)
                        global_idx = global_indices[local_idx]
                        v_vec[global_idx] += coeff * b_coeff
                
                # Process W
                for local_idx, coeff in enumerate(w_local):
                    if coeff == 0: continue
                    for (br, bc), c_coeff in c_contrib.items():
                        global_indices = get_block_indices_fixed('C', br, bc)
                        global_idx = global_indices[local_idx]
                        w_vec[global_idx] += coeff * c_coeff
                
                U_cols.append(u_vec)
                V_cols.append(v_vec)
                W_cols.append(w_vec)
        
        # Add Schoolbook columns for 4x4x3 (k=4,5,6)
        for i in range(n):
            for j in range(m):
                for k in range(4, 7):
                    u_vec = [0] * (n*m)
                    v_vec = [0] * (m*p)
                    w_vec = [0] * (p*n)
                    
                    u_vec[i*m + j] = 1
                    v_vec[j*p + k] = 1
                    w_vec[k*n + i] = 1
                    
                    U_cols.append(u_vec)
                    V_cols.append(v_vec)
                    W_cols.append(w_vec)
        
        rank = len(U_cols)
        U = np.array(U_cols).T
        V = np.array(V_cols).T
        W = np.array(W_cols).T
        
        # Verify
        # T_true[a, b, c]
        # a = i*4+j
        # b = j*7+k
        # c = k*4+i
        
        error = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m + j
                    b = j*p + k
                    c = k*n + i
                    
                    # Calculate T_decomposed[a, b, c]
                    # sum_r U[a, r] * V[b, r] * W[c, r]
                    val = 0
                    for r in range(rank):
                        val += U[a, r] * V[b, r] * W[c, r]
                    
                    if abs(val - 1) > 1e-9:
                        error += 1
        
        if error > 0:
            # Fallback to Schoolbook
            rank = n * m * p
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            column = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j, column] = 1
                        V[j * p + k, column] = 1
                        W[k * n + i, column] = 1
                        column += 1
        else:
            # Success
            pass
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    except Exception as e:
        # Fallback to Schoolbook
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
