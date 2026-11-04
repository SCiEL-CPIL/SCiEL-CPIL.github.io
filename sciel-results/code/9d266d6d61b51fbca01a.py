import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # We split B into B1 (4x4) and B2 (4x3)
    # C1 = A * B1 (4x4) using Strassen (rank 49)
    # C2 = A * B2 (4x3) using Schoolbook (rank 48)
    # Total rank = 97
    
    rank_strassen = 49
    rank_schoolbook = 48
    total_rank = rank_strassen + rank_schoolbook
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # --- Part 1: Strassen for 4x4 x 4x4 ---
    # We need to generate the U, V, W for the Strassen part.
    # The Strassen part corresponds to columns 0..48 of U, V, W.
    # V for Strassen part corresponds to the first 4 columns of B (B1).
    # W for Strassen part corresponds to the first 4 columns of C (C1).
    
    # Helper to add a 2x2x2 Strassen decomposition
    def add_strassen_2x2(offset_u, offset_v, offset_w, sign_u, sign_v, sign_w, col_offset):
        # This is too complex to do with offsets directly.
        # Let's just generate the 49 columns explicitly.
        pass

    # Let's use a recursive generator for Strassen decomposition
    def generate_strassen(n):
        # Returns U, V, W for n x n x n Strassen decomposition
        if n == 2:
            # 2x2x2 Strassen
            U_2 = np.zeros((4, 7))
            V_2 = np.zeros((4, 7))
            W_2 = np.zeros((4, 7))
            
            # Term 1: (A00+A11)(B00+B11) -> C00(+1), C11(+1)
            U_2[0, 0] = 1; U_2[3, 0] = 1
            V_2[0, 0] = 1; V_2[3, 0] = 1
            W_2[0, 0] = 1; W_2[3, 0] = 1 # C00 is idx 0, C11 is idx 3
            
            # Term 2: (A10+A11)B00 -> C10(+1)
            U_2[2, 1] = 1; U_2[3, 1] = 1
            V_2[0, 1] = 1
            W_2[2, 1] = 1 # C10 is idx 2
            
            # Term 3: A00(B11-B01) -> C01(+1), C11(-1)
            U_2[0, 2] = 1
            V_2[3, 2] = 1; V_2[1, 2] = -1
            W_2[1, 2] = 1; W_2[3, 2] = -1 # C01 is idx 1, C11 is idx 3
            
            # Term 4: A11(B00-B10) -> C00(+1), C10(-1)
            U_2[3, 3] = 1
            V_2[0, 3] = 1; V_2[2, 3] = -1
            W_2[0, 3] = 1; W_2[2, 3] = -1
            
            # Term 5: (A00+A01)B11 -> C00(-1), C01(+1)
            U_2[0, 4] = 1; U_2[1, 4] = 1
            V_2[3, 4] = 1
            W_2[0, 4] = -1; W_2[1, 4] = 1
            
            # Term 6: (A10-A00)(B00+B01) -> C11(+1)
            U_2[2, 5] = 1; U_2[0, 5] = -1
            V_2[0, 5] = 1; V_2[1, 5] = 1
            W_2[3, 5] = 1
            
            # Term 7: (A01-A11)(B10+B11) -> C00(+1), C11(-1)
            U_2[1, 6] = 1; U_2[3, 6] = -1
            V_2[2, 6] = 1; V_2[3, 6] = 1
            W_2[0, 6] = 1; W_2[3, 6] = -1
            
            return U_2, V_2, W_2
        
        else:
            # Recursive case
            half = n // 2
            U_half, V_half, W_half = generate_strassen(half)
            rank_half = U_half.shape[1]
            rank_full = 7 * rank_half
            
            U_full = np.zeros((n * n, rank_full))
            V_full = np.zeros((n * n, rank_full))
            W_full = np.zeros((n * n, rank_full))
            
            # Define the 7 Strassen terms
            # Each term is a tuple of (U_indices, U_signs, V_indices, V_signs, W_indices, W_signs)
            # Indices are relative to the n/2 x n/2 blocks.
            
            # Block indices:
            # 0: top-left, 1: top-right, 2: bottom-left, 3: bottom-right
            
            terms = [
                # 1: (A00+A11)(B00+B11) -> C00(+1), C11(+1)
                {
                    'U': [(0, 1), (3, 1)],
                    'V': [(0, 1), (3, 1)],
                    'W': [(0, 1), (3, 1)]
                },
                # 2: (A20+A21)B00 -> C20(+1)
                {
                    'U': [(2, 1), (3, 1)],
                    'V': [(0, 1)],
                    'W': [(2, 1)]
                },
                # 3: A00(B11-B01) -> C01(+1), C11(-1)
                {
                    'U': [(0, 1)],
                    'V': [(3, 1), (1, -1)],
                    'W': [(1, 1), (3, -1)]
                },
                # 4: A11(B00-B20) -> C00(+1), C20(-1)
                {
                    'U': [(3, 1)],
                    'V': [(0, 1), (2, -1)],
                    'W': [(0, 1), (2, -1)]
                },
                # 5: (A00+A01)B11 -> C00(-1), C01(+1)
                {
                    'U': [(0, 1), (1, 1)],
                    'V': [(3, 1)],
                    'W': [(0, -1), (1, 1)]
                },
                # 6: (A20-A00)(B00+B01) -> C11(+1)
                {
                    'U': [(2, 1), (0, -1)],
                    'V': [(0, 1), (1, 1)],
                    'W': [(3, 1)]
                },
                # 7: (A01-A11)(B20+B11) -> C00(+1), C11(-1)
                {
                    'U': [(1, 1), (3, -1)],
                    'V': [(2, 1), (3, 1)],
                    'W': [(0, 1), (3, -1)]
                }
            ]
            
            for t_idx, term in enumerate(terms):
                for k in range(rank_half):
                    col = t_idx * rank_half + k
                    
                    # Add U contributions
                    for block_idx, sign in term['U']:
                        # Block index to row range
                        row_start = (block_idx // 2) * half * half + (block_idx % 2) * half
                        # Actually, block 0 is rows 0..half-1, cols 0..half-1
                        # Block 1 is rows 0..half-1, cols half..n-1
                        # Block 2 is rows half..n-1, cols 0..half-1
                        # Block 3 is rows half..n-1, cols half..n-1
                        
                        r_start = (block_idx // 2) * half
                        c_start = (block_idx % 2) * half
                        # The U matrix for the half-size problem has indices r_half * half + c_half
                        # We need to map these to the full U matrix indices.
                        # Full U index = (r_start + r_half) * n + (c_start + c_half)
                        
                        for r_h in range(half):
                            for c_h in range(half):
                                val = U_half[r_h * half + c_h, k]
                                if val != 0:
                                    full_r = r_start + r_h
                                    full_c = c_start + c_h
                                    full_idx = full_r * n + full_c
                                    U_full[full_idx, col] += sign * val
                                    
                    # Add V contributions
                    for block_idx, sign in term['V']:
                        r_start = (block_idx // 2) * half
                        c_start = (block_idx % 2) * half
                        for r_h in range(half):
                            for c_h in range(half):
                                val = V_half[r_h * half + c_h, k]
                                if val != 0:
                                    full_r = r_start + r_h
                                    full_c = c_start + c_h
                                    full_idx = full_r * n + full_c
                                    V_full[full_idx, col] += sign * val
                                    
                    # Add W contributions
                    for block_idx, sign in term['W']:
                        r_start = (block_idx // 2) * half
                        c_start = (block_idx % 2) * half
                        for r_h in range(half):
                            for c_h in range(half):
                                val = W_half[r_h * half + c_h, k]
                                if val != 0:
                                    full_r = r_start + r_h
                                    full_c = c_start + c_h
                                    full_idx = full_r * n + full_c
                                    W_full[full_idx, col] += sign * val
                                    
            return U_full, V_full, W_full

    U_strassen, V_strassen, W_strassen = generate_strassen(4)
    
    # Copy Strassen part to U, V, W
    # U_strassen is 16x49
    # V_strassen is 16x49 (corresponds to B1, which is 4x4, so 16 elements)
    # W_strassen is 16x49 (corresponds to C1, which is 4x4, so 16 elements)
    
    U[:, :49] = U_strassen
    V[:, :49] = V_strassen
    W[:, :49] = W_strassen
    
    # --- Part 2: Schoolbook for 4x4 x 4x3 ---
    # This corresponds to columns 49..96 of U, V, W.
    # V for this part corresponds to the last 3 columns of B (B2).
    # W for this part corresponds to the last 3 columns of C (C2).
    
    # B2 is 4x3. V indices for B2 are 16..27 (since B1 is 0..15).
    # C2 is 4x3. W indices for C2 are 16..27 (since C1 is 0..15).
    
    col_offset = 49
    for i in range(n):
        for j in range(m):
            for k in range(3): # Only 3 columns for B2
                col = col_offset + (i * m * 3) + (j * 3) + k
                # Wait, the schoolbook loop is usually i, j, k.
                # U[i*m+j, col] = 1
                # V[j*p+k, col] = 1 -> Here p=3 for this part, so V[j*3+k, col] = 1
                # W[k*n+i, col] = 1
                
                # But we need to map these to the global V and W indices.
                # Global V index for B2: j * 7 + (4 + k)
                # Global W index for C2: (4 + k) * 4 + i
                
                U[i * m + j, col] = 1
                V[j * 7 + (4 + k), col] = 1
                W[(4 + k) * 4 + i, col] = 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

