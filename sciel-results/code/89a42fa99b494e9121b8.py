import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # We use a block decomposition: 6x6 = 2x2 blocks of 3x3 matrices.
    # Outer: 2x2 matrix mult (Strassen, rank 7)
    # Inner: 3x3 matrix mult (Schoolbook, rank 27)
    # Total Rank = 7 * 27 = 189
    
    rank_3x3 = 27
    rank_2x2 = 7
    total_rank = rank_2x2 * rank_3x3
    
    # Initialize factor matrices
    # U: (n*m, total_rank) = (36, 189)
    # V: (m*p, total_rank) = (36, 189)
    # W: (p*n, total_rank) = (36, 189)
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # Define the 3x3 schoolbook decomposition
    # A is 3x3, B is 3x3, C is 3x3
    # C[i,j] = sum_k A[i,k] * B[k,j]
    # Indices for 3x3 matrices (flattened row-major for A and B, but we need to map to global)
    # Local A index: i3 * 3 + k3
    # Local B index: k3 * 3 + j3
    # Local C index: j3 * 3 + i3 (to match the global W convention: col * dim + row)
    # Wait, let's verify the W convention again.
    # Global W index: col_global * n + row_global.
    # Inside a 3x3 block, col_local is j3, row_local is i3.
    # The block size is 3.
    # So local W index should be j3 * 3 + i3.
    
    # Strassen 2x2 decomposition
    # A_blocks: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B_blocks: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # C_blocks: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # Strassen Terms:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    
    # C reconstruction:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each term t in 0..6 corresponds to a product of two 3x3 matrices.
    # A_t = sum_{a in A_offsets} coeff_a * A_block[a]
    # B_t = sum_{b in B_offsets} coeff_b * B_block[b]
    # Product P_t = A_t * B_t (3x3 matrix)
    # This product is computed with 27 scalar multiplications.
    # For each scalar multiplication (i3, k3, j3) in the 3x3 product:
    #   Term in P_t[i3, j3] is A_t[i3, k3] * B_t[k3, j3]
    #   A_t[i3, k3] is a linear combination of global A entries.
    #   B_t[k3, j3] is a linear combination of global B entries.
    #   The product contributes to C_block[c_off] with coefficient W_t.
    
    # However, the rank-1 decomposition of the 3x3 product is:
    # For each (i3, k3, j3), we have a rank-1 term:
    #   U_part = A_t[i3, k3] (which is a linear combo of A entries)
    #   V_part = B_t[k3, j3] (which is a linear combo of B entries)
    #   W_part = 1 (contribution to P_t[i3, j3])
    # Then this P_t[i3, j3] is added to C with coefficient W_t.
    
    # So for each Strassen term t, and each (i3, k3, j3):
    #   We add a rank-1 term to the global decomposition.
    #   The global U factor is the linear combination of A entries forming A_t[i3, k3].
    #   The global V factor is the linear combination of B entries forming B_t[k3, j3].
    #   The global W factor is the contribution to the final C.
    #     C_entry = sum_t W_t * P_t[i3, j3]
    #     So for a specific t and (i3, k3, j3), the contribution to C is W_t * 1 * (value of C_entry corresponding to block (bi, bj) and local (i3, j3)).
    #     Wait, P_t is the product of the combined blocks.
    #     The result of the 2x2 Strassen is C_block[bi, bj].
    #     C_block[bi, bj][i3, j3] = sum_t W_t * P_t[i3, j3].
    #     So for a fixed t and (i3, j3), the contribution to C is W_t * P_t[i3, j3].
    #     Since P_t[i3, j3] is formed by the rank-1 term (A_t[i3, k3] * B_t[k3, j3]), the W factor for this specific rank-1 term is W_t.
    #     And it contributes to the global C entry at global row (2*bi + i3) and global col (2*bj + j3).
    #     Global W index = global_col * n + global_row.
    
    # Let's define the Strassen terms data
    # Each entry: (A_blocks, B_blocks, C_blocks_contribs)
    # A_blocks: list of (block_idx, coeff)
    # B_blocks: list of (block_idx, coeff)
    # C_blocks_contribs: list of (block_idx, coeff)
    
    strassen_terms = [
        # M1
        {
            'A': [(0, 1), (3, 1)],
            'B': [(0, 1), (3, 1)],
            'C': [(0, 1), (3, 1)]
        },
        # M2
        {
            'A': [(2, 1), (3, 1)],
            'B': [(0, 1)],
            'C': [(2, 1), (3, -1)]
        },
        # M3
        {
            'A': [(0, 1)],
            'B': [(1, 1), (3, -1)],
            'C': [(1, 1), (3, 1)]
        },
        # M4
        {
            'A': [(3, 1)],
            'B': [(2, 1), (0, -1)],
            'C': [(0, 1), (2, 1)]
        },
        # M5
        {
            'A': [(0, 1), (1, 1)],
            'B': [(3, 1)],
            'C': [(1, 1), (0, -1)]
        },
        # M6
        {
            'A': [(2, 1), (0, -1)],
            'B': [(0, 1), (1, 1)],
            'C': [(3, 1)]
        },
        # M7
        {
            'A': [(1, 1), (3, -1)],
            'B': [(2, 1), (3, 1)],
            'C': [(0, 1)]
        }
    ]
    
    col = 0
    for t in range(7):
        term = strassen_terms[t]
        A_blocks = term['A']
        B_blocks = term['B']
        C_blocks = term['C']
        
        # Iterate over the 27 schoolbook terms for the 3x3 product
        for i3 in range(3):
            for k3 in range(3):
                for j3 in range(3):
                    # This corresponds to the scalar product A_t[i3, k3] * B_t[k3, j3]
                    
                    # 1. Construct U factor (linear combo of A entries)
                    # A_t[i3, k3] = sum_{(ba, ca) in A_blocks} coeff * A_block[ba, ca][i3, k3]
                    # Global A index: row = 2*ba + i3, col = 2*ca + k3
                    # Flat index: row * m + col
                    for ba, ca_coeff in A_blocks:
                        # ca is the column block index, but we need to map it to A's column
                        # In Strassen, A is 2x2 blocks.
                        # Block (ba, ca) corresponds to A rows 2*ba..2*ba+2, cols 2*ca..2*ca+2.
                        # Wait, the variable name in the list is 'ca_coeff' but it's actually (block_idx, coeff).
                        # Let's unpack correctly.
                        pass
                    
                    # Let's re-loop A_blocks correctly
                    for block_idx, coeff in A_blocks:
                        rb, cb = divmod(block_idx, 2)
                        # Global row in A: 2*rb + i3
                        # Global col in A: 2*cb + k3
                        g_row = 2 * rb + i3
                        g_col = 2 * cb + k3
                        U_idx = g_row * m + g_col
                        U[U_idx, col] += coeff
                    
                    # 2. Construct V factor (linear combo of B entries)
                    # B_t[k3, j3] = sum_{(bb, bc) in B_blocks} coeff * B_block[bb, bc][k3, j3]
                    # Global B index: row = 2*bb + k3, col = 2*bc + j3
                    # Flat index: row * p + col
                    for block_idx, coeff in B_blocks:
                        rb, cb = divmod(block_idx, 2)
                        g_row = 2 * rb + k3
                        g_col = 2 * cb + j3
                        V_idx = g_row * p + g_col
                        V[V_idx, col] += coeff
                    
                    # 3. Construct W factor (contribution to C)
                    # This term contributes to C_block[bc_idx] with coefficient coeff_c.
                    # C_block[bc_idx][i3, j3]
                    # Global C row: 2*bc_row + i3
                    # Global C col: 2*bc_col + j3
                    # Flat index for W: col * n + row
                    for block_idx, coeff_c in C_blocks:
                        rb, cb = divmod(block_idx, 2)
                        g_row = 2 * rb + i3
                        g_col = 2 * cb + j3
                        W_idx = g_col * n + g_row
                        W[W_idx, col] += coeff_c
                    
                    col += 1
                    if col >= total_rank:
                        break
                if col >= total_rank:
                    break
            if col >= total_rank:
                break
        if col >= total_rank:
            break

    # Verification step (optional, but good for safety)
    # Check if time allows
    if time.time() < deadline - 10.0:
        try:
            # Reconstruct tensor T[i,k,j] = sum_r U[i,k,r] * V[k,j,r] * W[j,i,r] ?
            # Our indices:
            # U is indexed by (i, k) -> i*m+k
            # V is indexed by (k, j) -> k*p+j
            # W is indexed by (j, i) -> j*n+i (based on col*n+row)
            
            # Let's verify a few random entries or use einsum
            # T should be 1 if i==j and k==k? No.
            # The tensor for matrix mult is T_{ik, kj, ji} = 1 if the indices match up?
            # Actually, the decomposition represents the bilinear form.
            # C_{ij} = sum_k A_{ik} B_{kj}
            # So the tensor T has indices (i,k), (k,j), (i,j).
            # T[(i,k), (k,j), (i,j)] = 1.
            
            # Let's check if the sum of products equals 1 for all valid indices.
            # We can check a subset if it's too slow.
            
            # Reshape U, V, W to 3D tensors
            U_3d = U.reshape(n, m, -1)
            V_3d = V.reshape(m, p, -1)
            W_3d = W.reshape(p, n, -1) # Note: W is stored as col*n+row, so dim0 is col (p), dim1 is row (n)
            
            # Compute T[i,k,j] = sum_r U[i,k,r] * V[k,j,r] * W[j,i,r]
            # Note: W index is (j, i) in the tensor sense if we align with C[i,j].
            # In our storage, W[j*n+i] corresponds to row i, col j.
            # So W_3d[j, i, r] is the element.
            
            T = np.einsum('ikr,kjr,jir->ijk', U_3d, V_3d, W_3d)
            
            # The expected tensor is 1 if i==j and k==k? No.
            # The matrix multiplication tensor is defined such that
            # sum_r U[i,k,r] V[k,j,r] W[j,i,r] = 1 if the indices are consistent?
            # Actually, the standard tensor T_M has T_{ik, kj, ji} = 1.
            # So T[i,k,j] should be 1 for all i,j,k?
            # Wait. C_{ij} = sum_k A_{ik} B_{kj}.
            # The bilinear form is sum_{i,j,k} C_{ij} (A_{ik} B_{kj})^T ?
            # The tensor T satisfies T_{ik, kj, ji} = 1.
            # So T[i,k,j] should be 1 for all i, k, j.
            
            # Let's check max deviation
            diff = np.max(np.abs(T - 1.0))
            if diff > 1e-9:
                # Fallback to schoolbook 216
                rank = 216
                U_f = np.zeros((36, rank))
                V_f = np.zeros((36, rank))
                W_f = np.zeros((36, rank))
                for i in range(n):
                    for j in range(m): # k in A
                        for k in range(p): # j in B
                            col = i * 36 + j * 6 + k
                            U_f[i * m + j, col] = 1.0
                            V_f[j * p + k, col] = 1.0
                            W_f[k * n + i, col] = 1.0
                return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}
        except Exception:
            pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
