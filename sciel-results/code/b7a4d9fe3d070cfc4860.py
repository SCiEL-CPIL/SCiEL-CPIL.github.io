"""Matrix multiplication 5x5x5: Strassen-based block decomposition."""
import numpy as np

DIMENSIONS = [5, 5, 5]

def _strassen_rank7_decomp():
    """Return (U,V,W) for 2x2x2 matrix multiplication with rank 7.
    
    Indexing: U is (4,7), V is (4,7), W is (4,7)
    U index: i*2+j for row i, col j of A
    V index: j*2+k for row j, col k of B
    W index: k*2+i for row k, col i of C
    """
    n = 2
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # Strassen's 7 products for 2x2 matrix multiplication
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # For each Strassen product, define U and V entries
    # U[i*2+j, t] and V[j*2+k, t] such that sum_t U[i*2+j,t]*V[j*2+k,t] gives the coefficient
    
    # Product 1: M1 = (a11+a22)(b11+b22)
    # U contributions: a11 -> U[0,0], a22 -> U[3,0] (index 3 = 1*2+1)
    # V contributions: b11 -> V[0,0], b22 -> V[3,0] (index 3 = 1*2+1)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    
    # Product 2: M2 = (a21+a22)b11
    # U: a21 -> U[2,1], a22 -> U[3,1] (index 2 = 1*2+0)
    # V: b11 -> V[0,1]
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    
    # Product 3: M3 = a11(b12-b22)
    # U: a11 -> U[0,2]
    # V: b12 -> V[1,2], -b22 -> V[3,2]
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    
    # Product 4: M4 = a22(b21-b11)
    # U: a22 -> U[3,3]
    # V: b21 -> V[2,3], -b11 -> V[0,3]
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    
    # Product 5: M5 = (a11+a12)b22
    # U: a11 -> U[0,4], a12 -> U[1,4]
    # V: b22 -> V[3,4]
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    
    # Product 6: M6 = (a21-a11)(b11+b12)
    # U: a21 -> U[2,5], -a11 -> U[0,5]
    # V: b11 -> V[0,5], b12 -> V[1,5]
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    
    # Product 7: M7 = (a12-a22)(b21+b22)
    # U: a12 -> U[1,6], -a22 -> U[3,6]
    # V: b21 -> V[2,6], b22 -> V[3,6]
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    
    # Now build W: W[k*2+i, t] for each Strassen product
    # C11 = M1+M4-M5+M7 -> C index (0,0) = k=0,i=0 -> W[0, t]
    # C12 = M3+M5 -> C index (0,1) = k=0,i=1 -> W[2, t]
    # C21 = M2+M4 -> C index (1,0) = k=1,i=0 -> W[1, t]
    # C22 = M1-M2+M3+M6 -> C index (1,1) = k=1,i=1 -> W[3, t]
    
    W[0, 0] = 1   # C11 gets M1
    W[0, 3] = 1   # C11 gets M4
    W[0, 4] = -1  # C11 gets -M5
    W[0, 6] = 1   # C11 gets M7
    
    W[2, 2] = 1   # C12 gets M3
    W[2, 4] = 1   # C12 gets M5
    
    W[1, 1] = 1   # C21 gets M2
    W[1, 3] = 1   # C21 gets M4
    
    W[3, 0] = 1   # C22 gets M1
    W[3, 1] = -1  # C22 gets -M2
    W[3, 2] = 1   # C22 gets M3
    W[3, 5] = 1   # C22 gets M6
    
    return U.tolist(), V.tolist(), W.tolist()

def _combine_decomps(decomps):
    """Combine multiple decompositions that partition the indices.
    
    Each decomps entry is (U, V, W, U_offset, V_offset, W_offset, U_size, V_size, W_size)
    where U has shape (U_size, rank_i), etc.
    
    Returns combined (U, V, W) with total rank = sum of ranks.
    """
    total_rank = sum(d[1].shape[1] if isinstance(d[1], np.ndarray) else len(d[1][0]) for d in decomps)
    
    n_full = max(d[6] for d in decomps)
    m_full = max(d[7] for d in decomps)
    p_full = max(d[8] for d in decomps)
    
    U_full = [[0] * total_rank for _ in range(n_full)]
    V_full = [[0] * total_rank for _ in range(m_full)]
    W_full = [[0] * total_rank for _ in range(p_full)]
    
    col_offset = 0
    for U, V, W, u_off, v_off, w_off, u_size, v_size, w_size in decomps:
        if isinstance(U, np.ndarray):
            U = U.tolist()
        if isinstance(V, np.ndarray):
            V = V.tolist()
        if isinstance(W, np.ndarray):
            W = W.tolist()
        
        rank_i = len(U[0]) if U else 0
        for i in range(u_size):
            for c in range(rank_i):
                U_full[u_off + i][col_offset + c] = U[i][c]
        for j in range(v_size):
            for c in range(rank_i):
                V_full[v_off + j][col_offset + c] = V[j][c]
        for k in range(w_size):
            for c in range(rank_i):
                W_full[w_off + k][col_offset + c] = W[k][c]
        col_offset += rank_i
    
    return U_full, V_full, W_full

def construct():
    """Construct rank-minimal (or near-minimal) tensor decomposition for 5x5x5 matrix mult."""
    n, m, p = DIMENSIONS  # 5, 5, 5
    
    # Strategy: Split 5x5 into 4x4 + 1x1 blocks
    # C = [[C44, C41], [C14, C11]]
    # A = [[A44, A41], [A14, A11]]
    # B = [[B44, B41], [B14, B11]]
    # where C44 is 4x4, C41 is 4x1, C14 is 1x4, C11 is 1x1
    
    # C44 = A44*B44 + A41*B14
    # C41 = A44*B41 + A41*B11
    # C14 = A14*B44 + A11*B14
    # C11 = A14*B41 + A11*B11
    
    # We decompose each part:
    # 1. A44*B44: 4x4x4 -> use Strassen recursively: rank 49
    # 2. A41*B14: 4x1x4 -> outer product: rank 4
    # 3. A44*B41: 4x4x1 -> rank 4
    # 4. A41*B11: 4x1x1 -> rank 4
    # 5. A14*B44: 1x4x4 -> rank 4
    # 6. A11*B14: 1x1x4 -> rank 4
    # 7. A14*B41: 1x4x1 -> rank 4
    # 8. A11*B11: 1x1x1 -> rank 1
    # Total: 49 + 4 + 4 + 4 + 4 + 4 + 4 + 1 = 74
    
    # But we can be smarter. Let's use the full structure.
    
    # Actually, let's just build it block by block with proper index mapping.
    
    # Define the 5x5 indexing:
    # A[i,j] for i,j in 0..4
    # B[j,k] for j,k in 0..4  
    # C[i,k] for i,k in 0..4
    
    # Block structure:
    # A44: rows 0-3, cols 0-3
    # A41: rows 0-3, col 4
    # A14: row 4, cols 0-3
    # A11: row 4, col 4
    
    # B44: rows 0-3, cols 0-3
    # B41: rows 0-3, col 4
    # B14: row 4, cols 0-3
    # B11: row 4, col 4
    
    # C44: rows 0-3, cols 0-3
    # C41: rows 0-3, col 4
    # C14: row 4, cols 0-3
    # C11: row 4, col 4
    
    # Contributions:
    # C44 = A44*B44 + A41*B14
    # C41 = A44*B41 + A41*B11
    # C14 = A14*B44 + A11*B14
    # C11 = A14*B41 + A11*B11
    
    # Build each sub-decomposition
    
    # 1. A44 * B44 -> C44 (4x4x4)
    # Use Strassen for 4x4: split 4x4 into four 2x2 blocks, apply Strassen to each pair
    # This gives rank 7*7 = 49
    
    # For the 4x4x4 block, I need to generate the decomposition
    # Let's build it by applying Strassen twice (for the 4x4)
    
    def _strassen_4x4_decomp():
        """Return (U, V, W) for 4x4x4 matrix multiplication using Strassen, rank 49.
        
        U index: i*4+j for A[i,j]
        V index: j*4+k for B[j,k]
        W index: k*4+i for C[i,k]
        """
        # Split 4x4 into four 2x2 blocks:
        # A = [[A00, A01], [A10, A11]] where each is 2x2
        # B = [[B00, B01], [B10, B11]]
        # C = A*B = [[A00*B00+A01*B10, A00*B01+A01*B11],
        #             [A10*B00+A11*B10, A10*B01+A11*B11]]
        
        # For each of the 4 output blocks, use Strassen (rank 7 each)
        # Total rank = 4*7 = 28? No, that's not right.
        
        # Actually, Strassen for 4x4 by splitting into 2x2 blocks:
        # C00 = A00*B00 + A01*B10
        # C01 = A00*B01 + A01*B11
        # C10 = A10*B00 + A11*B10
        # C11 = A10*B01 + A11*B11
        
        # Each of these is a sum of two 2x2x2 multiplications.
        # Each 2x2x2 has rank 7.
        # So each Ci has rank at most 14 (sum of two rank-7 tensors).
        # Total rank = 4 * 14 = 56.
        
        # Better: Apply Strassen at the block level.
        # Treat the 4x4 as 2x2 blocks of 2x2 matrices.
        # Use Strassen's 7 products on the block level, but each "product" 
        # is a matrix multiplication of 2x2 matrices, which has rank 7.
        # Total rank = 7 * 7 = 49.
        
        # Strassen products for block level:
        # M1 = (A00+A11)*(B00+B11)
        # M2 = (A10+A11)*B00
        # M3 = A00*(B01-B11)
        # M4 = A11*(B10-B00)
        # M5 = (A00+A01)*B11
        # M6 = (A10-A00)*(B00+B01)
        # M7 = (A01-A11)*(B10+B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Each Mi is a 2x2x2 matrix product with rank 7.
        
        # I'll build the 49-rank decomposition by combining 7 groups of 7.
        
        rank = 49
        U = np.zeros((16, rank))  # 4*4=16 entries for 4x4 matrix
        V = np.zeros((16, rank))
        W = np.zeros((16, rank))
        
        # Strassen 2x2 decomp for each sub-block
        sU, sV, sW = _strassen_rank7_decomp()
        
        # For each of the 7 Strassen products at the block level,
        # we need to map the 2x2 block indices to 4x4 indices.
        
        # Block mapping:
        # A00: rows 0-1, cols 0-1 -> 4x4 indices (i*4+j) for i in 0-1, j in 0-1
        # A01: rows 0-1, cols 2-3
        # A10: rows 2-3, cols 0-1
        # A11: rows 2-3, cols 2-3
        
        # Similarly for B and C.
        
        # For each Strassen product t (0-6), and each sub-index s (0-6 for the rank-7 decomp):
        # The column in the 49-rank decomp is t*7 + s
        
        # Define block-to-index mappings
        # A block (bi, bj) where bi, bj in {0,1} maps to 4x4 indices:
        # row = bi*2 + i, col = bj*2 + j, where i,j in {0,1}
        # index in 4x4 flattened (row-major, 4 cols) = row*4 + col
        
        def block_A_idx(bi, bj, si, sj):
            """Map block (bi,bj) and sub-indices (si,sj) to 4x4 flattened index."""
            row = bi * 2 + si
            col = bj * 2 + sj
            return row * 4 + col
        
        def block_B_idx(bi, bj, si, sj):
            """Map block (bi,bj) and sub-indices (si,sj) to 4x4 flattened index for B."""
            row = bi * 2 + si  # This is the column index j of A
            col = bj * 2 + sj  # This is the row index of B
            # B[j,k] -> index j*4+k
            return row * 4 + col
        
        def block_C_idx(bi, bj, si, sj):
            """Map block (bi,bj) and sub-indices (si,sj) to 4x4 flattened index for C."""
            # C[i,k] where i = bi*2+si, k = bj*2+sj
            # W index: k*4+i
            row_c = bj * 2 + sj  # k
            col_c = bi * 2 + si  # i
            return row_c * 4 + col_c
        
        # Strassen block-level products and their constituent blocks:
        # M1 = (A00+A11)*(B00+B11)
        # M2 = (A10+A11)*B00
        # M3 = A00*(B01-B11)
        # M4 = A11*(B10-B00)
        # M5 = (A00+A01)*B11
        # M6 = (A10-A00)*(B00+B01)
        # M7 = (A01-A11)*(B10+B11)
        
        # For each product, define the A-blocks with coefficients, B-blocks with coefficients,
        # and C-blocks with coefficients.
        
        products = [
            # (A_blocks, B_blocks, C_blocks)
            # Each block entry: (block_row, block_col, coeff)
            # M1
            ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1)]),
            # M2
            ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1)]),
            # M3
            ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1)]),
            # M4
            ([(1,1,1)], [(1,0,1),(0,0,-1)], [(1,0,1)]),
            # M5
            ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,1,1)]),
            # M6
            ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
            # M7
            ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(1,1,1)]),
        ]
        
        # C block contributions:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        c_contributions = {
            (0,0): [(0,1),(3,1),(4,-1),(6,1)],   # C00: M1, M4, -M5, M7
            (0,1): [(2,1),(4,1)],                  # C01: M3, M5
            (1,0): [(1,1),(3,1)],                  # C10: M2, M4
            (1,1): [(0,1),(1,-1),(2,1),(5,1)],     # C11: M1, -M2, M3, M6
        }
        
        for t in range(7):
            a_blocks, b_blocks, _ = products[t]
            for si in range(2):
                for sj in range(2):
                    for k in range(2):
                        for l in range(2):
                            # This is getting complicated. Let me use a different approach.
                            pass
        
        # Simpler approach: just use the rank-7 decomp and expand
        # For each Strassen product t, we have a rank-7 decomp (sU, sV, sW)
        # where sU[i*2+j, s], sV[j*2+k, s], sW[k*2+i, s] for i,j,k in 0,1
        
        # For product t, the A-blocks are given by a_blocks
        # For each A-block (bi, bj, coeff_a):
        #   For each sub-index (si, sj) in 2x2:
        #     The 4x4 index is block_A_idx(bi, bj, si, sj)
        #     The rank-7 sub-index is si*2+sj
        #     U[4x4_idx, t*7+s] += coeff_a * sU[si*2+sj, s]
        
        for t in range(7):
            a_blocks, b_blocks, _ = products[t]
            for s in range(7):
                # Process A blocks
                for (bi, bj, coeff_a) in a_blocks:
                    for si in range(2):
                        for sj in range(2):
                            idx4 = block_A_idx(bi, bj, si, sj)
                            sub_idx = si * 2 + sj
                            U[idx4, t*7+s] += coeff_a * sU[sub_idx, s]
                
                # Process B blocks
                for (bi, bj, coeff_b) in b_blocks:
                    for si in range(2):
                        for sj in range(2):
                            idx4 = block_B_idx(bi, bj, si, sj)
                            sub_idx = si * 2 + sj
                            V[idx4, t*7+s] += coeff_b * sV[sub_idx, s]
                
                # Process C blocks
                for (ci, cj, coeff_c) in c_contributions.get((t, None), []):
                    pass
        
        # Hmm, I need to handle C contributions differently.
        # Let me rebuild this more carefully.
        
        U = np.zeros((16, 49))
        V = np.zeros((16, 49))
        W = np.zeros((16, 49))
        
        for t in range(7):
            a_blocks, b_blocks, _ = products[t]
            for s in range(7):
                # A side
                for (bi, bj, coeff_a) in a_blocks:
                    for si in range(2):
                        for sj in range(2):
                            idx4 = block_A_idx(bi, bj, si, sj)
                            sub_idx = si * 2 + sj
                            U[idx4, t*7+s] += coeff_a * sU[sub_idx, s]
                
                # B side
                for (bi, bj, coeff_b) in b_blocks:
                    for si in range(2):
                        for sj in range(2):
                            idx4 = block_B_idx(bi, bj, si, sj)
                            sub_idx = si * 2 + sj
                            V[idx4, t*7+s] += coeff_b * sV[sub_idx, s]
        
        # Now for W: for each C block (ci, cj), it receives contributions from various (t, s)
        for (ci, cj), contributions in c_contributions.items():
            for si in range(2):
                for sj in range(2):
                    idx4 = block_C_idx(ci, cj, si, sj)
                    sub_idx = si * 2 + sj
                    for (t, coeff_c) in contributions:
                        for s in range(7):
                            W[idx4, t*7+s] += coeff_c * sW[sub_idx, s]
        
        return U.tolist(), V.tolist(), W.tolist()
    
    U44, V44, W44 = _strassen_4x4_decomp()
    rank_44 = 49
    
    # Now build the full 5x5x5 decomposition
    # Total rank = 49 (for 4x4x4) + ranks for the strip terms
    
    # Strip terms:
    # A41*B14 -> C41: 4x1x4, rank 4
    # A44*B41 -> C41: 4x4x1, rank 4
    # A14*B44 -> C14: 1x4x4, rank 4
    # A11*B14 -> C14: 1x1x4, rank 4
    # A14*B41 -> C11: 1x4x1, rank 4
    # A11*B11 -> C11: 1x1x1, rank 1
    
    # Wait, let me reconsider. The total rank should be:
    # 49 + 4 + 4 + 4 + 4 + 4 + 1 = 70
    
    total_rank = 49 + 4 + 4 + 4 + 4 + 4 + 1  # = 70
    
    U = [[0] * total_rank for _ in range(25)]
    V = [[0] * total_rank for _ in range(25)]
    W = [[0] * total_rank for _ in range(25)]
    
    col = 0
    
    # 1. A44*B44 -> C44
    # A44: A[i,j] for i,j in 0-3 -> U index i*5+j
    # B44: B[j,k] for j,k in 0-3 -> V index j*5+k
    # C44: C[i,k] for i,k in 0-3 -> W index k*5+i
    
    for i in range(4):
        for j in range(4):
            u_idx_44 = i * 4 + j  # index in U44
            for c in range(49):
                U[i*5+j][col+c] = U44[u_idx_44][c]
    
    for j in range(4):
        for k in range(4):
            v_idx_44 = j * 4 + k  # index in V44
            for c in range(49):
                V[j*5+k][col+c] = V44[v_idx_44][c]
    
    for k in range(4):
        for i in range(4):
            w_idx_44 = k * 4 + i  # index in W44
            for c in range(49):
                W[k*5+i][col+c] = W44[w_idx_44][c]
    
    col += 49
    
    # 2. A41*B14 -> C41
    # A41: A[i,4] for i in 0-3 -> U index i*5+4
    # B14: B[4,k] for k in 0-3 -> V index 4*5+k = 20+k
    # C41: C[i,4] for i in 0-3 -> W index 4*5+i = 20+i
    # This is an outer product: C[i,4] = A[i,4]*B[4,4]... no wait
    # A41 is 4x1, B14 is 1x4, so C41 = A41*B14 is 4x4
    # C41[i,k] = A[i,4]*B[4,k]
    # Rank 4: one product per i
    
    for i in range(4):
        U[i*5+4][col] = 1
        for k in range(4):
            V[20+k][col] = 1
            W[20+i][col] = 1  # C[i,4] -> W index 4*5+i
        col += 1
    
    # 3. A44*B41 -> C41
    # A44: A[i,j] for i,j in 0-3
    # B41: B[j,4] for j in 0-3 -> V index j*5+4
    # C41: C[i,4] for i in 0-3
    # C41[i,4] = sum_j A[i,j]*B[j,4]
    # Rank 4: one product per j
    
    for j in range(4):
        for i in range(4):
            U[i*5+j][col] = 1
        V[j*5+4][col] = 1
        for i in range(4):
            W[20+i][col] = 1
        col += 1
    
    # 4. A14*B44 -> C14
    # A14: A[4,j] for j in 0-3 -> U index 4*5+j = 20+j
    # B44: B[j,k] for j,k in 0-3
    # C14: C[4,k] for k in 0-3 -> W index k*5+4
    # C14[4,k] = sum_j A[4,j]*B[j,k]
    # Rank 4: one product per j
    
    for j in range(4):
        U[20+j][col] = 1
        for k in range(4):
            V[j*5+k][col] = 1
            W[k*5+4][col] = 1
        col += 1
    
    # 5. A11*B14 -> C14
    # A11: A[4,4] -> U index 24
    # B14: B[4,k] for k in 0-3
    # C14: C[4,k] for k in 0-3
    # C14[4,k] += A[4,4]*B[4,k]
    # Rank 4: one product per k
    
    for k in range(4):
        U[24][col] = 1
        V[20+k][col] = 1
        W[k*5+4][col] = 1
        col += 1
    
    # 6. A14*B41 -> C11
    # A14: A[4,j] for j in 0-3
    # B41: B[j,4] for j in 0-3
    # C11: C[4,4] -> W index 4*5+4 = 24
    # C11 += sum_j A[4,j]*B[j,4]
    # Rank 4: one product per j
    
    for j in range(4):
        U[20+j][col] = 1
        V[j*5+4][col] = 1
        W[24][col] = 1
        col += 1
    
    # 7. A11*B11 -> C11
    # A11: A[4,4], B11: B[4,4], C11: C[4,4]
    # Rank 1
    
    U[24][col] = 1
    V[24][col] = 1
    W[24][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
