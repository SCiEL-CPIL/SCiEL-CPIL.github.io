import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Use Strassen decomposition for 6x6 matrices by treating them as 2x2 block matrices
    # of 3x3 sub-matrices.
    # Rank of 3x3 schoolbook is 27.
    # Strassen uses 7 such multiplications.
    # Total rank = 7 * 27 = 189.
    
    sub_dim = 3
    rank_sub = sub_dim * sub_dim * sub_dim # 27
    total_rank = 7 * rank_sub
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Helper to add a schoolbook decomposition for a block
    # block_u_offset: row offset in U for the A-block
    # block_v_offset: row offset in V for the B-block
    # block_w_offset: row offset in W for the C-block
    # scale: multiplier for the coefficients
    # rank_offset: starting column in U, V, W
    
    def add_block_decomp(u_off, v_off, w_off, scale, rank_off):
        for i in range(sub_dim):
            for j in range(sub_dim):
                for k in range(sub_dim):
                    col = rank_off + (i * sub_dim + j) * sub_dim + k
                    U[u_off + i * sub_dim + j][col] += scale
                    V[v_off + j * sub_dim + k][col] += scale
                    W[w_off + k * sub_dim + i][col] += scale

    # Block indices in 6x6 matrix:
    # A00: rows 0-2, cols 0-2
    # A01: rows 0-2, cols 3-5
    # A10: rows 3-5, cols 0-2
    # A11: rows 3-5, cols 3-5
    
    # U indices (A):
    # A00: 0..8
    # A01: 0..8 (cols 3..5) -> Wait, U shape is (n*m, rank) = (36, rank).
    # Index in U: i*m + j.
    # A00: i in 0..2, j in 0..2 -> idx 0..8
    # A01: i in 0..2, j in 3..5 -> idx 3..5, 6..8, 9..11 -> 3..11
    # A10: i in 3..5, j in 0..2 -> idx 18..20, 21..23, 24..26 -> 18..26
    # A11: i in 3..5, j in 3..5 -> idx 21..23, 24..26, 27..29 -> 21..29
    
    # V indices (B):
    # B00: j in 0..2, k in 0..2 -> idx 0..8
    # B01: j in 0..2, k in 3..5 -> idx 3..5, 6..8, 9..11 -> 3..11
    # B10: j in 3..5, k in 0..2 -> idx 18..20, 21..23, 24..26 -> 18..26
    # B11: j in 3..5, k in 3..5 -> idx 21..23, 24..26, 27..29 -> 21..29
    
    # W indices (C):
    # C00: k in 0..2, i in 0..2 -> idx 0..2, 6..8, 12..14 -> 0..2, 6..8, 12..14
    # C01: k in 0..2, i in 3..5 -> idx 3..5, 9..11, 15..17 -> 3..5, 9..11, 15..17
    # C10: k in 3..5, i in 0..2 -> idx 18..20, 24..26, 30..32 -> 18..20, 24..26, 30..32
    # C11: k in 3..5, i in 3..5 -> idx 21..23, 27..29, 33..35 -> 21..23, 27..29, 33..35

    # Define offsets for the 3x3 blocks in the flattened vectors
    # A blocks (U):
    u_00 = 0
    u_01 = 3
    u_10 = 18
    u_11 = 21
    
    # B blocks (V):
    v_00 = 0
    v_01 = 3
    v_10 = 18
    v_11 = 21
    
    # C blocks (W):
    # W index is k*n + i.
    # C00: k in 0..2, i in 0..2.
    # k=0: i=0,1,2 -> 0,1,2
    # k=1: i=0,1,2 -> 6,7,8
    # k=2: i=0,1,2 -> 12,13,14
    # This is not a contiguous block in the flattened W vector!
    # My `add_block_decomp` assumes contiguous blocks.
    # I need a more flexible helper or manual mapping.
    
    # Let's rewrite the helper to take explicit indices or handle strides.
    # Or simply iterate over the 27 elements of the 3x3 block and map them correctly.
    
    def add_block_decomp_flex(u_indices, v_indices, w_indices, scale, rank_off):
        for idx in range(27):
            i = idx // 9
            j = (idx // 3) % 3
            k = idx % 3
            
            u_idx = u_indices[i][j]
            v_idx = v_indices[j][k]
            w_idx = w_indices[k][i]
            
            col = rank_off + idx
            U[u_idx][col] += scale
            V[v_idx][col] += scale
            W[w_idx][col] += scale

    # Precompute index maps for 3x3 blocks
    # A block (U): i in 0..2, j in 0..2 relative to block start
    # But we need global indices.
    
    # U indices for A00 (rows 0-2, cols 0-2)
    u_00_idx = [[i * 6 + j for j in range(3)] for i in range(3)]
    # U indices for A01 (rows 0-2, cols 3-5)
    u_01_idx = [[i * 6 + (j + 3) for j in range(3)] for i in range(3)]
    # U indices for A10 (rows 3-5, cols 0-2)
    u_10_idx = [[(i + 3) * 6 + j for j in range(3)] for i in range(3)]
    # U indices for A11 (rows 3-5, cols 3-5)
    u_11_idx = [[(i + 3) * 6 + (j + 3) for j in range(3)] for i in range(3)]
    
    # V indices for B00 (rows 0-2, cols 0-2)
    v_00_idx = [[j * 6 + k for k in range(3)] for j in range(3)]
    # V indices for B01 (rows 0-2, cols 3-5)
    v_01_idx = [[j * 6 + (k + 3) for k in range(3)] for j in range(3)]
    # V indices for B10 (rows 3-5, cols 0-2)
    v_10_idx = [[(j + 3) * 6 + k for k in range(3)] for j in range(3)]
    # V indices for B11 (rows 3-5, cols 3-5)
    v_11_idx = [[(j + 3) * 6 + (k + 3) for k in range(3)] for j in range(3)]
    
    # W indices for C00 (cols 0-2, rows 0-2) -> k in 0..2, i in 0..2
    # W index: k * 6 + i
    w_00_idx = [[k * 6 + i for i in range(3)] for k in range(3)]
    # W indices for C01 (cols 0-2, rows 3-5) -> k in 0..2, i in 3..5
    w_01_idx = [[k * 6 + (i + 3) for i in range(3)] for k in range(3)]
    # W indices for C10 (cols 3-5, rows 0-2) -> k in 3..5, i in 0..2
    w_10_idx = [[(k + 3) * 6 + i for i in range(3)] for k in range(3)]
    # W indices for C11 (cols 3-5, rows 3-5) -> k in 3..5, i in 3..5
    w_11_idx = [[(k + 3) * 6 + (i + 3) for i in range(3)] for k in range(3)]

    rank_off = 0
    
    # M1 = (A00 + A11)(B00 + B11)
    # This is (A00 + A11) * (B00 + B11)
    # = A00*B00 + A00*B11 + A11*B00 + A11*B11
    # We need to add 4 decompositions? No, Strassen combines them.
    # The tensor decomposition is linear.
    # M1 contributes to the product.
    # We can just add the decomposition of the sum?
    # No, the decomposition is for the product.
    # We need to represent the product (A00+A11)(B00+B11).
    # This is not a single "block product" in the simple sense unless we expand.
    # Actually, we can just add the rank-27 decompositions for each term in the expansion?
    # That would be 4 * 27 = 108 for M1.
    # Total rank would be huge.
    
    # Wait, the Strassen algorithm replaces 8 multiplications with 7.
    # Each multiplication is of 3x3 matrices.
    # If we implement the 7 multiplications explicitly using schoolbook rank-27 decompositions,
    # we need to ensure the U, V, W matrices correctly reflect the sums.
    
    # For M1 = (A00 + A11)(B00 + B11):
    # U part: We need vectors u such that u * a = a00 + a11.
    # This means for each entry in A00, we add 1. For each entry in A11, we add 1.
    # So we can have a single set of 27 columns for M1?
    # No, the product is (A00+A11) * (B00+B11).
    # The schoolbook decomposition of X*Y where X=A00+A11 and Y=B00+B11.
    # X is not a contiguous block.
    # But we can define U, V, W for the "virtual" matrix X and Y.
    
    # Let's define the U, V, W contributions for each of the 7 Strassen steps.
    
    # Step 1: M1 = (A00 + A11)(B00 + B11)
    # U: A00 entries get +1, A11 entries get +1.
    # V: B00 entries get +1, B11 entries get +1.
    # W: C00 gets +1, C11 gets +1? No.
    # M1 contributes to C00, C01, C10, C11 with coefficients.
    # C00 += M1
    # C11 += M1
    # C01 -= M1? No.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So M1 adds to C00 with coeff 1, and C11 with coeff 1.
    
    def add_strassen_step(u_blocks, v_blocks, w_blocks, coeffs, rank_off):
        # u_blocks: list of (idx_map, scale)
        # v_blocks: list of (idx_map, scale)
        # w_blocks: list of (idx_map, scale)
        
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    col = rank_off + i*9 + j*3 + k
                    
                    # Accumulate U
                    u_val = 0
                    for idx_map, scale in u_blocks:
                        u_val += scale * 1 # The value is 1 for each element in the block
                    # Wait, the decomposition is sum_u U_u V_u W_u.
                    # For a specific term like A00*B00, U has 1 at A00, V has 1 at B00, W has 1 at C.
                    # If we have A00+A11, U has 1 at A00 AND 1 at A11.
                    # So for a given column, U[idx] is the sum of scales for all blocks containing idx.
                    
                    u_sum = 0
                    for idx_map, scale in u_blocks:
                        # idx_map[i][j] is the global index
                        # We are iterating i,j,k for the 3x3 block structure.
                        # The "local" i,j,k correspond to the 3x3 sub-block.
                        # The global index is determined by the block.
                        pass # This logic is getting complicated.
                        
    # Simpler approach:
    # Just add the 27 columns for each "virtual" multiplication.
    # For M1, we have 27 columns.
    # For each column c (0..26) corresponding to (i,j,k) in the 3x3 product:
    # U[A00[i][j]] += 1
    # U[A11[i][j]] += 1
    # V[B00[j][k]] += 1
    # V[B11[j][k]] += 1
    # W[C00[k][i]] += 1
    # W[C11[k][i]] += 1
    
    def add_step(u_blocks, v_blocks, w_blocks, rank_off):
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    col = rank_off + i*9 + j*3 + k
                    for ub in u_blocks:
                        U[ub[i][j]][col] += 1
                    for vb in v_blocks:
                        V[vb[j][k]][col] += 1
                    for wb in w_blocks:
                        W[wb[k][i]][col] += 1

    # M1
    add_step([u_00_idx, u_11_idx], [v_00_idx, v_11_idx], [w_00_idx, w_11_idx], rank_off)
    rank_off += 27
    
    # M2 = (A10 + A11) B00
    add_step([u_10_idx, u_11_idx], [v_00_idx], [w_10_idx], rank_off)
    rank_off += 27
    
    # M3 = A00 (B01 - B11)
    # V: B01 +1, B11 -1
    # Need to handle negative scales?
    # The problem allows integer/half-integer entries.
    # So we can use -1.
    
    def add_step_scaled(u_blocks, v_blocks, w_blocks, rank_off):
        # u_blocks: list of (idx_map, scale)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    col = rank_off + i*9 + j*3 + k
                    for idx_map, scale in u_blocks:
                        U[idx_map[i][j]][col] += scale
                    for idx_map, scale in v_blocks:
                        V[idx_map[j][k]][col] += scale
                    for idx_map, scale in w_blocks:
                        W[idx_map[k][i]][col] += scale

    # M1
    add_step_scaled([(u_00_idx, 1), (u_11_idx, 1)], [(v_00_idx, 1), (v_11_idx, 1)], [(w_00_idx, 1), (w_11_idx, 1)], 0)
    rank_off += 27
    
    # M2
    add_step_scaled([(u_10_idx, 1), (u_11_idx, 1)], [(v_00_idx, 1)], [(w_10_idx, 1)], rank_off)
    rank_off += 27
    
    # M3
    add_step_scaled([(u_00_idx, 1)], [(v_01_idx, 1), (v_11_idx, -1)], [(w_01_idx, 1)], rank_off)
    rank_off += 27
    
    # M4 = A11 (B10 - B00)
    add_step_scaled([(u_11_idx, 1)], [(v_10_idx, 1), (v_00_idx, -1)], [(w_00_idx, 1)], rank_off)
    rank_off += 27
    
    # M5 = (A00 + A01) B11
    add_step_scaled([(u_00_idx, 1), (u_01_idx, 1)], [(v_11_idx, 1)], [(w_01_idx, 1)], rank_off)
    rank_off += 27
    
    # M6 = (A10 - A00) (B00 + B01)
    add_step_scaled([(u_10_idx, 1), (u_00_idx, -1)], [(v_00_idx, 1), (v_01_idx, 1)], [(w_11_idx, 1)], rank_off)
    rank_off += 27
    
    # M7 = (A01 - A11) (B10 + B11)
    add_step_scaled([(u_01_idx, 1), (u_11_idx, -1)], [(v_10_idx, 1), (v_11_idx, 1)], [(w_11_idx, 1)], rank_off)
    rank_off += 27
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
