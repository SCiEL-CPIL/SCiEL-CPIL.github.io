"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2_2x2():
    """Returns U, V, W for 2x2 * 2x2 -> 2x2 with rank 7."""
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    # M1: (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2: (A10+A11)B00 -> C10 += M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1
    
    # M3: A00(B01-B11) -> C01 += M3, C11 += M3
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # M4: A11(B10-B00) -> C00 += M4, C10 += M4
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5: (A00+A01)B11 -> C00 -= M5, C01 += M5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6: (A10-A00)(B00+B01) -> C11 += M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7: (A01-A11)(B10+B11) -> C00 += M7, C11 += M7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[3, 6] = 1
    
    return U, V, W

def _schoolbook_2x2_2x1():
    """Returns U, V, W for 2x2 * 2x1 -> 2x1 with rank 4."""
    n, m, p = 2, 2, 1
    rank = 4
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    # M1: A00 B00 -> C00 += M1
    U[0, 0] = 1
    V[0, 0] = 1
    W[0, 0] = 1
    
    # M2: A01 B10 -> C00 += M2
    U[1, 1] = 1
    V[1, 1] = 1
    W[0, 1] = 1
    
    # M3: A10 B00 -> C10 += M3
    U[2, 2] = 1
    V[0, 2] = 1
    W[1, 2] = 1
    
    # M4: A11 B10 -> C10 += M4
    U[3, 3] = 1
    V[1, 3] = 1
    W[1, 3] = 1
    
    return U, V, W

def _build_strassen_decomp(n, m, p, base_U, base_V, base_W, bn, bm, bp):
    """
    Builds decomposition for n x m x p using Strassen with block sizes bn x bm and bp x bn?
    Actually, Strassen splits A (n x m) into 2x2 blocks of size (n/2 x m/2).
    And B (m x p) into 2x2 blocks of size (m/2 x p/2).
    Here we assume n=2*bn, m=2*bm, p=2*bp.
    """
    rank_base = base_U.shape[1]
    total_rank = 7 * rank_base
    
    U = np.zeros((n*m, total_rank), dtype=int)
    V = np.zeros((m*p, total_rank), dtype=int)
    W = np.zeros((p*n, total_rank), dtype=int)
    
    # Strassen terms definitions:
    # (Left A blocks, Right B blocks, Output C blocks with signs)
    # A blocks: 00, 01, 10, 11
    # B blocks: 00, 01, 10, 11
    # C blocks: 00, 01, 10, 11
    
    terms = [
        # (A_coefs, B_coefs, C_coefs)
        # Coefs are dict: block_id -> sign
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),       # M1
        ({2: 1, 3: 1}, {0: 1}, {1: 1}),                   # M2
        ({0: 1}, {1: 1, 3: -1}, {2: 1, 3: 1}),            # M3
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 1: 1}),            # M4
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 2: 1}),            # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),            # M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1, 3: 1}),      # M7
    ]
    
    for t_idx, (a_coefs, b_coefs, c_coefs) in enumerate(terms):
        col_offset = t_idx * rank_base
        
        for c_base in range(rank_base):
            col = col_offset + c_base
            
            # Fill U
            for a_block, sign in a_coefs.items():
                # a_block 0: rows 0..bn-1, cols 0..bm-1
                # a_block 1: rows 0..bn-1, cols bm..m-1
                # a_block 2: rows bn..n-1, cols 0..bm-1
                # a_block 3: rows bn..n-1, cols bm..m-1
                r_start = (a_block // 2) * bn
                c_start = (a_block % 2) * bm
                
                for r_loc in range(bn):
                    for c_loc in range(bm):
                        # Local index in base_U
                        r_base = r_loc * bm + c_loc
                        val = base_U[r_base, c_base] * sign
                        if val != 0:
                            # Global index in U
                            r_glob = r_start + r_loc
                            c_glob = c_start + c_loc
                            U[r_glob * m + c_glob, col] += val

            # Fill V
            for b_block, sign in b_coefs.items():
                # b_block 0: rows 0..bm-1, cols 0..bp-1
                # b_block 1: rows 0..bm-1, cols bp..p-1
                # b_block 2: rows bm..m-1, cols 0..bp-1
                # b_block 3: rows bm..m-1, cols bp..p-1
                r_start = (b_block // 2) * bm
                c_start = (b_block % 2) * bp
                
                for r_loc in range(bm):
                    for c_loc in range(bp):
                        r_base = r_loc * bp + c_loc
                        val = base_V[r_base, c_base] * sign
                        if val != 0:
                            r_glob = r_start + r_loc
                            c_glob = c_start + c_loc
                            V[r_glob * p + c_glob, col] += val

            # Fill W
            for c_block, sign in c_coefs.items():
                # c_block 0: rows 0..bp-1, cols 0..bn-1
                # c_block 1: rows 0..bp-1, cols bn..n-1
                # c_block 2: rows bp..p-1, cols 0..bn-1
                # c_block 3: rows bp..p-1, cols bn..n-1
                r_start = (c_block // 2) * bp
                c_start = (c_block % 2) * bn
                
                for r_loc in range(bp):
                    for c_loc in range(bn):
                        r_base = r_loc * bn + c_loc
                        val = base_W[r_base, c_base] * sign
                        if val != 0:
                            r_glob = r_start + r_loc
                            c_glob = c_start + c_loc
                            W[r_glob * n + c_glob, col] += val

    return U, V, W

def construct():
    start_time = time.time()
    deadline = 150.0
    
    # Part 1: 4x4 * 4x4 -> 4x4
    # Blocks: 2x2
    base_U1, base_V1, base_W1 = _strassen_2x2_2x2()
    U1, V1, W1 = _build_strassen_decomp(4, 4, 4, base_U1, base_V1, base_W1, 2, 2, 2)
    
    if time.time() > deadline - 10:
        # Fallback to schoolbook if taking too long (unlikely)
        return _schoolbook_construct()

    # Part 2: 4x4 * 4x2 -> 4x2
    # Blocks: 2x2 for A, 2x1 for B
    base_U2, base_V2, base_W2 = _schoolbook_2x2_2x1()
    # Note: _build_strassen_decomp assumes square-ish blocks for simplicity in index mapping?
    # My implementation handles bn, bm, bp.
    # A is 4x4, blocks 2x2. bn=2, bm=2.
    # B is 4x2, blocks 2x1. bm=2, bp=1.
    # C is 4x2, blocks 2x1. bp=1, bn=2.
    U2, V2, W2 = _build_strassen_decomp(4, 4, 2, base_U2, base_V2, base_W2, 2, 2, 1)
    
    if time.time() > deadline - 5:
        return _schoolbook_construct()

    # Combine
    rank1 = U1.shape[1]
    rank2 = U2.shape[1]
    total_rank = rank1 + rank2
    
    n, m, p = 4, 4, 6
    
    U = np.zeros((n*m, total_rank), dtype=int)
    V = np.zeros((m*p, total_rank), dtype=int)
    W = np.zeros((p*n, total_rank), dtype=int)
    
    # Copy Part 1
    U[:, :rank1] = U1
    V[:m*4, :rank1] = V1  # V1 is 16x49. m*p=24. First 16 rows correspond to B's first 4 cols.
    W[:p*4, :rank1] = W1  # W1 is 16x49. p*n=24. First 16 rows correspond to C's first 4 cols.
    
    # Copy Part 2
    U[:, rank1:] = U2
    # V2 is 8x28. Corresponds to B's last 2 cols (rows 16-23 of B).
    # B is 4x6. Rows 0-3, Cols 0-5.
    # Wait, V index is j*p + k.
    # For Part 1, B is 4x4. j in 0-3, k in 0-3.
    # For Part 2, B is 4x2. j in 0-3, k in 0-1 (local).
    # Global k for Part 2 is 4, 5.
    # So V rows for Part 2 should be filled in V[16:24, rank1:].
    # V2 shape is (m*bp, rank) = (4*1, 28) = (4, 28)?
    # No, in _build_strassen_decomp, V shape is (m*p, rank).
    # For Part 2, m=4, p=2. V shape (8, 28).
    # Indices in V2: j_glob * 2 + k_glob.
    # j_glob in 0-3, k_glob in 0-1.
    # Global V index for Part 2: j * 6 + k_global.
    # k_global = 4 + k_local.
    # So we need to map V2 indices to global V indices.
    
    for r in range(V2.shape[0]):
        j_local, k_local = divmod(r, 2) # Since p_local=2? No, p_local=1 in base, but here p=2.
        # Wait, V2 was built with p=2.
        # V2 index r corresponds to j in 0-3, k in 0-1.
        # r = j * 2 + k.
        j = r // 2
        k = r % 2
        k_global = 4 + k
        V[j * 6 + k_global, rank1:] = V2[r, :]
        
    # Similarly for W
    # W2 shape (p*n, rank) = (2*4, 28) = (8, 28).
    # W2 index r corresponds to k in 0-1, i in 0-3.
    # r = k * 4 + i.
    for r in range(W2.shape[0]):
        k_local, i = divmod(r, 4)
        k_global = 4 + k_local
        W[k_global * 4 + i, rank1:] = W2[r, :]
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _schoolbook_construct():
    n, m, p = DIMENSIONS
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
