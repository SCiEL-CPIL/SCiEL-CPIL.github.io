import time
import random
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def _generate_strassen_2x2():
    """
    Returns U, V, W matrices for 2x2x2 Strassen decomposition.
    Shapes: (4, 7), (4, 7), (4, 7)
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # P1 = (a00+a11)(b00+b11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # P2 = (a10+a11)b00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # P3 = a00(b01-b11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # P4 = a11(b10-b00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # P5 = (a00+a01)b11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # P6 = (a10-a00)(b00+b01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # P7 = (a01-a11)(b10+b11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _construct_4x4x4_strassen():
    """
    Constructs 4x4x4 Strassen decomposition.
    Rank: 49
    U: (16, 49), V: (16, 49), W: (16, 49)
    """
    U2, V2, W2 = _generate_strassen_2x2()
    rank = 49
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    # Strassen coefficients for 2x2 blocks
    # Each of the 7 products is computed 7 times (for each 2x2x2 sub-block in the 4x4x4 grid? No, 4x4x4 is 2x2 blocks of 2x2 matrices)
    # Actually, 4x4x4 can be seen as 2x2 blocks where each block is 2x2.
    # The 4x4x4 Strassen uses 7 multiplications of 2x2x2 matrices.
    # Wait, standard Strassen on 4x4x4:
    # Split A, B, C into 2x2 blocks of size 2x2.
    # A = [[A00, A01], [A10, A11]] where each Ai is 2x2.
    # This requires 7 multiplications of 2x2x2 tensors.
    # Each 2x2x2 multiplication has rank 7.
    # Total rank = 7 * 7 = 49.
    
    # The 7 Strassen steps for the block level:
    # S0 = (A00+A11)(B00+B11) -> C00 += S0, C11 += S0
    # S1 = (A10+A11)B00 -> C10 += S1, C11 -= S1
    # S2 = A00(B01-B11) -> C01 += S2, C11 += S2
    # S3 = A11(B10-B00) -> C00 += S3, C10 += S3
    # S4 = (A00+A01)B11 -> C00 -= S4, C01 += S4
    # S5 = (A10-A00)(B00+B01) -> C11 += S5
    # S6 = (A01-A11)(B10+B11) -> C00 += S6
    
    # For each of the 7 block-level products, we perform a 2x2x2 Strassen decomposition.
    # So for block product k (0..6), we use the 7 columns of U2, V2, W2.
    # The rows of U2, V2, W2 correspond to the 4 elements of the 2x2 sub-matrices.
    
    # Mapping of block indices to global indices:
    # A block (bi, bj) corresponds to rows 2*bi + 0..1 and 2*bj + 0..1 in the 4x4 matrix.
    # Specifically, row index in 4x4 is 2*bi + li, col is 2*bj + lj.
    # U index is row * 4 + col.
    
    block_defs = [
        # (U_blocks, V_blocks, W_blocks)
        # U_blocks: list of (bi, bj, coeff)
        # V_blocks: list of (bj, bk, coeff)
        # W_blocks: list of (bk, bi, coeff)
        # Note: The local 2x2 Strassen U2/V2/W2 already handles the internal 2x2 structure.
        # We just need to map the 2x2 block coordinates to the 4x4 global coordinates.
        
        # Product 0: (A00+A11)(B00+B11)
        # U: A00 (1), A11 (1)
        # V: B00 (1), B11 (1)
        # W: C00 (1), C11 (1)
        ([((0,0), 1), ((1,1), 1)],
         [((0,0), 1), ((1,1), 1)],
         [((0,0), 1), ((1,1), 1)]),
         
        # Product 1: (A10+A11)B00
        # U: A10 (1), A11 (1)
        # V: B00 (1)
        # W: C10 (1), C11 (-1)
        ([((1,0), 1), ((1,1), 1)],
         [((0,0), 1)],
         [((1,0), 1), ((1,1), -1)]),
         
        # Product 2: A00(B01-B11)
        # U: A00 (1)
        # V: B01 (1), B11 (-1)
        # W: C01 (1), C11 (1)
        ([((0,0), 1)],
         [((0,1), 1), ((1,1), -1)],
         [((0,1), 1), ((1,1), 1)]),
         
        # Product 3: A11(B10-B00)
        # U: A11 (1)
        # V: B10 (1), B00 (-1)
        # W: C00 (1), C10 (1)
        ([((1,1), 1)],
         [((1,0), 1), ((0,0), -1)],
         [((0,0), 1), ((1,0), 1)]),
         
        # Product 4: (A00+A01)B11
        # U: A00 (1), A01 (1)
        # V: B11 (1)
        # W: C00 (-1), C01 (1)
        ([((0,0), 1), ((0,1), 1)],
         [((1,1), 1)],
         [((0,0), -1), ((0,1), 1)]),
         
        # Product 5: (A10-A00)(B00+B01)
        # U: A10 (1), A00 (-1)
        # V: B00 (1), B01 (1)
        # W: C11 (1)
        ([((1,0), 1), ((0,0), -1)],
         [((0,0), 1), ((0,1), 1)],
         [((1,1), 1)]),
         
        # Product 6: (A01-A11)(B10+B11)
        # U: A01 (1), A11 (-1)
        # V: B10 (1), B11 (1)
        # W: C00 (1)
        ([((0,1), 1), ((1,1), -1)],
         [((1,0), 1), ((1,1), 1)],
         [((0,0), 1)]),
    ]
    
    for s_idx, (u_blocks, v_blocks, w_blocks) in enumerate(block_defs):
        for t_idx in range(7):
            col = s_idx * 7 + t_idx
            
            # Fill U
            for local_row in range(4):
                coeff_local = U2[local_row][t_idx]
                if coeff_local == 0: continue
                li = local_row // 2
                lj = local_row % 2
                for bi, bj, sigma in u_blocks:
                    gi = bi * 2 + li
                    gj = bj * 2 + lj
                    u_idx = gi * 4 + gj
                    U[u_idx][col] += coeff_local * sigma
            
            # Fill V
            for local_row in range(4):
                coeff_local = V2[local_row][t_idx]
                if coeff_local == 0: continue
                lj = local_row // 2
                lk = local_row % 2
                for bj, bk, sigma in v_blocks:
                    gj = bj * 2 + lj
                    gk = bk * 2 + lk
                    v_idx = gj * 4 + gk
                    V[v_idx][col] += coeff_local * sigma
            
            # Fill W
            for local_row in range(4):
                coeff_local = W2[local_row][t_idx]
                if coeff_local == 0: continue
                lk = local_row // 2
                li = local_row % 2
                for bk, bi, sigma in w_blocks:
                    gk = bk * 2 + lk
                    gi = bi * 2 + li
                    w_idx = gk * 4 + gi
                    W[w_idx][col] += coeff_local * sigma
                    
    return U, V, W

def _construct_4x4x2_strassen():
    """
    Constructs 4x4x2 decomposition using Strassen 2x2 blocks.
    Rank: 28
    U: (16, 28), V: (8, 28), W: (8, 28)
    """
    U2, V2, W2 = _generate_strassen_2x2()
    rank = 28
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(8)]
    W = [[0]*rank for _ in range(8)]
    
    # 4x4x2 can be split into four 2x2x2 blocks?
    # A is 4x4, B is 4x2, C is 4x2.
    # Split A into 2x2 blocks: A00, A01, A10, A11.
    # Split B into 2x2 blocks? No, B is 4x2.
    # If we split the middle dimension (4) into 2+2, B becomes two 2x2 blocks: B0, B1.
    # C is 4x2. Split into two 2x2 blocks: C0, C1.
    # 
    # The multiplication is C = A * B.
    # C0 = A00*B0 + A10*B1? No.
    # Let's index A as 4x4, B as 4x2.
    # C[i,k] = sum_j A[i,j] B[j,k].
    # 
    # We can view this as two independent 4x4x1 multiplications?
    # Or we can use the 2x2x2 Strassen on the 2x2x2 sub-blocks.
    # 
    # Let's partition the indices:
    # i: 0..3, j: 0..3, k: 0..1
    # 
    # If we use 2x2x2 Strassen, we need 2x2 blocks in all dimensions.
    # k is only 2, so it's one block of size 2.
    # i is 4, so two blocks of size 2.
    # j is 4, so two blocks of size 2.
    # 
    # This creates a 2 (i-blocks) x 2 (j-blocks) x 1 (k-blocks) grid of 2x2x2 blocks.
    # Total 4 blocks.
    # Each block uses 7 multiplications.
    # Total rank = 4 * 7 = 28.
    
    # Block (bi, bj) where bi in {0,1}, bj in {0,1}
    # A block: rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
    # B block: rows 2*bj..2*bj+1, cols 0..1 (since k is 0..1, it's always the full k-range)
    # C block: rows 2*bi..2*bi+1, cols 0..1
    
    for bi in range(2):
        for bj in range(2):
            base_rank = (bi * 2 + bj) * 7
            for t in range(7):
                col = base_rank + t
                
                # U contribution
                for li in range(2):
                    for lj in range(2):
                        local_u = li * 2 + lj
                        coeff = U2[local_u][t]
                        if coeff == 0: continue
                        gi = bi * 2 + li
                        gj = bj * 2 + lj
                        u_idx = gi * 4 + gj
                        U[u_idx][col] += coeff
                        
                # V contribution
                # B is 4x2. The block corresponds to rows 2*bj..2*bj+1 and cols 0..1.
                # Local V index in 2x2x2 is row (j-local) * 2 + k-local.
                # Here j-local is lj (0..1), k-local is lk (0..1).
                for lj in range(2):
                    for lk in range(2):
                        local_v = lj * 2 + lk
                        coeff = V2[local_v][t]
                        if coeff == 0: continue
                        gj = bj * 2 + lj
                        gk = lk # k is 0..1, no offset
                        v_idx = gj * 2 + gk
                        V[v_idx][col] += coeff
                        
                # W contribution
                # C is 4x2. Block corresponds to rows 2*bi..2*bi+1 and cols 0..1.
                # Local W index in 2x2x2 is row (k-local) * 2 + i-local.
                for lk in range(2):
                    for li in range(2):
                        local_w = lk * 2 + li
                        coeff = W2[local_w][t]
                        if coeff == 0: continue
                        gk = lk
                        gi = bi * 2 + li
                        w_idx = gk * 4 + gi
                        W[w_idx][col] += coeff
                        
    return U, V, W

def _construct_4x1x6_schoolbook():
    """
    Constructs 4x1x6 schoolbook decomposition.
    Rank: 24
    U: (4, 24), V: (6, 24), W: (24, 24) -> Wait, W is (p*n, rank) = (24, 24).
    Actually, U is (n*m, rank) = (4, 24).
    V is (m*p, rank) = (6, 24).
    W is (p*n, rank) = (24, 24).
    """
    n, m, p = 4, 1, 6
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return U, V, W

def _combine_parts(parts, n, m, p):
    """
    Combines partial decompositions into the full n x m x p decomposition.
    parts: list of (U_sub, V_sub, W_sub, i_range, j_range, k_range)
    """
    total_rank = sum(len(U_sub[0]) for U_sub, _, _, _, _, _ in parts)
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    current_rank = 0
    for U_sub, V_sub, W_sub, (i_s, i_e), (j_s, j_e), (k_s, k_e) in parts:
        n_sub = i_e - i_s
        m_sub = j_e - j_s
        p_sub = k_e - k_s
        rank_sub = len(U_sub[0])
        
        for r_sub in range(n_sub * m_sub):
            i_loc = r_sub // m_sub
            j_loc = r_sub % m_sub
            i_glob = i_s + i_loc
            j_glob = j_s + j_loc
            r_glob = i_glob * m + j_glob
            for c_sub in range(rank_sub):
                U[r_glob][current_rank + c_sub] = U_sub[r_sub][c_sub]
                
        for r_sub in range(m_sub * p_sub):
            j_loc = r_sub // p_sub
            k_loc = r_sub % p_sub
            j_glob = j_s + j_loc
            k_glob = k_s + k_loc
            r_glob = j_glob * p + k_glob
            for c_sub in range(rank_sub):
                V[r_glob][current_rank + c_sub] = V_sub[r_sub][c_sub]
                
        for r_sub in range(p_sub * n_sub):
            k_loc = r_sub // n_sub
            i_loc = r_sub % n_sub
            k_glob = k_s + k_loc
            i_glob = i_s + i_loc
            r_glob = k_glob * n + i_glob
            for c_sub in range(rank_sub):
                W[r_glob][current_rank + c_sub] = W_sub[r_sub][c_sub]
                
        current_rank += rank_sub
        
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    n, m, p = DIMENSIONS
    
    # Part 1: 4x4x4 (i:0-4, j:0-4, k:0-4)
    U1, V1, W1 = _construct_4x4x4_strassen()
    
    # Part 2: 4x1x6 (i:0-4, j:4-5, k:0-6)
    U2, V2, W2 = _construct_4x1x6_schoolbook()
    
    # Part 3: 4x4x2 (i:0-4, j:0-4, k:4-6)
    U3, V3, W3 = _construct_4x4x2_strassen()
    
    parts = [
        (U1, V1, W1, (0, 4), (0, 4), (0, 4)),
        (U2, V2, W2, (0, 4), (4, 5), (0, 6)),
        (U3, V3, W3, (0, 4), (0, 4), (4, 6))
    ]
    
    try:
        result = _combine_parts(parts, n, m, p)
        if time.time() > deadline:
            raise TimeoutError()
        return result
    except Exception:
        # Fallback to schoolbook if something goes wrong
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
