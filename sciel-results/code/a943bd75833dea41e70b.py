"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def _build_strassen_2x2():
    """
    Build the 2x2 Strassen decomposition.
    Returns (U, V, W) where U, V, W each have shape (4, 7).
    Row indexing: U[i*2+j], V[j*2+k], W[k*2+i] for i,j,k in {0,1}.
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # P1 = (a00+a11)*(b00+b11)
    U[0][0] = 1; U[3][0] = 1   # a00, a11
    V[0][0] = 1; V[3][0] = 1   # b00, b11
    W[0][0] = 1; W[3][0] = 1   # c00, c11
    
    # P2 = (a10+a11)*b00
    U[2][1] = 1; U[3][1] = 1   # a10, a11
    V[0][1] = 1                # b00
    W[2][1] = 1; W[3][1] = -1  # c10, -c11
    
    # P3 = a00*(b01-b11)
    U[0][2] = 1                # a00
    V[1][2] = 1; V[3][2] = -1  # b01, -b11
    W[1][2] = 1; W[3][2] = 1   # c01, c11
    
    # P4 = a11*(b10-b00)
    U[3][3] = 1                # a11
    V[2][3] = 1; V[0][3] = -1  # b10, -b00
    W[0][3] = 1; W[2][3] = 1   # c00, c10
    
    # P5 = (a00+a01)*b11
    U[0][4] = 1; U[1][4] = 1   # a00, a01
    V[3][4] = 1                # b11
    W[0][4] = -1; W[1][4] = 1  # -c00, c01
    
    # P6 = (a10-a00)*(b00+b01)
    U[2][5] = 1; U[0][5] = -1  # a10, -a00
    V[0][5] = 1; V[1][5] = 1   # b00, b01
    W[3][5] = 1                # c11
    
    # P7 = (a01-a11)*(b10+b11)
    U[1][6] = 1; U[3][6] = -1  # a01, -a11
    V[2][6] = 1; V[3][6] = 1   # b10, b11
    W[0][6] = 1                # c00
    
    return U, V, W

def _build_strassen_nxn(n):
    """
    Recursive Strassen decomposition for n x n matrix multiplication.
    n must be a power of 2.
    Returns (U, V, W) where U, V, W each have shape (n^2, rank).
    Row indexing: U[i*n+j], V[j*n+k], W[k*n+i].
    """
    if n == 2:
        U, V, W = _build_strassen_2x2()
        # U has shape (4, 7), V has shape (4, 7), W has shape (4, 7)
        return U, V, W
    
    n_half = n // 2
    U_sub, V_sub, W_sub = _build_strassen_nxn(n_half)
    rank_sub = len(U_sub[0])
    rank = 7 * rank_sub
    
    U = [[0]*rank for _ in range(n*n)]
    V = [[0]*rank for _ in range(n*n)]
    W = [[0]*rank for _ in range(n*n)]
    
    # 7 Strassen combinations with (a_block, b_block, c_block)
    # Each block is (row_start, col_start) for the n/2 x n/2 submatrix
    strassen_defs = [
        # (a_row, a_col), (b_row, b_col), (c_row, c_col)
        (0, 0, 0, 0, 0, 0),      # P1: A00 * B00 -> C00
        (0, 0, 0, 0, 1, 0),      # P2: A10 * B00 -> C10
        (0, 0, 0, 1, 1, 1),      # P3: A00 * B01 -> C01
        (0, 0, 0, 0, 0, 1),      # P4: A11 * B10 -> C10
        (0, 0, 0, 1, 1, 0),      # P5: A01 * B11 -> C11
        (0, 0, 1, 0, 1, 1),      # P6: A10 * B01 -> C11
        (0, 1, 0, 1, 0, 1),      # P7: A01 * B10 -> C01
    ]
    
    for s in range(7):
        a_r0, a_c0, b_r0, b_c0, c_r0, c_c0 = strassen_defs[s]
        
        for t in range(rank_sub):
            col = s * rank_sub + t
            
            # A part: extract from U_sub
            for li in range(n_half):
                for lj in range(n_half):
                    coeff_a = U_sub[li * n_half + lj][t]
                    if coeff_a != 0:
                        global_i = a_r0 * n_half + li
                        global_j = a_c0 * n_half + lj
                        U[global_i * n + global_j][col] += coeff_a
            
            # B part: extract from V_sub
            for lj in range(n_half):
                for lk in range(n_half):
                    coeff_b = V_sub[lj * n_half + lk][t]
                    if coeff_b != 0:
                        global_j = b_r0 * n_half + lj
                        global_k = b_c0 * n_half + lk
                        V[global_j * n + global_k][col] += coeff_b
            
            # C part: extract from W_sub
            for lk in range(n_half):
                for li in range(n_half):
                    coeff_c = W_sub[lk * n_half + li][t]
                    if coeff_c != 0:
                        global_k = c_r0 * n_half + lk
                        global_i = c_c0 * n_half + li
                        W[global_k * n + global_i][col] += coeff_c
    
    return U, V, W

def _build_strassen_rect(n, m, p):
    """
    Build Strassen-like decomposition for n x m * m x p -> n x p
    by splitting into blocks and using 2x2 Strassen where possible.
    
    Returns (U, V, W) where:
      U has shape (n*m, rank)
      V has shape (m*p, rank)
      W has shape (p*n, rank)
    """
    if n == 1 and m == 1 and p == 1:
        return [[1]], [[1]], [[1]]
    
    # Try to split along the largest dimension
    # We want to use Strassen 2x2 blocks where possible
    
    # If all dimensions are even, split into 2x2 blocks of (n/2)x(m/2)x(p/2)
    if n % 2 == 0 and m % 2 == 0 and p % 2 == 0:
        n_half, m_half, p_half = n // 2, m // 2, p // 2
        
        # Build the half-size decomposition
        U_sub, V_sub, W_sub = _build_strassen_rect(n_half, m_half, p_half)
        rank_sub = len(U_sub[0])
        rank = 7 * rank_sub
        
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        
        # 7 Strassen combinations
        # (a_block_i, a_block_j, b_block_j, b_block_k, c_block_i, c_block_k)
        strassen_defs = [
            (0, 0, 0, 0, 0, 0),      # P1: A00 * B00 -> C00
            (1, 0, 0, 0, 1, 0),      # P2: A10 * B00 -> C10
            (0, 0, 0, 1, 0, 1),      # P3: A00 * B01 -> C01
            (1, 1, 1, 0, 1, 0),      # P4: A11 * B10 -> C10
            (0, 1, 1, 1, 0, 1),      # P5: A01 * B11 -> C11
            (1, 0, 0, 1, 1, 1),      # P6: A10 * B01 -> C11
            (0, 1, 1, 0, 0, 1),      # P7: A01 * B10 -> C01
        ]
        
        for s in range(7):
            ai, aj, bj, bk, ci, ck = strassen_defs[s]
            
            for t in range(rank_sub):
                col = s * rank_sub + t
                
                # A part
                for li in range(n_half):
                    for lj in range(m_half):
                        coeff_a = U_sub[li * m_half + lj][t]
                        if coeff_a != 0:
                            gi = ai * n_half + li
                            gj = aj * m_half + lj
                            U[gi * m + gj][col] += coeff_a
                
                # B part
                for lj in range(m_half):
                    for lk in range(p_half):
                        coeff_b = V_sub[lj * p_half + lk][t]
                        if coeff_b != 0:
                            gj = bj * m_half + lj
                            gk = bk * p_half + lk
                            V[gj * p + gk][col] += coeff_b
                
                # C part
                for lk in range(p_half):
                    for li in range(n_half):
                        coeff_c = W_sub[lk * n_half + li][t]
                        if coeff_c != 0:
                            gk = ck * p_half + lk
                            gi = ci * n_half + li
                            W[gk * n + gi][col] += coeff_c
        
        return U, V, W
    
    # If we can't do a full 2x2 split, fall back to schoolbook
    # Or try a partial split
    return _build_schoolbook(n, m, p)

def _build_schoolbook(n, m, p):
    """
    Build schoolbook decomposition for n x m * m x p -> n x p.
    Returns U, V, W with rank n*m*p.
    """
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

def _combine_decompositions(decomps, n, m, p):
    """
    Combine a list of decompositions into one.
    Each decomp is (U_sub, V_sub, W_sub, (i_s, i_e), (j_s, j_e), (k_s, k_e))
    where i,j,k refer to the sub-matrix indices.
    """
    total_rank = sum(len(d[0][0]) for d in decomps)
    
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    current_rank = 0
    for U_sub, V_sub, W_sub, (i_s, i_e), (j_s, j_e), (k_s, k_e) in decomps:
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
    
    return U, V, W

def _build_rectangular_optimal(n, m, p):
    """
    Try to find a good block decomposition for n x m * m x p.
    """
    if n == 4 and m == 5 and p == 6:
        # Try different block splits and pick the best
        # Option 1: 4x4x4 Strassen (49) + 4x1x6 schoolbook (24) + 4x4x2 Strassen (28) = 101
        # Option 2: 4x3x6 Strassen (42) + 4x2x6 Strassen (28) = 70
        # But 3 and 2 aren't powers of 2, so Strassen won't work directly.
        
        # Let's try: split m=5 into 3+2, and p=6 stays as is.
        # A is 4x3 * B is 3x6 -> C1 is 4x6: use _build_strassen_rect(4,3,6)
        # A is 4x2 * B is 2x6 -> C2 is 4x6: use _build_strassen_rect(4,2,6)
        
        # For 4x3x6: can't do 2x2 Strassen on m=3, so schoolbook: 4*3*6 = 72
        # For 4x2x6: can split 4 into 2+2, but m=2, p=6... let's try schoolbook: 4*2*6 = 48
        # Total: 72 + 48 = 120. That's worse.
        
        # Better approach: use the existing strategy which gives 101.
        U1, V1, W1 = _build_strassen_rect(4, 4, 4)
        decomp1 = (U1, V1, W1, (0, 4), (0, 4), (0, 4))
        
        U2, V2, W2 = _build_schoolbook(4, 1, 6)
        decomp2 = (U2, V2, W2, (0, 4), (4, 5), (0, 6))
        
        U3, V3, W3 = _build_strassen_rect(4, 4, 2)
        decomp3 = (U3, V3, W3, (0, 4), (0, 4), (4, 6))
        
        return _combine_decompositions([decomp1, decomp2, decomp3], n, m, p)
    
    # General case: try Strassen if possible, else schoolbook
    return _build_strassen_rect(n, m, p)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    
    try:
        U, V, W = _build_rectangular_optimal(n, m, p)
        return {"U": U, "V": W[0] and W, "W": W}
        
    except Exception:
        # Fallback to schoolbook if something goes wrong
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
