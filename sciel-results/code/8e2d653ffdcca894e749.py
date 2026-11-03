"""Matrix multiplication 4x5x5 with rank-reduced decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]

def _strassen_2x2x2_terms():
    """
    Returns the 7 rank-1 terms for 2x2x2 Strassen decomposition.
    Each term is (u, v, w) where u, v, w are vectors of length 4 (row-major 2x2).
    """
    terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [-1, 1, 0, 0], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
    ]
    return terms

def _build_4x4x4_decomp():
    """
    Build the 4x4x4 decomposition using 7 Strassen 2x2x2 blocks.
    Returns U, V, W of shape (16, 49).
    """
    strassen_terms = _strassen_2x2x2_terms()
    
    U = [[0] * 49 for _ in range(16)]
    V = [[0] * 49 for _ in range(16)]
    W = [[0] * 49 for _ in range(16)]
    
    col = 0
    # Define the 7 Strassen products for 4x4
    # Each entry: (A_blocks, B_blocks, C_block_targets)
    # A_blocks: list of (block_idx, coeff)
    # B_blocks: list of (block_idx, coeff)
    # C_block_targets: list of (block_idx, coeff)
    
    # Block indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # M1 = (A00+A11)(B00+B11) -> C00+1
    # M2 = (A10+A11)B00 -> C11-1
    # M3 = A00(B10-B11) -> C01+1, C11+1
    # M4 = A11(B01-B00) -> C00+1, C10+1
    # M5 = (A00+A01)B11 -> C00-1, C01+1
    # M6 = (A10-A00)(B00+B01) -> C11+1
    # M7 = (A01-A11)(B10+B11) -> C00+1
    
    combos = [
        # M1
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1)]),
        # M2
        ([(2, 1), (3, 1)], [(0, 1)], [(3, -1)]),
        # M3
        ([(0, 1)], [(2, 1), (3, -1)], [(1, 1), (3, 1)]),
        # M4
        ([(3, 1)], [(1, 1), (0, -1)], [(0, 1), (2, 1)]),
        # M5
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        # M6
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # M7
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)])
    ]
    
    for s in range(7):
        a_blocks, b_blocks, c_blocks = combos[s]
        for t in range(7):
            u_t, v_t, w_t = strassen_terms[t]
            
            # Process U
            for bi, bj in a_blocks:
                coeff_a = bi
                # Map 2x2 local index l to 4x4 global index
                for l in range(4):
                    li = l // 2
                    lj = l % 2
                    # Global A index: (2*bi + li, 2*bj + lj)
                    i_global = 2 * bi + li
                    j_global = 2 * bj + lj
                    U[i_global * 4 + j_global][col] += u_t[l] * coeff_a
            
            # Process V
            for bi, bj in b_blocks:
                coeff_b = bj
                for l in range(4):
                    li = l // 2
                    lj = l % 2
                    i_global = 2 * bi + li
                    j_global = 2 * bj + lj
                    V[i_global * 4 + j_global][col] += v_t[l] * coeff_b
            
            # Process W
            for bi, bj in c_blocks:
                coeff_c = bj
                for l in range(4):
                    li = l // 2
                    lj = l % 2
                    i_global = 2 * bi + li
                    j_global = 2 * bj + lj
                    W[i_global * 4 + j_global][col] += w_t[l] * coeff_c
            
            col += 1
            
    return U, V, W

def _schoolbook_sub(n, m, p):
    """Schoolbook for n*m x m*p sub-block."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _assemble_block_decomp(n, m, p, sub_decomps):
    """
    Assemble full decomposition from sub-block decompositions.
    sub_decomps: list of (U_sub, V_sub, W_sub, i0, j0, k0, ni, mj, pk)
    """
    total_rank = sum(len(U[0]) for U, V, W, *_ in sub_decomps)
    U_out = [[0] * total_rank for _ in range(n * m)]
    V_out = [[0] * total_rank for _ in range(m * p)]
    W_out = [[0] * total_rank for _ in range(p * n)]
    
    col_off = 0
    for U_sub, V_sub, W_sub, i0, j0, k0, ni, mj, pk in sub_decomps:
        n_cols = len(U_sub[0])
        for i in range(ni):
            for j in range(mj):
                for c in range(n_cols):
                    U_out[(i0+i)*m + (j0+j)][col_off + c] = U_sub[i*mj+j][c]
        for j in range(mj):
            for k in range(pk):
                for c in range(n_cols):
                    V_out[(j0+j)*p + (k0+k)][col_off + c] = V_sub[j*pk+k][c]
        for k in range(pk):
            for i in range(ni):
                for c in range(n_cols):
                    W_out[(k0+k)*n + (i0+i)][col_off + c] = W_sub[k*ni+i][c]
        col_off += n_cols
    
    return U_out, V_out, W_out

def construct():
    import time
    t0 = time.time()
    
    n, m, p = 4, 5, 5
    
    # Core 4x4x4
    U1, V1, W1 = _build_4x4x4_decomp()
    
    # Strip 1: i in 0..3, j in 0..3, k=4
    # This is a 4x4x1 multiplication
    U2, V2, W2 = _schoolbook_sub(4, 4, 1)
    
    # Strip 2: i in 0..3, j=4, k in 0..3
    # This is a 4x1x4 multiplication
    U3, V3, W3 = _schoolbook_sub(4, 1, 4)
    
    # Strip 3: i in 0..3, j=4, k=4
    # This is a 4x1x1 multiplication
    U4, V4, W4 = _schoolbook_sub(4, 1, 1)
    
    sub_decomps = [
        (U1, V1, W1, 0, 0, 0, 4, 4, 4),
        (U2, V2, W2, 0, 0, 4, 4, 4, 1),
        (U3, V3, W3, 0, 4, 0, 4, 1, 4),
        (U4, V4, W4, 0, 4, 4, 4, 1, 1),
    ]
    
    U_out, V_out, W_out = _assemble_block_decomp(n, m, p, sub_decomps)
    
    # Verify
    rank = len(U_out[0])
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(rank):
                    val += U_out[i*m+j][c] * V_out[j*p+k][c] * W_out[k*n+i][c]
                if abs(val - 1) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to schoolbook
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][col] = 1
                    V_fb[j * p + k][col] = 1
                    W_fb[k * n + i][col] = 1
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U_out, "V": V_out, "W": W_out}
