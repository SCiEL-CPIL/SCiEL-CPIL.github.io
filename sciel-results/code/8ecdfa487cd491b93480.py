"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 16, 20, 20
    
    # Rank 72: Strassen 4x4x4 (rank 56) + Schoolbook 4x4x1 (rank 16)
    rank_s = 56
    rank_sb = 16
    total_rank = rank_s + rank_sb
    
    U = np.zeros((nm, total_rank), dtype=float)
    V = np.zeros((mp, total_rank), dtype=float)
    W = np.zeros((pn, total_rank), dtype=float)
    
    # --- Schoolbook part: columns 0..3 of B (4x4 * 4x4 = 4x4) ---
    # Use Strassen for this part
    # Block structure: 2x2 blocks of 2x2
    # A blocks: A00(rows0-1,cols0-1), A01(rows0-1,cols2-3), A10(rows2-3,cols0-1), A11(rows2-3,cols2-3)
    # B blocks: B00(rows0-1,cols0-1), B01(rows0-1,cols2-3), B10(rows2-3,cols0-1), B11(rows2-3,cols2-3)
    
    # Precompute block index mappings
    # For A: element at global (i,j) -> block (i//2, j//2), local (i%2, j%2)
    # a_idx[bi][u][v] = global index in A (i*m+j)
    # b_idx[bi][u][v] = global index in B (j*p+k) where j=row, k=col
    # c_idx[bi][u][v] = global index in C output (k*n+i) where i=row, k=col
    
    def a_idx(bi, u, v):
        i = (bi // 2) * 2 + u
        j = (bi % 2) * 2 + v
        return i * m + j
    
    def b_idx(bi, u, v):
        j = (bi // 2) * 2 + u  # row of B
        k = (bi % 2) * 2 + v  # col of B
        return j * p + k
    
    def c_idx(bi, u, v):
        i = (bi // 2) * 2 + u  # row of C
        k = (bi % 2) * 2 + v  # col of C
        return k * n + i
    
    # Strassen 4x4x4: 7 products, each 2x2x2 schoolbook = 8 terms each
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    
    # Output:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Define for each Mk: list of (block_idx, coeff) for A side and B side
    # A_side[mk] = [(bi, coeff), ...] meaning A' = sum coeff * A_bi
    # B_side[mk] = [(bi, coeff), ...] meaning B' = sum coeff * B_bi
    # out_coeff[mk] = {bi: coeff} for output combination
    
    A_side = [
        [(0, 1.0), (3, 1.0)],   # M1: A00+A11
        [(2, 1.0), (3, 1.0)],   # M2: A10+A11
        [(0, 1.0)],             # M3: A00
        [(3, 1.0)],             # M4: A11
        [(0, 1.0), (1, 1.0)],   # M5: A00+A01
        [(2, 1.0), (0, -1.0)],  # M6: A10-A00
        [(1, 1.0), (3, -1.0)],  # M7: A01-A11
    ]
    
    B_side = [
        [(0, 1.0), (3, 1.0)],   # M1: B00+B11
        [(0, 1.0)],             # M2: B00
        [(1, 1.0), (3, -1.0)],  # M3: B01-B11
        [(2, 1.0), (0, -1.0)],  # M4: B10-B00
        [(3, 1.0)],             # M5: B11
        [(0, 1.0), (1, 1.0)],   # M6: B00+B01
        [(2, 1.0), (3, 1.0)],   # M7: B10+B11
    ]
    
    out_coeff = [
        {0: 1.0, 3: 1.0},       # M1 -> C00, C11
        {2: 1.0, 3: -1.0},      # M2 -> C10, -C11
        {1: 1.0, 3: 1.0},       # M3 -> C01, C11
        {0: 1.0, 2: 1.0},       # M4 -> C00, C10
        {0: -1.0, 1: 1.0},      # M5 -> -C00, C01
        {3: 1.0},               # M6 -> C11
        {0: 1.0},               # M7 -> C00
    ]
    
    for mk in range(7):
        # For each of 8 schoolbook terms (u, w, v) for 2x2x2
        for u in range(2):
            for w in range(2):
                for v in range(2):
                    col = mk * 8 + u * 4 + w * 2 + v
                    
                    # U: A'_{u,w} = sum of coeff * A_{bi, u, w}
                    for (bi, coeff) in A_side[mk]:
                        U[a_idx(bi, u, w), col] += coeff
                    
                    # V: B'_{w,v} = sum of coeff * B_{bi, w, v}
                    for (bi, coeff) in B_side[mk]:
                        V[b_idx(bi, w, v), col] += coeff
                    
                    # W: C'_{u,v} contributes to global C blocks
                    for (bi, coeff) in out_coeff[mk].items():
                        W[c_idx(bi, u, v), col] += coeff
    
    # --- Schoolbook part: column 4 of B (4x4 * 4x1 = 4x1) ---
    col_offset = rank_s
    for i in range(n):
        for j in range(m):
            col = col_offset + i * m + j
            U[i * m + j, col] = 1.0
            V[j * p + 4, col] = 1.0
            W[4 * n + i, col] = 1.0
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
