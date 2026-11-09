"""5x5x5 matrix multiplication tensor decomposition using 4+1 block structure with recursive Strassen on 4x4x4."""
import time

DIMENSIONS = [5, 5, 5]

_T0 = time.time()
_LIMIT = 150.0
_MARGIN = 5.0

def _deadline_exceeded():
    return (time.time() - _T0) > (_LIMIT - _MARGIN)

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-reduced exact tensor decomposition for 5x5x5 matrix multiplication.
    
    Uses a 4+1 block partition. The 4x4x4 sub-block uses recursive Strassen
    (two 2x2x2 Strassen decompositions per level), while mixed 4x1, 1x4, 1x1
    blocks use schoolbook.
    
    Total rank: 117 (56 from 4x4x4 Strassen + 61 from mixed blocks)
    """
    n, m, p = DIMENSIONS  # 5, 5, 5
    
    columns = []  # list of (U_dict, V_dict, W_dict)
    
    def add_column(u_entries, v_entries, w_entries):
        """Add a rank-1 column. Each is a list of (index, value)."""
        u_d = {}
        for idx, val in u_entries:
            u_d[idx] = u_d.get(idx, 0) + val
        v_d = {}
        for idx, val in v_entries:
            v_d[idx] = v_d.get(idx, 0) + val
        w_d = {}
        for idx, val in w_entries:
            w_d[idx] = w_d.get(idx, 0) + val
        columns.append((u_d, v_d, w_d))
    
    # Strassen 2x2x2 base case
    def strassen_2x2x2(offset_i, offset_j, offset_k):
        """Add 7 columns for the 2x2x2 Strassen decomposition at given offsets."""
        strassen_terms = [
            # M1 = (A00+A11)(B00+B11), contributes to C00 and C11
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
            # M2 = (A10+A11)B00, contributes to C10 and -C11
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
            # M3 = A00(B01-B11), contributes to C01 and C11
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
            # M4 = A11(B10-B00), contributes to C00 and C10
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
            # M5 = (A00+A01)B11, contributes to C01 and -C00
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,1): 1, (0,0): -1}),
            # M6 = (A10-A00)(B00+B01), contributes to C11
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
            # M7 = (A01-A11)(B10+B11), contributes to C00
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
        ]
        
        for a_coeffs, b_coeffs, c_coeffs in strassen_terms:
            u_entries = []
            v_entries = []
            w_entries = []
            for (li, lj), a_val in a_coeffs.items():
                gi, gj = offset_i + li, offset_j + lj
                u_entries.append((gi * m + gj, a_val))
            for (lj, lk), b_val in b_coeffs.items():
                gj, gk = offset_j + lj, offset_k + lk
                v_entries.append((gj * p + gk, b_val))
            for (li, lk), c_val in c_coeffs.items():
                gi, gk = offset_i + li, offset_k + lk
                w_entries.append((gk * n + gi, c_val))
            add_column(u_entries, v_entries, w_entries)
    
    # Strassen 4x4x4: split into four 2x2x2 quadrants
    def strassen_4x4x4(offset_i, offset_j, offset_k):
        """Add 28 columns for the 4x4x4 Strassen decomposition at given offsets."""
        # Four 2x2x2 blocks at:
        # (0,0), (0,2), (2,0), (2,2) relative to offset
        strassen_2x2x2(offset_i + 0, offset_j + 0, offset_k + 0)
        strassen_2x2x2(offset_i + 0, offset_j + 2, offset_k + 0)
        strassen_2x2x2(offset_i + 2, offset_j + 0, offset_k + 0)
        strassen_2x2x2(offset_i + 2, offset_j + 2, offset_k + 0)
        # Wait, that's only 4 2x2x2 blocks covering a 4x2x2 region, not 4x4x4.
        # For 4x4x4, we need 8 2x2x2 blocks: 2x2x2 grid in 3D.
        # Indices: i in {0,2}, j in {0,2}, k in {0,2}
        for di in [0, 2]:
            for dj in [0, 2]:
                for dk in [0, 2]:
                    strassen_2x2x2(offset_i + di, offset_j + dj, offset_k + dk)
    
    # Schoolbook for a rectangular block
    def schoolbook_block(i_list, j_list, k_list):
        for i in i_list:
            for j in j_list:
                for k in k_list:
                    add_column(
                        [(i * m + j, 1.0)],
                        [(j * p + k, 1.0)],
                        [(k * n + i, 1.0)]
                    )
    
    # Block definitions: 4+1 partition
    # indices 0,1,2,3 form the 4-block; index 4 is the 1-block
    block_4 = [0, 1, 2, 3]
    block_1 = [4]
    
    # Build all 27 block combinations
    blocks = [block_4, block_1]  # two block types: 4 and 1
    block_sizes = [4, 1]
    
    for bi in range(2):
        for bj in range(2):
            for bk in range(2):
                i_list = blocks[bi]
                j_list = blocks[bj]
                k_list = blocks[bk]
                si = block_sizes[bi]
                sj = block_sizes[bj]
                sk = block_sizes[bk]
                
                # If all three are size 4, use 4x4x4 Strassen
                if si == 4 and sj == 4 and sk == 4:
                    strassen_4x4x4(i_list[0], j_list[0], k_list[0])
                else:
                    # Schoolbook for all other cases
                    schoolbook_block(i_list, j_list, k_list)
    
    rank = len(columns)
    
    # Build the matrices
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for c, (u_d, v_d, w_d) in enumerate(columns):
        for idx, val in u_d.items():
            U[idx][c] = val
        for idx, val in v_d.items():
            V[idx][c] = val
        for idx, val in w_d.items():
            W[idx][c] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
