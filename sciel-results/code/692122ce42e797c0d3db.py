"""5x5x5 matrix multiplication tensor decomposition using 2+2+1 block structure with Strassen on all 2x2x2 blocks."""
import time
import numpy as np

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
    
    Uses a 2+2+1 block partition. For each 2x2x2 sub-block product,
    uses Strassen's rank-7 decomposition instead of schoolbook rank 8.
    
    Blocks:
    - 2x2x2 blocks (8 of them): Strassen rank 7 each = 56
    - 2x1x2, 2x2x1, 1x2x2, 1x2x1, 2x1x1, 1x2x1, 1x1x2, 1x1x1 blocks: schoolbook
    - 2x2x1 (4 blocks): 4 each = 16
    - 2x1x2 (4 blocks): 4 each = 16
    - 1x2x2 (4 blocks): 4 each = 16
    - 1x2x1 (4 blocks): 2 each = 8
    - 2x1x1 (4 blocks): 2 each = 8
    - 1x1x2 (4 blocks): 2 each = 8
    - 1x1x1 (1 block): 1
    - 2x2x1 already counted... let me recount
    
    Actually let me enumerate all 3x3x3 = 27 block products:
    Block sizes for i, j, k dimensions are each in {2,2,1}
    
    (2,2,2): 8 blocks × 7 (Strassen) = 56
    (2,2,1): 4 blocks × 4 = 16
    (2,1,2): 4 blocks × 4 = 16
    (1,2,2): 4 blocks × 4 = 16
    (2,1,1): 4 blocks × 2 = 8
    (1,2,1): 4 blocks × 2 = 8
    (1,1,2): 4 blocks × 2 = 8
    (1,1,1): 1 block × 1 = 1
    
    Total: 56 + 16 + 16 + 16 + 8 + 8 + 8 + 1 = 129... that's worse.
    
    Wait, I need to recount. The 2+2+1 partition means:
    - i-blocks: [0,1], [2,3], [4]
    - j-blocks: [0,1], [2,3], [4]  
    - k-blocks: [0,1], [2,3], [4]
    
    For a (a,b,c) block product, schoolbook rank = a*b*c.
    For a (2,2,2) block, Strassen rank = 7 instead of 8.
    
    Count of (2,2,2) blocks: 2×2×2 = 8 blocks, saving 1 each = 8 saved
    Schoolbook total: 5×5×5 = 125
    With Strassen on 2x2x2: 125 - 8 = 117
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
    
    # Block index groups
    # Group 0: indices [0,1]
    # Group 1: indices [2,3]
    # Group 2: index [4]
    blocks = [[0, 1], [2, 3], [4]]
    
    # Schoolbook for a block product
    def schoolbook_block(i_list, j_list, k_list):
        for i in i_list:
            for j in j_list:
                for k in k_list:
                    add_column(
                        [(i * m + j, 1.0)],
                        [(j * p + k, 1.0)],
                        [(k * n + i, 1.0)]
                    )
    
    # Strassen for 2x2x2 block, with given base offsets
    def strassen_2x2x2(base_i, base_j, base_k):
        """Add 7 columns for the 2x2x2 Strassen decomposition at given base offsets."""
        # Local indices: 0,1 map to global base_i+0, base_i+1, etc.
        # A[i,j] -> global A[base_i+i, base_j+j], U index = (base_i+i)*5 + (base_j+j)
        # B[j,k] -> global B[base_j+j, base_k+k], V index = (base_j+j)*5 + (base_k+k)
        # C[i,k] -> global C[base_i+i, base_k+k], W index = (base_k+k)*5 + (base_i+i)
        
        # Strassen terms: (A_coeffs, B_coeffs, C_coeffs)
        strassen_terms = [
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,1): 1, (0,0): -1}),
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
        ]
        
        for a_coeffs, b_coeffs, c_coeffs in strassen_terms:
            u_entries = []
            v_entries = []
            w_entries = []
            for (li, lj), a_val in a_coeffs.items():
                gi, gj = base_i + li, base_j + lj
                u_entries.append((gi * m + gj, a_val))
            for (lj, lk), b_val in b_coeffs.items():
                gj, gk = base_j + lj, base_k + lk
                v_entries.append((gj * p + gk, b_val))
            for (li, lk), c_val in c_coeffs.items():
                gi, gk = base_i + li, base_k + lk
                w_entries.append((gk * n + gi, c_val))
            add_column(u_entries, v_entries, w_entries)
    
    # Build all 27 block products
    # Block group bases: group 0 starts at 0, group 1 starts at 2, group 2 starts at 4
    bases = [0, 2, 4]
    sizes = [2, 2, 1]
    
    for gi in range(3):
        for gj in range(3):
            for gk in range(3):
                i_list = blocks[gi]
                j_list = blocks[gj]
                k_list = blocks[gk]
                
                # If all three blocks have size 2, use Strassen
                if sizes[gi] == 2 and sizes[gj] == 2 and sizes[gk] == 2:
                    strassen_2x2x2(bases[gi], bases[gj], bases[gk])
                else:
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
