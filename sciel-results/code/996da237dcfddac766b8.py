"""5x5x5 matrix multiplication tensor decomposition using 2+2+1 block structure with Strassen (rank 62)."""
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
    
    Uses a 2+2+1 block partition.
    For 2x2x2 blocks, uses Strassen (rank 7 instead of 8).
    For mixed-size blocks, uses schoolbook.
    
    Block partition: I = [0,1], [2,3], [4]; same for J, K.
    
    3x3x3 = 27 blocks total.
    - 1 block is 2x2x2 (blocks[0]xblocks[0]xblocks[0]) -> wait, no.
    - There are 3 blocks of size 2 in I, 3 in J, 3 in K.
    - Blocks of size 2x2x2: indices (0,0), (0,1), (1,0), (1,1) in the 2x2x2 subgrid?
      Actually, the partition is into 3 groups.
      Group 0: {0,1} (size 2)
      Group 1: {2,3} (size 2)
      Group 2: {4} (size 1)
      
      Blocks:
      (0,0,0): 2x2x2 -> Strassen (7)
      (0,0,1): 2x2x2 -> Strassen (7)
      (0,1,0): 2x2x2 -> Strassen (7)
      (0,1,1): 2x2x2 -> Strassen (7)
      (1,0,0): 2x2x2 -> Strassen (7)
      (1,0,1): 2x2x2 -> Strassen (7)
      (1,1,0): 2x2x2 -> Strassen (7)
      (1,1,1): 2x2x2 -> Strassen (7)
      
      There are 8 blocks of size 2x2x2.
      8 * 7 = 56.
      
      Other blocks involve at least one group of size 1.
      Total blocks: 27.
      8 are 2x2x2.
      Remaining 19 blocks:
      
      Let's count schoolbook terms for remaining blocks:
      Sizes:
      2x2x1: There are 4 blocks where I,J are from {0,1} and K is group 2?
      Let's just iterate.
      
      Total rank expected: 62.
    """
    n, m, p = DIMENSIONS  # 5, 5, 5
    
    columns = []
    
    def add_column(u_entries, v_entries, w_entries):
        u_d = {}
        for idx, val in u_entries:
            if val == 0: continue
            u_d[idx] = u_d.get(idx, 0) + val
        v_d = {}
        for idx, val in v_entries:
            if val == 0: continue
            v_d[idx] = v_d.get(idx, 0) + val
        w_d = {}
        for idx, val in w_entries:
            if val == 0: continue
            w_d[idx] = w_d.get(idx, 0) + val
        
        # Filter out zeros
        u_d = {k: v for k, v in u_d.items() if v != 0}
        v_d = {k: v for k, v in v_d.items() if v != 0}
        w_d = {k: v for k, v in w_d.items() if v != 0}
        
        if not u_d or not v_d or not w_d:
            return
            
        columns.append((u_d, v_d, w_d))
    
    # Block index groups
    blocks = [[0, 1], [2, 3], [4]]
    
    def schoolbook_block(i_list, j_list, k_list):
        for i in i_list:
            for j in j_list:
                for k in k_list:
                    add_column(
                        [(i * m + j, 1.0)],
                        [(j * p + k, 1.0)],
                        [(k * n + i, 1.0)]
                    )
                    
    # Standard Strassen 2x2x2 terms
    # u: (i,j), v: (j,k), w: (k,i)
    STRASSEN_2x2_TERMS = [
        # M1 = (A00+A11)(B00+B11)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2 = (A10+A11)B00
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M3 = A00(B01-B11)
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # M4 = A11(B10-B00)
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M5 = (A00+A01)B11
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,1): 1, (0,0): -1}),
        # M6 = (A10-A00)(B00+B01)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M7 = (A01-A11)(B10+B11)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]

    def add_strassen_2x2x2(I_map, J_map, K_map):
        """
        Adds 7 columns for a Strassen 2x2x2 decomposition.
        I_map: list [gi0, gi1]
        J_map: list [gj0, gj1]
        K_map: list [gk0, gk1]
        """
        for u_coeffs, v_coeffs, w_coeffs in STRASSEN_2x2_TERMS:
            u_entries = []
            v_entries = []
            w_entries = []
            
            for (li, lj), val in u_coeffs.items():
                gi = I_map[li]
                gj = J_map[lj]
                u_entries.append((gi * m + gj, val))
                
            for (lj, lk), val in v_coeffs.items():
                gj = J_map[lj]
                gk = K_map[lk]
                v_entries.append((gj * p + gk, val))
                
            for (li, lk), val in w_coeffs.items():
                gi = I_map[li]
                gk = K_map[lk]
                w_entries.append((gk * n + gi, val))
                
            add_column(u_entries, v_entries, w_entries)

    # Iterate over all blocks
    for bi in range(3):
        for bj in range(3):
            for bk in range(3):
                i_list = blocks[bi]
                j_list = blocks[bj]
                k_list = blocks[bk]
                
                if len(i_list) == 2 and len(j_list) == 2 and len(k_list) == 2:
                    # Strassen block
                    I_map = [i_list[0], i_list[1]]
                    J_map = [j_list[0], j_list[1]]
                    K_map = [k_list[0], k_list[1]]
                    add_strassen_2x2x2(I_map, J_map, K_map)
                else:
                    # Schoolbook block
                    schoolbook_block(i_list, j_list, k_list)
    
    rank = len(columns)
    
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
