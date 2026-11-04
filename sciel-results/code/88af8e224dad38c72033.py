"""Rewritten program for 2x5x5 matrix multiplication tensor rank minimization."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def _winograd_5x5():
    """
    Winograd-style 5x5 matrix multiplication using 21 scalar multiplications.
    
    Decompose 5 = 2+2+1. Use Strassen (rank 7) for 2x2*2x2 blocks,
    and schoolbook (rank 4) for the remaining cross terms.
    
    Partition indices: {0,1}, {2,3}, {4}
    
    Block structure:
    - C[0:2, 0:2] = A[0:2]*B[0:2]  -> Strassen (7 mults)
    - C[0:2, 2:4] = A[0:2]*B[2:4]  -> Strassen (7 mults)  
    - C[0:2, 4:5] = A[0:2]*B[4:5]  -> schoolbook (2 mults)
    - C[2:4, 0:2] = A[2:4]*B[0:2]  -> Strassen (7 mults)
    - C[2:4, 2:4] = A[2:4]*B[2:4]  -> Strassen (7 mults)
    - C[2:4, 4:5] = A[2:4]*B[4:5]  -> schoolbook (2 mults)
    - C[4:5, 0:2] = A[4:5]*B[0:2]  -> schoolbook (2 mults)
    - C[4:5, 2:4] = A[4:5]*B[2:4]  -> schoolbook (2 mults)
    - C[4:5, 4:5] = A[4:5]*B[4:5]  -> schoolbook (1 mult)
    
    Total: 7+7+2+7+7+2+2+2+1 = 37 scalar multiplications
    """
    # We'll build the U, V, W matrices for a single 5x5 matmul
    # U has shape (5, rank), V has shape (5, rank), W has shape (5, rank)
    # For the full 2x5x5: U_full has shape (10, rank_total), etc.
    
    # Strassen 2x2: C = A*B where A, B are 2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B10-B11)
    # M4 = A11*(B01-B10)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B10)
    # M7 = (A01-A11)*(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # For each Strassen block, we need 7 "multiplication slots"
    # Each slot has a U coefficient (for A), V coefficient (for B), and W coefficients (for C)
    
    # Let's define the decomposition as a list of (u_vec, v_vec, w_vec) tuples
    # where u_vec is a dict mapping (i,j) -> coeff for A[i,j]
    # v_vec is a dict mapping (j,k) -> coeff for B[j,k]  
    # w_vec is a dict mapping (i,k) -> coeff for C[i,k]
    
    decomp = []
    
    def add_strassen(a_rows, b_cols, c_rows, c_cols):
        """Add Strassen decomposition for A[a_rows]*B[b_cols] contributing to C[c_rows, c_cols]"""
        # a_rows = [r0, r1], b_cols = [c0, c1], c_rows = [cr0, cr1], c_cols = [cc0, cc1]
        r0, r1 = a_rows
        c0, c1 = b_cols
        cr0, cr1 = c_rows
        cc0, cc1 = c_cols
        
        # M1 = (A[r0,c0]+A[r1,c1]) * (B[c0,c0]+B[c1,c1])
        decomp.append({
            'u': {(r0,c0): 1, (r1,c1): 1},
            'v': {(c0,c0): 1, (c1,c1): 1},
            'w': {(cr0,cc0): 1, (cr1,cc1): 1}
        })
        # M2 = (A[r1,c0]+A[r1,c1]) * B[c0,c0]
        decomp.append({
            'u': {(r1,c0): 1, (r1,c1): 1},
            'v': {(c0,c0): 1},
            'w': {(cr1,cc0): 1, (cr1,cc1): -1}
        })
        # M3 = A[r0,c0] * (B[c0,c1]-B[c1,c1])
        decomp.append({
            'u': {(r0,c0): 1},
            'v': {(c0,c1): 1, (c1,c1): -1},
            'w': {(cr0,cc1): 1, (cr1,cc1): 1}
        })
        # M4 = A[r1,c1] * (B[c0,c1]-B[c0,c0])
        decomp.append({
            'u': {(r1,c1): 1},
            'v': {(c0,c1): 1, (c0,c0): -1},
            'w': {(cr0,cc0): 1, (cr1,cc0): 1}
        })
        # M5 = (A[r0,c0]+A[r0,c1]) * B[c1,c1]
        decomp.append({
            'u': {(r0,c0): 1, (r0,c1): 1},
            'v': {(c1,c1): 1},
            'w': {(cr0,cc0): -1, (cr0,cc1): 1}
        })
        # M6 = (A[r1,c0]-A[r0,c0]) * (B[c0,c0]+B[c0,c1])
        decomp.append({
            'u': {(r1,c0): 1, (r0,c0): -1},
            'v': {(c0,c0): 1, (c0,c1): 1},
            'w': {(cr1,cc1): 1}
        })
        # M7 = (A[r0,c1]-A[r1,c1]) * (B[c1,c0]+B[c1,c1])
        decomp.append({
            'u': {(r0,c1): 1, (r1,c1): -1},
            'v': {(c1,c0): 1, (c1,c1): 1},
            'w': {(cr0,cc0): 1}
        })
    
    def add_schoolbook_2x1(a_rows, b_col, c_rows, c_col):
        """Add schoolbook for A[a_rows]*B[:,b_col] -> C[c_rows, c_col] (2 mults)"""
        for r in a_rows:
            for cr in c_rows:
                decomp.append({
                    'u': {(r, b_col): 1},
                    'v': {(b_col, b_col): 1},
                    'w': {(cr, c_col): 1}
                })
    
    def add_schoolbook_1x2(a_row, b_cols, c_row, c_cols):
        """Add schoolbook for A[a_row,:]*B[b_cols,:] -> C[c_row, c_cols] (2 mults)"""
        for c in b_cols:
            for cc in c_cols:
                decomp.append({
                    'u': {(a_row, c): 1},
                    'v': {(c, c): 1},
                    'w': {(c_row, cc): 1}
                })
    
    def add_schoolbook_1x1(a_row, b_col, c_row, c_col):
        """Add schoolbook for single element (1 mult)"""
        decomp.append({
            'u': {(a_row, b_col): 1},
            'v': {(b_col, b_col): 1},
            'w': {(c_row, c_col): 1}
        })
    
    # Block [0:2, 0:2] = A[0:2] * B[0:2]
    add_strassen([0,1], [0,1], [0,1], [0,1])
    # Block [0:2, 2:4] = A[0:2] * B[2:4]
    add_strassen([0,1], [2,3], [0,1], [2,3])
    # Block [0:2, 4:5] = A[0:2] * B[:,4]
    add_schoolbook_2x1([0,1], 4, [0,1], 4)
    # Block [2:4, 0:2] = A[2:4] * B[0:2]
    add_strassen([2,3], [0,1], [2,3], [0,1])
    # Block [2:4, 2:4] = A[2:4] * B[2:4]
    add_strassen([2,3], [2,3], [2,3], [2,3])
    # Block [2:4, 4:5] = A[2:4] * B[:,4]
    add_schoolbook_2x1([2,3], 4, [2,3], 4)
    # Block [4:5, 0:2] = A[4,:] * B[0:2]
    add_schoolbook_1x2(4, [0,1], 4, [0,1])
    # Block [4:5, 2:4] = A[4,:] * B[2:4]
    add_schoolbook_1x2(4, [2,3], 4, [2,3])
    # Block [4:5, 4:5] = A[4,:] * B[:,4]
    add_schoolbook_1x1(4, 4, 4, 4)
    
    return decomp

def _build_full_decomposition():
    """
    Build the full 2x5x5 decomposition using Strassen on 2x2 subblocks.
    
    Partition 5 = 2+2+1. Use Strassen (7 mults) for 2x2*2x2 blocks.
    
    For the 5x5 matmul (per row of A):
    - 4 corner 2x2 blocks: 4 * 7 = 28 mults (Strassen)
    - Edge blocks involving index 4: handled with schoolbook
    - Center block [2:4,2:4]: 7 mults (Strassen) -- wait, this is one of the 4 corners
    
    Let me re-index. Blocks:
    Rows: R0={0,1}, R1={2,3}, R2={4}
    Cols: C0={0,1}, C1={2,3}, C2={4}
    
    Block (R0,C0): Strassen, 7 mults
    Block (R0,C1): Strassen, 7 mults
    Block (R0,C2): A[0:2]*B[:,4] -> 2 mults (each row of A[0:2] times B[:,4])
    Block (R1,C0): Strassen, 7 mults
    Block (R1,C1): Strassen, 7 mults
    Block (R1,C2): A[2:4]*B[:,4] -> 2 mults
    Block (R2,C0): A[4,:]*B[0:2] -> 2 mults
    Block (R2,C1): A[4,:]*B[2:4] -> 2 mults
    Block (R2,C2): A[4,4]*B[4,4] -> 1 mult
    
    Total: 7+7+2+7+7+2+2+2+1 = 37 mults per 5x5 row.
    For 2 rows: 74 mults. This is worse than schoolbook (50).
    
    This approach doesn't work. Let me use schoolbook.
    """
    n, m, p = 2, 5, 5
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

def construct():
    return _build_full_decomposition()
# EVOLVE-BLOCK-END
