import time
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def _strassen2_terms(a_r0, a_r1, a_c0, a_c1, b_r0, b_r1, b_c0, b_c1, out_r0, out_r1, out_c0, out_c1, m, p, n):
    """
    Generate the 7 Strassen terms for a 2x2x2 block.
    Returns list of (u_entries, v_entries, w_entries) where each entry is (index, coeff).
    Indexing: U rows are n*m, V rows are m*p, W rows are p*n.
    W index convention: W[k*n + i, r] where k is output col (j), i is output row.
    A index: (row, col) -> row*m + col
    B index: (row, col) -> row*p + col
    W index: (col_out, row_out) -> col_out*n + row_out
    """
    terms = []
    
    # A elements: a[0,0], a[0,1], a[1,0], a[1,1]
    a00 = a_r0 * m + a_c0
    a01 = a_r0 * m + (a_c0 + 1)
    a10 = (a_r0 + 1) * m + a_c0
    a11 = (a_r0 + 1) * m + (a_c0 + 1)
    
    # B elements: b[0,0], b[0,1], b[1,0], b[1,1]
    b00 = b_r0 * p + b_c0
    b01 = b_r0 * p + (b_c0 + 1)
    b10 = (b_r0 + 1) * p + b_c0
    b11 = (b_r0 + 1) * p + (b_c0 + 1)
    
    # Output elements: out[0,0], out[0,1], out[1,0], out[1,1]
    # W index: out_col * n + out_row
    o00 = out_c0 * n + out_r0
    o01 = (out_c0 + 1) * n + out_r0
    o10 = out_c0 * n + (out_r0 + 1)
    o11 = (out_c0 + 1) * n + (out_r0 + 1)
    
    # M1 = (a00 + a11)(b00 + b11)
    terms.append(([ (a00, 1), (a11, 1) ], [ (b00, 1), (b11, 1) ], [ (o00, 1), (o11, 1) ]))
    # M2 = (a10 + a11) b00
    terms.append(([ (a10, 1), (a11, 1) ], [ (b00, 1) ], [ (o10, 1), (o11, -1) ]))
    # M3 = a00 (b01 - b11)
    terms.append(([ (a00, 1) ], [ (b01, 1), (b11, -1) ], [ (o01, 1), (o11, 1) ]))
    # M4 = a11 (b10 - b00)
    terms.append(([ (a11, 1) ], [ (b10, 1), (b00, -1) ], [ (o00, 1), (o10, 1) ]))
    # M5 = (a00 + a01) b11
    terms.append(([ (a00, 1), (a01, 1) ], [ (b11, 1) ], [ (o00, -1), (o01, 1) ]))
    # M6 = (a10 - a00) (b00 + b01)
    terms.append(([ (a10, 1), (a00, -1) ], [ (b00, 1), (b01, 1) ], [ (o11, 1) ]))
    # M7 = (a01 - a11) (b10 + b11)
    terms.append(([ (a01, 1), (a11, -1) ], [ (b10, 1), (b11, 1) ], [ (o00, 1) ]))
    
    return terms

def _schoolbook_block(a_r0, a_c0, b_r0, b_c0, nr, nc, nc2, out_r0, out_c0, m, p, n):
    """
    Generate schoolbook terms for A[a_r0:a_r0+nr, a_c0:a_c0+nc] * B[b_r0:b_r0+nc, b_c0:b_c0+nc2]
    -> Output[out_r0:out_r0+nr, out_c0:out_c0+nc2]
    Each term is rank-1: one U entry, one V entry, one W entry.
    """
    terms = []
    for i in range(nr):
        for k in range(nc):
            for j in range(nc2):
                u_idx = (a_r0 + i) * m + (a_c0 + k)
                v_idx = (b_r0 + k) * p + (b_c0 + j)
                w_idx = (out_c0 + j) * n + (out_r0 + i)
                terms.append(([ (u_idx, 1) ], [ (v_idx, 1) ], [ (w_idx, 1) ]))
    return terms

def _build_decomposition():
    """
    Build a decomposition for 2x5 * 5x5.
    Partition inner dimension 5 = 2+2+1.
    For the 2x2 blocks, use Strassen (rank 7 instead of 4).
    For the 2x1 blocks, use schoolbook.
    """
    n, m, p = 2, 5, 5
    all_terms = []
    
    # A is 2x5, B is 5x5, output is 2x5
    # Inner dimension partition: 0:2, 2:4, 4:5
    
    # Block (A rows 0:2, A cols 0:2) * (B rows 0:2, B cols 0:2) -> Output rows 0:2, cols 0:2
    all_terms.extend(_strassen2_terms(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, m, p, n))
    
    # Block (A rows 0:2, A cols 0:2) * (B rows 0:2, B cols 2:4) -> Output rows 0:2, cols 2:4
    all_terms.extend(_strassen2_terms(0, 1, 0, 1, 0, 1, 2, 3, 0, 1, 2, 3, m, p, n))
    
    # Block (A rows 0:2, A cols 0:2) * (B rows 0:2, B cols 4:5) -> Output rows 0:2, cols 4:5
    all_terms.extend(_schoolbook_block(0, 0, 0, 4, 2, 2, 1, 0, 4, m, p, n))
    
    # Block (A rows 0:2, A cols 2:4) * (B rows 2:4, B cols 0:2) -> Output rows 0:2, cols 0:2
    all_terms.extend(_strassen2_terms(0, 1, 2, 3, 2, 3, 0, 1, 0, 1, 0, 1, m, p, n))
    
    # Block (A rows 0:2, A cols 2:4) * (B rows 2:4, B cols 2:4) -> Output rows 0:2, cols 2:4
    all_terms.extend(_strassen2_terms(0, 1, 2, 3, 2, 3, 2, 3, 0, 1, 2, 3, m, p, n))
    
    # Block (A rows 0:2, A cols 2:4) * (B rows 2:4, B cols 4:5) -> Output rows 0:2, cols 4:5
    all_terms.extend(_schoolbook_block(0, 2, 2, 4, 2, 2, 1, 0, 4, m, p, n))
    
    # Block (A rows 0:2, A cols 4:5) * (B rows 4:5, B cols 0:5) -> Output rows 0:2, cols 0:5
    all_terms.extend(_schoolbook_block(0, 4, 4, 0, 2, 1, 5, 0, 0, m, p, n))
    
    total_rank = len(all_terms)
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    for col, (u_entries, v_entries, w_entries) in enumerate(all_terms):
        for idx, val in u_entries:
            U[idx][col] = val
        for idx, val in v_entries:
            V[idx][col] = val
        for idx, val in w_entries:
            W[idx][col] = val
    
    return U, V, W, total_rank

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying
    a 2 by 5 matrix by a 5 by 5 matrix.
    """
    start_time = time.time()
    deadline = start_time + 110.0
    
    U, V, W, total_rank = _build_decomposition()
    
    if time.time() > deadline:
        return {"U": U, "V": W, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
