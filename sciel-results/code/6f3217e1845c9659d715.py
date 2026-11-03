import time
import numpy as np

_START = time.time()
_DEADLINE = 140.0

def _check_deadline():
    if time.time() > _DEADLINE:
        raise TimeoutError("deadline")

def _strassen_222():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition.
    U: 4x7, V: 4x7, W: 4x7
    Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    # Strassen 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B00)
    # M4 = A11(B01-B11)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Tensors are defined by (i, j, k) -> (u_idx, v_idx, w_idx)
    # U[i*2+j], V[j*2+k], W[k*2+i]
    
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # M1
    # (A00+A11)(B00+B11)
    # A00: i=0,j=0. A11: i=1,j=1.
    # B00: j=0,k=0. B11: j=1,k=1.
    # Products:
    # A00*B00 -> i=0, j=0, k=0. C00 += 1
    # A00*B11 -> i=0, j=1, k=1. C01 += 1
    # A11*B00 -> i=1, j=0, k=0. C10 += 1
    # A11*B11 -> i=1, j=1, k=1. C11 += 1
    
    # U col 0: A00 (1), A11 (1). Others 0.
    U[0, 0] = 1
    U[3, 0] = 1
    # V col 0: B00 (1), B11 (1).
    V[0, 0] = 1
    V[3, 0] = 1
    # W col 0: C00 (1), C01 (1), C10 (1), C11 (1).
    W[0, 0] = 1 # k=0, i=0 -> W[0*2+0]
    W[1, 0] = 1 # k=1, i=0 -> W[1*2+0]
    W[2, 0] = 1 # k=0, i=1 -> W[0*2+1]
    W[3, 0] = 1 # k=1, i=1 -> W[1*2+1]
    
    # M2: (A10+A11)B00
    # A10: i=1, j=0. A11: i=1, j=1.
    # B00: j=0, k=0.
    # Products:
    # A10*B00 -> i=1, j=0, k=0. C10 += 1
    # A11*B00 -> i=1, j=0, k=0. C10 += 1
    # Wait, A11 is i=1, j=1. B00 is j=0, k=0.
    # A11*B00 is not a direct term in C.
    # Strassen M2 = (A10+A11)B00.
    # This computes a scalar.
    # How does it map to C?
    # C10 = M2 - M3 + M4 - M6? No.
    # Standard Strassen reconstruction:
    # C00 = M1 - M2 + M4 - M5
    # C01 = M2 + M5
    # C10 = M3 + M6
    # C11 = M1 - M3 + M4 + M7
    
    # Let's use the standard coefficients.
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B00)
    # M4 = A11(B01-B11)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # We need to fill U, V, W for each M_r.
    # U_r corresponds to the A part.
    # V_r corresponds to the B part.
    # W_r corresponds to the C part (coefficients in reconstruction).
    
    # M1
    U[0, 0] = 1; U[3, 0] = 1 # A00, A11
    V[0, 0] = 1; V[3, 0] = 1 # B00, B11
    # C coeffs:
    # C00: +1 (from M1)
    # C01: 0
    # C10: 0
    # C11: +1 (from M1)
    W[0, 0] = 1 # C00
    W[3, 0] = 1 # C11
    
    # M2
    U[2, 1] = 1; U[3, 1] = 1 # A10, A11
    V[0, 1] = 1 # B00
    # C coeffs:
    # C00: -1
    # C01: +1
    # C10: 0
    # C11: 0
    W[0, 1] = -1 # C00
    W[1, 1] = 1 # C01
    
    # M3
    U[0, 2] = 1 # A00
    V[2, 2] = 1; V[0, 2] = -1 # B10, -B00
    # C coeffs:
    # C00: 0
    # C01: 0
    # C10: +1
    # C11: -1
    W[2, 2] = 1 # C10
    W[3, 2] = -1 # C11
    
    # M4
    U[3, 3] = 1 # A11
    V[1, 3] = 1; V[3, 3] = -1 # B01, -B11
    # C coeffs:
    # C00: +1
    # C01: 0
    # C10: 0
    # C11: +1
    W[0, 3] = 1 # C00
    W[3, 3] = 1 # C11
    
    # M5
    U[0, 4] = 1; U[1, 4] = 1 # A00, A01
    V[3, 4] = 1 # B11
    # C coeffs:
    # C00: -1
    # C01: +1
    # C10: 0
    # C11: 0
    W[0, 4] = -1 # C00
    W[1, 4] = 1 # C01
    
    # M6
    U[2, 5] = 1; U[0, 5] = -1 # A10, -A00
    V[0, 5] = 1; V[1, 5] = 1 # B00, B01
    # C coeffs:
    # C00: 0
    # C01: 0
    # C10: +1
    # C11: 0
    W[2, 5] = 1 # C10
    
    # M7
    U[1, 6] = 1; U[3, 6] = -1 # A01, -A11
    V[2, 6] = 1; V[3, 6] = 1 # B10, B11
    # C coeffs:
    # C00: 0
    # C01: 0
    # C10: 0
    # C11: +1
    W[3, 6] = 1 # C11
    
    return U, V, W

def _schoolbook(n, m, p):
    """
    Returns U, V, W for n x m x p schoolbook decomposition.
    U: (n*m) x (n*m*p)
    V: (m*p) x (n*m*p)
    W: (p*n) x (n*m*p)
    """
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _combine_blocks(blocks):
    """
    blocks: list of tuples (u, v, w, u_offset, v_offset, w_offset)
    u: (n_sub * m_sub) x rank_sub
    v: (m_sub * p_sub) x rank_sub
    w: (p_sub * n_sub) x rank_sub
    """
    total_rank = sum(b[0].shape[1] for b in blocks)
    n_total = 3 * 5
    m_total = 5 * 5
    p_total = 5 * 3
    
    U = np.zeros((n_total, total_rank), dtype=int)
    V = np.zeros((m_total, total_rank), dtype=int)
    W = np.zeros((p_total, total_rank), dtype=int)
    
    rank_offset = 0
    for u, v, w, u_off, v_off, w_off in blocks:
        r = u.shape[1]
        U[u_off:u_off+u.shape[0], rank_offset:rank_offset+r] = u
        V[v_off:v_off+v.shape[0], rank_offset:rank_offset+r] = v
        W[w_off:w_off+w.shape[0], rank_offset:rank_offset+r] = w
        rank_offset += r
        
    return U, V, W

def construct():
    n, m, p = 3, 5, 5
    
    # Split A (3x5) into A1 (3x2) and A2 (3x3)
    # Split B (5x5) into B1 (2x5) and B2 (3x5)
    
    # Block 1: A1 * B1 (3x2x5)
    # Split B1 (2x5) into B11 (2x2) and B12 (2x3)
    # Term 1.1: A1 * B11 (3x2x2) -> Rank 11
    # Term 1.2: A1 * B12 (3x2x3) -> Rank 17
    
    # Block 2: A2 * B2 (3x3x5)
    # Split B2 (3x5) into B21 (3x2) and B22 (3x3)
    # Term 2.1: A2 * B21 (3x3x2) -> Rank 17
    # Term 2.2: A2 * B22 (3x3x3) -> Rank 27 (Schoolbook)
    
    blocks = []
    
    # --- Term 1.1: A1 (3x2) * B11 (2x2) ---
    # Split A1 into A1a (2x2) and A1b (1x2)
    # A1a * B11 (2x2x2) -> Strassen (Rank 7)
    # A1b * B11 (1x2x2) -> Schoolbook (Rank 4)
    
    u11, v11, w11 = _strassen_222()
    # A1a is rows 0,1 of A1 (global rows 0,1), cols 0,1 (global cols 0,1)
    # B11 is rows 0,1 of B1 (global rows 0,1), cols 0,1 (global cols 0,1)
    # C part is rows 0,1 of C, cols 0,1 of C
    # Offsets:
    # u_off: 0 (A rows 0,1, cols 0,1)
    # v_off: 0 (B rows 0,1, cols 0,1)
    # w_off: 0 (C cols 0,1, rows 0,1)
    blocks.append((u11, v11, w11, 0, 0, 0))
    
    u11b, v11b, w11b = _schoolbook(1, 2, 2)
    # A1b is row 2 of A1 (global row 2), cols 0,1 (global cols 0,1)
    # B11 is rows 0,1 of B1 (global rows 0,1), cols 0,1 (global cols 0,1)
    # C part is row 2 of C, cols 0,1 of C
    # u_off: 2*5 + 0 = 10? No. U index is i*m + j.
    # A1b: i=2, j=0,1. Indices: 2*5+0=10, 2*5+1=11.
    # But u11b is 1x2 size. It corresponds to i=0, j=0,1 in local.
    # We need to place it at global i=2.
    # So u_off = 2*5 + 0 = 10.
    # v_off = 0.
    # w_off: C cols 0,1, row 2. W index k*n + i.
    # k=0,1. i=2.
    # W indices: 0*3+2=2, 1*3+2=5.
    # w11b is 2x2 size (p*n = 2*1=2? No, p=2, n=1. W is 2x1? No.
    # Schoolbook(1,2,2): n=1, m=2, p=2.
    # U: 1x2 x 4. V: 2x2 x 4. W: 2x1 x 4.
    # W index k*n + i. n=1. k=0,1. i=0.
    # W indices 0, 1.
    # Global W indices for i=2, k=0,1 are 2, 5.
    # So w_off = 2.
    blocks.append((u11b, v11b, w11b, 10, 0, 2))
    
    # --- Term 1.2: A1 (3x2) * B12 (2x3) ---
    # Split B12 into B12a (2x2) and B12b (2x1)
    # A1 * B12a (3x2x2) -> Rank 11
    # A1 * B12b (3x2x1) -> Rank 6
    
    # A1 * B12a (3x2x2)
    # Split A1 into A1a (2x2) and A1b (1x2)
    # A1a * B12a (2x2x2) -> Strassen (Rank 7)
    # A1b * B12a (1x2x2) -> Schoolbook (Rank 4)
    
    u12a, v12a, w12a = _strassen_222()
    # A1a: i=0,1; j=0,1. Global u_off = 0.
    # B12a: B12 is cols 2,3,4 of B1. B12a is cols 2,3.
    # B1 rows 0,1.
    # Global v_off: rows 0,1; cols 2,3.
    # V index j*p + k. p=5.
    # j=0,1. k=2,3.
    # Indices: 0*5+2=2, 0*5+3=3, 1*5+2=7, 1*5+3=8.
    # v12a is 4x7. It covers j=0,1; k=0,1 local.
    # We need to map local k=0,1 to global k=2,3.
    # This requires shifting the V matrix columns? No, V is (m*p) x rank.
    # The rows of V correspond to (j,k).
    # Local V rows: (0,0), (0,1), (1,0), (1,1).
    # Global V rows needed: (0,2), (0,3), (1,2), (1,3).
    # We can't just use an offset. We need to permute or embed.
    # _combine_blocks assumes contiguous blocks.
    # We must create a zero-padded V for this block.
    
    v12a_pad = np.zeros((25, 7), dtype=int)
    v12a_pad[2, :] = v12a[0, :]
    v12a_pad[3, :] = v12a[1, :]
    v12a_pad[7, :] = v12a[2, :]
    v12a_pad[8, :] = v12a[3, :]
    
    # W for C part: rows 0,1; cols 2,3.
    # W index k*n + i. n=3.
    # k=2,3. i=0,1.
    # Indices: 2*3+0=6, 2*3+1=7, 3*3+0=9, 3*3+1=10.
    # w12a is 4x7. Local rows: (0,0), (0,1), (1,0), (1,1) for k=0,1; i=0,1.
    # We need to map to global k=2,3.
    w12a_pad = np.zeros((15, 7), dtype=int)
    w12a_pad[6, :] = w12a[0, :]
    w12a_pad[7, :] = w12a[1, :]
    w12a_pad[9, :] = w12a[2, :]
    w12a_pad[10, :] = w12a[3, :]
    
    blocks.append((u12a, v12a_pad, w12a_pad, 0, 0, 0))
    
    u12ab, v12ab, w12ab = _schoolbook(1, 2, 2)
    # A1b: i=2; j=0,1. Global u_off = 10.
    # B12a: j=0,1; k=2,3.
    # v12ab is 4x4.
    v12ab_pad = np.zeros((25, 4), dtype=int)
    v12ab_pad[2, :] = v12ab[0, :]
    v12ab_pad[3, :] = v12ab[1, :]
    v12ab_pad[7, :] = v12ab[2, :]
    v12ab_pad[8, :] = v12ab[3, :]
    
    # W: i=2; k=2,3.
    # W indices: 6, 7, 9, 10? No.
    # n=1 local. Global n=3.
    # Local W rows: k=0,1; i=0.
    # Global W rows: k=2,3; i=2.
    # Indices: 6, 9.
    w12ab_pad = np.zeros((15, 4), dtype=int)
    w12ab_pad[6, :] = w12ab[0, :]
    w12ab_pad[9, :] = w12ab[1, :]
    
    blocks.append((u12ab, v12ab_pad, w12ab_pad, 10, 0, 0))
    
    # A1 * B12b (3x2x1) -> Schoolbook Rank 6
    u12b, v12b, w12b = _schoolbook(3, 2, 1)
    # A1: i=0,1,2; j=0,1. Global u_off = 0.
    # B12b: j=0,1; k=4.
    # v12b is 2x6.
    v12b_pad = np.zeros((25, 6), dtype=int)
    v12b_pad[4, :] = v12b[0, :]
    v12b_pad[9, :] = v12b[1, :]
    
    # W: i=0,1,2; k=4.
    # W indices: 4*3+0=12, 4*3+1=13, 4*3+2=14.
    w12b_pad = np.zeros((15, 6), dtype=int)
    w12b_pad[12, :] = w12b[0, :]
    w12b_pad[13, :] = w12b[1, :]
    w12b_pad[14, :] = w12b[2, :]
    
    blocks.append((u12b, v12b_pad, w12b_pad, 0, 0, 0))
    
    # --- Term 2.1: A2 (3x3) * B21 (3x2) ---
    # Split A2 into A2a (2x3) and A2b (1x3)
    # A2a * B21 (2x3x2) -> Rank 11
    # A2b * B21 (1x3x2) -> Rank 6
    
    # A2a * B21 (2x3x2)
    # Split B21 into B21a (3x1) and B21b (3x1)? No, B21 is 3x2.
    # Split middle dim 3? No, B21 is 3x2.
    # A2a is 2x3. B21 is 3x2.
    # Split A2a cols? No.
    # Split B21 cols?
    # Let's split the 3 in A2a rows? No, A2a is 2x3.
    # Let's use the previous logic: 2x3x2.
    # Split 3 into 2+1.
    # A2a = [A2a1 (2x2), A2a2 (2x1)].
    # B21 = [B211 (2x2), B212 (1x2)].
    # A2a1 * B211 (2x2x2) -> Strassen (7)
    # A2a2 * B212 (2x1x2) -> Schoolbook (4)
    # Wait, A2a1 * B212? No.
    # A2a * B21 = A2a1 B211 + A2a2 B212?
    # Indices: A (2x3) * B (3x2).
    # A = [A1 (2x2) A2 (2x1)].
    # B = [B1 (2x2); B2 (1x2)].
    # Product = A1 B1 + A2 B2.
    
    u21a1, v21a1, w21a1 = _strassen_222()
    # A2a1: A2 is rows 2,3,4 of A? No.
    # A2 is cols 2,3,4 of A.
    # A2a is rows 0,1 of A2 (global rows 0,1), cols 0,1 of A2 (global cols 2,3).
    # A2a1: rows 0,1; cols 0,1.
    # Global u_off: 0*5+2=2.
    # B21: B2 is rows 2,3,4 of B.
    # B21 is cols 0,1 of B2.
    # B211: rows 0,1 of B21 (global rows 2,3), cols 0,1 (global cols 0,1).
    # Global v_off: 2*5+0=10.
    # W: C rows 0,1; cols 0,1.
    # Global w_off: 0.
    
    v21a1_pad = np.zeros((25, 7), dtype=int)
    v21a1_pad[10, :] = v21a1[0, :]
    v21a1_pad[11, :] = v21a1[1, :]
    v21a1_pad[15, :] = v21a1[2, :]
    v21a1_pad[16, :] = v21a1[3, :]
    
    w21a1_pad = np.zeros((15, 7), dtype=int)
    w21a1_pad[0, :] = w21a1[0, :]
    w21a1_pad[1, :] = w21a1[1, :]
    w21a1_pad[3, :] = w21a1[2, :]
    w21a1_pad[4, :] = w21a1[3, :]
    
    blocks.append((u21a1, v21a1_pad, w21a1_pad, 2, 0, 0))
    
    u21a2, v21a2, w21a2 = _schoolbook(2, 1, 2)
    # A2a2: rows 0,1; col 2 of A2 (global col 4).
    # Global u_off: 0*5+4=4.
    # B212: row 2 of B21 (global row 4), cols 0,1.
    # Global v_off: 4*5+0=20.
    # W: C rows 0,1; cols 0,1.
    # Global w_off: 0.
    
    v21a2_pad = np.zeros((25, 4), dtype=int)
    v21a2_pad[20, :] = v21a2[0, :]
    v21a2_pad[21, :] = v21a2[1, :]
    
    w21a2_pad = np.zeros((15, 4), dtype=int)
    w21a2_pad[0, :] = w21a2[0, :]
    w21a2_pad[1, :] = w21a2[1, :]
    w21a2_pad[3, :] = w21a2[2, :]
    w21a2_pad[4, :] = w21a2[3, :]
    
    blocks.append((u21a2, v21a2_pad, w21a2_pad, 4, 0, 0))
    
    # A2b * B21 (1x3x2) -> Schoolbook Rank 6
    u21b, v21b, w21b = _schoolbook(1, 3, 2)
    # A2b: row 2 of A2 (global row 2), cols 0,1,2 (global cols 2,3,4).
    # Global u_off: 2*5+2=12.
    # B21: rows 0,1,2 of B2 (global rows 2,3,4), cols 0,1.
    # Global v_off: 2*5+0=10.
    # W: C row 2; cols 0,1.
    # Global w_off: 2*3+2=8? No.
    # W index k*n+i. k=0,1. i=2.
    # Indices: 2, 5.
    
    v21b_pad = np.zeros((25, 6), dtype=int)
    v21b_pad[10, :] = v21b[0, :]
    v21b_pad[11, :] = v21b[1, :]
    v21b_pad[15, :] = v21b[2, :]
    v21b_pad[16, :] = v21b[3, :]
    v21b_pad[20, :] = v21b[4, :]
    v21b_pad[21, :] = v21b[5, :]
    
    w21b_pad = np.zeros((15, 6), dtype=int)
    w21b_pad[2, :] = w21b[0, :]
    w21b_pad[5, :] = w21b[1, :]
    w21b_pad[8, :] = w21b[2, :]
    w21b_pad[11, :] = w21b[3, :]
    w21b_pad[14, :] = w21b[4, :]
    # Wait, n=1 local. W is 2x1 x 6.
    # W rows: k=0,1. i=0.
    # Global i=2.
    # W indices: 0*3+2=2, 1*3+2=5.
    # w21b has 6 columns.
    # w21b[0, :] corresponds to i=0, j=0, k=0.
    # We need to map these to global i=2.
    # So w21b_pad[2, :] should sum the contributions for i=2, k=0.
    # w21b[0, :] is for j=0, k=0.
    # w21b[3, :] is for j=1, k=0.
    # w21b[6, :] is for j=2, k=0.
    # So w21b_pad[2, :] = w21b[0, :] + w21b[3, :] + w21b[6, :].
    
    # Let's redo W padding for schoolbook.
    # It's easier to just construct the full W matrix directly if padding is complex.
    # But let's try to fix the mapping.
    
    w21b_pad = np.zeros((15, 6), dtype=int)
    # k=0, i=2 -> idx 2
    w21b_pad[2, :] = w21b[0, :] + w21b[3, :] + w21b[6, :]
    # k=1, i=2 -> idx 5
    w21b_pad[5, :] = w21b[1, :] + w21b[4, :] + w21b[7, :]
    
    blocks.append((u21b, v21b_pad, w21b_pad, 12, 0, 0))
    
    # --- Term 2.2: A2 (3x3) * B22 (3x3) ---
    # Schoolbook Rank 27
    u22, v22, w22 = _schoolbook(3, 3, 3)
    # A2: rows 0,1,2; cols 2,3,4.
    # Global u_off: 0*5+2=2.
    # B22: rows 2,3,4; cols 2,3,4.
    # Global v_off: 2*5+2=12.
    # W: C rows 0,1,2; cols 2,3,4.
    # Global w_off: 2*3+0=6.
    
    v22_pad = np.zeros((25, 27), dtype=int)
    for j in range(3):
        for k in range(3):
            local_idx = j * 3 + k
            global_idx = (2 + j) * 5 + (2 + k)
            v22_pad[global_idx, :] = v22[local_idx, :]
            
    w22_pad = np.zeros((15, 27), dtype=int)
    for k in range(3):
        for i in range(3):
            local_idx = k * 3 + i
            global_idx = (2 + k) * 3 + i
            w22_pad[global_idx, :] = w22[local_idx, :]
            
    blocks.append((u22, v22_pad, w22_pad, 2, 0, 0))
    
    U, V, W = _combine_blocks(blocks)
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    result = construct()
    print("Rank:", result["U"].shape[1])
