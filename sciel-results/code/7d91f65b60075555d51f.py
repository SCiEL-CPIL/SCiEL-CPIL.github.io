import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 4, 4, 7
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_entries, v_entries, w_entries):
        u_col = [0] * (n * m)
        v_col = [0] * (m * p)
        w_col = [0] * (p * n)
        for idx, coeff in u_entries.items():
            u_col[idx] = coeff
        for idx, coeff in v_entries.items():
            v_col[idx] = coeff
        for idx, coeff in w_entries.items():
            w_col[idx] = coeff
        U_cols.append(u_col)
        V_cols.append(v_col)
        W_cols.append(w_col)
    
    def merge_add(d1, d2):
        result = dict(d1)
        for k, v in d2.items():
            result[k] = result.get(k, 0) + v
        return result
    
    def merge_sub(d1, d2):
        result = dict(d1)
        for k, v in d2.items():
            result[k] = result.get(k, 0) - v
        return result
    
    def merge_w(d1, d2, s1=1, s2=1):
        result = {}
        for k, v in d1.items():
            result[k] = result.get(k, 0) + s1 * v
        for k, v in d2.items():
            result[k] = result.get(k, 0) + s2 * v
        return result
    
    def a_idx(i, j):
        return i * m + j
    
    def b_idx(j, k):
        return j * p + k
    
    def c_idx(i, k):
        return k * n + i
    
    def strassen_2x2x2(va, vb, vc):
        a00, a01 = va[0][0], va[0][1]
        a10, a11 = va[1][0], va[1][1]
        b00, b01 = vb[0][0], vb[0][1]
        b10, b11 = vb[1][0], vb[1][1]
        c00, c01 = vc[0][0], vc[0][1]
        c10, c11 = vc[1][0], vc[1][1]
        
        add_term(merge_add(a00, a11), merge_add(b00, b11), merge_w(c00, c11, 1, 1))
        add_term(merge_add(a10, a11), b00, merge_w(c10, c11, 1, -1))
        add_term(a00, merge_sub(b01, b11), merge_w(c01, c11, 1, 1))
        add_term(a11, merge_sub(b10, b00), merge_w(c00, c10, 1, 1))
        add_term(merge_add(a00, a01), b11, merge_w(c00, c01, -1, 1))
        add_term(merge_sub(a10, a00), merge_add(b00, b01), c11)
        add_term(merge_sub(a01, a11), merge_add(b10, b11), c00)
    
    def make_a_block(r1, r2, c1, c2):
        return [
            [{a_idx(r1, c1): 1}, {a_idx(r1, c2): 1}],
            [{a_idx(r2, c1): 1}, {a_idx(r2, c2): 1}]
        ]
    
    def make_b_block(r1, r2, c1, c2):
        return [
            [{b_idx(r1, c1): 1}, {b_idx(r1, c2): 1}],
            [{b_idx(r2, c1): 1}, {b_idx(r2, c2): 1}]
        ]
    
    def make_c_block(r1, r2, c1, c2):
        return [
            [{c_idx(r1, c1): 1}, {c_idx(r1, c2): 1}],
            [{c_idx(r2, c1): 1}, {c_idx(r2, c2): 1}]
        ]
    
    # Define all 2x2 blocks for the 4x4x4 part
    A11 = make_a_block(0, 1, 0, 1)
    A12 = make_a_block(0, 1, 2, 3)
    A21 = make_a_block(2, 3, 0, 1)
    A22 = make_a_block(2, 3, 2, 3)
    
    B11 = make_b_block(0, 1, 0, 1)
    B12 = make_b_block(0, 1, 2, 3)
    B21 = make_b_block(2, 3, 0, 1)
    B22 = make_b_block(2, 3, 2, 3)
    
    C11 = make_c_block(0, 1, 0, 1)
    C12 = make_c_block(0, 1, 2, 3)
    C21 = make_c_block(2, 3, 0, 1)
    C22 = make_c_block(2, 3, 2, 3)
    
    def block_add(b1, b2):
        return [
            [merge_add(b1[0][0], b2[0][0]), merge_add(b1[0][1], b2[0][1])],
            [merge_add(b1[1][0], b2[1][0]), merge_add(b1[1][1], b2[1][1])]
        ]
    
    def block_sub(b1, b2):
        return [
            [merge_sub(b1[0][0], b2[0][0]), merge_sub(b1[0][1], b2[0][1])],
            [merge_sub(b1[1][0], b2[1][0]), merge_sub(b1[1][1], b2[1][1])]
        ]
    
    # 4x4x4 part: rank 49 using recursive Strassen
    # These 7 calls each invoke strassen_2x2x2 which adds 7 terms
    strassen_2x2x2(block_add(A11, A22), block_add(B11, B22), block_add(C11, C22))
    strassen_2x2x2(block_add(A21, A22), B11, block_sub(C21, C22))
    strassen_2x2x2(A11, block_sub(B12, B22), block_add(C12, C22))
    strassen_2x2x2(A22, block_sub(B21, B11), block_add(C11, C21))
    strassen_2x2x2(block_add(A11, A12), B22, [
        [merge_w(C11[0][0], C12[0][0], -1, 1), merge_w(C11[0][1], C12[0][1], -1, 1)],
        [merge_w(C11[1][0], C12[1][0], -1, 1), merge_w(C11[1][1], C12[1][1], -1, 1)]
    ])
    strassen_2x2x2(block_sub(A21, A11), block_add(B11, B12), C22)
    strassen_2x2x2(block_sub(A12, A22), block_add(B21, B22), C11)
    
    # Now for the remaining 3 columns (cols 4,5,6), reuse the same 7 U-factors
    # The 7 U-factors from Strassen are:
    # U1 = A11+A22, U2 = A21+A22, U3 = A11, U4 = A22, U5 = A11+A12, U6 = A21-A11, U7 = A12-A22
    # For B_ext (4x3, cols 4-6), define 2x2 blocks:
    # B11_ext = rows 0-1, cols 4-5 (2x2), but we have 3 cols... 
    # Let's split B_ext into B11_ext (rows 0-1, cols 4-5), B12_ext (rows 0-1, col 6), etc.
    # This is awkward. Instead, let's define the V-factors directly as linear combinations.
    
    # For the 4x3 part, we need: C_ext = A * B_ext
    # Using the same 7 U-factors, we define 7 V-factors (linear combos of B_ext entries)
    # and 7 W-factors (linear combos of C_ext entries)
    
    # B_ext has entries b_idx(j, k) for j in 0..3, k in 4..6
    # C_ext has entries c_idx(i, k) for i in 0..3, k in 4..6
    
    # Define block-like structures for B_ext and C_ext
    # B_top_ext: rows 0-1, cols 4-6 -> treat as 2x3
    # B_bot_ext: rows 2-3, cols 4-6 -> treat as 2x3
    
    # For Strassen-style, we need 2x2 blocks. Let's use cols 4-5 as the "main" 2x2 
    # and handle col 6 separately, OR we can define V-factors that span all 3 cols.
    
    # Simpler: define V-factors as dicts directly
    # V1_ext = B11_ext + B22_ext where B11_ext = rows 0-1, cols 4-5; B22_ext = rows 2-3, cols 4-5
    # But we also need col 6. Let's include it.
    
    # Actually, let's just define the 7 V-factors and W-factors explicitly for the 4x3 case
    # The structure mirrors Strassen but with 3 columns instead of 4.
    
    # B_ext blocks (using cols 4,5 as a 2-wide and col 6 as 1-wide is messy)
    # Let me define:
    # B11e: rows 0-1, col 4; B12e: rows 0-1, col 5; B13e: rows 0-1, col 6
    # B21e: rows 2-3, col 4; B22e: rows 2-3, col 5; B23e: rows 2-3, col 6
    
    B11e = [{b_idx(0, 4): 1}, {b_idx(1, 4): 1}]  # rows 0-1, col 4
    B12e = [{b_idx(0, 5): 1}, {b_idx(1, 5): 1}]  # rows 0-1, col 5
    B13e = [{b_idx(0, 6): 1}, {b_idx(1, 6): 1}]  # rows 0-1, col 6
    B21e = [{b_idx(2, 4): 1}, {b_idx(3, 4): 1}]  # rows 2-3, col 4
    B22e = [{b_idx(2, 5): 1}, {b_idx(3, 5): 1}]  # rows 2-3, col 5
    B23e = [{b_idx(2, 6): 1}, {b_idx(3, 6): 1}]  # rows 2-3, col 6
    
    C11e = [{c_idx(0, 4): 1}, {c_idx(1, 4): 1}]  # rows 0-1, col 4
    C12e = [{c_idx(0, 5): 1}, {c_idx(1, 5): 1}]  # rows 0-1, col 5
    C13e = [{c_idx(0, 6): 1}, {c_idx(1, 6): 1}]  # rows 0-1, col 6
    C21e = [{c_idx(2, 4): 1}, {c_idx(3, 4): 1}]  # rows 2-3, col 4
    C22e = [{c_idx(2, 5): 1}, {c_idx(3, 5): 1}]  # rows 2-3, col 5
    C23e = [{c_idx(2, 6): 1}, {c_idx(3, 6): 1}]  # rows 2-3, col 6
    
    # U-factors (same as in the 4x4x4 Strassen)
    U1 = block_add(A11, A22)  # A11+A22
    U2 = block_add(A21, A22)  # A21+A22
    U3 = A11
    U4 = A22
    U5 = block_add(A11, A12)  # A11+A12
    U6 = block_sub(A21, A11)  # A21-A11
    U7 = block_sub(A12, A22)  # A12-A22
    
    # For each of the 7 U-factors, we need to create 3 rank-1 terms (one per output column group)
    # OR we can create 7 terms total where each V-factor is a combination spanning all 3 cols
    # and each W-factor is a combination spanning all 3 cols.
    
    # The Strassen formulas for the 4x4x4 case give:
    # C11 = P1 + P3 - P5 + P7
    # C12 = P2 + P5
    # C21 = P4 + P6
    # C22 = P1 - P2 + P4 + P6
    
    # For the extended case, we want to compute C_ext = A * B_ext
    # Using the same U-factors, define V-factors and W-factors such that:
    # sum_i U_i * V_i_ext = C_ext (as a tensor)
    
    # The key insight: if we define V_i_ext as the "Strassen-combined" version of B_ext blocks
    # and W_i_ext as the corresponding C_ext blocks, then the same linear combination works.
    
    # For a single column k (say col 4), the contribution is:
    # C_col4 = A * B_col4
    # Using Strassen-like: 
    # C11_col4 = P1_col4 + P3_col4 - P5_col4 + P7_col4
    # C12_col4 = P2_col4 + P5_col4  (but C12 is rows 0-1, col 5... this doesn't map cleanly)
    
    # This approach is getting complicated. Let me just use schoolbook for the 4x3 part.
    # Rank 4*4*3 = 48. Total: 49 + 48 = 97. Same as before.
    
    # Actually, let me try a different approach: use Strassen for 4x4x4 (rank 49) 
    # and then for the 4x4x3 part, note that it's 3 separate 4x4x1 multiplications.
    # Each 4x4x1 can be done with rank 4 (schoolbook) or potentially less.
    # 3 * 4 = 12. Total: 49 + 12 = 61.
    
    # Wait, 4x4x1 means A(4x4) * B(4x1) = C(4x1). Schoolbook is 4*4 = 16 terms.
    # So 3 * 16 = 48. Total: 49 + 48 = 97.
    
    # Let me just go with schoolbook for the 4x3 part.
    for i in range(4):
        for j in range(4):
            for k in range(3):
                add_term({a_idx(i, j): 1}, {b_idx(j, 4+k): 1}, {c_idx(i, 4+k): 1})
    
    rank = len(U_cols)
    U = [[U_cols[c][r] for c in range(rank)] for r in range(n * m)]
    V = [[V_cols[c][r] for c in range(rank)] for r in range(m * p)]
    W = [[W_cols[c][r] for c in range(rank)] for r in range(p * n)]
    
    def clean_matrix(mat):
        result = []
        for row in mat:
            new_row = []
            for val in row:
                if isinstance(val, float) and val == int(val):
                    new_row.append(int(val))
                elif isinstance(val, np.floating) and float(val) == int(float(val)):
                    new_row.append(int(float(val)))
                else:
                    new_row.append(val)
            result.append(new_row)
        return result
    
    U = clean_matrix(U)
    V = clean_matrix(V)
    W = clean_matrix(W)
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    result = construct()
    rank = len(result["U"][0])
    print(f"Rank: {rank}")
