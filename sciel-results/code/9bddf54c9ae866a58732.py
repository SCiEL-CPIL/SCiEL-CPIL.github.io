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
    
    def schoolbook_2x2x2(va, vb, vc):
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    add_term(va[i][j], vb[j][k], vc[i][k])
    
    def schoolbook_2x2x1(va, vb, vc):
        for i in range(2):
            for j in range(2):
                add_term(va[i][j], vb[j], vc[i])
    
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
    
    def make_b_col_block(r1, r2, c):
        return [
            {b_idx(r1, c): 1},
            {b_idx(r2, c): 1}
        ]
    
    def make_c_block(r1, r2, c1, c2):
        return [
            [{c_idx(r1, c1): 1}, {c_idx(r1, c2): 1}],
            [{c_idx(r2, c1): 1}, {c_idx(r2, c2): 1}]
        ]
    
    def make_c_col_block(r1, r2, c):
        return [
            {c_idx(r1, c): 1},
            {c_idx(r2, c): 1}
        ]
    
    # Define all 2x2 blocks
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
    strassen_2x2x2(block_add(A11, A22), block_add(B11, B22), block_add(C11, C22))
    strassen_2x2x2(block_add(A21, A22), B11, block_sub(C21, C22))
    strassen_2x2x2(A11, block_sub(B12, B22), block_add(C12, C22))
    strassen_2x2x2(A22, block_sub(B21, B11), block_add(C11, C21))
    strassen_2x2x2(block_add(A11, A12), B22, merge_w(C11, C12, -1, 1) if False else [
        [merge_w(C11[0][0], C12[0][0], -1, 1), merge_w(C11[0][1], C12[0][1], -1, 1)],
        [merge_w(C11[1][0], C12[1][0], -1, 1), merge_w(C11[1][1], C12[1][1], -1, 1)]
    ])
    strassen_2x2x2(block_sub(A21, A11), block_add(B11, B12), C22)
    strassen_2x2x2(block_sub(A12, A22), block_add(B21, B22), C11)
    
    if time.time() > deadline:
        rank = len(U_cols)
        U = [[U_cols[c][r] for c in range(rank)] for r in range(n * m)]
        V = [[V_cols[c][r] for c in range(rank)] for r in range(m * p)]
        W = [[W_cols[c][r] for c in range(rank)] for r in range(p * n)]
        return {"U": U, "V": V, "W": W}
    
    # 4x4x2 part: rank 28 using Strassen with 2x2x2 base
    # Split B into B1 (cols 4,5) as 2x2 blocks, C into C1 (cols 4,5) as 2x2 blocks
    B11_45 = make_b_block(0, 1, 4, 5)
    B12_45 = make_b_block(0, 1, 6, 6)  # This won't work, let me reconsider
    
    # Actually for 4x4x2: A is 4x4, B is 4x2, C is 4x2
    # B has rows 0-3, cols 4-5
    # Split A into 2x2 blocks (A11, A12, A21, A22)
    # Split B into top 2x2 (rows 0-1, cols 4-5) and bottom 2x2 (rows 2-3, cols 4-5)
    # Split C into top 2x2 (rows 0-1, cols 4-5) and bottom 2x2 (rows 2-3, cols 4-5)
    
    B_top_45 = make_b_block(0, 1, 4, 5)
    B_bot_45 = make_b_block(2, 3, 4, 5)
    C_top_45 = make_c_block(0, 1, 4, 5)
    C_bot_45 = make_c_block(2, 3, 4, 5)
    
    # C_top = A11 * B_top + A12 * B_bot  (2x2 = 2x2 x 2x2 + 2x2 x 2x2)
    # C_bot = A21 * B_top + A22 * B_bot  (2x2 = 2x2 x 2x2 + 2x2 x 2x2)
    
    # Use Strassen for each 2x2x2x2 (sum of two 2x2x2 multiplications)
    # Actually, we need to compute A11*B_top + A12*B_bot which is a 2x2x2x2 tensor
    # This can be done with rank 7 using Strassen-like approach
    
    # For 2x2x2x2: M = A*B + C*D where A,B,C,D are 2x2
    # Using Strassen: 7 multiplications
    # M = (A+C)(B+D) - C(B-D) ... no, that's not quite right
    
    # Let me use the standard approach:
    # (A+C)(B+D) = AB + AD + CB + CD
    # We want AB + CD
    # So: AB + CD = (A+C)(B+D) - AD - CB
    # AD = A(D) and CB = C(B)
    # This gives 4 multiplications total, not better than 8
    
    # Better: just use schoolbook for each 2x2x2, giving 4*4 = 16 per 2x2x2x2
    # Total for 4x4x2: 2 * 16 = 32
    
    # Or use Strassen for the 2x2x2x2:
    # Actually, the rank of 2x2x2x2 tensor is known to be 7 (same as 2x2x2)
    # No wait, 2x2x2x2 is different. Let me just use schoolbook.
    
    # Schoolbook: 4x4x2 = 4*4*2 = 32 scalar multiplications
    # But we can do better with block Strassen
    
    # Let me use: for each of the 4 output blocks (2x2), compute as sum of two 2x2x2 products
    # Each 2x2x2 has rank 7 (Strassen) or 4 (schoolbook)
    # Using Strassen: 4 * 7 = 28
    # Using schoolbook: 4 * 4 = 16... but that's not right either
    
    # C_top[i,k] = sum_j A11[i,j]*B_top[j,k] + sum_j A12[i,j]*B_bot[j,k]
    # This is a 2x2x2 matrix = sum of two 2x2x2 matrices
    # Each 2x2x2 has tensor rank 7 (Strassen)
    # So C_top requires 7+7 = 14 rank-1 terms
    # Similarly C_bot requires 14 rank-1 terms
    # Total: 28
    
    # Let me implement this properly
    def strassen_for_sum(va1, vb1, va2, vb2, vc):
        """Compute vc = va1*vb1 + va2*vb2 using 2 Strassen calls (14 terms)"""
        # First product
        a00, a01 = va1[0][0], va1[0][1]
        a10, a11 = va1[1][0], va1[1][1]
        b00, b01 = vb1[0][0], vb1[0][1]
        b10, b11 = vb1[1][0], vb1[1][1]
        
        add_term(merge_add(a00, a11), merge_add(b00, b11), vc[0][0])
        add_term(merge_add(a10, a11), b00, vc[1][0])
        add_term(a00, merge_sub(b01, b11), vc[0][1])
        add_term(a11, merge_sub(b10, b00), vc[0][0])
        add_term(merge_add(a00, a01), b11, vc[0][1])
        add_term(merge_sub(a10, a00), merge_add(b00, b01), vc[1][1])
        add_term(merge_sub(a01, a11), merge_add(b10, b11), vc[0][0])
        
        # Second product
        c00, c01 = va2[0][0], va2[0][1]
        c10, c11 = va2[1][0], va2[1][1]
        d00, d01 = vb2[0][0], vb2[0][1]
        d10, d11 = vb2[1][0], vb2[1][1]
        
        add_term(merge_add(c00, c11), merge_add(d00, d11), vc[0][0])
        add_term(merge_add(c10, c11), d00, vc[1][0])
        add_term(c00, merge_sub(d01, d11), vc[0][1])
        add_term(c11, merge_sub(d10, d00), vc[0][0])
        add_term(merge_add(c00, c01), d11, vc[0][1])
        add_term(merge_sub(c10, c00), merge_add(d00, d01), vc[1][1])
        add_term(merge_sub(c01, c11), merge_add(d10, d11), vc[0][0])
    
    # This approach has issues with overlapping W terms. Let me use a cleaner method.
    # Just use schoolbook for the 4x4x2 part: rank 32
    # Total: 49 + 32 = 81
    
    # Actually, let me reconsider. Let me just use schoolbook for 4x4x2.
    for i in range(4):
        for j in range(4):
            for k in range(2):
                add_term({a_idx(i, j): 1}, {b_idx(j, 4+k): 1}, {c_idx(i, 4+k): 1})
    
    if time.time() > deadline:
        rank = len(U_cols)
        U = [[U_cols[c][r] for c in range(rank)] for r in range(n * m)]
        V = [[V_cols[c][r] for c in range(rank)] for r in range(m * p)]
        W = [[W_cols[c][r] for c in range(rank)] for r in range(p * n)]
        return {"U": U, "V": V, "W": W}
    
    # 4x4x1 part: rank 16 using schoolbook
    for i in range(4):
        for j in range(4):
            add_term({a_idx(i, j): 1}, {b_idx(j, 6): 1}, {c_idx(i, 6): 1})
    
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
