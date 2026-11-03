import time
import sys

def construct():
    """
    Returns a rank-minimal (or near-minimal) exact tensor decomposition for multiplying
    a 3x3 matrix by a 3x5 matrix.
    
    Uses block decomposition with Strassen for 2x2x2 blocks.
    Target rank: 43.
    """
    # U: (9, R), V: (15, R), W: (15, R)
    # U index: i*3 + j
    # V index: j*5 + k
    # W index: k*3 + i
    
    U = []
    V = []
    W = []
    
    def add_term(u_idx, v_idx, w_idx, val=1):
        # We append to lists and then convert to arrays at the end?
        # Or we can build the columns.
        # Since R is fixed at 43, we can pre-allocate.
        pass
        
    # Let's build the terms as a list of (u_idx, v_idx, w_idx, val)
    terms = []
    
    def term(u_idx, v_idx, w_idx, val=1):
        terms.append((u_idx, v_idx, w_idx, val))
        
    # Helper to add Strassen 2x2x2
    # a_rows, a_cols, b_rows, b_cols, c_rows, c_cols
    # a_rows: [i1, i2], a_cols: [j1, j2]
    # b_rows: [j1, j2], b_cols: [k1, k2]
    # c_rows: [i1, i2], c_cols: [k1, k2]
    def add_strassen_222(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        a00 = a_rows[0]*3 + a_cols[0]
        a01 = a_rows[0]*3 + a_cols[1]
        a10 = a_rows[1]*3 + a_cols[0]
        a11 = a_rows[1]*3 + a_cols[1]
        
        b00 = b_rows[0]*5 + b_cols[0]
        b01 = b_rows[0]*5 + b_cols[1]
        b10 = b_rows[1]*5 + b_cols[0]
        b11 = b_rows[1]*5 + b_cols[1]
        
        c00 = c_rows[0]*3 + c_cols[0]
        c01 = c_rows[0]*3 + c_cols[1]
        c10 = c_rows[1]*3 + c_cols[0]
        c11 = c_rows[1]*3 + c_cols[1]
        
        # M1 = (a00+a11)*(b00+b11)
        # C00 += M1, C01 += M1, C10 += M1, C11 += M1
        # Wait, Strassen output:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We need to add rank-1 terms for each Mi.
        # Mi = u_i(A) * v_i(B).
        # This contributes to C_ik via w_ik,i * Mi.
        # So for each Mi, we add 4 terms (one for each C element it affects).
        # Actually, we add terms to the global sum.
        # term(u_A, v_B, w_C, val)
        
        # M1: u=(a00+a11), v=(b00+b11)
        # u is not a single index, it's a sum.
        # So we decompose u = a00 + a11 into two terms.
        # term(a00, b00, c00, 1)
        # term(a00, b11, c00, 1)
        # term(a11, b00, c00, 1)
        # term(a11, b11, c00, 1)
        # And similarly for c01, c10, c11.
        
        # This approach expands the linear forms.
        # M1 contributes 4 terms to C00, 4 to C01, 4 to C10, 4 to C11. Total 16 terms?
        # No, the rank of the decomposition is the number of *multiplications* Mi.
        # In the tensor decomposition, each "multiplication" Mi corresponds to a set of rank-1 terms?
        # No. The rank is the number of terms in the sum $\sum_r U_r V_r W_r$.
        # If $M_1 = (a_{00}+a_{11})(b_{00}+b_{11})$, this is a bilinear form.
        # It can be written as sum of 4 rank-1 tensors:
        # $a_{00}b_{00}, a_{00}b_{11}, a_{11}b_{00}, a_{11}b_{11}$.
        # So M1 alone has rank 4 in the tensor sense?
        # But Strassen says rank 7 for the whole 2x2x2 multiplication.
        # This means the 7 multiplications are the "rank-1" components in the *bilinear* sense,
        # but when expanded to the tensor rank, each $M_l$ might contribute multiple tensor rank-1 terms?
        # No. The tensor rank of $M_{2,2,2}$ is 7.
        # This means there exist 7 triples of vectors $(u_l, v_l, w_l)$ such that
        # $\sum_{l=1}^7 u_{ij,l} v_{jk,l} w_{ki,l} = 1$ for all $i,j,k \in \{0,1\}$.
        # The vectors $u_l, v_l$ are not sparse indicators, they have coefficients.
        # So we don't expand into 4 terms. We just use the coefficients.
        
        # Strassen coefficients:
        # M1: u=(1,0,0,1), v=(1,0,0,1) -> indices (00,11) for A, (00,11) for B
        # Actually, let's list the (u, v, w) for the 7 terms.
        
        # Term 1: M1
        # u: a00=1, a11=1. Others 0.
        # v: b00=1, b11=1. Others 0.
        # w: c00=1, c01=1, c10=1, c11=1.
        # Wait, w is the coefficient for C.
        # C00 gets +M1. C01 gets +M1. C10 gets +M1. C11 gets +M1.
        # So w00=1, w01=1, w10=1, w11=1.
        
        # Term 2: M2
        # u: a10=1, a11=1.
        # v: b00=1.
        # w: C10 gets +M2. C11 gets -M2.
        # w10=1, w11=-1.
        
        # Term 3: M3
        # u: a00=1.
        # v: b01=1, b11=-1.
        # w: C01 gets +M3. C11 gets +M3.
        # w01=1, w11=1.
        
        # Term 4: M4
        # u: a11=1.
        # v: b10=1, b00=-1.
        # w: C00 gets +M4. C10 gets +M4.
        # w00=1, w10=1.
        
        # Term 5: M5
        # u: a00=1, a01=1.
        # v: b11=1.
        # w: C00 gets -M5. C01 gets +M5.
        # w00=-1, w01=1.
        
        # Term 6: M6
        # u: a10=1, a00=-1.
        # v: b00=1, b01=1.
        # w: C11 gets +M6.
        # w11=1.
        
        # Term 7: M7
        # u: a01=1, a11=-1.
        # v: b10=1, b11=1.
        # w: C00 gets +M7.
        # w00=1.
        
        # We add these 7 terms.
        
        def add_u(u_dict):
            for idx, val in u_dict.items():
                if val != 0:
                    terms.append((idx, -1, -1, val)) # Placeholder, we need to group by term r
            # This is getting complicated to manage indices.
            # Let's just append the full (u_idx, u_val, v_idx, v_val, w_idx, w_val) and then combine?
            # No, the rank is the number of terms in the sum.
            # Each "Term" above is one rank-1 component in the decomposition of the *block*.
            # But it has multiple non-zero entries in u, v, w.
            # The global decomposition is the sum of these.
            # So we just add these 7 "terms" to the global list.
            # But a "term" in the global decomposition is a triple of vectors.
            # So we need to store the vectors.
            
        # Let's store the terms as dictionaries of non-zero entries for U, V, W.
        
        # Term 1
        u = {a00: 1, a11: 1}
        v = {b00: 1, b11: 1}
        w = {c00: 1, c01: 1, c10: 1, c11: 1}
        terms.append((u, v, w))
        
        # Term 2
        u = {a10: 1, a11: 1}
        v = {b00: 1}
        w = {c10: 1, c11: -1}
        terms.append((u, v, w))
        
        # Term 3
        u = {a00: 1}
        v = {b01: 1, b11: -1}
        w = {c01: 1, c11: 1}
        terms.append((u, v, w))
        
        # Term 4
        u = {a11: 1}
        v = {b10: 1, b00: -1}
        w = {c00: 1, c10: 1}
        terms.append((u, v, w))
        
        # Term 5
        u = {a00: 1, a01: 1}
        v = {b11: 1}
        w = {c00: -1, c01: 1}
        terms.append((u, v, w))
        
        # Term 6
        u = {a10: 1, a00: -1}
        v = {b00: 1, b01: 1}
        w = {c11: 1}
        terms.append((u, v, w))
        
        # Term 7
        u = {a01: 1, a11: -1}
        v = {b10: 1, b11: 1}
        w = {c00: 1}
        terms.append((u, v, w))
        
    # Add Block 1: A11 B11 -> C11
    # A rows 0,1 cols 0,1. B rows 0,1 cols 0,1. C rows 0,1 cols 0,1.
    add_strassen_222([0,1], [0,1], [0,1], [0,1], [0,1], [0,1])
    
    # Add Block 3: A11 B13 -> C13
    # A rows 0,1 cols 0,1. B rows 0,1 cols 3,4. C rows 0,1 cols 3,4.
    add_strassen_222([0,1], [0,1], [0,1], [3,4], [0,1], [3,4])
    
    # Other blocks are smaller.
    # We can implement them similarly or just use direct rank-1 terms if rank is small.
    # For 2x2x1 (rank 4):
    # C0 = A00 B0 + A01 B1
    # C1 = A10 B0 + A11 B1
    # This is just 4 terms.
    
    def add_direct_221(a_rows, a_cols, b_rows, b_col, c_rows, c_col):
        # A is 2x2, B is 2x1, C is 2x1
        a00 = a_rows[0]*3 + a_cols[0]
        a01 = a_rows[0]*3 + a_cols[1]
        a10 = a_rows[1]*3 + a_cols[0]
        a11 = a_rows[1]*3 + a_cols[1]
        
        b0 = b_rows[0]*5 + b_col
        b1 = b_rows[1]*5 + b_col
        
        c0 = c_rows[0]*3 + c_col
        c1 = c_rows[1]*3 + c_col
        
        # C0 = A00 B0 + A01 B1
        # C1 = A10 B0 + A11 B1
        
        # Term 1: A00 B0 -> C0
        terms.append(({a00: 1}, {b0: 1}, {c0: 1}))
        # Term 2: A01 B1 -> C0
        terms.append(({a01: 1}, {b1: 1}, {c0: 1}))
        # Term 3: A10 B0 -> C1
        terms.append(({a10: 1}, {b0: 1}, {c1: 1}))
        # Term 4: A11 B1 -> C1
        terms.append(({a11: 1}, {b1: 1}, {c1: 1}))
        
    # Add Block 2: A11 B12 -> C12
    # A 2x2, B 2x1, C 2x1
    add_direct_221([0,1], [0,1], [0,1], 2, [0,1], 2)
    
    # Add Block 4: A12 B21 -> C21? No, A12 is 2x1, B21 is 1x2. C is 2x2.
    # A12: rows 0,1 col 2. B21: row 2 cols 0,1. C21: rows 0,1 cols 0,1.
    # This is 2x1x2. Rank 4.
    def add_direct_212(a_rows, a_col, b_row, b_cols, c_rows, c_cols):
        a0 = a_rows[0]*3 + a_col
        a1 = a_rows[1]*3 + a_col
        
        b0 = b_row*5 + b_cols[0]
        b1 = b_row*5 + b_cols[1]
        
        c00 = c_rows[0]*3 + c_cols[0]
        c01 = c_rows[0]*3 + c_cols[1]
        c10 = c_rows[1]*3 + c_cols[0]
        c11 = c_rows[1]*3 + c_cols[1]
        
        # C00 = A0 B0
        # C01 = A0 B1
        # C10 = A1 B0
        # C11 = A1 B1
        
        terms.append(({a0: 1}, {b0: 1}, {c00: 1}))
        terms.append(({a0: 1}, {b1: 1}, {c01: 1}))
        terms.append(({a1: 1}, {b0: 1}, {c10: 1}))
        terms.append(({a1: 1}, {b1: 1}, {c11: 1}))
        
    add_direct_212([0,1], 2, 2, [0,1], [0,1], [0,1])
    
    # Add Block 6: A12 B23 -> C13? No, C13 is rows 0,1 cols 3,4.
    # A12: rows 0,1 col 2. B23: row 2 cols 3,4.
    add_direct_212([0,1], 2, 2, [3,4], [0,1], [3,4])
    
    # Add Block 5: A12 B22 -> C12? No, C12 is rows 0,1 col 2.
    # A12: rows 0,1 col 2. B22: row 2 col 2.
    # This is 2x1x1. Rank 2.
    def add_direct_211(a_rows, a_col, b_row, b_col, c_rows, c_col):
        a0 = a_rows[0]*3 + a_col
        a1 = a_rows[1]*3 + a_col
        
        b = b_row*5 + b_col
        
        c0 = c_rows[0]*3 + c_col
        c1 = c_rows[1]*3 + c_col
        
        terms.append(({a0: 1}, {b: 1}, {c0: 1}))
        terms.append(({a1: 1}, {b: 1}, {c1: 1}))
        
    add_direct_211([0,1], 2, 2, 2, [0,1], 2)
    
    # Add Block 7: A21 B11 -> C21.
    # A21: row 2 cols 0,1. B11: rows 0,1 cols 0,1.
    # This is 1x2x2. Rank 4.
    def add_direct_122(a_row, a_cols, b_rows, b_cols, c_row, c_cols):
        a0 = a_row*3 + a_cols[0]
        a1 = a_row*3 + a_cols[1]
        
        b00 = b_rows[0]*5 + b_cols[0]
        b01 = b_rows[0]*5 + b_cols[1]
        b10 = b_rows[1]*5 + b_cols[0]
        b11 = b_rows[1]*5 + b_cols[1]
        
        c0 = c_row*3 + c_cols[0]
        c1 = c_row*3 + c_cols[1]
        
        # C0 = A0 B00 + A1 B10
        # C1 = A0 B01 + A1 B11
        
        terms.append(({a0: 1}, {b00: 1}, {c0: 1}))
        terms.append(({a1: 1}, {b10: 1}, {c0: 1}))
        terms.append(({a0: 1}, {b01: 1}, {c1: 1}))
        terms.append(({a1: 1}, {b11: 1}, {c1: 1}))
        
    add_direct_122(2, [0,1], [0,1], [0,1], 2, [0,1])
    
    # Add Block 9: A21 B13 -> C23.
    # A21: row 2 cols 0,1. B13: rows 0,1 cols 3,4.
    add_direct_122(2, [0,1], [0,1], [3,4], 2, [3,4])
    
    # Add Block 8: A21 B12 -> C22.
    # A21: row 2 cols 0,1. B12: rows 0,1 col 2.
    # This is 1x2x1. Rank 2.
    def add_direct_121(a_row, a_cols, b_rows, b_col, c_row, c_col):
        a0 = a_row*3 + a_cols[0]
        a1 = a_row*3 + a_cols[1]
        
        b0 = b_rows[0]*5 + b_col
        b1 = b_rows[1]*5 + b_col
        
        c = c_row*3 + c_col
        
        # C = A0 B0 + A1 B1
        
        terms.append(({a0: 1}, {b0: 1}, {c: 1}))
        terms.append(({a1: 1}, {b1: 1}, {c: 1}))
        
    add_direct_121(2, [0,1], [0,1], 2, 2, 2)
    
    # Add Block 10: A22 B21 -> C21.
    # A22: row 2 col 2. B21: row 2 cols 0,1.
    # This is 1x1x2. Rank 2.
    def add_direct_112(a_row, a_col, b_row, b_cols, c_row, c_cols):
        a = a_row*3 + a_col
        
        b0 = b_row*5 + b_cols[0]
        b1 = b_row*5 + b_cols[1]
        
        c0 = c_row*3 + c_cols[0]
        c1 = c_row*3 + c_cols[1]
        
        terms.append(({a: 1}, {b0: 1}, {c0: 1}))
        terms.append(({a: 1}, {b1: 1}, {c1: 1}))
        
    add_direct_112(2, 2, 2, [0,1], 2, [0,1])
    
    # Add Block 12: A22 B23 -> C23.
    add_direct_112(2, 2, 2, [3,4], 2, [3,4])
    
    # Add Block 11: A22 B22 -> C22.
    # 1x1x1. Rank 1.
    a = 2*3 + 2
    b = 2*5 + 2
    c = 2*3 + 2
    terms.append(({a: 1}, {b: 1}, {c: 1}))
    
    # Now convert terms to U, V, W matrices
    R = len(terms)
    U = [[0]*R for _ in range(9)]
    V = [[0]*R for _ in range(15)]
    W = [[0]*R for _ in range(15)]
    
    for r, (u, v, w) in enumerate(terms):
        for idx, val in u.items():
            U[idx][r] = val
        for idx, val in v.items():
            V[idx][r] = val
        for idx, val in w.items():
            W[idx][r] = val
            
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
