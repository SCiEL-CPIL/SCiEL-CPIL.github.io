import time

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # n=3, m=4, p=4
    # U size: n*m = 12
    # V size: m*p = 16
    # W size: p*n = 12
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    # u_idx: index in U
    # v_idx: index in V
    # w_idx: index in W
    # coeff: coefficient (1 or -1)
    def add_term(u_idx, v_idx, w_idx, coeff):
        nonlocal col
        U[u_idx][col] += coeff
        V[v_idx][col] += coeff
        W[w_idx][col] += coeff
        col += 1

    # Strassen helper for 2x2 blocks
    # x_indices: list of (i, j) for X
    # y_indices: list of (j, k) for Y
    # z_indices: list of (k, i) for Z
    # We map local (0,0), (0,1), (1,0), (1,1) to global indices
    
    def strassen_2x2(x_offsets, y_offsets, z_offsets):
        # x_offsets: (i0, j0) top-left of X block
        # y_offsets: (j0, k0) top-left of Y block
        # z_offsets: (k0, i0) top-left of Z block
        
        i0, j0 = x_offsets
        j1, k0 = y_offsets
        k1, i1 = z_offsets
        
        # X elements global indices:
        # X00: i0, j0
        # X01: i0, j0+1
        # X10: i0+1, j0
        # X11: i0+1, j0+1
        
        # Y elements global indices:
        # Y00: j1, k0
        # Y01: j1, k0+1
        # Y10: j1+1, k0
        # Y11: j1+1, k0+1
        
        # Z elements global indices:
        # Z00: k1, i1
        # Z01: k1+1, i1
        # Z10: k1, i1+1
        # Z11: k1+1, i1+1
        
        # U indices: i * m + j
        # V indices: j * p + k
        # W indices: k * n + i
        
        u00 = i0 * m + j0
        u01 = i0 * m + (j0 + 1)
        u10 = (i0 + 1) * m + j0
        u11 = (i0 + 1) * m + (j0 + 1)
        
        v00 = j1 * p + k0
        v01 = j1 * p + (k0 + 1)
        v10 = (j1 + 1) * p + k0
        v11 = (j1 + 1) * p + (k0 + 1)
        
        w00 = k1 * n + i1
        w01 = (k1 + 1) * n + i1
        w10 = k1 * n + (i1 + 1)
        w11 = (k1 + 1) * n + (i1 + 1)
        
        # M1 = (x00 + x11)(y00 + y11)
        # This is a product of sums. We need to expand?
        # No, Strassen terms are bilinear forms.
        # The tensor decomposition of Strassen uses 7 terms.
        # Each term is a product of a linear form in X and a linear form in Y.
        # We add the contributions to U, V, W.
        
        # Term 1: (x00 + x11) * (y00 + y11)
        # U part: x00 (coeff 1), x11 (coeff 1)
        # V part: y00 (coeff 1), y11 (coeff 1)
        # W part: z00 (coeff 1), z01 (coeff 0), z10 (coeff 0), z11 (coeff 1)
        # Wait, z00 = M1 + M4 - M5 + M7
        # z11 = M1 - M2 + M3 + M6
        # So M1 contributes to z00 and z11.
        
        # We add 1 column for M1.
        # U: u00 += 1, u11 += 1
        # V: v00 += 1, v11 += 1
        # W: w00 += 1, w11 += 1
        
        U[u00][col] += 1
        U[u11][col] += 1
        V[v00][col] += 1
        V[v11][col] += 1
        W[w00][col] += 1
        W[w11][col] += 1
        col += 1
        
        # Term 2: (x10 + x11) * y00
        # U: u10 += 1, u11 += 1
        # V: v00 += 1
        # W: z10 (coeff 1), z11 (coeff -1)
        U[u10][col] += 1
        U[u11][col] += 1
        V[v00][col] += 1
        W[w10][col] += 1
        W[w11][col] -= 1
        col += 1
        
        # Term 3: x00 * (y10 - y11)
        # U: u00 += 1
        # V: v10 += 1, v11 -= 1
        # W: z01 (coeff 1), z11 (coeff 1)
        U[u00][col] += 1
        V[v10][col] += 1
        V[v11][col] -= 1
        W[w01][col] += 1
        W[w11][col] += 1
        col += 1
        
        # Term 4: x11 * (y01 - y00)
        # U: u11 += 1
        # V: v01 += 1, v00 -= 1
        # W: z00 (coeff 1), z10 (coeff 1)
        U[u11][col] += 1
        V[v01][col] += 1
        V[v00][col] -= 1
        W[w00][col] += 1
        W[w10][col] += 1
        col += 1
        
        # Term 5: (x00 + x01) * y11
        # U: u00 += 1, u01 += 1
        # V: v11 += 1
        # W: z01 (coeff 1), z00 (coeff -1)
        U[u00][col] += 1
        U[u01][col] += 1
        V[v11][col] += 1
        W[w01][col] += 1
        W[w00][col] -= 1
        col += 1
        
        # Term 6: (x10 - x00) * (y00 + y01)
        # U: u10 += 1, u00 -= 1
        # V: v00 += 1, v01 += 1
        # W: z11 (coeff 1)
        U[u10][col] += 1
        U[u00][col] -= 1
        V[v00][col] += 1
        V[v01][col] += 1
        W[w11][col] += 1
        col += 1
        
        # Term 7: (x01 - x11) * (y10 + y11)
        # U: u01 += 1, u11 -= 1
        # V: v10 += 1, v11 += 1
        # W: z00 (coeff 1)
        U[u01][col] += 1
        U[u11][col] -= 1
        V[v10][col] += 1
        V[v11][col] += 1
        W[w00][col] += 1
        col += 1

    # Block 1: A11 B11 -> C11
    # A11: i in {0,1}, j in {0,1}
    # B11: j in {0,1}, k in {0,1}
    # C11: k in {0,1}, i in {0,1}
    strassen_2x2((0, 0), (0, 0), (0, 0))
    
    # Block 2: A12 B21 -> C11
    # A12: i in {0,1}, j in {2,3}
    # B21: j in {2,3}, k in {0,1}
    # C11: k in {0,1}, i in {0,1}
    strassen_2x2((0, 2), (2, 0), (0, 0))
    
    # Block 3: A11 B12 -> C12
    # A11: i in {0,1}, j in {0,1}
    # B12: j in {0,1}, k in {2,3}
    # C12: k in {2,3}, i in {0,1}
    strassen_2x2((0, 0), (0, 2), (2, 0))
    
    # Block 4: A12 B22 -> C12
    # A12: i in {0,1}, j in {2,3}
    # B22: j in {2,3}, k in {2,3}
    # C12: k in {2,3}, i in {0,1}
    strassen_2x2((0, 2), (2, 2), (2, 0))
    
    # Block 5: A21 B11 -> C21
    # A21: i in {2}, j in {0,1}
    # B11: j in {0,1}, k in {0,1}
    # C21: k in {0,1}, i in {2}
    # Rank 4 terms
    # A21 = [a0, a1] (global U indices: 2*m+0, 2*m+1)
    # B11 = [[b00, b01], [b10, b11]] (global V indices: 0*p+0, 0*p+1, 1*p+0, 1*p+1)
    # C21 = [c0, c1] (global W indices: 0*n+2, 1*n+2)
    
    u_a0 = 2 * m + 0
    u_a1 = 2 * m + 1
    v_b00 = 0 * p + 0
    v_b10 = 1 * p + 0
    v_b01 = 0 * p + 1
    v_b11 = 1 * p + 1
    w_c0 = 0 * n + 2
    w_c1 = 1 * n + 2
    
    # t1 = a0 b00 -> c0
    U[u_a0][col] += 1
    V[v_b00][col] += 1
    W[w_c0][col] += 1
    col += 1
    
    # t2 = a1 b10 -> c0
    U[u_a1][col] += 1
    V[v_b10][col] += 1
    W[w_c0][col] += 1
    col += 1
    
    # t3 = a0 b01 -> c1
    U[u_a0][col] += 1
    V[v_b01][col] += 1
    W[w_c1][col] += 1
    col += 1
    
    # t4 = a1 b11 -> c1
    U[u_a1][col] += 1
    V[v_b11][col] += 1
    W[w_c1][col] += 1
    col += 1
    
    # Block 6: A22 B21 -> C21
    # A22: i in {2}, j in {2,3}
    # B21: j in {2,3}, k in {0,1}
    # C21: k in {0,1}, i in {2}
    
    u_a2 = 2 * m + 2
    u_a3 = 2 * m + 3
    v_b20 = 2 * p + 0
    v_b30 = 3 * p + 0
    v_b21 = 2 * p + 1
    v_b31 = 3 * p + 1
    # w_c0, w_c1 same as above
    
    # t5 = a2 b20 -> c0
    U[u_a2][col] += 1
    V[v_b20][col] += 1
    W[w_c0][col] += 1
    col += 1
    
    # t6 = a3 b30 -> c0
    U[u_a3][col] += 1
    V[v_b30][col] += 1
    W[w_c0][col] += 1
    col += 1
    
    # t7 = a2 b21 -> c1
    U[u_a2][col] += 1
    V[v_b21][col] += 1
    W[w_c1][col] += 1
    col += 1
    
    # t8 = a3 b31 -> c1
    U[u_a3][col] += 1
    V[v_b31][col] += 1
    W[w_c1][col] += 1
    col += 1
    
    # Block 7: A21 B12 -> C22
    # A21: i in {2}, j in {0,1}
    # B12: j in {0,1}, k in {2,3}
    # C22: k in {2,3}, i in {2}
    
    # u_a0, u_a1 same as above
    v_b02 = 0 * p + 2
    v_b12 = 1 * p + 2
    v_b03 = 0 * p + 3
    v_b13 = 1 * p + 3
    w_c2 = 2 * n + 2
    w_c3 = 3 * n + 2
    
    # t9 = a0 b02 -> c2
    U[u_a0][col] += 1
    V[v_b02][col] += 1
    W[w_c2][col] += 1
    col += 1
    
    # t10 = a1 b12 -> c2
    U[u_a1][col] += 1
    V[v_b12][col] += 1
    W[w_c2][col] += 1
    col += 1
    
    # t11 = a0 b03 -> c3
    U[u_a0][col] += 1
    V[v_b03][col] += 1
    W[w_c3][col] += 1
    col += 1
    
    # t12 = a1 b13 -> c3
    U[u_a1][col] += 1
    V[v_b13][col] += 1
    W[w_c3][col] += 1
    col += 1
    
    # Block 8: A22 B22 -> C22
    # A22: i in {2}, j in {2,3}
    # B22: j in {2,3}, k in {2,3}
    # C22: k in {2,3}, i in {2}
    
    # u_a2, u_a3 same as above
    v_b22 = 2 * p + 2
    v_b32 = 3 * p + 2
    v_b23 = 2 * p + 3
    v_b33 = 3 * p + 3
    # w_c2, w_c3 same as above
    
    # t13 = a2 b22 -> c2
    U[u_a2][col] += 1
    V[v_b22][col] += 1
    W[w_c2][col] += 1
    col += 1
    
    # t14 = a3 b32 -> c2
    U[u_a3][col] += 1
    V[v_b32][col] += 1
    W[w_c2][col] += 1
    col += 1
    
    # t15 = a2 b23 -> c3
    U[u_a2][col] += 1
    V[v_b23][col] += 1
    W[w_c3][col] += 1
    col += 1
    
    # t16 = a3 b33 -> c3
    U[u_a3][col] += 1
    V[v_b33][col] += 1
    W[w_c3][col] += 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
