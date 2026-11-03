import time

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # U: (n*m) x rank
    # V: (m*p) x rank
    # W: (p*n) x rank
    
    U = []
    V = []
    W = []
    
    def add_strassen_block(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        """
        Adds 7 columns corresponding to Strassen decomposition for 2x2x2 block.
        a_rows: [r1, r2] indices in A
        a_cols: [c1, c2] indices in A
        b_rows: [r1, r2] indices in B
        b_cols: [c1, c2] indices in B
        c_rows: [r1, r2] indices in C (k indices)
        c_cols: [c1, c2] indices in C (i indices)
        """
        # M1 = (A00 + A11) * (B00 + B11)
        u1 = [0]*(n*m); v1 = [0]*(m*p); w1 = [0]*(p*n)
        u1[a_rows[0]*m + a_cols[0]] = 1
        u1[a_rows[1]*m + a_cols[1]] = 1
        v1[b_rows[0]*p + b_cols[0]] = 1
        v1[b_rows[1]*p + b_cols[1]] = 1
        w1[c_rows[0]*n + c_cols[0]] = 1 # C00
        w1[c_rows[1]*n + c_cols[1]] = 1 # C11
        U.append(u1); V.append(v1); W.append(w1)

        # M2 = (A10 + A11) * B00
        u2 = [0]*(n*m); v2 = [0]*(m*p); w2 = [0]*(p*n)
        u2[a_rows[1]*m + a_cols[0]] = 1
        u2[a_rows[1]*m + a_cols[1]] = 1
        v2[b_rows[0]*p + b_cols[0]] = 1
        w2[c_rows[1]*n + c_cols[0]] = 1 # C10
        w2[c_rows[1]*n + c_cols[1]] = -1 # C11
        U.append(u2); V.append(v2); W.append(w2)

        # M3 = A00 * (B01 - B11)
        u3 = [0]*(n*m); v3 = [0]*(m*p); w3 = [0]*(p*n)
        u3[a_rows[0]*m + a_cols[0]] = 1
        v3[b_rows[0]*p + b_cols[1]] = 1
        v3[b_rows[1]*p + b_cols[1]] = -1
        w3[c_rows[0]*n + c_cols[1]] = 1 # C01
        w3[c_rows[1]*n + c_cols[1]] = 1 # C11
        U.append(u3); V.append(v3); W.append(w3)

        # M4 = A11 * (B10 - B00)
        u4 = [0]*(n*m); v4 = [0]*(m*p); w4 = [0]*(p*n)
        u4[a_rows[1]*m + a_cols[1]] = 1
        v4[b_rows[1]*p + b_cols[0]] = 1
        v4[b_rows[0]*p + b_cols[0]] = -1
        w4[c_rows[0]*n + c_cols[0]] = 1 # C00
        w4[c_rows[1]*n + c_cols[0]] = 1 # C10
        U.append(u4); V.append(v4); W.append(w4)

        # M5 = (A00 + A01) * B11
        u5 = [0]*(n*m); v5 = [0]*(m*p); w5 = [0]*(p*n)
        u5[a_rows[0]*m + a_cols[0]] = 1
        u5[a_rows[0]*m + a_cols[1]] = 1
        v5[b_rows[1]*p + b_cols[1]] = 1
        w5[c_rows[0]*n + c_cols[0]] = -1 # C00
        w5[c_rows[0]*n + c_cols[1]] = 1  # C01
        U.append(u5); V.append(v5); W.append(w5)

        # M6 = (A10 - A00) * (B00 + B01)
        u6 = [0]*(n*m); v6 = [0]*(m*p); w6 = [0]*(p*n)
        u6[a_rows[1]*m + a_cols[0]] = 1
        u6[a_rows[0]*m + a_cols[0]] = -1
        v6[b_rows[0]*p + b_cols[0]] = 1
        v6[b_rows[0]*p + b_cols[1]] = 1
        w6[c_rows[1]*n + c_cols[1]] = 1 # C11
        U.append(u6); V.append(v6); W.append(w6)

        # M7 = (A01 - A11) * (B10 + B11)
        u7 = [0]*(n*m); v7 = [0]*(m*p); w7 = [0]*(p*n)
        u7[a_rows[0]*m + a_cols[1]] = 1
        u7[a_rows[1]*m + a_cols[1]] = -1
        v7[b_rows[1]*p + b_cols[0]] = 1
        v7[b_rows[1]*p + b_cols[1]] = 1
        w7[c_rows[0]*n + c_cols[0]] = 1 # C00
        U.append(u7); V.append(v7); W.append(w7)

    def add_schoolbook_term(i, j, k):
        """
        Adds 1 column for term A[i,j] * B[j,k] contributing to C[k,i]
        """
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[i*m + j] = 1
        v[j*p + k] = 1
        w[k*n + i] = 1
        U.append(u); V.append(v); W.append(w)

    # 1. P1 = A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    add_strassen_block([0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1])
    
    # 2. P2 = A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    add_strassen_block([0, 1], [2, 3], [2, 3], [0, 1], [0, 1], [0, 1])
    
    # 3. P3 = A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    add_strassen_block([0, 1], [0, 1], [0, 1], [2, 3], [2, 3], [0, 1])
    
    # 4. P4 = A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    add_strassen_block([0, 1], [2, 3], [2, 3], [2, 3], [2, 3], [0, 1])

    # Top-right: C[0, 4] and C[1, 4]
    # C[0, 4] = A[0,0]B[0,4] + A[0,1]B[1,4] + A[0,2]B[2,4] + A[0,3]B[3,4]
    add_schoolbook_term(0, 0, 4)
    add_schoolbook_term(0, 1, 4)
    add_schoolbook_term(0, 2, 4)
    add_schoolbook_term(0, 3, 4)
    # C[1, 4] = A[1,0]B[0,4] + A[1,1]B[1,4] + A[1,2]B[2,4] + A[1,3]B[3,4]
    add_schoolbook_term(1, 0, 4)
    add_schoolbook_term(1, 1, 4)
    add_schoolbook_term(1, 2, 4)
    add_schoolbook_term(1, 3, 4)

    # Bottom-left: C[2, 0] and C[2, 1]
    # C[2, 0] = A[2,0]B[0,0] + A[2,1]B[1,0] + A[2,2]B[2,0] + A[2,3]B[3,0]
    add_schoolbook_term(2, 0, 0)
    add_schoolbook_term(2, 1, 0)
    add_schoolbook_term(2, 2, 0)
    add_schoolbook_term(2, 3, 0)
    # C[2, 1] = A[2,0]B[0,1] + A[2,1]B[1,1] + A[2,2]B[2,1] + A[2,3]B[3,1]
    add_schoolbook_term(2, 0, 1)
    add_schoolbook_term(2, 1, 1)
    add_schoolbook_term(2, 2, 1)
    add_schoolbook_term(2, 3, 1)

    # Bottom-middle: C[2, 2] and C[2, 3]
    # C[2, 2] = A[2,0]B[0,2] + A[2,1]B[1,2] + A[2,2]B[2,2] + A[2,3]B[3,2]
    add_schoolbook_term(2, 0, 2)
    add_schoolbook_term(2, 1, 2)
    add_schoolbook_term(2, 2, 2)
    add_schoolbook_term(2, 3, 2)
    # C[2, 3] = A[2,0]B[0,3] + A[2,1]B[1,3] + A[2,2]B[2,3] + A[2,3]B[3,3]
    add_schoolbook_term(2, 0, 3)
    add_schoolbook_term(2, 1, 3)
    add_schoolbook_term(2, 2, 3)
    add_schoolbook_term(2, 3, 3)

    # Bottom-right: C[2, 4]
    # C[2, 4] = A[2,0]B[0,4] + A[2,1]B[1,4] + A[2,2]B[2,4] + A[2,3]B[3,4]
    add_schoolbook_term(2, 0, 4)
    add_schoolbook_term(2, 1, 4)
    add_schoolbook_term(2, 2, 4)
    add_schoolbook_term(2, 3, 4)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
