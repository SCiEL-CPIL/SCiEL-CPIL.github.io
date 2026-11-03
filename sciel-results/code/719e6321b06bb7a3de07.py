import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # We will build the U, V, W matrices by appending columns for each term.
    # Each term is a rank-1 tensor: u * v * w^T
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_term(u_idx, v_idx, w_idx, coeff=1):
        """
        Adds a rank-1 term to the decomposition.
        u_idx: index in U (i*m + j)
        v_idx: index in V (j*p + k)
        w_idx: index in W (k*n + i)
        """
        U_list.append([0] * (n*m) if not U_list else None) # Placeholder, we'll build columns
        # Actually, it's easier to build columns as lists of zeros and set the value.
        
    # Better approach: Build U, V, W as lists of columns.
    # Each column is a vector of length n*m, m*p, p*n respectively.
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_idx, v_idx, w_idx, coeff=1):
        u_vec = [0] * (n*m)
        v_vec = [0] * (m*p)
        w_vec = [0] * (p*n)
        
        u_vec[u_idx] = coeff
        v_vec[v_idx] = 1
        w_vec[w_idx] = 1
        
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
        
    def add_strassen(a_start, b_start, c_start):
        """
        Adds a 2x2x2 Strassen decomposition.
        a_start: (i, j) start index for A (2x2 block)
        b_start: (j, k) start index for B (2x2 block)
        c_start: (k, i) start index for C (2x2 block)
        
        A block:
        a11 a12
        a21 a22
        Indices: (i, j), (i, j+1), (i+1, j), (i+1, j+1)
        
        B block:
        b11 b12
        b21 b22
        Indices: (j, k), (j, k+1), (j+1, k), (j+1, k+1)
        
        C block:
        c11 c12
        c21 c22
        Indices: (k, i), (k, i+1), (k+1, i), (k+1, i+1)
        """
        i, j = a_start
        jb, kb = b_start
        kc, ic = c_start
        
        # A indices
        a11 = i * m + j
        a12 = i * m + (j + 1)
        a21 = (i + 1) * m + j
        a22 = (i + 1) * m + (j + 1)
        
        # B indices
        b11 = jb * p + kb
        b12 = jb * p + (kb + 1)
        b21 = (jb + 1) * p + kb
        b22 = (jb + 1) * p + (kb + 1)
        
        # C indices
        c11 = kc * n + ic
        c12 = kc * n + (ic + 1)
        c21 = (kc + 1) * n + ic
        c22 = (kc + 1) * n + (ic + 1)
        
        # Strassen terms:
        # m1 = (a11+a12)(b11+b22) -> U: a11, a12; V: b11, b22; W: c11, c12, c21, c22
        # We need to distribute the sum in U and V to single indices?
        # No, U and V can have multiple non-zeros.
        # But the rank-1 term requires U, V, W to be single non-zero vectors?
        # No, U, V, W are vectors. They can have multiple non-zero entries.
        # The term is sum_u sum_v sum_w U[u] V[v] W[w] T[u,v,w] = 1?
        # The decomposition is T = sum_r U_r V_r W_r^T.
        # So U_r is a vector.
        
        # For Strassen:
        # Term 1: (a11+a12) * (b11+b22) * (c11 - c22 ...?)
        # Let's write out the 7 terms explicitly.
        
        # m1 = (a11 + a12) (b11 + b22)
        # Contribution to C:
        # c11 += m1
        # c12 += m1
        # c21 -= m1
        # c22 -= m1
        # Wait, Strassen formulas:
        # C11 = m1 + m4 - m5 + m7
        # C12 = m3 + m5
        # C21 = m2 + m5
        # C22 = m1 - m2 + m4 + m6
        
        # Let's add 7 terms.
        
        # Term 1: m1
        u1 = [0]*(n*m); u1[a11] = 1; u1[a12] = 1
        v1 = [0]*(m*p); v1[b11] = 1; v1[b22] = 1
        w1 = [0]*(p*n); w1[c11] = 1; w1[c12] = 1; w1[c21] = -1; w1[c22] = -1
        U_cols.append(u1); V_cols.append(v1); W_cols.append(w1)
        
        # Term 2: m2 = (a21+a22) b11
        u2 = [0]*(n*m); u2[a21] = 1; u2[a22] = 1
        v2 = [0]*(m*p); v2[b11] = 1
        w2 = [0]*(p*n); w2[c21] = 1; w2[c22] = -1
        U_cols.append(u2); V_cols.append(v2); W_cols.append(w2)
        
        # Term 3: m3 = a11 (b12 - b22)
        u3 = [0]*(n*m); u3[a11] = 1
        v3 = [0]*(m*p); v3[b12] = 1; v3[b22] = -1
        w3 = [0]*(p*n); w3[c12] = 1
        U_cols.append(u3); V_cols.append(v3); W_cols.append(w3)
        
        # Term 4: m4 = a22 (b21 - b11)
        u4 = [0]*(n*m); u4[a22] = 1
        v4 = [0]*(m*p); v4[b21] = 1; v4[b11] = -1
        w4 = [0]*(p*n); w4[c11] = 1; w4[c22] = 1
        U_cols.append(u4); V_cols.append(v4); W_cols.append(w4)
        
        # Term 5: m5 = (a11+a22) b22
        u5 = [0]*(n*m); u5[a11] = 1; u5[a22] = 1
        v5 = [0]*(m*p); v5[b22] = 1
        w5 = [0]*(p*n); w5[c12] = 1; w5[c21] = 1; w5[c11] = -1
        U_cols.append(u5); V_cols.append(v5); W_cols.append(w5)
        
        # Term 6: m6 = (a12-a21) (b21+b22)
        u6 = [0]*(n*m); u6[a12] = 1; u6[a21] = -1
        v6 = [0]*(m*p); v6[b21] = 1; v6[b22] = 1
        w6 = [0]*(p*n); w6[c22] = 1
        U_cols.append(u6); V_cols.append(v6); W_cols.append(w6)
        
        # Term 7: m7 = (a11-a12) b12
        u7 = [0]*(n*m); u7[a11] = 1; u7[a12] = -1
        v7 = [0]*(m*p); v7[b12] = 1
        w7 = [0]*(p*n); w7[c11] = 1
        U_cols.append(u7); V_cols.append(v7); W_cols.append(w7)
        
    # 1. Top 2 rows of A, Left 4 cols of B
    # A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    add_strassen((0, 0), (0, 0), (0, 0))
    # A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    add_strassen((0, 0), (0, 2), (2, 0))
    # A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    add_strassen((0, 2), (2, 0), (0, 0))
    # A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    add_strassen((0, 2), (2, 2), (2, 0))
    
    # 2. Bottom 1 row of A, Left 4 cols of B
    # A[2, :] * B[:, 0:4] -> C[2, 0:4]
    for j in range(4):
        for k in range(4):
            add_term(2 * m + j, j * p + k, k * n + 2)
            
    # 3. Top 2 rows of A, Right 2 cols of B
    # A[0:2, 0:2] * B[0:2, 4:6] -> C[0:2, 4:6]
    add_strassen((0, 0), (0, 4), (4, 0))
    # A[0:2, 2:4] * B[2:4, 4:6] -> C[0:2, 4:6]
    add_strassen((0, 2), (2, 4), (4, 0))
    
    # 4. Bottom 1 row of A, Right 2 cols of B
    # A[2, :] * B[:, 4:6] -> C[2, 4:6]
    for j in range(4):
        for k in range(4, 6):
            add_term(2 * m + j, j * p + k, k * n + 2)
            
    # Convert lists to matrices
    rank = len(U_cols)
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    for r in range(rank):
        for i in range(n*m):
            U[i, r] = U_cols[r][i]
        for i in range(m*p):
            V[i, r] = V_cols[r][i]
        for i in range(p*n):
            W[i, r] = W_cols[r][i]
            
    return {"U": U, "V": V, "W": W}
