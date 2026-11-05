"""Matrix multiplication 3,4,7 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 7
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_term(u_val, v_val, w_val):
        U_list.append(u_val)
        V_list.append(v_val)
        W_list.append(w_val)
    
    def add_schoolbook(i, j, k):
        """Add a single rank-1 term for the (i,j,k) triple."""
        u = [0] * (n * m)
        v = [0] * (m * p)
        w = [0] * (p * n)
        u[i * m + j] = 1
        v[j * p + k] = 1
        w[k * n + i] = 1
        add_term(u, v, w)
    
    def add_strassen_block(i0, j0, k0):
        """
        Add 7 Strassen terms for a 2x2x2 block starting at (i0, j0, k0).
        
        Strassen formulas:
        M1 = (A11 + A22) * (B11 + B22)
        M2 = (A21 + A22) * B11
        M3 = A11 * (B12 - B22)
        M4 = A22 * (B21 - B11)
        M5 = (A11 + A12) * B22
        M6 = (A21 - A11) * (B11 + B12)
        M7 = (A12 - A22) * (B21 + B22)
        
        C11 = M1 + M4 - M5 + M7
        C12 = M3 + M5
        C21 = M2 + M4
        C22 = M1 - M2 + M3 + M6
        """
        # Define the 7 terms
        # Each term: (u_dict, v_dict, w_dict) where dict maps index -> coefficient
        
        # M1 = (A11+A22)(B11+B22)
        # U: A11 at (i0,j0), A22 at (i0+1,j0+1)
        # V: B11 at (j0,k0), B22 at (j0+1,k0+1)
        # W: contributes to C11 (k0,i0) and C22 (k0+1,i0+1)
        u1 = {(i0, j0): 1, (i0+1, j0+1): 1}
        v1 = {(j0, k0): 1, (j0+1, k0+1): 1}
        w1 = {(k0, i0): 1, (k0+1, i0+1): 1}
        
        # M2 = (A21+A22) * B11
        # U: A21 at (i0+1,j0), A22 at (i0+1,j0+1)
        # V: B11 at (j0,k0)
        # W: contributes to C21 (k0, i0+1)
        u2 = {(i0+1, j0): 1, (i0+1, j0+1): 1}
        v2 = {(j0, k0): 1}
        w2 = {(k0, i0+1): 1}
        
        # M3 = A11 * (B12 - B22)
        # U: A11 at (i0,j0)
        # V: B12 at (j0,k0+1), -B22 at (j0+1,k0+1)
        # W: contributes to C12 (k0+1, i0)
        u3 = {(i0, j0): 1}
        v3 = {(j0, k0+1): 1, (j0+1, k0+1): -1}
        w3 = {(k0+1, i0): 1}
        
        # M4 = A22 * (B21 - B11)
        # U: A22 at (i0+1,j0+1)
        # V: B21 at (j0+1,k0), -B11 at (j0,k0)
        # W: contributes to C11 (k0,i0) and C21 (k0,i0+1)
        u4 = {(i0+1, j0+1): 1}
        v4 = {(j0+1, k0): 1, (j0, k0): -1}
        w4 = {(k0, i0): 1, (k0, i0+1): 1}
        
        # M5 = (A11+A12) * B22
        # U: A11 at (i0,j0), A12 at (i0,j0+1)
        # V: B22 at (j0+1,k0+1)
        # W: contributes to C12 (k0+1,i0) with -1, and C22 (k0+1,i0+1) with +1... wait
        # C11 = M1 + M4 - M5 + M7, so M5 contributes -1 to C11
        # C12 = M3 + M5, so M5 contributes +1 to C12
        # But M5 is a single rank-1 term, it can only have one W vector.
        # The W vector for M5 must encode both contributions.
        # Actually no - each M is a scalar product, and it gets added to specific C entries.
        # In the tensor decomposition, the W vector for term M5 should have:
        # W[k0+1, i0] = 1 (for C12)
        # W[k0, i0] = -1 (for C11)
        # Wait, but M5 = (A11+A12)*B22 is a single scalar. It's added as:
        # C11 += -M5, C12 += +M5
        # So the W vector needs entries at (k0, i0) = -1 and (k0+1, i0) = 1
        u5 = {(i0, j0): 1, (i0, j0+1): 1}
        v5 = {(j0+1, k0+1): 1}
        w5 = {(k0, i0): -1, (k0+1, i0): 1}
        
        # M6 = (A21-A11) * (B11+B12)
        # C22 += M6
        u6 = {(i0+1, j0): 1, (i0, j0): -1}
        v6 = {(j0, k0): 1, (j0, k0+1): 1}
        w6 = {(k0+1, i0+1): 1}
        
        # M7 = (A12-A22) * (B21+B22)
        # C11 += M7
        u7 = {(i0, j0+1): 1, (i0+1, j0+1): -1}
        v7 = {(j0+1, k0): 1, (j0+1, k0+1): 1}
        w7 = {(k0, i0): 1}
        
        terms = [(u1, v1, w1), (u2, v2, w2), (u3, v3, w3), 
                 (u4, v4, w4), (u5, v5, w5), (u6, v6, w6), (u7, v7, w7)]
        
        for u_dict, v_dict, w_dict in terms:
            u = [0] * (n * m)
            v = [0] * (m * p)
            w = [0] * (p * n)
            for (ii, jj), coeff in u_dict.items():
                u[ii * m + jj] = coeff
            for (jj, kk), coeff in v_dict.items():
                v[jj * p + kk] = coeff
            for (kk, ii), coeff in w_dict.items():
                w[kk * n + ii] = coeff
            add_term(u, v, w)
    
    # Apply Strassen to 2x2 blocks where possible
    # n=3, m=4, p=7
    # Use Strassen for i in {0,1}, j in {0,1} or {2,3}, k in {0,1} or {2,3}
    
    # 4 Strassen blocks
    add_strassen_block(0, 0, 0)  # i∈{0,1}, j∈{0,1}, k∈{0,1}
    add_strassen_block(0, 2, 0)  # i∈{0,1}, j∈{2,3}, k∈{0,1}
    add_strassen_block(0, 0, 2)  # i∈{0,1}, j∈{0,1}, k∈{2,3}
    add_strassen_block(0, 2, 2)  # i∈{0,1}, j∈{2,3}, k∈{2,3}
    
    # Schoolbook for remaining (i,j,k) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Skip if covered by any Strassen block
                if i in [0, 1] and j in [0, 1, 2, 3] and k in [0, 1, 2, 3]:
                    continue
                add_schoolbook(i, j, k)
    
    rank = len(U_list)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        for i in range(n * m):
            U[i][r] = U_list[r][i]
        for j in range(m * p):
            V[j][r] = V_list[r][j]
        for k in range(p * n):
            W[k][r] = W_list[r][k]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
