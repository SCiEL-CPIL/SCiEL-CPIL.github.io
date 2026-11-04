"""Matrix multiplication 4x5 x 5x5 via Strassen-based recursive decomposition."""
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # Pad to next power of 2 for Strassen recursion
    def next_pow2(x):
        p = 1
        while p < x:
            p *= 2
        return p
    
    N = max(n, m, p)
    N = next_pow2(N) if N > 1 else 1
    
    # We'll build the decomposition for N x N x N matrix multiplication
    # then extract the (n, m, p) sub-decomposition
    
    # Strassen decomposition for size N
    # Each rank-1 term is a triple of linear forms
    # U: linear form in A (size N x N)
    # V: linear form in B (size N x N)  
    # W: linear form in C (size N x N)
    
    # We represent each linear form as a dict from (row, col) to coefficient
    # This allows us to extract sub-forms for smaller matrices
    
    def strassen_terms(N):
        """Generate the Strassen decomposition for N x N x N matrix multiplication.
        Returns list of (U_form, V_form, W_form) where each form is a dict {(i,j): coeff}.
        """
        if N == 1:
            return [({(0,0): 1}, {(0,0): 1}, {(0,0): 1})]
        
        half = N // 2
        # Get Strassen terms for half-size
        sub_terms = strassen_terms(half)
        
        # Strassen: 7 multiplications
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        terms = []
        
        for (u_form, v_form, w_form) in sub_terms:
            # Each sub-term contributes to each of the 7 M's
            
            # M1: (A11+A22)(B11+B22) -> C11, C22
            u1 = dict(u_form)
            v1 = dict(v_form)
            # A11+A22: rows 0..half-1 and half..N-1 for first half of cols
            for (i, j), c in u_form.items():
                u1[(i, j)] = u1.get((i, j), 0) + c
                u1[(i + half, j)] = u1.get((i + half, j), 0) + c
            for (i, j), c in v_form.items():
                v1[(i, j)] = v1.get((i, j), 0) + c
                v1[(i + half, j)] = v1.get((i + half, j), 0) + c
            # W for M1: C11 and C22
            w1 = {}
            for (i, j), c in w_form.items():
                w1[(i, j)] = w1.get((i, j), 0) + c
                w1[(i + half, j + half)] = w1.get((i + half, j + half), 0) + c
            terms.append((u1, v1, w1))
            
            # M2: (A21+A22)B11 -> C21
            u2 = dict(u_form)
            v2 = dict(v_form)
            for (i, j), c in u_form.items():
                u2[(i + half, j)] = u2.get((i + half, j), 0) + c
                u2[(i + half, j + half)] = u2.get((i + half, j + half), 0) + c
            # B11: rows 0..half-1, cols 0..half-1
            # v_form already has this structure
            # W for M2: C21
            w2 = {}
            for (i, j), c in w_form.items():
                w2[(i + half, j)] = w2.get((i + half, j), 0) + c
            terms.append((u2, v2, w2))
            
            # M3: A11(B21-B22) -> C12, C21
            u3 = dict(u_form)
            v3 = dict(v_form)
            # A11: rows 0..half-1, cols 0..half-1 (already in u_form)
            # B21-B22: rows half..N-1
            for (i, j), c in v_form.items():
                v3[(i + half, j)] = v3.get((i + half, j), 0) + c
                v3[(i + half, j + half)] = v3.get((i + half, j + half), 0) - c
            # W for M3: C12 and C21
            w3 = {}
            for (i, j), c in w_form.items():
                w3[(i, j + half)] = w3.get((i, j + half), 0) + c
                w3[(i + half, j)] = w3.get((i + half, j), 0) + c
            terms.append((u3, v3, w3))
            
            # M4: A22(B12-B11) -> C11, C21
            u4 = dict(u_form)
            v4 = dict(v_form)
            # A22: rows half..N-1, cols half..N-1
            for (i, j), c in u_form.items():
                u4[(i + half, j + half)] = u4.get((i + half, j + half), 0) + c
            # B12-B11: rows 0..half-1
            for (i, j), c in v_form.items():
                v4[(i, j)] = v4.get((i, j), 0) - c
                v4[(i, j + half)] = v4.get((i, j + half), 0) + c
            # W for M4: C11 and C21
            w4 = {}
            for (i, j), c in w_form.items():
                w4[(i, j)] = w4.get((i, j), 0) + c
                w4[(i + half, j)] = w4.get((i + half, j), 0) + c
            terms.append((u4, v4, w4))
            
            # M5: (A11+A12)B22 -> C11
            u5 = dict(u_form)
            v5 = dict(v_form)
            # A11+A12: cols 0..half-1 and half..N-1
            for (i, j), c in u_form.items():
                u5[(i, j)] = u5.get((i, j), 0) + c
                u5[(i, j + half)] = u5.get((i, j + half), 0) + c
            # B22: rows half..N-1, cols half..N-1
            for (i, j), c in v_form.items():
                v5[(i + half, j + half)] = v5.get((i + half, j + half), 0) + c
            # W for M5: C11 with negative sign
            w5 = {}
            for (i, j), c in w_form.items():
                w5[(i, j)] = w5.get((i, j), 0) - c
            terms.append((u5, v5, w5))
            
            # M6: (A21-A11)(B11+B12) -> C22
            u6 = dict(u_form)
            v6 = dict(v_form)
            # A21-A11: rows half..N-1 minus rows 0..half-1
            for (i, j), c in u_form.items():
                u6[(i, j)] = u6.get((i, j), 0) - c
                u6[(i + half, j)] = u6.get((i + half, j), 0) + c
            # B11+B12: cols 0..half-1 and half..N-1
            for (i, j), c in v_form.items():
                v6[(i, j)] = v6.get((i, j), 0) + c
                v6[(i, j + half)] = v6.get((i, j + half), 0) + c
            # W for M6: C22
            w6 = {}
            for (i, j), c in w_form.items():
                w6[(i + half, j + half)] = w6.get((i + half, j + half), 0) + c
            terms.append((u6, v6, w6))
            
            # M7: (A12-A22)(B21+B22) -> C11
            u7 = dict(u_form)
            v7 = dict(v_form)
            # A12-A22: cols half..N-1 minus rows half..N-1
            for (i, j), c in u_form.items():
                u7[(i, j + half)] = u7.get((i, j + half), 0) + c
                u7[(i + half, j + half)] = u7.get((i + half, j + half), 0) - c
            # B21+B22: rows half..N-1
            for (i, j), c in v_form.items():
                v7[(i + half, j)] = v7.get((i + half, j), 0) + c
                v7[(i + half, j + half)] = v7.get((i + half, j + half), 0) + c
            # W for M7: C11
            w7 = {}
            for (i, j), c in w_form.items():
                w7[(i, j)] = w7.get((i, j), 0) + c
            terms.append((u7, v7, w7))
        
        return terms
    
    # Generate Strassen terms for size N
    strassen = strassen_terms(N)
    
    # Now extract the (n, m, p) sub-decomposition
    # A is n x m, B is m x p, C is p x n
    # In the N x N x N problem:
    #   A occupies rows 0..n-1, cols 0..m-1 of the N x N matrix
    #   B occupies rows 0..m-1, cols 0..p-1 of the N x N matrix
    #   C occupies rows 0..p-1, cols 0..n-1 of the N x N matrix
    
    # U factor: shape (n*m, rank) - indexed by (i, j) for i in range(n), j in range(m)
    # V factor: shape (m*p, rank) - indexed by (j, k) for j in range(m), k in range(p)
    # W factor: shape (p*n, rank) - indexed by (k, i) for k in range(p), i in range(n)
    
    rank = len(strassen)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u_form, v_form, w_form) in enumerate(strassen):
        # Extract U entries: for (i, j) in u_form, if i < n and j < m
        for (i, j), coeff in u_form.items():
            if 0 <= i < n and 0 <= j < m:
                U[i * m + j][r] = coeff
        
        # Extract V entries: for (j, k) in v_form, if j < m and k < p
        for (j, k), coeff in v_form.items():
            if 0 <= j < m and 0 <= k < p:
                V[j * p + k][r] = coeff
        
        # Extract W entries: for (k, i) in w_form, if k < p and i < n
        # W index is [k*n + i, r]
        for (k, i), coeff in w_form.items():
            if 0 <= k < p and 0 <= i < n:
                W[k * n + i][r] = coeff
    
    # Remove zero columns to minimize rank
    keep = []
    for r in range(rank):
        u_col = [U[i][r] for i in range(n * m)]
        v_col = [V[i][r] for i in range(m * p)]
        w_col = [W[i][r] for i in range(p * n)]
        if any(c != 0 for c in u_col) and any(c != 0 for c in v_col) and any(c != 0 for c in w_col):
            keep.append(r)
    
    if len(keep) < rank:
        U = [[U[i][r] for r in keep] for i in range(n * m)]
        V = [[V[i][r] for r in keep] for i in range(m * p)]
        W = [[W[i][r] for r in keep] for i in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
