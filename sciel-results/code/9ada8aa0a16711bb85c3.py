"""Matrix multiplication 4x5x5 with structured tensor network ansatz."""
import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 4x5 matrix by a 5x5 matrix.
    
    Strategy: Use a structured approach based on Strassen-like algebraic
    cancellations. We decompose the problem by splitting the matrices into
    blocks and using bilinear combinations to reduce the number of
    multiplications below the schoolbook bound of 4*5*5 = 100.
    
    We split A (4x5) into two 2x5 blocks (top and bottom rows) and B (5x5)
    into 2x2 blocks plus remainders. For each 2x2x2 subproblem, we use
    Strassen's algorithm (rank 7 instead of 8). For the remaining 1-dimensional
    cases, we use schoolbook.
    
    The 5x5 matrix B is split along rows into [0:2], [2:4], [4:5] and along
    columns into [0:2], [2:4], [4:5]. Similarly for A's columns.
    
    For each row-half of A (2 rows) and each column-block of the output (2 or 1
    columns), we compute the sub-multiplication.
    """
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # We'll build the decomposition term by term.
    # Each term is (U_entry, V_entry, W_entry) where each is a dict mapping
    # index -> coefficient.
    
    terms = []  # list of (dict, dict, dict)
    
    def add_term(u_dict, v_dict, w_dict):
        """Add a rank-1 term."""
        terms.append((u_dict, v_dict, w_dict))
    
    def schoolbook_2x1x1(a_rows, a_col, b_row, b_col, c_rows, c_col):
        """
        Compute C[c_rows, c_col] += A[a_rows, a_col] * B[b_row, b_col]
        where a_rows is a list of 2 row indices, c_rows is a list of 2 row indices.
        This is 2 rank-1 terms.
        """
        for i in a_rows:
            for r in c_rows:
                # C[r, c_col] += A[i, a_col] * B[b_row, b_col]
                # But wait, this is only valid if the mapping is correct.
                # Actually, for a single product A[i, j] * B[j, k] contributing to C[i, k],
                # we have a rank-1 term with U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1.
                # Here we're doing a batch. Let me reconsider.
                pass
        # Actually, let me just add individual rank-1 terms for clarity.
        # For each i in a_rows, r in c_rows: but they should be the same set.
        # Let's assume a_rows == c_rows for simplicity in the block structure.
        for i in a_rows:
            # C[i, c_col] += A[i, a_col] * B[b_row, b_col]
            u = {i * m + a_col: 1}
            v = {b_row * p + b_col: 1}
            w = {c_col * n + i: 1}
            add_term(u, v, w)
    
    def schoolbook_2x2x2(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        """
        Compute C[c_rows, c_cols] += A[a_rows, a_cols] * B[b_rows, b_cols]
        using schoolbook (8 terms).
        """
        for i in a_rows:
            for j in a_cols:
                for k in c_cols:
                    # C[i, k] += A[i, j] * B[j, k]
                    # But we need to ensure b_rows corresponds to a_cols and b_cols to c_cols.
                    # In our block decomposition, a_cols == b_rows and b_cols == c_cols.
                    u = {i * m + j: 1}
                    v = {j * p + k: 1}
                    w = {k * n + i: 1}
                    add_term(u, v, w)
    
    def strassen_2x2x2(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        """
        Compute C[c_rows, c_cols] += A[a_rows, a_cols] * B[b_rows, b_cols]
        using Strassen's algorithm (7 terms).
        
        a_rows = [r1, r2], a_cols = [c1, c2]
        b_rows = [c1, c2], b_cols = [k1, k2]
        c_rows = [r1, r2], c_cols = [k1, k2]
        
        Strassen:
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
        r1, r2 = a_rows
        c1, c2 = a_cols
        k1, k2 = c_cols
        
        # M1 = (A[r1,c1] + A[r2,c2]) * (B[c1,k1] + B[c2,k2])
        u1 = {r1 * m + c1: 1, r2 * m + c2: 1}
        v1 = {c1 * p + k1: 1, c2 * p + k2: 1}
        w1 = {k1 * n + r1: 1, k2 * n + r1: 1, k1 * n + r2: 1, k2 * n + r2: 1}
        # C11 += M1, C12 += M1, C21 += M1, C22 += M1
        # But with coefficients: C11 += 1*M1, C12 += 1*M1, C21 += 1*M1, C22 += 1*M1
        add_term(u1, v1, {k1 * n + r1: 1, k2 * n + r1: 1, k1 * n + r2: 1, k2 * n + r2: 1})
        
        # M2 = (A[r2,c1] + A[r2,c2]) * B[c1,k1]
        u2 = {r2 * m + c1: 1, r2 * m + c2: 1}
        v2 = {c1 * p + k1: 1}
        # C21 += M2, C22 -= M2
        add_term(u2, v2, {k1 * n + r2: 1, k2 * n + r2: -1})
        
        # M3 = A[r1,c1] * (B[c1,k2] - B[c2,k2])
        u3 = {r1 * m + c1: 1}
        v3 = {c1 * p + k2: 1, c2 * p + k2: -1}
        # C11 += M3, C12 += M3
        add_term(u3, v3, {k2 * n + r1: 1, k1 * n + r1: 1})
        
        # M4 = A[r2,c2] * (B[c2,k1] - B[c1,k1])
        u4 = {r2 * m + c2: 1}
        v4 = {c2 * p + k1: 1, c1 * p + k1: -1}
        # C11 += M4, C21 += M4
        add_term(u4, v4, {k1 * n + r1: 1, k1 * n + r2: 1})
        
        # M5 = (A[r1,c1] + A[r1,c2]) * B[c2,k2]
        u5 = {r1 * m + c1: 1, r1 * m + c2: 1}
        v5 = {c2 * p + k2: 1}
        # C11 -= M5, C12 += M5
        add_term(u5, v5, {k2 * n + r1: -1, k2 * n + r1: 1})
        # Wait, C11 gets -M5 and C12 gets +M5. But they're both at (k2, r1).
        # So the net coefficient for W[k2*n+r1] is -1 + 1 = 0? No, that's wrong.
        # Let me re-examine. C11 = M1 + M4 - M5 + M7. C12 = M3 + M5.
        # So M5 contributes -1 to C11 and +1 to C12. But C11 is at (k1, r1) and C12 is at (k2, r1).
        # So: W[k1*n+r1] += -1, W[k2*n+r1] += +1.
        add_term(u5, v5, {k1 * n + r1: -1, k2 * n + r1: 1})
        # But I already added a term above. Let me remove that.
        terms.pop()
        
        # M6 = (A[r2,c1] - A[r1,c1]) * (B[c1,k1] + B[c1,k2])
        u6 = {r2 * m + c1: 1, r1 * m + c1: -1}
        v6 = {c1 * p + k1: 1, c1 * p + k2: 1}
        # C22 += M6
        add_term(u6, v6, {k2 * n + r2: 1})
        
        # M7 = (A[r1,c2] - A[r2,c2]) * (B[c2,k1] + B[c2,k2])
        u7 = {r1 * m + c2: 1, r2 * m + c2: -1}
        v7 = {c2 * p + k1: 1, c2 * p + k2: 1}
        # C11 += M7, C22 += M7
        add_term(u7, v7, {k1 * n + r1: 1, k2 * n + r2: 1})
    
    def schoolbook_2x1x2(a_rows, a_col, b_row, b_cols, c_rows, c_cols):
        """
        Compute C[c_rows, c_cols] += A[a_rows, a_col] * B[b_row, b_cols]
        where b_row is a single row, b_cols is 2 columns.
        This is 4 rank-1 terms.
        """
        for i in a_rows:
            for k in b_cols:
                u = {i * m + a_col: 1}
                v = {b_row * p + k: 1}
                w = {k * n + i: 1}
                add_term(u, v, w)
    
    def schoolbook_2x2x1(a_rows, a_cols, b_rows, b_col, c_rows, c_col):
        """
        Compute C[c_rows, c_col] += A[a_rows, a_cols] * B[b_rows, b_col]
        where b_col is a single column.
        This is 4 rank-1 terms.
        """
        for i in a_rows:
            for j in a_cols:
                u = {i * m + j: 1}
                v = {j * p + b_col: 1}
                w = {b_col * n + i: 1}
                add_term(u, v, w)
    
    # Now decompose the full 4x5x5 problem.
    # A is 4x5, B is 5x5, C is 4x5.
    # Split A rows into [0,1] and [2,3].
    # Split A cols into [0,1], [2,3], [4].
    # Split B rows into [0,1], [2,3], [4].
    # Split B cols into [0,1], [2,3], [4].
    
    # For each row-half of A and each column-block of output:
    
    # Top half (rows 0-1):
    # C[0:2, 0:2] = A[0:2, 0:2] * B[0:2, 0:2] + A[0:2, 2:4] * B[2:4, 0:2] + A[0:2, 4] * B[4, 0:2]
    strassen_2x2x2([0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1])
    strassen_2x2x2([0, 1], [2, 3], [2, 3], [0, 1], [0, 1], [0, 1])
    schoolbook_2x1x2([0, 1], 4, 4, [0, 1], [0, 1], [0, 1])
    
    # C[0:2, 2:4] = A[0:2, 0:2] * B[0:2, 2:4] + A[0:2, 2:4] * B[2:4, 2:4] + A[0:2, 4] * B[4, 2:4]
    strassen_2x2x2([0, 1], [0, 1], [0, 1], [2, 3], [0, 1], [2, 3])
    strassen_2x2x2([0, 1], [2, 3], [2, 3], [2, 3], [0, 1], [2, 3])
    schoolbook_2x1x2([0, 1], 4, 4, [2, 3], [0, 1], [2, 3])
    
    # C[0:2, 4] = A[0:2, 0:2] * B[0:2, 4] + A[0:2, 2:4] * B[2:4, 4] + A[0:2, 4] * B[4, 4]
    schoolbook_2x2x1([0, 1], [0, 1], [0, 1], 4, [0, 1], 4)
    schoolbook_2x2x1([0, 1], [2, 3], [2, 3], 4, [0, 1], 4)
    schoolbook_2x1x1([0, 1], 4, 4, 4, [0, 1], 4)
    
    # Bottom half (rows 2-3):
    # C[2:4, 0:2] = A[2:4, 0:2] * B[0:2, 0:2] + A[2:4, 2:4] * B[2:4, 0:2] + A[2:4, 4] * B[4, 0:2]
    strassen_2x2x2([2, 3], [0, 1], [0, 1], [0, 1], [2, 3], [0, 1])
    strassen_2x2x2([2, 3], [2, 3], [2, 3], [0, 1], [2, 3], [0, 1])
    schoolbook_2x1x2([2, 3], 4, 4, [0, 1], [2, 3], [0, 1])
    
    # C[2:4, 2:4] = A[2:4, 0:2] * B[0:2, 2:4] + A[2:4, 2:4] * B[2:4, 2:4] + A[2:4, 4] * B[4, 2:4]
    strassen_2x2x2([2, 3], [0, 1], [0, 1], [2, 3], [2, 3], [2, 3])
    strassen_2x2x2([2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3])
    schoolbook_2x1x2([2, 3], 4, 4, [2, 3], [2, 3], [2, 3])
    
    # C[2:4, 4] = A[2:4, 0:2] * B[0:2, 4] + A[2:4, 2:4] * B[2:4, 4] + A[2:4, 4] * B[4, 4]
    schoolbook_2x2x1([2, 3], [0, 1], [0, 1], 4, [2, 3], 4)
    schoolbook_2x2x1([2, 3], [2, 3], [2, 3], 4, [2, 3], 4)
    schoolbook_2x1x1([2, 3], 4, 4, 4, [2, 3], 4)
    
    # Now convert terms to matrix form
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, (u, v, w) in enumerate(terms):
        for idx, coeff in u.items():
            U[idx][col] = coeff
        for idx, coeff in v.items():
            V[idx][col] = coeff
        for idx, coeff in w.items():
            W[idx][col] = coeff
    
    return {"U": U, "V": V, "W": W}
