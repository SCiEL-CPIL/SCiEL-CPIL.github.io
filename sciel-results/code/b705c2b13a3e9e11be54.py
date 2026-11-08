import numpy as np
from collections import defaultdict

def construct():
    n, m, p = 2, 4, 7
    
    # We'll build up U, V, W as sparse dictionaries
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # Each entry is a list of (rank_index, value) pairs
    
    U = [defaultdict(float) for _ in range(n*m)]  # U[row][rank] = val
    V = [defaultdict(float) for _ in range(m*p)]  # V[row][rank] = val
    W = [defaultdict(float) for _ in range(p*n)]  # W[row][rank] = val
    
    rank = 0
    
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        """Add one tensor rank-1 term.
        u_coeffs: dict mapping (i,j) -> coeff for U
        v_coeffs: dict mapping (j,k) -> coeff for V
        w_coeffs: dict mapping (i,k) -> coeff for W
        """
        nonlocal rank
        # Collect all involved indices
        for (i,j), c in u_coeffs.items():
            if c != 0:
                U[i*m + j][rank] += c
        for (j,k), c in v_coeffs.items():
            if c != 0:
                V[j*p + k][rank] += c
        for (i,k), c in w_coeffs.items():
            if c != 0:
                W[k*n + i][rank] += c
        rank += 1
    
    # Strassen 2x2x2 decomposition
    # For multiplying A[i,j] (i=0,1; j=0,1) by B[j,k] (j=0,1; k=0,1) to get C[i,k]
    # The 7 terms:
    strassen_terms = [
        # M1 = (A00+A11)(B00+B11) -> contributes to C00, C11
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # M2 = (A10+A11)B00 -> contributes to C10, C11
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        # M3 = A00(B10-B11) -> contributes to C01, C11
        ({(0,0):1}, {(0,1):-1, (1,1):1}, {(0,1):1, (1,1):1}),
        # M4 = A11(B01-B00) -> contributes to C00, C10
        ({(1,1):1}, {(0,0):-1, (0,1):1}, {(0,0):1, (1,0):1}),
        # M5 = (A00+A01)B11 -> contributes to C00, C01
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # M6 = (A10-A00)(B00+B01) -> contributes to C11
        ({(0,0):-1, (1,0):1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # M7 = (A01-A11)(B10+B11) -> contributes to C00
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    # Verify Strassen decomposition
    for (i,j), (_,_,w_c) in enumerate([(i,j,k) for k in range(2) for j in range(2) for i in range(2)]):
        pass
    
    # Let me verify properly
    for i in range(2):
        for j in range(2):
            for k in range(2):
                total = 0
                for uc, vc, wc in strassen_terms:
                    total += uc.get((i,j), 0) * vc.get((j,k), 0) * wc.get((i,k), 0)
                if total != 1:
                    print(f"Strassen error at ({i},{j},{k}): got {total}, expected 1")
    
    # Now we need to decompose 2x4x7 into blocks.
    # A is 2x4, B is 4x7, C is 2x7
    # Inner dimension is m=4.
    # Split A's columns into two groups of 2: j=0,1 and j=2,3
    # For each group, we have a 2x2 submatrix of A
    # B's rows corresponding to j=0,1 form a 2x7 submatrix, j=2,3 form another 2x7
    
    # For each 2x2 block of A (rows i=0,1, cols j in {j0,j0+1}), 
    # and the corresponding 2x7 block of B (rows j in {j0,j0+1}, all 7 cols k=0..6),
    # we need to compute a 2x7 result.
    
    # For the 2x7 part, we can further split the 7 columns into 2,2,2,1.
    # For each 2x2x2 block, use Strassen (7 terms)
    # For the remaining 2x2x1 block, use schoolbook (4 terms)
    
    def add_strassen_2x2x2_block(i0, j0, k0):
        """Add Strassen decomposition for A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2] -> C[i0:i0+2, k0:k0+2]"""
        # Map local indices to global
        # A local (li, lj) -> global (i0+li, j0+lj) -> index (i0+li)*m + (j0+lj)
        # B local (lj, lk) -> global (j0+lj, k0+lk) -> index (j0+lj)*p + (k0+lk)
        # C local (li, lk) -> global (i0+li, k0+lk) -> index (k0+lk)*n + (i0+li)
        
        for uc, vc, wc in strassen_terms:
            # Transform coefficients
            u_new = {}
            for (li, lj), c in uc.items():
                gi = i0 + li
                gj = j0 + lj
                u_new[(gi, gj)] = c
            
            v_new = {}
            for (lj, lk), c in vc.items():
                gj = j0 + lj
                gk = k0 + lk
                v_new[(gj, gk)] = c
            
            w_new = {}
            for (li, lk), c in wc.items():
                gi = i0 + li
                gk = k0 + lk
                w_new[(gi, gk)] = c
            
            add_term(u_new, v_new, w_new)
    
    def add_schoolbook_2x2x1_block(i0, j0, k0):
        """Add schoolbook for A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+1] -> C[i0:i0+2, k0:k0+1]"""
        # 4 terms: a*e, b*g, c*e, d*g
        # A local (li, lj) -> global (i0+li, j0+lj)
        # B local (lj, 0) -> global (j0+lj, k0)
        # C local (li, 0) -> global (i0+li, k0)
        
        # Term 1: A[0,0]*B[0,0] -> C[0,0]
        add_term(
            {(i0, j0): 1},
            {(j0, k0): 1},
            {(i0, k0): 1}
        )
        # Term 2: A[0,1]*B[1,0] -> C[0,0]
        add_term(
            {(i0, j0+1): 1},
            {(j0+1, k0): 1},
            {(i0, k0): 1}
        )
        # Term 3: A[1,0]*B[0,0] -> C[1,0]
        add_term(
            {(i0+1, j0): 1},
            {(j0, k0): 1},
            {(i0+1, k0): 1}
        )
        # Term 4: A[1,1]*B[1,0] -> C[1,0]
        add_term(
            {(i0+1, j0+1): 1},
            {(j0+1, k0): 1},
            {(i0+1, k0): 1}
        )
    
    # Process all blocks
    # j0 = 0: A columns 0,1; B rows 0,1
    # j0 = 2: A columns 2,3; B rows 2,3
    # For each, k splits: 0,1 / 2,3 / 4,5 / 6
    
    for j0 in [0, 2]:
        # Strassen blocks for k=0,1; k=2,3; k=4,5
        for k0 in [0, 2, 4]:
            add_strassen_2x2x2_block(0, j0, k0)
        # Schoolbook for k=6
        add_schoolbook_2x2x1_block(0, j0, 6)
    
    # Now convert to dense matrices
    # U is n*m = 8 rows, rank columns
    # V is m*p = 28 rows, rank columns
    # W is p*n = 14 rows, rank columns
    
    U_dense = [[0]*rank for _ in range(n*m)]
    V_dense = [[0]*rank for _ in range(m*p)]
    W_dense = [[0]*rank for _ in range(p*n)]
    
    for i in range(n*m):
        for r, v in U[i].items():
            U_dense[i][r] = v
    
    for i in range(m*p):
        for r, v in V[i].items():
            V_dense[i][r] = v
    
    for i in range(p*n):
        for r, v in W[i].items():
            W_dense[i][r] = v
    
    # Verify the decomposition
    # The multiplication tensor T[i,j,k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For matrix multiplication, T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # Actually, the tensor for matmul is T[i,j,k] = delta (it's 1 for all i,j,k since each term A[i,j]*B[j,k] contributes to C[i,k])
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U_dense[i*m+j][r] * V_dense[j*p+k][r] * W_dense[k*n+i][r]
                if total != 1:
                    print(f"Verification error at ({i},{j},{k}): got {total}, expected 1")
                    break
    
    return {"U": U_dense, "V": V_dense, "W": W_dense}
