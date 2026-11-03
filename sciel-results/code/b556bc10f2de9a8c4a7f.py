"""Matrix multiplication 3,4,5 - rank-minimal decomposition."""
DIMENSIONS = [3, 4, 5]
import time
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin
    
    n, m, p = DIMENSIONS  # 3, 4, 5
    
    # Schoolbook as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def add_strassen_block(U, V, W, i0, j0, k0, col_offset):
        """Add a Strassen 2x2x2 block for indices i in {i0,i0+1}, j in {j0,j0+1}, k in {k0,k0+1}."""
        # Strassen's 7 multiplications for 2x2 matrices:
        # A = [[A00, A01], [A10, A11]] where A00=A[i0,j0], A01=A[i0,j0+1], A10=A[i0+1,j0], A11=A[i0+1,j0+1]
        # B = [[B00, B01], [B10, B11]] where B00=B[j0,k0], B01=B[j0,k0+1], B10=B[j0+1,k0], B11=B[j0+1,k0+1]
        # C = A*B = [[C00, C01], [C10, C11]] where C00=C[i0,k0], etc.
        #
        # M1 = (A00+A11)*(B00+B11)
        # M2 = (A10+A11)*B00
        # M3 = A00*(B10-B11)
        # M4 = A11*(B01-B00)
        # M5 = (A00+A01)*B11
        # M6 = (A10-A00)*(B00+B01)
        # M7 = (A01-A11)*(B10+B11)
        #
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        strassen_factors = [
            # (U_coeffs for A indices, V_coeffs for B indices, W_coeffs for C indices)
            # M1 = (A00+A11)*(B00+B11)
            ([(i0, j0, 1), (i0+1, j0+1, 1)], [(j0, k0, 1), (j0+1, k0+1, 1)], [(i0, k0, 1), (i0+1, k0+1, 1)]),
            # M2 = (A10+A11)*B00
            ([(i0+1, j0, 1), (i0+1, j0+1, 1)], [(j0, k0, 1)], [(i0+1, k0, 1)]),
            # M3 = A00*(B10-B11)
            ([(i0, j0, 1)], [(j0+1, k0, 1), (j0+1, k0+1, -1)], [(i0, k0+1, 1)]),
            # M4 = A11*(B01-B00)
            ([(i0+1, j0+1, 1)], [(j0, k0+1, 1), (j0, k0, -1)], [(i0+1, k0, 1), (i0, k0, 1)]),
            # M5 = (A00+A01)*B11
            ([(i0, j0, 1), (i0, j0+1, 1)], [(j0+1, k0+1, 1)], [(i0, k0, -1), (i0, k0+1, 1)]),
            # M6 = (A10-A00)*(B00+B01)
            ([(i0+1, j0, 1), (i0, j0, -1)], [(j0, k0, 1), (j0, k0+1, 1)], [(i0+1, k0+1, 1)]),
            # M7 = (A01-A11)*(B10+B11)
            ([(i0, j0+1, 1), (i0+1, j0+1, -1)], [(j0+1, k0, 1), (j0+1, k0+1, 1)], [(i0, k0, 1), (i0+1, k0+1, -1)]),
        ]
        
        # Wait, I need to recheck the Strassen formulas more carefully.
        # Let me use the standard Strassen:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        # C00 = M1 - M5 + M4 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M6 + M3
        
        # Let me redo with correct C formulas:
        strassen_factors = [
            # M1: U=(A00+A11), V=(B00+B11), W contributes to C00(+1) and C11(+1)
            ([(i0, j0, 1), (i0+1, j0+1, 1)], [(j0, k0, 1), (j0+1, k0+1, 1)], [(i0, k0, 1), (i0+1, k0+1, 1)]),
            # M2: U=(A10+A11), V=B00, W contributes to C10(+1) and C11(-1)
            ([(i0+1, j0, 1), (i0+1, j0+1, 1)], [(j0, k0, 1)], [(i0+1, k0, 1), (i0+1, k0+1, -1)]),
            # M3: U=A00, V=(B10-B11), W contributes to C01(+1) and C11(+1)
            ([(i0, j0, 1)], [(j0+1, k0, 1), (j0+1, k0+1, -1)], [(i0, k0+1, 1), (i0+1, k0+1, 1)]),
            # M4: U=A11, V=(B01-B00), W contributes to C00(+1) and C10(+1)
            ([(i0+1, j0+1, 1)], [(j0, k0+1, 1), (j0, k0, -1)], [(i0, k0, 1), (i0+1, k0, 1)]),
            # M5: U=(A00+A01), V=B11, W contributes to C00(-1) and C01(+1)
            ([(i0, j0, 1), (i0, j0+1, 1)], [(j0+1, k0+1, 1)], [(i0, k0, -1), (i0, k0+1, 1)]),
            # M6: U=(A10-A00), V=(B00+B01), W contributes to C11(+1)
            ([(i0+1, j0, 1), (i0, j0, -1)], [(j0, k0, 1), (j0, k0+1, 1)], [(i0+1, k0+1, 1)]),
            # M7: U=(A01-A11), V=(B10+B11), W contributes to C00(+1)
            ([(i0, j0+1, 1), (i0+1, j0+1, -1)], [(j0+1, k0, 1), (j0+1, k0+1, 1)], [(i0, k0, 1)]),
        ]
        
        for u_coeffs, v_coeffs, w_coeffs in strassen_factors:
            col = col_offset
            for (i, j, val) in u_coeffs:
                U[i * m + j][col] += val
            for (j, k, val) in v_coeffs:
                V[j * p + k][col] += val
            for (i, k, val) in w_coeffs:
                W[k * n + i][col] += val
            col_offset += 1
        
        return col_offset
    
    def add_schoolbook_term(U, V, W, i, j, k, col):
        U[i * m + j][col] = 1
        V[j * p + k][col] = 1
        W[k * n + i][col] = 1
    
    # Strategy: Use 4 Strassen blocks for the 2x2x2 subspaces, then schoolbook for the rest
    # Strassen blocks cover: i in {0,1}, j in {0,1} or {2,3}, k in {0,1} or {2,3}
    # That covers i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3}
    # Remaining: i in {0,1}, k=4 (all j) -> 8 terms
    #           i=2, all j, all k -> 20 terms
    # Total: 4*7 + 8 + 20 = 28 + 28 = 56
    
    rank = 4 * 7 + 8 + 20  # 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Add 4 Strassen blocks
    for i0 in [0]:  # i in {0,1}
        for j0 in [0, 2]:  # j in {0,1} or {2,3}
            for k0 in [0, 2]:  # k in {0,1} or {2,3}
                col = add_strassen_block(U, V, W, i0, j0, k0, col)
    
    # Add schoolbook terms for i in {0,1}, k=4
    for i in [0, 1]:
        for j in range(m):
            add_schoolbook_term(U, V, W, i, j, 4, col)
            col += 1
    
    # Add schoolbook terms for i=2
    for j in range(m):
        for k in range(p):
            add_schoolbook_term(U, V, W, 2, j, k, col)
            col += 1
    
    # Verify the decomposition
    def verify(U, V, W):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for c in range(rank):
                        total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(total - 1.0) > 1e-10:
                        return False
        return True
    
    if not verify(U, V, W):
        # Fallback to schoolbook
        return schoolbook()
    
    return {"U": U, "V": V, "W": W}
