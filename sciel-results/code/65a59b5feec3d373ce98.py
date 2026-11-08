import numpy as np

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 4
    # We construct a rank-22 decomposition using block Strassen + outer product terms.
    # A is 2x3, B is 3x4, C is 2x4.
    # A indices: A00=0, A01=1, A02=2, A10=3, A11=4, A12=5
    # B indices: B00=0, B01=1, B02=2, B03=3, B10=4, B11=5, B12=6, B13=7, B20=8, B21=9, B22=10, B23=11
    # C indices (W[k*n+i, r]): C00=0, C01=1, C02=2, C03=3, C10=4, C11=5, C12=6, C13=7
    # Wait, let's re-verify the trace convention W[k*n+i, r].
    # C[i, j] = sum_k A[i,k] B[k,j].
    # U has rows for A indices. V has rows for B indices. W has rows for C indices.
    # C[i, j] corresponds to W[i, j] if n=2, p=4? No, W[k*n+i, r] is the evaluator's internal index.
    # The task says: "output factor follows the official trace convention W[k*n+i,r]".
    # Actually, the return is {'U', 'V', 'W'} with shapes (n*m, rank), (m*p, rank), (p*n, rank).
    # U shape: (2*3, rank) = (6, rank). Rows correspond to A elements.
    # V shape: (3*4, rank) = (12, rank). Rows correspond to B elements.
    # W shape: (4*2, rank) = (8, rank). Rows correspond to C elements.
    # Let's check the index mapping for W.
    # "W[k*n+i, r]" suggests row index in W is `k*n + i`.
    # C is n x p. So i in [0, n), k in [0, p)?
    # Usually C[i, j]. Here i is row of C, j is col of C.
    # If row index is `k*n + i`, and k ranges over columns of C (p of them), i ranges over rows of C (n of them).
    # So W index = col * n + row.
    # Let's assume W[row_in_W] corresponds to C[i, j] where row_in_W = j * n + i.
    # Let's verify with the seed program's usage.
    # Seed: W shape (p*n, rank) = (4*2, 8).
    # Seed M1: adds to W[0] and W[3].
    # Strassen M1: (A00+A11)(B00+B11) -> C00 + C11.
    # C00 is i=0, j=0. Index = 0*2 + 0 = 0.
    # C11 is i=1, j=1. Index = 1*2 + 1 = 3.
    # This matches! So W index = j * n + i, where j is column of C, i is row of C.
    
    rank = 22
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term(u_indices, v_indices, w_indices, col):
        for idx, val in u_indices:
            U[idx][col] += val
        for idx, val in v_indices:
            V[idx][col] += val
        for idx, val in w_indices:
            W[idx][col] += val

    # Block 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2]
    # A submatrix:
    # 0 1
    # 3 4
    # B submatrix:
    # 0 1
    # 4 5
    # C submatrix (cols 0,1):
    # C00 C01
    # C10 C11
    # W indices:
    # C00: j=0, i=0 -> 0
    # C01: j=1, i=0 -> 2
    # C10: j=0, i=1 -> 1
    # C11: j=1, i=1 -> 3
    
    # M1: (A00+A11)(B00+B11) = C00 + C11
    add_term([(0, 1), (4, 1)], [(0, 1), (5, 1)], [(0, 1), (3, 1)], 0)
    
    # M2: (A10+A11)B00 = C10 - C11
    add_term([(3, 1), (4, 1)], [(0, 1)], [(1, 1), (3, -1)], 1)
    
    # M3: A00(B01-B11) = C01 + C11
    add_term([(0, 1)], [(1, 1), (5, -1)], [(2, 1), (3, 1)], 2)
    
    # M4: A11(B10-B00) = C00 + C10
    add_term([(4, 1)], [(4, 1), (0, -1)], [(0, 1), (1, 1)], 3)
    
    # M5: (A00+A01)B11 = -C00 + C01
    add_term([(0, 1), (1, 1)], [(5, 1)], [(0, -1), (2, 1)], 4)
    
    # M6: (A10-A00)(B00+B01) = C11
    add_term([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)], 5)
    
    # M7: (A01-A11)(B10+B11) = C00
    add_term([(1, 1), (4, -1)], [(4, 1), (5, 1)], [(0, 1)], 6)

    # Block 2: Strassen for A[0:2, 0:2] * B[0:2, 2:4]
    # A submatrix same as above:
    # 0 1
    # 3 4
    # B submatrix (cols 2,3):
    # 2 3
    # 6 7
    # C submatrix (cols 2,3):
    # C02 C03
    # C12 C13
    # W indices:
    # C02: j=2, i=0 -> 4
    # C03: j=3, i=0 -> 6
    # C12: j=2, i=1 -> 5
    # C13: j=3, i=1 -> 7

    # M1': (A00+A11)(B02+B13) = C02 + C13
    add_term([(0, 1), (4, 1)], [(2, 1), (7, 1)], [(4, 1), (7, 1)], 7)
    
    # M2': (A10+A11)B02 = C12 - C13
    add_term([(3, 1), (4, 1)], [(2, 1)], [(5, 1), (7, -1)], 8)
    
    # M3': A00(B03-B13) = C03 + C13
    add_term([(0, 1)], [(3, 1), (7, -1)], [(6, 1), (7, 1)], 9)
    
    # M4': A11(B12-B02) = C02 + C12
    add_term([(4, 1)], [(6, 1), (2, -1)], [(4, 1), (5, 1)], 10)
    
    # M5': (A00+A01)B13 = -C02 + C03
    add_term([(0, 1), (1, 1)], [(7, 1)], [(4, -1), (6, 1)], 11)
    
    # M6': (A10-A00)(B02+B03) = C13
    add_term([(3, 1), (0, -1)], [(2, 1), (3, 1)], [(7, 1)], 12)
    
    # M7': (A01-A11)(B12+B13) = C02
    add_term([(1, 1), (4, -1)], [(6, 1), (7, 1)], [(4, 1)], 13)

    # Outer Product: A[0:2, 2:3] * B[2:3, 0:4]
    # A col 2:
    # A02 = 2
    # A12 = 5
    # B row 2:
    # B20 = 8, B21 = 9, B22 = 10, B23 = 11
    # C contributions:
    # C0j += A02 * B2j
    # C1j += A12 * B2j
    
    # A02 * B20 -> C00 (j=0, i=0 -> 0)
    add_term([(2, 1)], [(8, 1)], [(0, 1)], 14)
    # A02 * B21 -> C01 (j=1, i=0 -> 2)
    add_term([(2, 1)], [(9, 1)], [(2, 1)], 15)
    # A02 * B22 -> C02 (j=2, i=0 -> 4)
    add_term([(2, 1)], [(10, 1)], [(4, 1)], 16)
    # A02 * B23 -> C03 (j=3, i=0 -> 6)
    add_term([(2, 1)], [(11, 1)], [(6, 1)], 17)
    
    # A12 * B20 -> C10 (j=0, i=1 -> 1)
    add_term([(5, 1)], [(8, 1)], [(1, 1)], 18)
    # A12 * B21 -> C11 (j=1, i=1 -> 3)
    add_term([(5, 1)], [(9, 1)], [(3, 1)], 19)
    # A12 * B22 -> C12 (j=2, i=1 -> 5)
    add_term([(5, 1)], [(10, 1)], [(5, 1)], 20)
    # A12 * B23 -> C13 (j=3, i=1 -> 7)
    add_term([(5, 1)], [(11, 1)], [(7, 1)], 21)

    return {"U": U, "V": V, "W": W}
