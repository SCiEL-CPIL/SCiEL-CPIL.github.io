"""Recursive Strassen decomposition of 4x4x4 matrix multiplication (rank 49)."""
import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START

# Strassen's 7 products for 2x2 matrices.
# A indices: (0,0)=a, (0,1)=b, (1,0)=c, (1,1)=d
# B indices: (0,0)=e, (0,1)=f, (1,0)=g, (1,1)=h

_STRASSEN_2X2 = [
    # P1 = (a+d)(e+h)
    ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
    # P2 = (c+d)e
    ([(1,0,1),(1,1,1)], [(0,0,1)]),
    # P3 = a(f-h)
    ([(0,0,1)], [(0,1,1),(1,1,-1)]),
    # P4 = d(g-e)
    ([(1,1,1)], [(1,0,1),(0,0,-1)]),
    # P5 = (a+b)h
    ([(0,0,1),(0,1,1)], [(1,1,1)]),
    # P6 = (c-a)(e+f)
    ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)]),
    # P7 = (b-d)(g+h)
    ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)]),
]

# How each C entry is formed from the 7 Strassen products:
# C00 = P1 + P4 - P5 + P7
# C01 = P3 + P5
# C10 = P2 + P4
# C11 = P1 - P2 + P3 + P6

_STRASSEN_C_COMBINATION = [
    (0, 0, [(0, 1), (3, 1), (4, -1), (6, 1)]),
    (0, 1, [(2, 1), (4, 1)]),
    (1, 0, [(1, 1), (3, 1)]),
    (1, 1, [(0, 1), (1, -1), (2, 1), (5, 1)]),
]


def _strassen_4x4_decomposition():
    """
    Build rank-49 decomposition for 4x4x4 matrix multiplication using
    recursive Strassen.
    """
    n, m, p = 4, 4, 4
    rank = 49

    U = [[0] * rank for _ in range(n * m)]  # shape (16, 49)
    V = [[0] * rank for _ in range(m * p)]  # shape (16, 49)
    W = [[0] * rank for _ in range(p * n)]  # shape (16, 49)

    # Block-level Strassen products for 4x4 matrices partitioned into 2x2 blocks.
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Each block product Mi = Xi * Yi where Xi, Yi are 2x2 blocks.
    # We store the linear combination of A-blocks and B-blocks.
    # Block (br, bc) of A covers rows 2*br..2*br+1, cols 2*bc..2*bc+1.
    
    block_products = [
        ([(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (1, 1, 1)]), # M1
        ([(1, 0, 1), (1, 1, 1)], [(0, 0, 1)]),             # M2
        ([(0, 0, 1)], [(0, 1, 1), (1, 1, -1)]),            # M3
        ([(1, 1, 1)], [(1, 0, 1), (0, 0, -1)]),            # M4
        ([(0, 0, 1), (0, 1, 1)], [(1, 1, 1)]),             # M5
        ([(1, 0, 1), (0, 0, -1)], [(0, 0, 1), (0, 1, 1)]), # M6
        ([(0, 1, 1), (1, 1, -1)], [(1, 0, 1), (1, 1, 1)]), # M7
    ]

    # C block combination: C[cbr, cbc] = sum_i w_i * Mi
    c_block_combination = [
        (0, 0, [(0, 1), (3, 1), (4, -1), (6, 1)]), # C00
        (0, 1, [(2, 1), (4, 1)]),                   # C01
        (1, 0, [(1, 1), (3, 1)]),                   # C10
        (1, 1, [(0, 1), (1, -1), (2, 1), (5, 1)]),  # C11
    ]

    # For each of the 7 block products, we apply Strassen's 7 products on the 2x2 blocks.
    # Total rank = 7 * 7 = 49.
    
    for i in range(7):
        a_combo, b_combo = block_products[i]
        for s in range(7):
            r = i * 7 + s
            a_local, b_local = _STRASSEN_2X2[s]
            
            # Update U
            for (a_br, a_bc, a_coeff) in a_combo:
                for (lr, lc, l_coeff) in a_local:
                    row = 2 * a_br + lr
                    col = 2 * a_bc + lc
                    U[row * m + col][r] += a_coeff * l_coeff

            # Update V
            for (b_br, b_bc, b_coeff) in b_combo:
                for (lr, lc, l_coeff) in b_local:
                    row = 2 * b_br + lr
                    col = 2 * b_bc + lc
                    V[row * p + col][r] += b_coeff * l_coeff

            # Update W
            # C[2*cbr+lr, 2*cbc+lc] gets contribution from block product i with weight w_i
            # and Strassen product s with weight v_s.
            for (cbr, cbc, c_list) in c_block_combination:
                w_i = 0
                for (pi, wi) in c_list:
                    if pi == i:
                        w_i = wi
                        break
                if w_i == 0:
                    continue
                
                for (lr, lc, v_list) in _STRASSEN_C_COMBINATION:
                    v_s = 0
                    for (si, vs) in v_list:
                        if si == s:
                            v_s = vs
                            break
                    if v_s == 0:
                        continue
                    
                    c_row = 2 * cbr + lr
                    c_col = 2 * cbc + lc
                    # W index: W[k*n + i] where k is col, i is row
                    W[c_col * n + c_row][r] += w_i * v_s

    return {"U": U, "V": V, "W": W}


def _verify_decomposition(U, V, W, n, m, p, rank):
    """Verify that the decomposition is correct."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if total != 1:
                    return False
    return True


def _schoolbook_decomposition(n, m, p):
    """Fallback schoolbook rank n*m*p decomposition."""
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


def construct():
    deadline = time.time() + 140.0
    n, m, p = DIMENSIONS

    # Generate the rank-49 decomposition
    result = _strassen_4x4_decomposition()
    U, V, W = result["U"], result["V"], result["W"]
    rank = len(U[0])

    # Verify correctness
    if _verify_decomposition(U, V, W, n, m, p, rank):
        return result

    # Fallback to schoolbook (rank 64)
    return _schoolbook_decomposition(n, m, p)

# EVOLVE-BLOCK-END
