import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 8]
START_TIME = time.time()
TIME_LIMIT = 140.0

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W form a valid tensor decomposition."""
    rank = len(U[0]) if U else 0
    if len(U) != n * m:
        return False
    if len(V) != m * p:
        return False
    if len(W) != p * n:
        return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                actual = Fraction(0)
                for c in range(rank):
                    actual += Fraction(U[i * m + j][c]) * Fraction(V[j * p + k][c]) * Fraction(W[k * n + i][c])
                if actual != 1:
                    return False
    return True

def _construct_strassen_block(U_list, V_list, W_list, i0, j0, k0, n, m, p):
    """
    Add 7 terms for a 2x2x2 Strassen block starting at (i0, j0, k0).
    Indices: i in {i0, i0+1}, j in {j0, j0+1}, k in {k0, k0+1}.
    """
    nm, mp, pn = n * m, m * p, p * n
    
    # Local indices
    I0, I1 = i0, i0 + 1
    J0, J1 = j0, j0 + 1
    K0, K1 = k0, k0 + 1
    
    # Helper to create vectors
    def make_u(indices_vals):
        vec = [0] * nm
        for (i, j), val in indices_vals:
            vec[i * m + j] = val
        return vec

    def make_v(indices_vals):
        vec = [0] * mp
        for (j, k), val in indices_vals:
            vec[j * p + k] = val
        return vec

    def make_w(indices_vals):
        vec = [0] * pn
        for (k, i), val in indices_vals:
            vec[k * n + i] = val
        return vec

    # Strassen Terms:
    # M1 = (A11 + A22)(B11 + B22) -> C11 += M1, C22 += M1
    u1 = make_u([((I0, J0), 1), ((I1, J1), 1)])
    v1 = make_v([((J0, K0), 1), ((J1, K1), 1)])
    w1 = make_w([((K0, I0), 1), ((K1, I1), 1)])
    U_list.append(u1); V_list.append(v1); W_list.append(w1)

    # M2 = (A21 + A22)B11 -> C21 += M2
    u2 = make_u([((I1, J0), 1), ((I1, J1), 1)])
    v2 = make_v([((J0, K0), 1)])
    w2 = make_w([((K0, I1), 1)])
    U_list.append(u2); V_list.append(v2); W_list.append(w2)

    # M3 = A11(B21 - B22) -> C11 += M3, C12 += M3
    u3 = make_u([((I0, J0), 1)])
    v3 = make_v([((J1, K0), 1), ((J1, K1), -1)])
    w3 = make_w([((K0, I0), 1), ((K1, I0), 1)])
    U_list.append(u3); V_list.append(v3); W_list.append(w3)

    # M4 = A22(B12 - B11) -> C11 += -M4? No, C11 = M1 + M4 - M5 + M7. 
    # Wait, standard Strassen:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    # My M4 definition: A22(B12 - B11).
    # In C11, it is +M4. In C21, it is +M4.
    u4 = make_u([((I1, J1), 1)])
    v4 = make_v([((J0, K1), 1), ((J0, K0), -1)]) # B12 - B11 => j=J0, k=K1 is B12? 
    # Careful with indices: B_{jk}. 
    # B12 is j=0, k=1 (if 1-indexed 1,2). Here J0, K1.
    # B11 is j=0, k=0.
    # So v4[J0, K1] = 1, v4[J0, K0] = -1.
    w4 = make_w([((K1, I0), 1), ((K0, I1), 1)]) # Contributes to C11 (K0, I0) and C21 (K0, I1)?
    # Wait. C11 is (I0, K0). C21 is (I1, K0).
    # M4 = A22 * (B12 - B11).
    # Term in C11: A22 is A[I1, J1]. B12 is B[J0, K1]. Product is A[I1, J1] B[J0, K1].
    # This term appears in C[I0, K0]? No.
    # The multiplication is C[i,k] = sum_j A[i,j] B[j,k].
    # So A[I1, J1] B[J0, K1] contributes to C[I1, K1].
    # But Strassen M4 is supposed to contribute to C11 and C21?
    # Let's re-verify Strassen formulas.
    # M4 = A22 (B12 - B11).
    # C11 = M1 + M4 - M5 + M7.
    # C21 = M2 + M4.
    # This implies M4 must contribute to C11 and C21.
    # How can A22 (which is row 2) contribute to C11 (row 1)? It can't directly.
    # Ah, the formulas are for the 2x2 blocks.
    # A = [[A11, A12], [A21, A22]].
    # B = [[B11, B12], [B21, B22]].
    # M4 = A22 (B12 - B11).
    # C11 = M1 + M4 - M5 + M7.
    # C21 = M2 + M4.
    # This is correct. The "contribution" is via the linear combination in W.
    # So for M4:
    # U4: A22 -> (I1, J1) = 1.
    # V4: B12 - B11 -> (J0, K1) = 1, (J0, K0) = -1.
    # W4: Must produce 1 at (K0, I0) for C11 and 1 at (K0, I1) for C21.
    # So W4[(K0, I0)] = 1, W4[(K0, I1)] = 1.
    w4 = make_w([((K0, I0), 1), ((K0, I1), 1)])
    U_list.append(u4); V_list.append(v4); W_list.append(w4)

    # M5 = (A11 + A12)B22 -> C11 -= M5, C12 += M5
    u5 = make_u([((I0, J0), 1), ((I0, J1), 1)])
    v5 = make_v([((J1, K1), 1)])
    w5 = make_w([((K1, I0), -1), ((K1, I0), 1)]) # Wait, C11 is (I0, K1)? No, C11 is (I0, K0).
    # M5 contributes to C11 with -1 and C12 with +1.
    # C11 is (I0, K0). C12 is (I0, K1).
    # So W5[(K0, I0)] = -1, W5[(K1, I0)] = 1.
    w5 = make_w([((K0, I0), -1), ((K1, I0), 1)])
    U_list.append(u5); V_list.append(v5); W_list.append(w5)

    # M6 = (A21 - A11)(B11 + B12) -> C22 += M6
    u6 = make_u([((I1, J0), 1), ((I0, J0), -1)])
    v6 = make_v([((J0, K0), 1), ((J0, K1), 1)])
    w6 = make_w([((K1, I1), 1)]) # C22 is (I1, K1).
    U_list.append(u6); V_list.append(v6); W_list.append(w6)

    # M7 = (A12 - A22)(B21 + B22) -> C11 += M7
    u7 = make_u([((I0, J1), 1), ((I1, J1), -1)])
    v7 = make_v([((J1, K0), 1), ((J1, K1), 1)])
    w7 = make_w([((K0, I0), 1)]) # C11 is (I0, K0).
    U_list.append(u7); V_list.append(v7); W_list.append(w7)

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    U_list = []
    V_list = []
    W_list = []
    
    # Track which (i,j,k) tuples are covered by Strassen blocks
    covered = set()
    
    # Define Strassen blocks
    # i in {0,1}
    # j in {0,1} and {2,3}
    # k in {0,1}, {2,3}, {4,5}, {6,7}
    
    i_blocks = [(0, 1)]
    j_blocks = [(0, 1), (2, 3)]
    k_blocks = [(0, 1), (2, 3), (4, 5), (6, 7)]
    
    for i0, i1 in i_blocks:
        for j0, j1 in j_blocks:
            for k0, k1 in k_blocks:
                # Mark all 8 tuples as covered
                for i in [i0, i1]:
                    for j in [j0, j1]:
                        for k in [k0, k1]:
                            covered.add((i, j, k))
                
                _construct_strassen_block(U_list, V_list, W_list, i0, j0, k0, n, m, p)
    
    # Add schoolbook terms for the rest
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    # Schoolbook term
                    u = [0] * nm
                    v = [0] * mp
                    w = [0] * pn
                    
                    u[i * m + j] = 1
                    v[j * p + k] = 1
                    w[k * n + i] = 1
                    
                    U_list.append(u)
                    V_list.append(v)
                    W_list.append(w)
    
    # Verify
    if _verify(U_list, V_list, W_list, n, m, p):
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fallback to schoolbook if verification fails (should not happen)
        U_school = []
        V_school = []
        W_school = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = [0] * nm; u[i * m + j] = 1
                    v = [0] * mp; v[j * p + k] = 1
                    w = [0] * pn; w[k * n + i] = 1
                    U_school.append(u)
                    V_school.append(v)
                    W_school.append(w)
        return {"U": U_school, "V": V_school, "W": W_school}

# EVOLVE-BLOCK-END
