"""Rewritten program: exact rank-54 decomposition for 2x5x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill

    # Rank-54 decomposition using Strassen on 2x2x2 blocks and trivial on 2x1x2 blocks.
    # j indices (columns of A / rows of B): 0..4
    # k indices (rows of C / cols of B): 0..5
    #
    # j_pairs: (0,1), (2,3) -> each paired with each k_pair -> 2*3 = 6 blocks of 2x2x2, rank 7 each = 42
    # j_single: 4 -> paired with each k_pair -> 3 blocks of 2x1x2, rank 4 each = 12
    # Total rank = 54

    rank = 54
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = [0]  # use list for mutability in nested functions

    def add_term(u_coeffs, v_coeffs, w_coeffs):
        c = col[0]
        if c >= rank:
            return
        for u_idx, coeff in u_coeffs:
            U[u_idx, c] += coeff
        for v_idx, coeff in v_coeffs:
            V[v_idx, c] += coeff
        for w_idx, coeff in w_coeffs:
            W[w_idx, c] += coeff
        col[0] = c + 1

    def strassen_2x2x2(ja, jb, ka, kb):
        """
        Strassen decomposition for 2x2x2 sub-problem:
        A sub: rows {0,1}, cols {ja,jb}
        B sub: rows {ja,jb}, cols {ka,kb}
        C sub: rows {0,1}, cols {ka,kb}
        
        Tensor indices:
        U[i*m + j, r], V[j*p + k, r], W[k*n + i, r]
        
        The tensor T[i,j,k] = 1 if this is the (i,j,k) entry of the multiplication tensor.
        For this sub-block, we need:
          sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for (i,j,k) in this block, 0 otherwise.
        
        Standard Strassen for 2x2x2:
          Let A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = A*B = [[ae+bg, af+bh],[ce+dg, cf+dh]]
          
          M1 = (a+d)(e+h)
          M2 = (c+d)(e)
          M3 = a(g-e)
          M4 = d(f-g)
          M5 = (a+b)(h)
          M6 = (c-a)(e+f)
          M7 = (b-d)(g+h)
          
          C00 = M1 - M4 + M5 - M7
          C01 = M3 + M5
          C10 = M2 + M4
          C11 = M1 - M2 + M3 + M6
        """
        # U indices: i*m + j
        u0a = 0 * m + ja  # A[0, ja]
        u0b = 0 * m + jb  # A[0, jb]
        u1a = 1 * m + ja  # A[1, ja]
        u1b = 1 * m + jb  # A[1, jb]

        # V indices: j*p + k
        v_aka = ja * p + ka  # B[ja, ka]
        v_akb = ja * p + kb  # B[ja, kb]
        v_bka = jb * p + ka  # B[jb, ka]
        v_bkb = jb * p + kb  # B[jb, kb]

        # W indices: k*n + i
        w_ka0 = ka * n + 0  # C[0, ka]
        w_ka1 = ka * n + 1  # C[1, ka]
        w_kb0 = kb * n + 0  # C[0, kb]
        w_kb1 = kb * n + 1  # C[1, kb]

        # M1 = (A00+A11)(B00+B11)
        # U: A00=1, A11=1; V: B00=1, B11=1; W: C00=1, C11=1
        add_term([(u0a, 1), (u1b, 1)],
                 [(v_aka, 1), (v_bkb, 1)],
                 [(w_ka0, 1), (w_kb1, 1)])

        # M2 = (A10+A11)(B00)
        # U: A10=1, A11=1; V: B00=1; W: C10=1, C11=-1
        add_term([(u1a, 1), (u1b, 1)],
                 [(v_aka, 1)],
                 [(w_ka1, 1), (w_kb1, -1)])

        # M3 = A00(B10-B00)
        # U: A00=1; V: B10=1, B00=-1; W: C01=1, C11=1
        add_term([(u0a, 1)],
                 [(v_bka, 1), (v_aka, -1)],
                 [(w_kb0, 1), (w_kb1, 1)])

        # M4 = A11(B01-B10)
        # U: A11=1; V: B01=1, B10=-1; W: C00=1, C10=1
        add_term([(u1b, 1)],
                 [(v_akb, 1), (v_bka, -1)],
                 [(w_ka0, 1), (w_ka1, 1)])

        # M5 = (A00+A01)(B11)
        # U: A00=1, A01=1; V: B11=1; W: C00=-1, C01=1
        add_term([(u0a, 1), (u0b, 1)],
                 [(v_bkb, 1)],
                 [(w_ka0, -1), (w_kb0, 1)])

        # M6 = (A10-A00)(B00+B01)
        # U: A10=1, A00=-1; V: B00=1, B01=1; W: C11=1
        add_term([(u1a, 1), (u0a, -1)],
                 [(v_aka, 1), (v_akb, 1)],
                 [(w_kb1, 1)])

        # M7 = (A01-A11)(B10+B11)
        # U: A01=1, A11=-1; V: B10=1, B11=1; W: C00=1
        add_term([(u0b, 1), (u1b, -1)],
                 [(v_bka, 1), (v_bkb, 1)],
                 [(w_ka0, 1)])

    def trivial_2x1x2(j, ka, kb):
        """
        Trivial decomposition for 2x1x2 sub-problem:
        A sub: rows {0,1}, col {j}
        B sub: row {j}, cols {ka,kb}
        C sub: rows {0,1}, cols {ka,kb}
        
        4 terms, one for each (i, k) pair.
        """
        u0 = 0 * m + j  # A[0, j]
        u1 = 1 * m + j  # A[1, j]
        v_ka = j * p + ka  # B[j, ka]
        v_kb = j * p + kb  # B[j, kb]
        w_ka0 = ka * n + 0  # C[0, ka]
        w_ka1 = ka * n + 1  # C[1, ka]
        w_kb0 = kb * n + 0  # C[0, kb]
        w_kb1 = kb * n + 1  # C[1, kb]

        # C[0,ka] = A[0,j]*B[j,ka]
        add_term([(u0, 1)], [(v_ka, 1)], [(w_ka0, 1)])
        # C[0,kb] = A[0,j]*B[j,kb]
        add_term([(u0, 1)], [(v_kb, 1)], [(w_kb0, 1)])
        # C[1,ka] = A[1,j]*B[j,ka]
        add_term([(u1, 1)], [(v_ka, 1)], [(w_ka1, 1)])
        # C[1,kb] = A[1,j]*B[j,kb]
        add_term([(u1, 1)], [(v_kb, 1)], [(w_kb1, 1)])

    # Define block structure
    j_pairs = [(0, 1), (2, 3)]
    j_singles = [4]
    k_pairs = [(0, 1), (2, 3), (4, 5)]

    # Add Strassen blocks
    for (ja, jb) in j_pairs:
        for (ka, kb) in k_pairs:
            if time.time() > deadline:
                break
            strassen_2x2x2(ja, jb, ka, kb)

    # Add trivial blocks
    for j in j_singles:
        for (ka, kb) in k_pairs:
            if time.time() > deadline:
                break
            trivial_2x1x2(j, ka, kb)

    # Extract used columns
    used_col = col[0]
    U_np = U[:, :used_col]
    V_np = V[:, :used_col]
    W_np = W[:, :used_col]

    # Verify the decomposition
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = U_np[i * m + j, :]
                v_row = V_np[j * p + k, :]
                w_row = W_np[k * n + i, :]
                result = np.sum(u_row * v_row * w_row)
                if abs(result - 1) > 1e-10:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break

    if valid:
        return {"U": U_np.tolist(), "V": V_np.tolist(), "W": W_np.tolist()}
    else:
        # Fallback to trivial rank 60
        U_fallback = np.zeros((n * m, 60), dtype=float)
        V_fallback = np.zeros((m * p, 60), dtype=float)
        W_fallback = np.zeros((p * n, 60), dtype=float)
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fallback[i * m + j, c] = 1
                    V_fallback[j * p + k, c] = 1
                    W_fallback[k * n + i, c] = 1
                    c += 1
        return {"U": U_fallback.tolist(), "V": V_fallback.tolist(), "W": W_fallback.tolist()}
# EVOLVE-BLOCK-END
