import time
import numpy as np

DIMENSIONS = [2, 2, 13]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 4, 26, 26
    
    # Rank calculation:
    # 13 columns of B can be grouped into 6 pairs and 1 single.
    # Each pair (2x2 block) uses Strassen (rank 7).
    # The single column (2x1 block) uses schoolbook (rank 4).
    # Total rank = 6 * 7 + 4 = 46.
    R = 46
    U = np.zeros((nm, R), dtype=float)
    V = np.zeros((mp, R), dtype=float)
    W = np.zeros((pn, R), dtype=float)
    
    def A_idx(i, j): return i * m + j
    def B_idx(j, k): return j * p + k
    def C_idx(i, k): return k * n + i
    
    def add_strassen_block(k0, col_offset):
        # Strassen decomposition for 2x2 * 2x2
        # A = [[a00, a01], [a10, a11]]
        # B = [[b00, b01], [b10, b11]] (columns k0, k0+1)
        # C = A * B
        prods = [
            # (A_coeffs, B_coeffs)
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),      # M1 = (a00+a11)(b00+b11)
            ({(1,0):1, (1,1):1}, {(0,0):1}),                # M2 = (a10+a11)b00
            ({(0,0):1}, {(0,1):1, (1,1):-1}),               # M3 = a00(b01-b11)
            ({(1,1):1}, {(1,0):1, (0,0):-1}),               # M4 = a11(b10-b00)
            ({(0,0):1, (0,1):1}, {(1,1):1}),                # M5 = (a00+a01)b11
            ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}),      # M6 = (a10-a00)(b00+b01)
            ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}),      # M7 = (a01-a11)(b10+b11)
        ]
        for t, (a_coeff, b_coeff) in enumerate(prods):
            col = col_offset + t
            for (i, j), c in a_coeff.items():
                U[A_idx(i, j), col] += c
            for (j, kl), c in b_coeff.items():
                V[B_idx(j, k0 + kl), col] += c
        
        # C = [[c00, c01], [c10, c11]]
        # c00 = M1 + M4 - M5 + M7
        # c01 = M3 + M5
        # c10 = M2 + M4
        # c11 = M1 - M2 + M3 + M6
        c_combos = [
            (0, k0,     [(0,1), (3,1), (4,-1), (6,1)]),
            (0, k0+1,   [(2,1), (4,1)]),
            (1, k0,     [(1,1), (3,1)]),
            (1, k0+1,   [(0,1), (1,-1), (2,1), (5,1)]),
        ]
        for (i, k, terms) in c_combos:
            for (t, coeff) in terms:
                W[C_idx(i, k), col_offset + t] += coeff
    
    def add_single_block(k, col_offset):
        # Schoolbook for 2x2 * 2x1 (column k)
        prods = [
            ((0,0), (0,k)),  # a00 * b0k -> c0k
            ((0,1), (1,k)),  # a01 * b1k -> c0k
            ((1,0), (0,k)),  # a10 * b0k -> c1k
            ((1,1), (1,k)),  # a11 * b1k -> c1k
        ]
        for t, ((i,j), (jj,kk)) in enumerate(prods):
            col = col_offset + t
            U[A_idx(i, j), col] += 1
            V[B_idx(jj, kk), col] += 1
        # c0k = p0 + p1
        # c1k = p2 + p3
        W[C_idx(0, k), col_offset+0] += 1
        W[C_idx(0, k), col_offset+1] += 1
        W[C_idx(1, k), col_offset+2] += 1
        W[C_idx(1, k), col_offset+3] += 1
    
    # 6 Strassen blocks for k0 = 0, 2, 4, 6, 8, 10
    col = 0
    for k0 in range(0, 12, 2):
        add_strassen_block(k0, col)
        col += 7
    # 1 single block for k=12
    add_single_block(12, col)
    col += 4
    assert col == R, f"col={col}, R={R}"
    
    # Verify with integer-exact check using fractions
    from fractions import Fraction
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_i = A_idx(i, j)
                b_i = B_idx(j, k)
                c_i = C_idx(i, k)
                total = Fraction(0)
                for r in range(R):
                    u_val = Fraction(U[a_i, r]).limit_denominator(10**6)
                    v_val = Fraction(V[b_i, r]).limit_denominator(10**6)
                    w_val = Fraction(W[c_i, r]).limit_denominator(10**6)
                    total += u_val * v_val * w_val
                if total != 1:
                    valid = False
                    break
        if not valid:
            break
    
    if not valid:
        # Fallback to schoolbook (Rank 52)
        R0 = n * m * p
        U0 = np.zeros((nm, R0), dtype=float)
        V0 = np.zeros((mp, R0), dtype=float)
        W0 = np.zeros((pn, R0), dtype=float)
        col_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U0[i * m + j, col_idx] = 1.0
                    V0[j * p + k, col_idx] = 1.0
                    W0[k * n + i, col_idx] = 1.0
                    col_idx += 1
        return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
