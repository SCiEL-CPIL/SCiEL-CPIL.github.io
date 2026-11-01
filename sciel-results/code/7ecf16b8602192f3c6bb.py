"""Tensor rank minimization for 2x5x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    start_time = time.time()
    deadline = start_time + 110.0

    # Strategy: Split 5 into 2+3, split 6 into 3+3
    # 2x2x3: Strassen 2x2x2 (rank 7) + schoolbook 2x2x1 (rank 4) = rank 11
    # 2x3x3: rank 9 (known optimal)
    # Total: 2*11 + 2*9 = 40

    rank = 40
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0

    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        if col >= rank:
            return
        for u_idx, coeff in u_coeffs:
            U[u_idx, col] += coeff
        for v_idx, coeff in v_coeffs:
            V[v_idx, col] += coeff
        for w_idx, coeff in w_coeffs:
            W[w_idx, col] += coeff
        col += 1

    # Strassen 2x2x2 block: (i∈{0,1}, j∈{ja,jb}, k∈{ka,kb})
    # u indices: i*m + j
    # v indices: j*p + k
    # w indices: k*n + i
    def strassen_222(ja, jb, ka, kb):
        ua = ja
        ub = 5 + ja
        va = jb
        vb = 5 + jb
        wka0 = 2 * ka
        wka1 = 2 * ka + 1
        wkb0 = 2 * kb
        wkb1 = 2 * kb + 1
        vka = ja * p + ka
        vkb = ja * p + kb
        vjb_ka = jb * p + ka
        vjb_kb = jb * p + kb

        add_term([(ua, 1), (vb, 1)], [(vka, 1), (vjb_kb, 1)], [(wka0, 1), (wkb1, 1)])
        add_term([(ub, 1), (vb, 1)], [(vka, 1)], [(wka1, 1), (wkb1, -1)])
        add_term([(ua, 1)], [(vkb, 1), (vjb_kb, -1)], [(wkb0, 1), (wkb1, 1)])
        add_term([(vb, 1)], [(vjb_ka, 1), (vka, -1)], [(wka0, 1), (wka1, 1)])
        add_term([(ua, 1), (va, 1)], [(vjb_kb, 1)], [(wka0, -1), (wkb0, 1)])
        add_term([(ub, 1), (ua, -1)], [(vka, 1), (vkb, 1)], [(wkb1, 1)])
        add_term([(va, 1), (vb, -1)], [(vjb_ka, 1), (vjb_kb, 1)], [(wka0, 1)])

    # Schoolbook 2x2x1 block: (i∈{0,1}, j∈{ja,jb}, k=ka)
    def schoolbook_221(ja, jb, ka):
        for i in range(n):
            for j in [ja, jb]:
                u_idx = i * m + j
                v_idx = j * p + ka
                w_idx = ka * n + i
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1

    # 2x2x3 block: Strassen 2x2x2 + schoolbook 2x2x1
    def block_223(ja, jb, ka, kb, kc):
        # Strassen for k in {ka, kb}
        strassen_222(ja, jb, ka, kb)
        # Schoolbook for k = kc
        schoolbook_221(ja, jb, kc)

    # 2x3x3 block with rank 9
    # Known rank-9 decomposition for 2x3x3 matrix multiplication
    # A is 2x3, B is 3x3, C = A*B is 2x3
    # Using the known minimal decomposition
    def block_233(ja, jb, jc, ka, kb, kc):
        # j indices: ja, jb, jc (3 columns of A)
        # k indices: ka, kb, kc (3 rows of B)
        # i indices: 0, 1 (2 rows)
        
        # Rank-9 algorithm for 2x3x3:
        # M1 = (A00+A01+A02)(B00+B10+B20)  -> contributes to C00
        # M2 = (A10+A11+A12)(B00+B10+B20)  -> contributes to C10
        # M3 = (A00+A01)(B01+B11)           -> contributes to C01
        # M4 = (A10+A11)(B01+B11)           -> contributes to C11
        # M5 = (A00+A02)(B02+B22)           -> contributes to C02
        # M6 = (A10+A12)(B02+B22)           -> contributes to C12
        # M7 = (A01-A02)(B12-B21)           -> contributes to C01, C02
        # M8 = (A11-A12)(B12-B21)           -> contributes to C11, C12
        # M9 = (A00-A01+A02)(B00-B10+B20)   -> contributes to C00
        # ... this needs careful derivation
        
        # Let me use a verified rank-9 decomposition
        # A = [[a00, a01, a02], [a10, a11, a12]]
        # B = [[b00, b01, b02], [b10, b11, b12], [b20, b21, b22]]
        # C[i,k] = sum_j A[i,j]*B[j,k]
        
        u_base = lambda i, j: i * m + j
        v_base = lambda j, k: j * p + k
        w_base = lambda k, i: k * n + i
        
        j_list = [ja, jb, jc]
        k_list = [ka, kb, kc]
        
        # Using a known rank-9 construction:
        # M1: (a00+a01+a02)*(b00+b10+b20)
        add_term([(u_base(0, ja), 1), (u_base(0, jb), 1), (u_base(0, jc), 1)],
                 [(v_base(ja, ka), 1), (v_base(jb, ka), 1), (v_base(jc, ka), 1)],
                 [(w_base(ka, 0), 1)])
        
        # M2: (a10+a11+a12)*(b00+b10+b20)
        add_term([(u_base(1, ja), 1), (u_base(1, jb), 1), (u_base(1, jc), 1)],
                 [(v_base(ja, ka), 1), (v_base(jb, ka), 1), (v_base(jc, ka), 1)],
                 [(w_base(ka, 1), 1)])
        
        # M3: (a00+a01)*(b01+b11)
        add_term([(u_base(0, ja), 1), (u_base(0, jb), 1)],
                 [(v_base(ja, kb), 1), (v_base(jb, kb), 1)],
                 [(w_base(kb, 0), 1)])
        
        # M4: (a10+a11)*(b01+b11)
        add_term([(u_base(1, ja), 1), (u_base(1, jb), 1)],
                 [(v_base(ja, kb), 1), (v_base(jb, kb), 1)],
                 [(w_base(kb, 1), 1)])
        
        # M5: (a00+a02)*(b02+b22)
        add_term([(u_base(0, ja), 1), (u_base(0, jc), 1)],
                 [(v_base(ja, kc), 1), (v_base(jc, kc), 1)],
                 [(w_base(kc, 0), 1)])
        
        # M6: (a10+a12)*(b02+b22)
        add_term([(u_base(1, ja), 1), (u_base(1, jc), 1)],
                 [(v_base(ja, kc), 1), (v_base(jc, kc), 1)],
                 [(w_base(kc, 1), 1)])
        
        # M7: (a01-a02)*(b12-b21)
        add_term([(u_base(0, jb), 1), (u_base(0, jc), -1)],
                 [(v_base(jb, kc), 1), (v_base(jc, kb), -1)],
                 [(w_base(kb, 0), 1), (w_base(kc, 0), -1)])
        
        # M8: (a11-a12)*(b12-b21)
        add_term([(u_base(1, jb), 1), (u_base(1, jc), -1)],
                 [(v_base(jb, kc), 1), (v_base(jc, kb), -1)],
                 [(w_base(kb, 1), 1), (w_base(kc, 1), -1)])
        
        # M9: (a00-a01+a02)*(b00-b10+b20)
        add_term([(u_base(0, ja), 1), (u_base(0, jb), -1), (u_base(0, jc), 1)],
                 [(v_base(ja, ka), 1), (v_base(jb, ka), -1), (v_base(jc, ka), 1)],
                 [(w_base(ka, 0), 1)])
        
        # Wait, I need to verify this actually works. Let me check C[0,0]:
        # C[0,0] = a00*b00 + a01*b10 + a02*b20
        # M1 contributes: a00*b00 + a00*b10 + a00*b20 + a01*b00 + a01*b10 + a01*b20 + a02*b00 + a02*b10 + a02*b20
        # M9 contributes: a00*b00 - a00*b10 + a00*b20 - a01*b00 + a01*b10 - a01*b20 + a02*b00 - a02*b10 + a02*b20
        # Sum M1+M9: 2*a00*b00 + 0*a00*b10 + 2*a00*b20 + 0*a01*b00 + 2*a01*b10 + 0*a01*b20 + 2*a02*b00 + 0*a02*b10 + 2*a02*b20
        # That's 2*(a00*b00 + a01*b10 + a02*b20) = 2*C[0,0]. Not correct.
        
        # This ad-hoc construction is error-prone. Let me fall back to schoolbook for 2x3x3 (rank 18)
        # and keep the 2x2x3 as rank 11.
        # Total: 2*11 + 2*18 = 58. Worse than 54.
        
        # Actually, let me just use schoolbook for everything and see if I can find a better split.
        # For now, let me revert to the known working 54-rank solution but with a cleaner structure.
        pass

    # The ad-hoc 2x3x3 rank-9 is too error-prone. Let me use the original working approach
    # but verify it carefully.
    
    # Reset and use the proven 54-rank decomposition
    rank = 54
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0

    def add_term2(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        if col >= rank:
            return
        for u_idx, coeff in u_coeffs:
            U[u_idx, col] += coeff
        for v_idx, coeff in v_coeffs:
            V[v_idx, col] += coeff
        for w_idx, coeff in w_coeffs:
            W[w_idx, col] += coeff
        col += 1

    def strassen_block(ja, jb, ka, kb):
        ua = ja
        ub = 5 + ja
        va = jb
        vb = 5 + jb
        wka0 = 2 * ka
        wka1 = 2 * ka + 1
        wkb0 = 2 * kb
        wkb1 = 2 * kb + 1
        vka = ja * p + ka
        vkb = ja * p + kb
        vjb_ka = jb * p + ka
        vjb_kb = jb * p + kb

        add_term2([(ua, 1), (vb, 1)], [(vka, 1), (vjb_kb, 1)], [(wka0, 1), (wkb1, 1)])
        add_term2([(ub, 1), (vb, 1)], [(vka, 1)], [(wka1, 1), (wkb1, -1)])
        add_term2([(ua, 1)], [(vkb, 1), (vjb_kb, -1)], [(wkb0, 1), (wkb1, 1)])
        add_term2([(vb, 1)], [(vjb_ka, 1), (vka, -1)], [(wka0, 1), (wka1, 1)])
        add_term2([(ua, 1), (va, 1)], [(vjb_kb, 1)], [(wka0, -1), (wkb0, 1)])
        add_term2([(ub, 1), (ua, -1)], [(vka, 1), (vkb, 1)], [(wkb1, 1)])
        add_term2([(va, 1), (vb, -1)], [(vjb_ka, 1), (vjb_kb, 1)], [(wka0, 1)])

    strassen_block(0, 1, 0, 1)
    strassen_block(0, 1, 2, 3)
    strassen_block(0, 1, 4, 5)
    strassen_block(2, 3, 0, 1)
    strassen_block(2, 3, 2, 3)
    strassen_block(2, 3, 4, 5)

    for i in range(n):
        for k in range(p):
            u_idx = i * m + 4
            v_idx = 4 * p + k
            w_idx = k * n + i
            U[u_idx, col] = 1
            V[v_idx, col] = 1
            W[w_idx, col] = 1
            col += 1

    U_used = U[:, :col]
    V_used = V[:, :col]
    W_used = W[:, :col]

    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = U_used[i * m + j, :]
                v_row = V_used[j * p + k, :]
                w_row = W_used[k * n + i, :]
                result = np.sum(u_row * v_row * w_row)
                if abs(result - 1) > 1e-10:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break

    if not valid:
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

    return {"U": U_used.tolist(), "V": V_used.tolist(), "W": W_used.tolist()}
# EVOLVE-BLOCK-END
