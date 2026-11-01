"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    start_time = time.time()
    deadline = start_time + 110.0

    # Use 6 Strassen blocks: j in {0,1} or {2,3}, k in {0,1} or {2,3} or {4,5}
    # Each block: i in {0,1}, 2 j-values, 2 k-values -> 7 terms instead of 8
    # j=4 triples are standard: 2*6 = 12 terms
    # Total rank: 6*7 + 12 = 54
    
    rank = 54
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

    # Strassen block for (i in {0,1}, j in {ja,jb}, k in {ka,kb})
    # u indices: i*m + j
    # v indices: j*p + k
    # w indices: k*n + i
    def strassen_block(ja, jb, ka, kb):
        # u indices
        u0a = 0 * m + ja  # (0, ja)
        u1a = 1 * m + ja  # (1, ja)
        u0b = 0 * m + jb  # (0, jb)
        u1b = 1 * m + jb  # (1, jb)
        
        # v indices
        v_aka = ja * p + ka  # (ja, ka)
        v_akb = ja * p + kb  # (ja, kb)
        v_bka = jb * p + ka  # (jb, ka)
        v_bkb = jb * p + kb  # (jb, kb)
        
        # w indices
        w_ka0 = ka * n + 0  # (ka, 0)
        w_ka1 = ka * n + 1  # (ka, 1)
        w_kb0 = kb * n + 0  # (kb, 0)
        w_kb1 = kb * n + 1  # (kb, 1)
        
        # Standard Strassen 2x2x2 decomposition
        # M1 = (A11+A22)(B11+B22)
        add_term([(u0a, 1), (u1b, 1)], [(v_aka, 1), (v_bkb, 1)], [(w_ka0, 1), (w_kb1, 1)])
        # M2 = (A21+A22)B11
        add_term([(u1a, 1), (u1b, 1)], [(v_aka, 1)], [(w_ka1, 1), (w_kb1, -1)])
        # M3 = A11(B21-B11)
        add_term([(u0a, 1)], [(v_bka, 1), (v_aka, -1)], [(w_kb0, 1), (w_kb1, 1)])
        # M4 = A22(B12-B01) -- wait, let me use the standard formulation
        
        # Let me use the well-known Strassen formulation:
        # For C = AB where A is 2x2, B is 2x2:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B00)
        # M4 = A11(B01-B10)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        # C00 = M1+M4-M5+M7
        # C01 = M3+M5
        # C10 = M2+M4
        # C11 = M1-M2+M3+M6
        
        # In our tensor: A[i,j] = U[i*m+j, :], B[j,k] = V[j*p+k, :], C[i,k] = W[k*n+i, :]
        # So A00=u0a, A01=u0b, A10=u1a, A11=u1b
        # B00=v_aka, B01=v_akb, B10=v_bka, B11=v_bkb
        # C00=w_ka0, C01=w_ka1, C10=w_kb0, C11=w_kb1
        
        # I already added M1 above. Let me redo all 7:
        # Actually I already added M1, so let me add M2-M7
        
        # M2 = (A10+A11)B00 = (u1a+u1b) * v_aka -> contributes to C10 and C11
        add_term([(u1a, 1), (u1b, 1)], [(v_aka, 1)], [(w_kb0, 1), (w_kb1, -1)])
        # M3 = A00(B10-B00) = u0a * (v_bka - v_aka) -> contributes to C01 and C11
        add_term([(u0a, 1)], [(v_bka, 1), (v_aka, -1)], [(w_ka1, 1), (w_kb1, 1)])
        # M4 = A11(B01-B10) = u1b * (v_akb - v_bka) -> contributes to C00 and C10
        add_term([(u1b, 1)], [(v_akb, 1), (v_bka, -1)], [(w_ka0, 1), (w_kb0, 1)])
        # M5 = (A00+A01)B11 = (u0a+u0b) * v_bkb -> contributes to C00 and C01
        add_term([(u0a, 1), (u0b, 1)], [(v_bkb, 1)], [(w_ka0, -1), (w_ka1, 1)])
        # M6 = (A10-A00)(B00+B01) = (u1a-u0a) * (v_aka+v_akb) -> contributes to C11
        add_term([(u1a, 1), (u0a, -1)], [(v_aka, 1), (v_akb, 1)], [(w_kb1, 1)])
        # M7 = (A01-A11)(B10+B11) = (u0b-u1b) * (v_bka+v_bkb) -> contributes to C00
        add_term([(u0b, 1), (u1b, -1)], [(v_bka, 1), (v_bkb, 1)], [(w_ka0, 1)])

    # Define the 6 Strassen blocks
    blocks = [
        (0, 1, 0, 1),  # j in {0,1}, k in {0,1}
        (0, 1, 2, 3),  # j in {0,1}, k in {2,3}
        (0, 1, 4, 5),  # j in {0,1}, k in {4,5}
        (2, 3, 0, 1),  # j in {2,3}, k in {0,1}
        (2, 3, 2, 3),  # j in {2,3}, k in {2,3}
        (2, 3, 4, 5),  # j in {2,3}, k in {4,5}
    ]
    
    for (ja, jb, ka, kb) in blocks:
        strassen_block(ja, jb, ka, kb)
        if time.time() > deadline:
            break

    # Standard terms for j=4 (not in any Strassen block)
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 4
            v_idx = 4 * p + k
            w_idx = k * n + i
            U[u_idx, col] = 1
            V[v_idx, col] = 1
            W[w_idx, col] = 1
            col += 1

    if col != rank:
        # Adjust if needed
        pass

    # Verify
    U_np = U[:, :col]
    V_np = V[:, :col]
    W_np = W[:, :col]
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
