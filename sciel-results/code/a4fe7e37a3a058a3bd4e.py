import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 115

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    # Strassen terms for 2x2 matrix multiplication
    # U, V, W coefficients for 7 terms
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # C = A @ B
    # 7 multiplications:
    # m1 = (a00 + a11) * (b00 + b11)
    # m2 = (a10 + a11) * b00
    # m3 = a00 * (b01 - b11)
    # m4 = a11 * (b10 - b00)
    # m5 = (a00 + a01) * b11
    # m6 = (a10 - a00) * (b00 + b01)
    # m7 = (a01 - a11) * (b10 + b11)
    #
    # C00 = m1 + m4 - m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    
    # The mapping of indices in the provided code:
    # Local indices 0,1,2,3 correspond to a00, a01, a10, a11
    # But the code used:
    # a11 = 0*m + 2*u (this is actually a_{0, 2u}) -> index 0 in 2x2 block?
    # Let's re-read the provided code's index mapping to ensure consistency.
    # a11 = 0 * m + 2 * u  -> Row 0, Col 2u
    # a12 = 0 * m + 2 * u + 1 -> Row 0, Col 2u+1
    # a21 = 1 * m + 2 * u  -> Row 1, Col 2u
    # a22 = 1 * m + 2 * u + 1 -> Row 1, Col 2u+1
    # So local 0 -> a11 (global 0,2u), local 1 -> a12 (global 0,2u+1), local 2 -> a21 (global 1,2u), local 3 -> a22 (global 1,2u+1)
    # This corresponds to A_sub = [[a00, a01], [a10, a11]] in standard notation.
    # So local 0 is top-left, 1 is top-right, 2 is bottom-left, 3 is bottom-right.
    
    # B mapping:
    # b11 = (2 * u) * p + 2 * v -> Row 2u, Col 2v
    # b12 = (2 * u) * p + 2 * v + 1 -> Row 2u, Col 2v+1
    # b21 = (2 * u + 1) * p + 2 * v -> Row 2u+1, Col 2v
    # b22 = (2 * u + 1) * p + 2 * v + 1 -> Row 2u+1, Col 2v+1
    # So local 0 is top-left, etc.
    
    # C mapping:
    # c11 = (2 * v) * n + 0 -> Row 2v, Col 0
    # c12 = (2 * v + 1) * n + 0 -> Row 2v+1, Col 0
    # c21 = (2 * v) * n + 1 -> Row 2v, Col 1
    # c22 = (2 * v + 1) * n + 1 -> Row 2v+1, Col 1
    # So local 0 is top-left, 1 is top-right, 2 is bottom-left, 3 is bottom-right.
    
    # Strassen terms defined as (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: list of (local_idx, coeff)
    # Local idx: 0:TL, 1:TR, 2:BL, 3:BR
    
    # m1: (a00+a11)*(b00+b11) -> C00, C11
    # u: 0(1), 3(1)
    # v: 0(1), 3(1)
    # w: 0(1), 3(1)
    
    # m2: (a10+a11)*b00 -> C10, C11(-1)
    # u: 2(1), 3(1)
    # v: 0(1)
    # w: 2(1), 3(-1)
    
    # m3: a00*(b01-b11) -> C00(-1), C01(1)
    # u: 0(1)
    # v: 1(1), 3(-1)
    # w: 0(-1), 1(1)
    
    # m4: a11*(b10-b00) -> C00(1), C10(1)
    # u: 3(1)
    # v: 2(1), 0(-1)
    # w: 0(1), 2(1)
    
    # m5: (a00+a01)*b11 -> C00(-1), C01(1)
    # u: 0(1), 1(1)
    # v: 3(1)
    # w: 0(-1), 1(1)
    
    # m6: (a10-a00)*(b00+b01) -> C11(1)
    # u: 2(1), 0(-1)
    # v: 0(1), 1(1)
    # w: 3(1)
    
    # m7: (a01-a11)*(b10+b11) -> C00(1)
    # u: 1(1), 3(-1)
    # v: 2(1), 3(1)
    # w: 0(1)
    
    strassen_terms = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        ([(0, 1)], [(1, 1), (3, -1)], [(0, -1), (1, 1)]),
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]

    def build_block_strassen():
        rank = 56
        U = np.zeros((nm, rank), dtype=float)
        V = np.zeros((mp, rank), dtype=float)
        W = np.zeros((pn, rank), dtype=float)

        for u in range(2):  # 2 blocks of A rows (0-1, 2-3)
            for v in range(4): # 4 blocks of B cols (0-1, 2-3, 4-5, 6-7)
                base_col = 7 * (4 * u + v)
                
                # Map local 0,1,2,3 to global indices
                # A: row i, col j
                a00 = 0 * m + (2 * u)
                a01 = 0 * m + (2 * u + 1)
                a10 = 1 * m + (2 * u)
                a11 = 1 * m + (2 * u + 1)
                a_local_to_global = {0: a00, 1: a01, 2: a10, 3: a11}

                # B: row j, col k
                b00 = (2 * u) * p + (2 * v)
                b01 = (2 * u) * p + (2 * v + 1)
                b10 = (2 * u + 1) * p + (2 * v)
                b11 = (2 * u + 1) * p + (2 * v + 1)
                b_local_to_global = {0: b00, 1: b01, 2: b10, 3: b11}

                # C: row k, col i (Trace convention W[k*n+i, r])
                c00 = (2 * v) * n + 0
                c01 = (2 * v + 1) * n + 0
                c10 = (2 * v) * n + 1
                c11 = (2 * v + 1) * n + 1
                c_local_to_global = {0: c00, 1: c01, 2: c10, 3: c11}

                for t in range(7):
                    u_coeffs, v_coeffs, w_coeffs = strassen_terms[t]
                    col = base_col + t

                    for local_idx, coeff in u_coeffs:
                        U[a_local_to_global[local_idx], col] += coeff
                    for local_idx, coeff in v_coeffs:
                        V[b_local_to_global[local_idx], col] += coeff
                    for local_idx, coeff in w_coeffs:
                        W[c_local_to_global[local_idx], col] += coeff

        return U, V, W

    def verify(U, V, W, n, m, p, atol=1e-8):
        # Check a few random or all indices if fast enough
        # n=2, m=4, p=8 -> 64 entries. Very fast.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                    if not np.isclose(val, 1.0, atol=atol):
                        return False
        return True

    U, V, W = build_block_strassen()

    if not verify(U, V, W, n, m, p):
        # Fallback to trivial rank 64
        rank = n * m * p
        U = np.zeros((nm, rank), dtype=float)
        V = np.zeros((mp, rank), dtype=float)
        W = np.zeros((pn, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    construct()
