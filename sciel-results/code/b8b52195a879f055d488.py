"""Tensor rank minimization for 2x5x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    start_time = time.time()
    deadline = start_time + 110.0

    # We expect rank 54
    # 6 Strassen blocks (7 terms each) = 42
    # 12 standard terms
    # Total 54
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

    def strassen_block(i0, j0, k0):
        # i in {i0, i0+1}, j in {j0, j0+1}, k in {k0, k0+1}
        
        # u indices:
        # (i0, j0), (i0+1, j0), (i0, j0+1), (i0+1, j0+1)
        u00 = i0 * m + j0
        u10 = (i0 + 1) * m + j0
        u01 = i0 * m + (j0 + 1)
        u11 = (i0 + 1) * m + (j0 + 1)
        
        # v indices:
        # (j0, k0), (j0+1, k0), (j0, k0+1), (j0+1, k0+1)
        v00 = j0 * p + k0
        v10 = (j0 + 1) * p + k0
        v01 = j0 * p + (k0 + 1)
        v11 = (j0 + 1) * p + (k0 + 1)
        
        # w indices:
        # (k0, i0), (k0, i0+1), (k0+1, i0), (k0+1, i0+1)
        w00 = k0 * n + i0
        w01 = k0 * n + (i0 + 1)
        w10 = (k0 + 1) * n + i0
        w11 = (k0 + 1) * n + (i0 + 1)

        # M1 = (A00 + A11)(B00 + B11) -> C00 +1, C11 +1
        add_term([(u00, 1), (u11, 1)], [(v00, 1), (v11, 1)], [(w00, 1), (w11, 1)])
        
        # M2 = (A10 + A11)B00 -> C10 +1, C11 -1
        # Note: A10 corresponds to (i0+1, j0), B00 to (j0, k0)
        # C10 corresponds to (k0, i0+1) -> w01
        # C11 corresponds to (k0+1, i0+1) -> w11
        add_term([(u10, 1), (u11, 1)], [(v00, 1)], [(w01, 1), (w11, -1)])
        
        # M3 = A00(B01 - B11) -> C01 +1, C11 +1
        # A00: (i0, j0) -> u00
        # B01: (j0, k0+1) -> v01
        # B11: (j0+1, k0+1) -> v11
        # C01: (k0, i0+1) -> w01
        # C11: (k0+1, i0+1) -> w11
        add_term([(u00, 1)], [(v01, 1), (v11, -1)], [(w01, 1), (w11, 1)])
        
        # M4 = A11(B10 - B00) -> C00 +1, C10 +1
        # A11: (i0+1, j0+1) -> u11
        # B10: (j0+1, k0) -> v10
        # B00: (j0, k0) -> v00
        # C00: (k0, i0) -> w00
        # C10: (k0, i0+1) -> w01
        add_term([(u11, 1)], [(v10, 1), (v00, -1)], [(w00, 1), (w01, 1)])
        
        # M5 = (A00 + A01)B11 -> C00 -1, C01 +1
        # A00: (i0, j0) -> u00
        # A01: (i0, j0+1) -> u01
        # B11: (j0+1, k0+1) -> v11
        # C00: (k0, i0) -> w00
        # C01: (k0, i0+1) -> w01
        add_term([(u00, 1), (u01, 1)], [(v11, 1)], [(w00, -1), (w01, 1)])
        
        # M6 = (A10 - A00)(B00 + B01) -> C11 +1
        # A10: (i0+1, j0) -> u10
        # A00: (i0, j0) -> u00
        # B00: (j0, k0) -> v00
        # B01: (j0, k0+1) -> v01
        # C11: (k0+1, i0+1) -> w11
        add_term([(u10, 1), (u00, -1)], [(v00, 1), (v01, 1)], [(w11, 1)])
        
        # M7 = (A01 - A11)(B10 + B11) -> C00 +1
        # A01: (i0, j0+1) -> u01
        # A11: (i0+1, j0+1) -> u11
        # B10: (j0+1, k0) -> v10
        # B11: (j0+1, k0+1) -> v11
        # C00: (k0, i0) -> w00
        add_term([(u01, 1), (u11, -1)], [(v10, 1), (v11, 1)], [(w00, 1)])

    # Apply Strassen to 6 blocks
    # j in {0,1} and {2,3}
    # k in {0,1}, {2,3}, {4,5}
    
    strassen_blocks = [
        (0, 0, 0),
        (0, 2, 0),
        (0, 0, 2),
        (0, 2, 2),
        (0, 0, 4),
        (0, 2, 4)
    ]
    
    for i0, j0, k0 in strassen_blocks:
        strassen_block(i0, j0, k0)

    # Add standard terms for j=4
    # i in {0,1}, j=4, k in {0..5}
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 4
            v_idx = 4 * p + k
            w_idx = k * n + i
            U[u_idx, col] = 1
            V[v_idx, col] = 1
            W[w_idx, col] = 1
            col += 1
            
    # col should be 42 + 12 = 54
    # Trim if necessary (though we allocated exactly 54)
    U_used = U[:, :col]
    V_used = V[:, :col]
    W_used = W[:, :col]

    # Verify
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

    if valid:
        return {"U": U_used.tolist(), "V": V_used.tolist(), "W": W_used.tolist()}
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
