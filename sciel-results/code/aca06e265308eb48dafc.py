import time
import numpy as np

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 2, 2, 8
    nr, mr, pr = n * m, m * p, p * n  # 4, 16, 16

    # ------------------------------------------------------------------
    # The multiplication tensor: T[a,b,c] = 1 with a=i*m+j, b=j*p+k, c=k*n+i
    # (the official trace convention for W).  T has 32 ones, one per (i,j,k).
    # ------------------------------------------------------------------
    T = np.zeros((nr, mr, pr), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def verify(U, V, W, tol=1e-8):
        Tc = np.einsum('at,bt,ct->abc', U, V, W)
        err = float(np.max(np.abs(Tc - T)))
        return err < tol, err

    # ------------------------------------------------------------------
    # Deterministic rank-28 construction: Strassen (rank-7) applied to each of the
    # 4 column-pairs (2x2 x 2x2 blocks).  This is exactly correct and minimal.
    # The rank of M_{2,2,8} is 28 because it is the direct sum of four M_{2,2,2}
    # tensors with disjoint supports in the B and C indices.
    # ------------------------------------------------------------------
    def get_strassen_28():
        rank = 28
        U = np.zeros((nr, rank), dtype=float)
        V = np.zeros((mr, rank), dtype=float)
        W = np.zeros((pr, rank), dtype=float)
        
        # Strassen's 7 products for 2x2 matrix multiplication.
        # A_sub is 2x2, B_sub is 2x2, C_sub = A_sub * B_sub.
        # Indices in sub-matrices: (0,0), (0,1), (1,0), (1,1)
        # 
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B01 - B11)
        # M4 = A11 * (B10 - B00)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        #
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6

        strassen_products = [
            {'A': {(0, 0): 1, (1, 1): 1}, 'B': {(0, 0): 1, (1, 1): 1}, 'C': {(0, 0): 1, (1, 1): 1}},
            {'A': {(1, 0): 1, (1, 1): 1}, 'B': {(0, 0): 1}, 'C': {(1, 0): 1, (1, 1): -1}},
            {'A': {(0, 0): 1}, 'B': {(0, 1): 1, (1, 1): -1}, 'C': {(0, 1): 1, (1, 1): 1}},
            {'A': {(1, 1): 1}, 'B': {(1, 0): 1, (0, 0): -1}, 'C': {(0, 0): 1, (1, 0): 1}},
            {'A': {(0, 0): 1, (0, 1): 1}, 'B': {(1, 1): 1}, 'C': {(0, 0): -1, (0, 1): 1}},
            {'A': {(1, 0): 1, (0, 0): -1}, 'B': {(0, 0): 1, (0, 1): 1}, 'C': {(1, 1): 1}},
            {'A': {(0, 1): 1, (1, 1): -1}, 'B': {(1, 0): 1, (1, 1): 1}, 'C': {(0, 0): 1}},
        ]

        for q in range(p // 2):
            k1 = 2 * q
            for r, prod in enumerate(strassen_products):
                t = q * 7 + r
                for (i, j), coeff in prod['A'].items():
                    U[i * m + j, t] = coeff
                for (j, k_sub), coeff in prod['B'].items():
                    V[j * p + (k1 + k_sub), t] = coeff
                for (i, k_sub), coeff in prod['C'].items():
                    W[(k1 + k_sub) * n + i, t] = coeff
        return U, V, W

    # Get the rank-28 solution
    U, V, W = get_strassen_28()
    
    # Verify
    ok, err = verify(U, V, W)
    if not ok:
        # Fallback to schoolbook rank 32 if something went wrong (should not happen)
        U = np.zeros((nr, 32)); V = np.zeros((mr, 32)); W = np.zeros((pr, 32))
        for idx in range(32):
            i = idx // 16
            j = (idx // 8) % 2
            k = idx % 8
            U[i * m + j, idx] = 1.0
            V[j * p + k, idx] = 1.0
            W[k * n + i, idx] = 1.0
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
