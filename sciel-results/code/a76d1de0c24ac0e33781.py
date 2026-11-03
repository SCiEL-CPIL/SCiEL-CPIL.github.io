"""Matrix multiplication 2x3x8 - rank-minimal tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 8
    t0 = time.time()
    DEADLINE = t0 + 140.0  # Safety margin before 150s kill

    def elapsed():
        return time.time() - t0

    def schoolbook():
        """Return schoolbook decomposition of rank n*m*p = 48."""
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
        return U, V, W

    def verify_decomp(U, V, W, rank):
        """Exact verification using Fraction arithmetic."""
        Uf = [[Fraction(x) for x in row] for row in U]
        Vf = [[Fraction(x) for x in row] for row in V]
        Wf = [[Fraction(x) for x in row] for row in W]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for c in range(rank):
                        s += Uf[i * m + j][c] * Vf[j * p + k][c] * Wf[k * n + i][c]
                    if s != 1:
                        return False
        return True

    def strassen_rank44():
        """
        Construct rank-44 decomposition using block Strassen.
        Split m=3 into m1=1 and m2=2.
        Part 1: 2x1x8 (Rank 16)
        Part 2: 2x2x8, split into 4 blocks of 2x2x2 (Rank 7 each) -> Total 28.
        Total Rank: 44.
        """
        rank = 44
        U = [[0] * rank for _ in range(n * m)]  # 6 x 44
        V = [[0] * rank for _ in range(m * p)]  # 24 x 44
        W = [[0] * rank for _ in range(p * n)]  # 16 x 44

        col = 0

        # --- Part 1: 2x1x8 (j=0) ---
        # A is 2x1, B is 1x8.
        # C[i, k] = A[i, 0] * B[0, k]
        # Rank 16.
        for i in range(n):
            for k in range(p):
                u_idx = i * m + 0
                v_idx = 0 * p + k
                w_idx = k * n + i
                
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
                
        if elapsed() > DEADLINE:
            return None

        # --- Part 2: 2x2x8 (j=1,2) ---
        # A is 2x2 (cols 1,2), B is 2x8 (rows 1,2).
        # Split B into 4 blocks of 2 columns.
        for block_idx in range(4):
            k_start = block_idx * 2
            
            # Strassen 2x2x2
            # Local A:
            # A00 -> Global A[0, 1] (u_idx = 0*m + 1 = 1)
            # A01 -> Global A[0, 2] (u_idx = 0*m + 2 = 2)
            # A10 -> Global A[1, 1] (u_idx = 1*m + 1 = 4)
            # A11 -> Global A[1, 2] (u_idx = 1*m + 2 = 5)
            
            # Local B (for this block):
            # B00 -> Global B[1, k_start] (v_idx = 1*p + k_start)
            # B01 -> Global B[1, k_start + 1] (v_idx = 1*p + k_start + 1)
            # B10 -> Global B[2, k_start] (v_idx = 2*p + k_start)
            # B11 -> Global B[2, k_start + 1] (v_idx = 2*p + k_start + 1)
            
            # Local C (for this block):
            # C00 -> Global C[0, k_start] (w_idx = k_start*n + 0)
            # C01 -> Global C[0, k_start + 1] (w_idx = (k_start+1)*n + 0)
            # C10 -> Global C[1, k_start] (w_idx = k_start*n + 1)
            # C11 -> Global C[1, k_start + 1] (w_idx = (k_start+1)*n + 1)

            # Define Strassen steps:
            # Each step is (U_coeffs, V_coeffs, W_coeffs)
            # U_coeffs: dict {global_u_idx: coeff}
            # V_coeffs: dict {global_v_idx: coeff}
            # W_coeffs: dict {global_w_idx: coeff}

            steps = []

            # M1 = (A00 + A11) * (B00 + B11)
            # C00 += M1, C11 += M1
            steps.append({
                'U': {1: 1, 5: 1},  # A00, A11
                'V': {1*p + k_start: 1, 2*p + (k_start+1): 1},  # B00, B11
                'W': {k_start*n + 0: 1, (k_start+1)*n + 1: 1}  # C00, C11
            })

            # M2 = (A10 + A11) * B00
            # C10 += M2, C11 -= M2
            steps.append({
                'U': {4: 1, 5: 1},  # A10, A11
                'V': {1*p + k_start: 1},  # B00
                'W': {k_start*n + 1: 1, (k_start+1)*n + 1: -1}  # C10, C11
            })

            # M3 = A00 * (B01 - B11)
            # C01 += M3, C11 += M3
            steps.append({
                'U': {1: 1},  # A00
                'V': {1*p + (k_start+1): 1, 2*p + (k_start+1): -1},  # B01, -B11
                'W': {(k_start+1)*n + 0: 1, (k_start+1)*n + 1: 1}  # C01, C11
            })

            # M4 = A11 * (B10 - B00)
            # C00 += M4, C10 += M4
            steps.append({
                'U': {5: 1},  # A11
                'V': {2*p + k_start: 1, 1*p + k_start: -1},  # B10, -B00
                'W': {k_start*n + 0: 1, k_start*n + 1: 1}  # C00, C10
            })

            # M5 = (A00 + A01) * B11
            # C00 -= M5, C01 += M5
            steps.append({
                'U': {1: 1, 2: 1},  # A00, A01
                'V': {2*p + (k_start+1): 1},  # B11
                'W': {k_start*n + 0: -1, (k_start+1)*n + 0: 1}  # C00, C01
            })

            # M6 = (A10 - A00) * (B00 + B01)
            # C11 += M6
            steps.append({
                'U': {4: 1, 1: -1},  # A10, -A00
                'V': {1*p + k_start: 1, 1*p + (k_start+1): 1},  # B00, B01
                'W': {(k_start+1)*n + 1: 1}  # C11
            })

            # M7 = (A01 - A11) * (B10 + B11)
            # C00 += M7
            steps.append({
                'U': {2: 1, 5: -1},  # A01, -A11
                'V': {2*p + k_start: 1, 2*p + (k_start+1): 1},  # B10, B11
                'W': {k_start*n + 0: 1}  # C00
            })

            for step in steps:
                for u_idx, val in step['U'].items():
                    U[u_idx][col] += val
                for v_idx, val in step['V'].items():
                    V[v_idx][col] += val
                for w_idx, val in step['W'].items():
                    W[w_idx][col] += val
                col += 1

        if col != rank:
            return None # Sanity check

        return U, V, W

    # 1. Try the deterministic rank-44 construction
    result = strassen_rank44()
    if result is not None:
        U, V, W = result
        # Verify to be absolutely sure
        if verify_decomp(U, V, W, 44):
            return {"U": U, "V": V, "W": W}
        # If verification fails (should not happen), fall through to schoolbook

    # 2. Fallback to schoolbook rank 48
    U_sch, V_sch, W_sch = schoolbook()
    return {"U": U_sch, "V": V_sch, "W": W_sch}
# EVOLVE-BLOCK-END
