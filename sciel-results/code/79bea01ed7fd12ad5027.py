import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 6]


def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    # We decompose the (2,4,6) multiplication tensor into two (2,2,3) sub-tensors
    # by splitting the middle dimension m=4 into two groups of 2.
    # Each (2,2,3) sub-tensor has rank 6 (not 7 like Strassen), giving total rank 12.
    #
    # For a (2,2,3) tensor: A is 2x2, B is 2x3, C is 2x3
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # Rank-6 decomposition for 2x2x3:
    # Let A = [[a0, a1], [b0, b1]], B = [[B0, B1, B2], [B3, B4, B5]] (each Bk is 1x1)
    # C[0,0] = a0*B0 + a1*B3
    # C[0,1] = a0*B1 + a1*B4
    # C[0,2] = a0*B2 + a1*B5
    # C[1,0] = b0*B0 + b1*B3
    # C[1,1] = b0*B1 + b1*B4
    # C[1,2] = b0*B2 + b1*B5
    #
    # The 2x2x3 tensor has 12 non-zero entries. We need rank r such that
    # sum_{r=1}^R u_{(i,j)}^{(r)} * v_{(j,k)}^{(r)} * w_{(i,k)}^{(r)} = T[i,j,k]
    #
    # For 2x2x3, the rank is known to be 6. Here's a construction:
    #
    # M1: (a0+b0) * (B0+B3) -> contributes to C[0,0] and C[1,0] and C[0,1] and C[1,1] etc.
    #
    # Actually, let me use a known rank-6 construction for 2x2x3:
    # 
    # The tensor T has T[i, j, k] = 1 if i' = i and j' = j and k' = k where
    # the indices are: A index = i*2+j, B index = j*3+k, C index = k*2+i
    #
    # Wait, let me be more careful. For the 2x2x3 subproblem:
    # A has shape (2,2), B has shape (2,3), C has shape (2,3)
    # A index: i*2 + j for i in {0,1}, j in {0,1}
    # B index: j*3 + k for j in {0,1}, k in {0,1,2}
    # C index: k*2 + i for k in {0,1,2}, i in {0,1}
    #
    # T[A_idx, B_idx, C_idx] = 1 iff A_idx = i*2+j, B_idx = j*3+k, C_idx = k*2+i
    #
    # Rank-6 construction:
    # Let's use the following 6 products:
    #
    # P1: (A[0,0] + A[1,0]) * (B[0,0] + B[1,0])
    # P2: A[0,0] * (B[0,1] - B[1,1])
    # P3: A[1,0] * (B[0,2] - B[1,2])
    # P4: (A[0,1] + A[1,1]) * (B[0,0] + B[1,0])
    # P5: A[0,1] * (B[0,1] - B[1,1])
    # P6: A[1,1] * (B[0,2] - B[1,2])
    #
    # Hmm, this doesn't look right. Let me think more carefully.
    #
    # Actually, for 2x2x3, we can use the following approach:
    # Think of it as computing C = A*B where A is 2x2 and B is 2x3.
    # C[0,0] = a00*B00 + a01*B10
    # C[0,1] = a00*B01 + a01*B11
    # C[0,2] = a00*B02 + a01*B12
    # C[1,0] = a10*B00 + a11*B10
    # C[1,1] = a10*B01 + a11*B11
    # C[1,2] = a10*B02 + a11*B12
    #
    # Let me try a direct rank-6 construction using bilinear forms.
    #
    # Define:
    # M1 = (a00 + a10) * (B00 + B10)
    # M2 = a00 * (B01 - B11)
    # M3 = a10 * (B01 - B11)
    # M4 = (a01 + a11) * (B00 + B10)
    # M5 = a01 * (B02 - B12)
    # M6 = a11 * (B02 - B12)
    #
    # Check C[0,0] = a00*B00 + a01*B10
    # From M1: a00*B00 + a00*B10 + a10*B00 + a10*B10
    # From M4: a01*B00 + a01*B10 + a11*B00 + a11*B10
    # Sum so far has too many terms.
    #
    # Let me try a different approach. Use the known rank-6 formula for 2x2x3.
    #
    # After some thought, here's a valid rank-6 decomposition:
    #
    # M1: (a00+a10)(B00+B10) -> W gets +1 at (0,0) and (1,0), -1 at (0,1) and (1,1) etc.
    #
    # Actually, let me just verify a known construction computationally.
    # I'll set up the system and solve for the factors.

    # For now, let's use a rank-6 construction that I'll verify:
    # Based on the structure, here's one approach:
    #
    # For 2x2x3, split into:
    # - 2x2x2 part (columns 0,1 of B) -> rank 7 via Strassen
    # - 2x1x1 part (column 2 of B) -> rank 2
    # Total: 9, not 6.
    #
    # So the rank-6 claim needs verification. Let me just use rank 7+2=9 per block, total 18.
    # But that's worse than 42/2=21... no wait, 9*2=18 < 21.
    #
    # Actually, let me just implement the straightforward decomposition:
    # Split m=4 into two halves of size 2.
    # For each half, we have a 2x2x6 multiplication.
    # 2x2x6 = 2x2x2 + 2x2x2 + 2x2x2 = 3 * rank(2x2x2)
    # Using Strassen: 3 * 7 = 21 per half, total 42. Same as before.
    #
    # Using schoolbook for 2x2x2: 3 * 4 = 12 per half, total 24.
    # But schoolbook 2x2x2 has rank 4.
    #
    # Wait, schoolbook for 2x2x2 is rank 4, not 8. Let me reconsider.
    # For 2x2x2: C[0,0] = a00*B00 + a01*B10, C[0,1] = a00*B01 + a01*B11, etc.
    # That's 4 independent products, so rank 4.
    #
    # So 2x2x6 via schoolbook: 3 blocks of 2x2x2, each rank 4, total rank 12 per half.
    # Total: 2 * 12 = 24.
    #
    # That's much better than 42! Let me implement this.

    rank = 24
    U = np.zeros((nm, rank), dtype=float)
    V = np.zeros((mp, rank), dtype=float)
    W = np.zeros((pn, rank), dtype=float)

    def add_2x2x2_schoolbook(i_start, j_start, k_start, offset):
        # A indices: (i_start+0)*m + (j_start+0), (i_start+0)*m + (j_start+1),
        #             (i_start+1)*m + (j_start+0), (i_start+1)*m + (j_start+1)
        # B indices: (j_start+0)*p + (k_start+0), (j_start+0)*p + (k_start+1),
        #             (j_start+1)*p + (k_start+0), (j_start+1)*p + (k_start+1)
        # C indices: (k_start+0)*n + (i_start+0), (k_start+0)*n + (i_start+1),
        #             (k_start+1)*n + (i_start+0), (k_start+1)*n + (i_start+1)

        a00 = (i_start + 0) * m + (j_start + 0)
        a01 = (i_start + 0) * m + (j_start + 1)
        a10 = (i_start + 1) * m + (j_start + 0)
        a11 = (i_start + 1) * m + (j_start + 1)

        b00 = (j_start + 0) * p + (k_start + 0)
        b01 = (j_start + 0) * p + (k_start + 1)
        b10 = (j_start + 1) * p + (k_start + 0)
        b11 = (j_start + 1) * p + (k_start + 1)

        w00 = (k_start + 0) * n + (i_start + 0)
        w01 = (k_start + 0) * n + (i_start + 1)
        w10 = (k_start + 1) * n + (i_start + 0)
        w11 = (k_start + 1) * n + (i_start + 1)

        # C[0,0] = a00*B00 + a01*B10
        U[a00, offset] += 1.0
        V[b00, offset] += 1.0
        W[w00, offset] += 1.0

        U[a01, offset + 1] += 1.0
        V[b10, offset + 1] += 1.0
        W[w00, offset + 1] += 1.0

        # C[0,1] = a00*B01 + a01*B11
        U[a00, offset + 2] += 1.0
        V[b01, offset + 2] += 1.0
        W[w10, offset + 2] += 1.0

        U[a01, offset + 3] += 1.0
        V[b11, offset + 3] += 1.0
        W[w10, offset + 3] += 1.0

        # C[1,0] = a10*B00 + a11*B10
        U[a10, offset + 4] += 1.0
        V[b00, offset + 4] += 1.0
        W[w01, offset + 4] += 1.0

        U[a11, offset + 5] += 1.0
        V[b10, offset + 5] += 1.0
        W[w01, offset + 5] += 1.0

        # C[1,1] = a10*B01 + a11*B11
        U[a10, offset + 6] += 1.0
        V[b01, offset + 6] += 1.0
        W[w11, offset + 6] += 1.0

        U[a11, offset + 7] += 1.0
        V[b11, offset + 7] += 1.0
        W[w11, offset + 7] += 1.0

    offset = 0
    for i_start in range(0, n, 2):
        for j_start in range(0, m, 2):
            for k_start in range(0, p, 2):
                add_2x2x2_schoolbook(i_start, j_start, k_start, offset)
                offset += 8

    # Verification
    def verify(U_mat, V_mat, W_mat, tol=1e-8):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = 0.0
                    for r in range(rank):
                        s += U_mat[a, r] * V_mat[b, r] * W_mat[c, r]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    if not verify(U, V, W):
        # Fallback to schoolbook rank 48
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * (m * p) + j * p + k
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
        return {"U": U, "V": V, "W": W}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
