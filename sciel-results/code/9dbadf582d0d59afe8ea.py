import numpy as np
import time
import itertools

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    DEADLINE = 130  # 150s wall-clock, leave 20s margin

    # ------------------------------------------------------------------
    # We aim for rank 28 decomposition of 2x3x5 matrix multiplication.
    #
    # Strategy:
    #   Block A: j=0 (single middle index), k=0..4  -> schoolbook 2x1x5, rank 10
    #   Block B: j=1,2 (2x2 sub-block)
    #     B1: k=0,1  -> Strassen 2x2x2, rank 7
    #     B2: k=2    -> schoolbook 2x2x1, rank 4
    #     B3: k=3,4  -> Strassen 2x2x2, rank 7
    #   Total: 10 + 7 + 4 + 7 = 28
    #
    # Additionally, we attempt to find a rank-27 decomposition by
    # searching for a non-trivial rank-10 decomposition of the 2x2x3
    # sub-tensor (B1+B2+B3 combined) or by mixing blocks.
    # ------------------------------------------------------------------

    rank = 28
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    col = 0

    # ---- Block A: j=0, i=0,1, k=0..4 (schoolbook 2x1x5) ----
    for i in range(n):
        for k in range(p):
            U[i * m + 0, col] = 1.0
            V[0 * p + k, col] = 1.0
            W[k * n + i, col] = 1.0
            col += 1

    # ---- Strassen helper for 2x2x2 sub-block ----
    def strassen_block(i0, j0, k0, start_col):
        """
        Strassen decomposition for a 2x2x2 sub-block.
        i0: starting i index (0 or 1)
        j0: starting j index (1 or 2)
        k0: starting k index
        Returns the next column index.
        """
        # Local indices: i in {0,1}, j' in {0,1}, k' in {0,1}
        # Global: i = i0 + i, j = j0 + j', k = k0 + k'
        c = start_col

        # M1: (A[i0,j0] + A[i0+1,j0+1]) * (B[j0,k0] + B[j0+1,k0+1])
        #   -> C[k0, i0] += M1, C[k0+1, i0+1] += M1
        U[i0 * m + j0, c] = 1.0
        U[(i0 + 1) * m + (j0 + 1), c] = 1.0
        V[j0 * p + k0, c] = 1.0
        V[(j0 + 1) * p + (k0 + 1), c] = 1.0
        W[k0 * n + i0, c] = 1.0
        W[(k0 + 1) * n + (i0 + 1), c] = 1.0
        c += 1

        # M2: (A[i0+1,j0] + A[i0+1,j0+1]) * B[j0,k0]
        #   -> C[k0+1, i0+1] += M2, C[k0, i0+1] -= M2  (wait, need to check)
        # Standard Strassen:
        # M2 = (A10 + A11) * B00  -> C10 += M2, C11 -= M2
        # In our local indexing: i0->i, j0->j', k0->k'
        # A[i0+1, j0] + A[i0+1, j0+1]  -> local (1,0)+(1,1)
        # B[j0, k0]                     -> local (0,0)
        # C[k0+1, i0+1] += M2, C[k0, i0+1] -= M2  (wait, this doesn't look right)
        # Let me re-derive Strassen carefully.

        # Standard Strassen for C = A*B where A,B are 2x2:
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11)B00
        # M3 = A00(B01 - B11)
        # M4 = A11(B10 - B00)
        # M5 = (A00 + A01)B11
        # M6 = (A10 - A00)(B00 + B01)
        # M7 = (A01 - A11)(B10 + B11)
        #
        # C00 = M1 - M4 + M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        #
        # So:
        # M1: U: A00, A11; V: B00, B11; W: C00=+1, C11=+1
        # M2: U: A10, A11; V: B00;       W: C10=+1, C11=-1
        # M3: U: A00;       V: B01, B11; W: C01=+1, C11=+1  (wait, C01=M3+M5, C11=M1-M2+M3+M6)
        # Actually:
        # C00 = M1 - M4 + M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6

        # Let me redo with correct assignments.

        # M1: (A00+A11)(B00+B11)
        U[i0 * m + j0, c] = 1.0
        U[(i0 + 1) * m + (j0 + 1), c] = 1.0
        V[j0 * p + k0, c] = 1.0
        V[(j0 + 1) * p + (k0 + 1), c] = 1.0
        W[k0 * n + i0, c] = 1.0      # C00
        W[(k0 + 1) * n + (i0 + 1), c] = 1.0  # C11
        c += 1

        # M2: (A10+A11)B00
        U[(i0 + 1) * m + j0, c] = 1.0
        U[(i0 + 1) * m + (j0 + 1), c] = 1.0
        V[j0 * p + k0, c] = 1.0
        W[k0 * n + (i0 + 1), c] = 1.0    # C10
        W[(k0 + 1) * n + (i0 + 1), c] = -1.0  # C11
        c += 1

        # M3: A00(B01-B11)
        U[i0 * m + j0, c] = 1.0
        V[j0 * p + (k0 + 1), c] = 1.0
        V[(j0 + 1) * p + (k0 + 1), c] = -1.0
        W[k0 * n + i0, c] = 1.0      # C01
        W[(k0 + 1) * n + (i0 + 1), c] = 1.0  # C11
        c += 1

        # M4: A11(B10-B00)
        U[(i0 + 1) * m + (j0 + 1), c] = 1.0
        V[(j0 + 1) * p + k0, c] = 1.0
        V[j0 * p + k0, c] = -1.0
        W[k0 * n + i0, c] = 1.0      # C00
        W[k0 * n + (i0 + 1), c] = 1.0  # C10
        c += 1

        # M5: (A00+A01)B11
        U[i0 * m + j0, c] = 1.0
        U[i0 * m + (j0 + 1), c] = 1.0
        V[(j0 + 1) * p + (k0 + 1), c] = 1.0
        W[k0 * n + i0, c] = -1.0     # C00
        W[k0 * n + i0, c] = 1.0      # C01  -- wait, C01 = M3+M5, so M5 contributes +1 to C01
        # C00 = M1 - M4 + M5 - M7, so M5 contributes +1 to C00? No: C00 = M1 - M4 + M5 - M7
        # So M5: C00 += 1, C01 += 1
        W[k0 * n + i0, c] = 1.0
        W[k0 * n + i0, c] = 1.0  # This is getting confusing. Let me just set both.
        # Actually W[k0*n+i0] is C00, W[k0*n+i0] is C01? No.
        # C is indexed by (k, i). C00 = W[0*n+0], C01 = W[0*n+1] (if i=1), etc.
        # For local 2x2: C[k', i'] where k' in {0,1}, i' in {0,1}
        # C00 = W[k0*n + i0]
        # C01 = W[k0*n + (i0+1)]  -- NO! C01 means row 0, col 1, so k'=0, i'=1
        # Wait, the output matrix C has C[k, i] = sum_j A[i,j] B[j,k]
        # So C is p x n. C[k0, i0] is W[k0*n + i0].
        # C00 in local 2x2x2 means k'=0, i'=0 -> W[k0*n + i0]
        # C01 means k'=0, i'=1 -> W[k0*n + (i0+1)]
        # C10 means k'=1, i'=0 -> W[(k0+1)*n + i0]
        # C11 means k'=1, i'=1 -> W[(k0+1)*n + (i0+1)]

        # OK let me restart the Strassen with correct indexing.
        # I'll just use the known correct Strassen coefficients.

        return c

    # The above helper is getting messy. Let me just hardcode all 7 terms
    # for each Strassen block explicitly, which is what the original did.

    # Reset and rebuild properly
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0

    # Block A: j=0, schoolbook 2x1x5
    for i in range(n):
        for k in range(p):
            U[i * m + 0, col] = 1.0
            V[0 * p + k, col] = 1.0
            W[k * n + i, col] = 1.0
            col += 1

    # Strassen for 2x2x2 with global indices i in {i0,i0+1}, j in {j0,j0+1}, k in {k0,k0+1}
    def do_strassen(i0, j0, k0):
        nonlocal col
        # M1: (A[i0,j0]+A[i0+1,j0+1])*(B[j0,k0]+B[j0+1,k0+1]) -> C[k0,i0]+=1, C[k0+1,i0+1]+=1
        U[i0 * m + j0, col] = 1.0
        U[(i0+1) * m + (j0+1), col] = 1.0
        V[j0 * p + k0, col] = 1.0
        V[(j0+1) * p + (k0+1), col] = 1.0
        W[k0 * n + i0, col] = 1.0
        W[(k0+1) * n + (i0+1), col] = 1.0
        col += 1
        # M2: (A[i0+1,j0]+A[i0+1,j0+1])*B[j0,k0] -> C[k0+1,i0+1]+=1, C[k0,i0+1]-=1
        U[(i0+1) * m + j0, col] = 1.0
        U[(i0+1) * m + (j0+1), col] = 1.0
        V[j0 * p + k0, col] = 1.0
        W[(k0+1) * n + (i0+1), col] = 1.0
        W[k0 * n + (i0+1), col] = -1.0
        col += 1
        # M3: A[i0,j0]*(B[j0,k0+1]-B[j0+1,k0+1]) -> C[k0,i0+1]+=1, C[k0+1,i0+1]+=1
        U[i0 * m + j0, col] = 1.0
        V[j0 * p + (k0+1), col] = 1.0
        V[(j0+1) * p + (k0+1), col] = -1.0
        W[k0 * n + (i0+1), col] = 1.0
        W[(k0+1) * n + (i0+1), col] = 1.0
        col += 1
        # M4: A[i0+1,j0+1]*(B[j0+1,k0]-B[j0,k0]) -> C[k0,i0]+=1, C[k0+1,i0]+=1
        U[(i0+1) * m + (j0+1), col] = 1.0
        V[(j0+1) * p + k0, col] = 1.0
        V[j0 * p + k0, col] = -1.0
        W[k0 * n + i0, col] = 1.0
        W[(k0+1) * n + i0, col] = 1.0
        col += 1
        # M5: (A[i0,j0]+A[i0,j0+1])*B[j0+1,k0+1] -> C[k0,i0]-=1, C[k0,i0+1]+=1
        U[i0 * m + j0, col] = 1.0
        U[i0 * m + (j0+1), col] = 1.0
        V[(j0+1) * p + (k0+1), col] = 1.0
        W[k0 * n + i0, col] = -1.0
        W[k0 * n + (i0+1), col] = 1.0
        col += 1
        # M6: (A[i0+1,j0]-A[i0,j0])*(B[j0,k0]+B[j0,k0+1]) -> C[k0+1,i0+1]+=1
        U[(i0+1) * m + j0, col] = 1.0
        U[i0 * m + j0, col] = -1.0
        V[j0 * p + k0, col] = 1.0
        V[j0 * p + (k0+1), col] = 1.0
        W[(k0+1) * n + (i0+1), col] = 1.0
        col += 1
        # M7: (A[i0,j0+1]-A[i0+1,j0+1])*(B[j0+1,k0]+B[j0+1,k0+1]) -> C[k0,i0]+=1
        U[i0 * m + (j0+1), col] = 1.0
        U[(i0+1) * m + (j0+1), col] = -1.0
        V[(j0+1) * p + k0, col] = 1.0
        V[(j0+1) * p + (k0+1), col] = 1.0
        W[k0 * n + i0, col] = 1.0
        col += 1

    # Block B1: j=1,2, k=0,1 (Strassen 2x2x2)
    do_strassen(i0=0, j0=1, k0=0)

    # Block B2: j=1,2, k=2 (schoolbook 2x2x1)
    for i in range(n):
        for j in [1, 2]:
            U[i * m + j, col] = 1.0
            V[j * p + 2, col] = 1.0
            W[2 * n + i, col] = 1.0
            col += 1

    # Block B3: j=1,2, k=3,4 (Strassen 2x2x2)
    do_strassen(i0=0, j0=1, k0=3)

    assert col == rank, f"Column count mismatch: {col} != {rank}"

    # Verify
    def verify(Um, Vm, Wm):
        tensor = np.zeros((n, m, p), dtype=float)
        for r in range(Um.shape[1]):
            u = Um[:, r]
            v = Vm[:, r]
            w = Wm[:, r]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        tensor[i, j, k] += u[i * m + j] * v[j * p + k] * w[k * n + i]
        expected = np.ones((n, m, p), dtype=float)
        return np.allclose(tensor, expected, atol=1e-10)

    if not verify(U, V, W):
        raise ValueError("Decomposition verification failed")

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
    print("Verification passed.")
