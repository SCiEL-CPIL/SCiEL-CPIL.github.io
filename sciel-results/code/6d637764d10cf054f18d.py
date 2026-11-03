import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    deadline = 140.0

    def _to_list(arr):
        return arr.tolist()

    def _verify(U, V, W, T):
        recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.allclose(recon, T, atol=1e-9)

    # Build rank-68 decomposition:
    # 7 Strassen terms for A[0:2,0:2] * B[0:2,0:8] (2x2 x 2x8)
    # 16 schoolbook terms for A[0:2,2] * B[2,0:8] (2x1 x 1x8)
    # 16 schoolbook terms for A[2,0:2] * B[0:2,0:8] (1x2 x 2x8)
    # 8 schoolbook terms for A[2,2] * B[2,0:8] (1x1 x 1x8)
    # Total: 7 + 16 + 16 + 8 = 47... wait, that's not 68.
    # Let me recalculate. Actually for 2x2 x 2x8:
    # Strassen gives 7 terms for 2x2x2, but we need 2x2x8.
    # We apply Strassen to each 2x2x2 sub-block of B: B[0:2, 0:2], B[0:2, 2:4], B[0:2, 4:6], B[0:2, 6:8]
    # That's 4 * 7 = 28 terms for A[0:2,0:2] * B[0:2,0:8]
    # 16 terms for A[0:2,2] * B[2,0:8]
    # 16 terms for A[2,0:2] * B[0:2,0:8]
    # 8 terms for A[2,2] * B[2,0:8]
    # Total: 28 + 16 + 16 + 8 = 68. Correct!

    rank = 68
    U = np.zeros((n * m, rank))  # 9 x 68
    V = np.zeros((m * p, rank))  # 24 x 68
    W = np.zeros((p * n, rank))  # 24 x 68

    r = 0  # current rank index

    # Strassen for 2x2 x 2x2 block, applied to each column-block of B
    # Strassen terms for multiplying 2x2 A by 2x2 B:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6

    def add_strassen_block(b_offset):
        """Add 7 Strassen terms for A[0:2,0:2] * B[0:2, b_offset:b_offset+2]."""
        nonlocal r
        # B indices: B[j, k] -> V index j*8+k
        # For b_offset, B columns are b_offset and b_offset+1
        b0 = b_offset
        b1 = b_offset + 1

        # M1 = (A00+A11)(B00+B11)
        # U: A00 at idx 0, A11 at idx 3
        # V: B00 at idx 0*8+b0, B11 at idx 1*8+b1
        # W: contributes to C00 (idx b0*3+0) and C11 (idx b1*3+1)
        U[0, r] = 1; U[3, r] = 1
        V[0*8+b0, r] = 1; V[1*8+b1, r] = 1
        W[b0*3+0, r] = 1; W[b1*3+1, r] = 1
        r += 1

        # M2 = (A10+A11)B00
        # U: A10 at idx 1, A11 at idx 3
        # V: B00 at idx 0*8+b0
        # W: C10 (idx b0*3+1), C11 (idx b1*3+1) with -1
        U[1, r] = 1; U[3, r] = 1
        V[0*8+b0, r] = 1
        W[b0*3+1, r] = 1; W[b1*3+1, r] = -1
        r += 1

        # M3 = A00(B10-B11)
        # U: A00 at idx 0
        # V: B10 at idx 1*8+b0, B11 at idx 1*8+b1
        # W: C01 (idx b1*3+0), C11 (idx b1*3+1)
        U[0, r] = 1
        V[1*8+b0, r] = 1; V[1*8+b1, r] = -1
        W[b1*3+0, r] = 1; W[b1*3+1, r] = 1
        r += 1

        # M4 = A11(B01-B00)
        # U: A11 at idx 3
        # V: B01 at idx 0*8+b1, B00 at idx 0*8+b0
        # W: C00 (idx b0*3+0), C10 (idx b0*3+1)
        U[3, r] = 1
        V[0*8+b1, r] = 1; V[0*8+b0, r] = -1
        W[b0*3+0, r] = 1; W[b0*3+1, r] = 1
        r += 1

        # M5 = (A00+A01)B11
        # U: A00 at idx 0, A01 at idx 2
        # V: B11 at idx 1*8+b1
        # W: C00 (idx b0*3+0) with -1, C01 (idx b1*3+0)
        U[0, r] = 1; U[2, r] = 1
        V[1*8+b1, r] = 1
        W[b0*3+0, r] = -1; W[b1*3+0, r] = 1
        r += 1

        # M6 = (A10-A00)(B00+B01)
        # U: A10 at idx 1, A00 at idx 0
        # V: B00 at idx 0*8+b0, B01 at idx 0*8+b1
        # W: C11 (idx b1*3+1)
        U[1, r] = 1; U[0, r] = -1
        V[0*8+b0, r] = 1; V[0*8+b1, r] = 1
        W[b1*3+1, r] = 1
        r += 1

        # M7 = (A01-A11)(B10+B11)
        # U: A01 at idx 2, A11 at idx 3
        # V: B10 at idx 1*8+b0, B11 at idx 1*8+b1
        # W: C00 (idx b0*3+0)
        U[2, r] = 1; U[3, r] = -1
        V[1*8+b0, r] = 1; V[1*8+b1, r] = 1
        W[b0*3+0, r] = 1
        r += 1

    # Apply Strassen to 4 column-blocks of B[0:2, 0:8]
    for b_off in [0, 2, 4, 6]:
        add_strassen_block(b_off)

    # Schoolbook: A[0:2, 2] * B[2, 0:8] -> C[0:2, 0:8]
    # A[i,2] at U idx i*3+2 for i=0,1
    # B[2,k] at V idx 2*8+k = 16+k
    # C[i,k] at W idx k*3+i
    for i in range(2):
        for k in range(8):
            U[i*3+2, r] = 1
            V[16+k, r] = 1
            W[k*3+i, r] = 1
            r += 1

    # Schoolbook: A[2, 0:2] * B[0:2, 0:8] -> C[2, 0:8]
    # A[2,j] at U idx 2*3+j = 6+j for j=0,1
    # B[j,k] at V idx j*8+k
    # C[2,k] at W idx k*3+2
    for j in range(2):
        for k in range(8):
            U[6+j, r] = 1
            V[j*8+k, r] = 1
            W[k*3+2, r] = 1
            r += 1

    # Schoolbook: A[2,2] * B[2, 0:8] -> C[2, 0:8]
    # A[2,2] at U idx 8
    # B[2,k] at V idx 16+k
    # C[2,k] at W idx k*3+2
    for k in range(8):
        U[8, r] = 1
        V[16+k, r] = 1
        W[k*3+2, r] = 1
        r += 1

    assert r == rank, f"Rank mismatch: {r} != {rank}"

    # Verify
    T = np.ones((n, m, p))
    if _verify(U, V, W, T):
        return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}

    # Fallback to schoolbook (rank 72)
    rank_sb = n * m * p
    U = np.zeros((n * m, rank_sb))
    V = np.zeros((m * p, rank_sb))
    W = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
