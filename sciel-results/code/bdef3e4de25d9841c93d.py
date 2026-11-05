"""
Tensor decomposition for 2x3 by 3x10 matrix multiplication.
Rank 55: Strassen on 5 (2x2x2) blocks + schoolbook for remaining 2x1x10 part.
"""
import time
import numpy as np


def construct():
    start_time = time.time()
    n, m, p = 2, 3, 10

    rank = 55
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    idx = 0

    # Part 1: Strassen blocks for rows 0,1 of A and rows 0,1 of B
    # We have 10 columns in B, group them into 5 pairs of 2 columns.
    # For each pair (k0, k0+1), apply Strassen 2x2x2.
    for block in range(5):
        k0 = 2 * block
        k1 = 2 * block + 1

        # Strassen M1 = (A[0,0]+A[1,1]) * (B[0,k0]+B[1,k1])
        # A indices: (0,0)=0, (1,1)=4
        # B indices: (0,k0)=k0, (1,k1)=m+k1=3+k1
        U[0, idx] = 1.0
        U[4, idx] = 1.0
        V[k0, idx] = 1.0
        V[3 + k1, idx] = 1.0
        # W: contributes to C[0,k0] and C[1,k1]
        # W index = k*n + i = k*2 + i
        W[k0 * 2 + 0, idx] = 1.0
        W[k1 * 2 + 1, idx] = 1.0
        idx += 1

        # M2 = (A[1,0]+A[1,1]) * B[0,k0]
        # A: (1,0)=3, (1,1)=4
        # B: (0,k0)=k0
        U[3, idx] = 1.0
        U[4, idx] = 1.0
        V[k0, idx] = 1.0
        # C = M1 + M2 - M5 (for C[1,k0]) and C = M1 - M2 (for C[1,k1])
        # Wait, let me re-derive Strassen carefully.
        # Standard Strassen:
        # C[0,0] = M1 - M4 + M6 + M7
        # C[1,0] = M2 + M4
        # C[0,1] = M3 + M5
        # C[1,1] = M1 - M2 + M3 + M5
        # where:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)

        # So C[1,0] = M2 + M4
        # C[1,1] = M1 - M2 + M3 + M5

        # Redo with correct assignments:
        # Let me restart the Strassen part properly.

    # Let me redo from scratch with correct Strassen formulas.
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    idx = 0

    for block in range(5):
        k0 = 2 * block
        k1 = 2 * block + 1

        # M1 = (A[0,0]+A[1,1]) * (B[0,k0]+B[1,k1])
        U[0, idx] = 1.0;   U[4, idx] = 1.0
        V[k0, idx] = 1.0;  V[3 + k1, idx] = 1.0
        W[k0*2+0, idx] = 1.0;  W[k1*2+1, idx] = 1.0
        idx += 1

        # M2 = (A[1,0]+A[1,1]) * B[0,k0]
        U[3, idx] = 1.0;   U[4, idx] = 1.0
        V[k0, idx] = 1.0
        W[k0*2+1, idx] = 1.0;  W[k1*2+1, idx] = -1.0
        idx += 1

        # M3 = A[0,0] * (B[0,k1]-B[1,k1])
        U[0, idx] = 1.0
        V[3+k1, idx] = -1.0;  V[k1, idx] = 1.0
        W[k1*2+0, idx] = 1.0;  W[k1*2+1, idx] = 1.0
        idx += 1

        # M4 = A[1,1] * (B[1,k0]-B[0,k0])
        U[4, idx] = 1.0
        V[k0, idx] = -1.0;  V[3+k0, idx] = 1.0
        W[k0*2+0, idx] = -1.0;  W[k0*2+1, idx] = 1.0
        idx += 1

        # M5 = (A[0,0]+A[0,1]) * B[1,k1]
        U[0, idx] = 1.0;   U[1, idx] = 1.0
        V[3+k1, idx] = 1.0
        W[k0*2+0, idx] = -1.0;  W[k1*2+0, idx] = 1.0
        idx += 1

        # M6 = (A[1,0]-A[0,0]) * (B[0,k0]+B[0,k1])
        U[3, idx] = 1.0;   U[0, idx] = -1.0
        V[k0, idx] = 1.0;  V[k1, idx] = 1.0
        W[k0*2+0, idx] = 1.0
        idx += 1

        # M7 = (A[0,1]-A[1,1]) * (B[1,k0]+B[1,k1])
        U[1, idx] = 1.0;   U[4, idx] = -1.0
        V[3+k0, idx] = 1.0;  V[3+k1, idx] = 1.0
        W[k0*2+0, idx] = 1.0
        idx += 1

    # Part 2: Schoolbook for A[2,:]*B[2,:] -> C[:,10]... no.
    # A is 2x3, B is 3x10. A row 2 doesn't exist. A is 2 rows.
    # B is 3 rows. The "extra" is B row 2 (index 2), which multiplies A columns 0,1.
    # Actually: C[i,k] = sum_j A[i,j]*B[j,k], j=0,1,2
    # We handled j=0,1 in Strassen (rows 0,1 of B).
    # Remaining: j=2 term: A[i,2]*B[2,k] for i=0,1 and k=0..9
    # This is 2*10 = 20 schoolbook terms.

    for i in range(2):
        for k in range(10):
            U[i*3 + 2, idx] = 1.0   # A[i,2] -> index i*3+2
            V[2*10 + k, idx] = 1.0  # B[2,k] -> index 2*10+k = 20+k
            W[k*2 + i, idx] = 1.0   # W[k*n+i] = W[k*2+i]
            idx += 1

    # Trim to actual rank used
    U = U[:, :idx]
    V = V[:, :idx]
    W = W[:, :idx]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


if __name__ == "__main__":
    result = construct()
    rank = len(result["U"][0])
    print(f"Rank: {rank}")

    n, m, p = 2, 3, 10
    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])
    rank = U.shape[1]

    np.random.seed(42)
    A = np.random.randint(-5, 6, size=(n, m)).astype(float)
    B = np.random.randint(-5, 6, size=(m, p)).astype(float)

    C_decomp = np.zeros((n, p))
    for r in range(rank):
        prod_A = 0.0
        for i in range(n):
            for j in range(m):
                prod_A += U[i * m + j, r] * A[i, j]

        prod_B = 0.0
        for j in range(m):
            for k in range(p):
                prod_B += V[j * p + k, r] * B[j, k]

        val = prod_A * prod_B
        for i in range(n):
            for k in range(p):
                C_decomp[i, k] += val * W[k * n + i, r]

    C_direct = A @ B
    error = np.max(np.abs(C_decomp - C_direct))
    print(f"Max error: {error}")
    if error > 1e-10:
        print("WARNING: Decomposition is incorrect!")

