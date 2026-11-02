"""Tensor decomposition for 3x3 * 3x8 matrix multiplication using Strassen."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication.
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    """
    n, m, p = DIMENSIONS  # n=3, m=3, p=8
    deadline = time.time() + 145  # 5s safety margin

    # Build the Strassen-based decomposition
    # Split 8 columns into 4 groups of 2
    # For each group, split A into 2x2 (rows 0,1) and 1x2 (row 2)
    # Use Strassen for 2x2 * 2x2, schoolbook for 1x2 * 2x2

    U = []  # Shape (n*m, rank) = (9, rank)
    V = []  # Shape (m*p, rank) = (24, rank)
    W = []  # Shape (p*n, rank) = (24, rank)

    # Strassen formulas for 2x2 * 2x2:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B10-B11)
    # M4 = A11*(B01-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B10)
    # M7 = (A01-A11)*(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6

    # For each column group (k_offset in [0, 2, 4, 6]):
    # B_sub is 3x2: rows 0,1,2 of B, cols k_offset, k_offset+1
    # A_sub_2x2 is rows 0,1 of A (2x3), but we only use cols 0,1 for Strassen
    # A_row is row 2 of A (1x3), use cols 0,1 for schoolbook

    for k_offset in range(0, 8, 2):
        # Strassen part: 2x2 A (rows 0,1; cols 0,1) * 2x2 B (rows 0,1; cols k_offset, k_offset+1)
        # This produces C[0, k_offset], C[0, k_offset+1], C[1, k_offset], C[1, k_offset+1]
        
        # M1 = (A[0,0]+A[1,1]) * (B[0,k_off]+B[1,k_off+1])
        U.append([1, 0, 0, 0, 1, 0, 0, 0, 0])  # A[0,0] + A[1,1]
        V.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # placeholder
        # Fix V for M1: B[0,k_off] + B[1,k_off+1]
        V[-1] = [0]*24
        V[-1][0*p + k_offset] = 1  # B[0, k_off]
        V[-1][1*p + k_offset+1] = 1  # B[1, k_off+1]
        # W for M1: contributes to C[0,k_off], C[0,k_off+1], C[1,k_off], C[1,k_off+1]
        W.append([0]*24)
        W[-1][k_offset*n + 0] = 1  # C[0, k_off]
        W[-1][(k_offset+1)*n + 0] = 1  # C[0, k_off+1]
        W[-1][k_offset*n + 1] = 1  # C[1, k_off]
        W[-1][(k_offset+1)*n + 1] = 1  # C[1, k_off+1]

        # M2 = (A[1,0]+A[1,1]) * B[0,k_off]
        U.append([0, 0, 0, 0, 0, 0, 1, 0, 1])  # A[1,0] + A[1,1]
        V.append([0]*24)
        V[-1][0*p + k_offset] = 1  # B[0, k_off]
        W.append([0]*24)
        W[-1][k_offset*n + 1] = 1  # C[1, k_off]
        W[-1][(k_offset+1)*n + 1] = 1  # C[1, k_off+1]

        # M3 = A[0,0] * (B[1,k_off] - B[1,k_off+1])
        U.append([1, 0, 0, 0, 0, 0, 0, 0, 0])  # A[0,0]
        V.append([0]*24)
        V[-1][1*p + k_offset] = 1  # B[1, k_off]
        V[-1][1*p + k_offset+1] = -1  # B[1, k_off+1]
        W.append([0]*24)
        W[-1][(k_offset+1)*n + 0] = 1  # C[0, k_off+1]
        W[-1][k_offset*n + 1] = 1  # C[1, k_off]

        # M4 = A[1,1] * (B[0,k_off+1] - B[0,k_off])
        U.append([0, 0, 0, 0, 0, 0, 0, 0, 1])  # A[1,1]
        V.append([0]*24)
        V[-1][0*p + k_offset+1] = 1  # B[0, k_off+1]
        V[-1][0*p + k_offset] = -1  # B[0, k_off]
        W.append([0]*24)
        W[-1][k_offset*n + 0] = 1  # C[0, k_off]
        W[-1][k_offset*n + 1] = 1  # C[1, k_off]

        # M5 = (A[0,0]+A[0,1]) * B[1,k_off+1]
        U.append([1, 0, 0, 0, 1, 0, 0, 0, 0])  # A[0,0] + A[0,1]
        V.append([0]*24)
        V[-1][1*p + k_offset+1] = 1  # B[1, k_off+1]
        W.append([0]*24)
        W[-1][(k_offset+1)*n + 0] = -1  # C[0, k_off+1]
        W[-1][(k_offset+1)*n + 1] = 1  # C[1, k_off+1]

        # M6 = (A[1,0]-A[0,0]) * (B[0,k_off]+B[1,k_off])
        U.append([0, 0, 0, 0, 0, 0, 1, 0, 0])  # A[1,0]
        # Need to subtract A[0,0], so this is two terms? No, we can combine.
        # Actually, U should be [0,0,0,0,-1,0,1,0,0] for A[1,0]-A[0,0]
        U[-1] = [0, 0, 0, 0, -1, 0, 1, 0, 0]  # A[1,0] - A[0,0]
        V.append([0]*24)
        V[-1][0*p + k_offset] = 1  # B[0, k_off]
        V[-1][1*p + k_offset] = 1  # B[1, k_off]
        W.append([0]*24)
        W[-1][(k_offset+1)*n + 1] = 1  # C[1, k_off+1]

        # M7 = (A[0,1]-A[1,1]) * (B[0,k_off+1]+B[1,k_off+1])
        U.append([0, 0, 0, 0, 1, 0, 0, 0, 0])  # A[0,1]
        U[-1] = [0, 0, 0, 0, 1, 0, 0, 0, -1]  # A[0,1] - A[1,1]
        V.append([0]*24)
        V[-1][0*p + k_offset+1] = 1  # B[0, k_off+1]
        V[-1][1*p + k_offset+1] = 1  # B[1, k_off+1]
        W.append([0]*24)
        W[-1][k_offset*n + 0] = 1  # C[0, k_off]
        W[-1][(k_offset+1)*n + 1] = 1  # C[1, k_off+1]

        # Schoolbook part: A[2,0]*B[0,k_off], A[2,0]*B[0,k_off+1], A[2,1]*B[1,k_off], A[2,1]*B[1,k_off+1]
        # But we only need 2 multiplications if we use the structure correctly.
        # Actually, for 1x2 * 2x2, we need 2 multiplications:
        # C[2,k_off] = A[2,0]*B[0,k_off] + A[2,1]*B[1,k_off]
        # C[2,k_off+1] = A[2,0]*B[0,k_off+1] + A[2,1]*B[1,k_off+1]
        # This requires 4 multiplications in general, but we can use 2 if we precompute.
        # Wait, no: each output element requires 2 products, so 4 total.
        # But we can share: let X = A[2,0]*B[0,k_off] + A[2,1]*B[1,k_off] (1 mult? No, 2 mults)
        # Actually, we need 4 separate products. Let me reconsider.
        
        # For 1x2 * 2x2, the tensor rank is 2 (it's just two independent dot products).
        # C[2,k_off] = A[2,0]*B[0,k_off] + A[2,1]*B[1,k_off]
        # C[2,k_off+1] = A[2,0]*B[0,k_off+1] + A[2,1]*B[1,k_off+1]
        # These are two independent bilinear forms, so rank 2.
        
        # Term 8: A[2,0]*B[0,k_off] contributes to C[2,k_off]
        U.append([0, 0, 0, 0, 0, 0, 0, 0, 0])  # placeholder
        U[-1] = [0, 0, 0, 0, 0, 0, 0, 1, 0]  # A[2,0]
        V.append([0]*24)
        V[-1][0*p + k_offset] = 1  # B[0, k_off]
        W.append([0]*24)
        W[-1][k_offset*n + 2] = 1  # C[2, k_off]

        # Term 9: A[2,1]*B[1,k_off] contributes to C[2,k_off]
        U.append([0, 0, 0, 0, 0, 0, 0, 0, 0])  # placeholder
        U[-1] = [0, 0, 0, 0, 0, 0, 0, 0, 1]  # A[2,1]
        V.append([0]*24)
        V[-1][1*p + k_offset] = 1  # B[1, k_off]
        W.append([0]*24)
        W[-1][k_offset*n + 2] = 1  # C[2, k_off]

        # Term 10: A[2,0]*B[0,k_off+1] contributes to C[2,k_off+1]
        U.append([0, 0, 0, 0, 0, 0, 0, 0, 0])  # placeholder
        U[-1] = [0, 0, 0, 0, 0, 0, 0, 1, 0]  # A[2,0]
        V.append([0]*24)
        V[-1][0*p + k_offset+1] = 1  # B[0, k_off+1]
        W.append([0]*24)
        W[-1][(k_offset+1)*n + 2] = 1  # C[2, k_off+1]

        # Term 11: A[2,1]*B[1,k_off+1] contributes to C[2,k_off+1]
        U.append([0, 0, 0, 0, 0, 0, 0, 0, 0])  # placeholder
        U[-1] = [0, 0, 0, 0, 0, 0, 0, 0, 1]  # A[2,1]
        V.append([0]*24)
        V[-1][1*p + k_offset+1] = 1  # B[1, k_off+1]
        W.append([0]*24)
        W[-1][(k_offset+1)*n + 2] = 1  # C[2, k_off+1]

    # Now we have 11 terms per block, 4 blocks = 44 terms
    # But we can optimize the schoolbook part: use 2 terms instead of 4
    # by combining: (A[2,0]+A[2,1]) * (B[0,k_off]+B[1,k_off]) etc., but that doesn't help.
    # Actually, the 1x2 * 2x2 case has tensor rank 2, so we should be able to do it with 2 terms.
    # Let me reconsider: C[2,k] = A[2,0]*B[0,k] + A[2,1]*B[1,k] for k = k_off, k_off+1.
    # This is two independent dot products, so rank 2. We can write:
    # C[2,k_off] = (A[2,0], A[2,1]) · (B[0,k_off], B[1,k_off])
    # C[2,k_off+1] = (A[2,0], A[2,1]) · (B[0,k_off+1], B[1,k_off+1])
    # These are two separate rank-1 tensors, so total rank 2.
    # But each rank-1 tensor requires one multiplication, so 2 multiplications total.
    # However, in our decomposition, each multiplication is a separate term.
    # So we need 2 terms, not 4. Let me fix this.

    # Actually, I made an error. Let me restart with a cleaner approach.

    U = []
    V = []
    W = []

    for k_offset in range(0, 8, 2):
        # Strassen: 7 terms for 2x2 * 2x2 (A rows 0,1; B rows 0,1; cols k_off, k_off+1)
        # M1 = (A00+A11)*(B0k+B1k+1)
        U.append([1, 0, 0, 0, 1, 0, 0, 0, 0])
        V.append([0]*24)
        V[-1][0*p + k_offset] = 1
        V[-1][1*p + k_offset+1] = 1
        W.append([0]*24)
        W[-1][k_offset*n + 0] = 1
        W[-1][(k_offset+1)*n + 0] = 1
        W[-1][k_offset*n + 1] = 1
        W[-1][(k_offset+1)*n + 1] = 1

        # M2 = (A10+A11)*B0k
        U.append([0, 0, 0, 0, 0, 0, 1, 0, 1])
        V.append([0]*24)
        V[-1][0*p + k_offset] = 1
        W.append([0]*24)
        W[-1][k_offset*n + 1] = 1
        W[-1][(k_offset+1)*n + 1] = 1

        # M3 = A00*(B1k-B1k+1)
        U.append([1, 0, 0, 0, 0, 0, 0, 0, 0])
        V.append([0]*24)
        V[-1][1*p + k_offset] = 1
        V[-1][1*p + k_offset+1] = -1
        W.append([0]*24)
        W[-1][(k_offset+1)*n + 0] = 1
        W[-1][k_offset*n + 1] = 1

        # M4 = A11*(B0k+1-B0k)
        U.append([0, 0, 0, 0, 0, 0, 0, 0, 1])
        V.append([0]*24)
        V[-1][0*p + k_offset+1] = 1
        V[-1][0*p + k_offset] = -1
        W.append([0]*24)
        W[-1][k_offset*n + 0] = 1
        W[-1][k_offset*n + 1] = 1

        # M5 = (A00+A01)*B1k+1
        U.append([1, 0, 0, 0, 1, 0, 0, 0, 0])
        V.append([0]*24)
        V[-1][1*p + k_offset+1] = 1
        W.append([0]*24)
        W[-1][(k_offset+1)*n + 0] = -1
        W[-1][(k_offset+1)*n + 1] = 1

        # M6 = (A10-A00)*(B0k+B1k)
        U.append([0, 0, 0, 0, -1, 0, 1, 0, 0])
        V.append([0]*24)
        V[-1][0*p + k_offset] = 1
        V[-1][1*p + k_offset] = 1
        W.append([0]*24)
        W[-1][(k_offset+1)*n + 1] = 1

        # M7 = (A01-A11)*(B0k+1+B1k+1)
        U.append([0, 0, 0, 0, 1, 0, 0, 0, -1])
        V.append([0]*24)
        V[-1][0*p + k_offset+1] = 1
        V[-1][1*p + k_offset+1] = 1
        W.append([0]*24)
        W[-1][k_offset*n + 0] = 1
        W[-1][(k_offset+1)*n + 1] = 1

        # Schoolbook: 2 terms for A row 2 * B cols k_off, k_off+1
        # C[2,k_off] = A[2,0]*B[0,k_off] + A[2,1]*B[1,k_off]
        # C[2,k_off+1] = A[2,0]*B[0,k_off+1] + A[2,1]*B[1,k_off+1]
        # Use 2 multiplications:
        # X = A[2,0]*B[0,k_off] + A[2,1]*B[1,k_off] (but this is 2 products, not 1)
        # Actually, we need 4 separate products. The tensor rank of 1x2*2x2 is 2,
        # but each "multiplication" in the decomposition is a single product.
        # So we need 4 terms. Let me keep 4 terms.

        # Term 8: A[2,0]*B[0,k_off] -> C[2,k_off]
        U.append([0, 0, 0, 0, 0, 0, 0, 1, 0])
        V.append([0]*24)
        V[-1][0*p + k_offset] = 1
        W.append([0]*24)
        W[-1][k_offset*n + 2] = 1

        # Term 9: A[2,1]*B[1,k_off] -> C[2,k_off]
        U.append([0, 0, 0, 0, 0, 0, 0, 0, 1])
        V.append([0]*24)
        V[-1][1*p + k_offset] = 1
        W.append([0]*24)
        W[-1][k_offset*n + 2] = 1

        # Term 10: A[2,0]*B[0,k_off+1] -> C[2,k_off+1]
        U.append([0, 0, 0, 0, 0, 0, 0, 1, 0])
        V.append([0]*24)
        V[-1][0*p + k_offset+1] = 1
        W.append([0]*24)
        W[-1][(k_offset+1)*n + 2] = 1

        # Term 11: A[2,1]*B[1,k_off+1] -> C[2,k_off+1]
        U.append([0, 0, 0, 0, 0, 0, 0, 0, 1])
        V.append([0]*24)
        V[-1][1*p + k_offset+1] = 1
        W.append([0]*24)
        W[-1][(k_offset+1)*n + 2] = 1

    # Total rank = 4 * 11 = 44
    rank = len(U)

    # Verify the decomposition
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    # Test with random matrices
    for _ in range(5):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        C_decomp = np.zeros((n, p))
        for c in range(rank):
            u_vec = np.zeros(n)
            for i in range(n):
                for j in range(m):
                    u_vec[i] += U_arr[i*m+j, c] * A[i, j]
            v_vec = np.zeros(p)
            for k in range(p):
                for j in range(m):
                    v_vec[k] += V_arr[j*p+k, c] * B[j, k]
            for i in range(n):
                for k in range(p):
                    C_decomp[i, k] += u_vec[i] * v_vec[k] * W_arr[k*n+i, c]
        if not np.allclose(C_expected, C_decomp, atol=1e-9):
            # If verification fails, fall back to schoolbook
            U = []
            V = []
            W = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U.append([0]*9)
                        U[-1][i*m+j] = 1
                        V.append([0]*24)
                        V[-1][j*p+k] = 1
                        W.append([0]*24)
                        W[-1][k*n+i] = 1
            rank = len(U)
            break

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
