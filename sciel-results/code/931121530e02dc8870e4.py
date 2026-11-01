"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import itertools

DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    # Build a rank-21 tensor decomposition for 2x2 * 2x6 matrix multiplication.
    # n=2, m=2, p=6.
    n, m, p = DIMENSIONS
    U_size = n * m  # 4
    V_size = m * p  # 12
    W_size = p * n  # 12

    rank = 21
    U = np.zeros((U_size, rank), dtype=np.int64)
    V = np.zeros((V_size, rank), dtype=np.int64)
    W = np.zeros((W_size, rank), dtype=np.int64)

    # We use a structured approach:
    # The multiplication tensor T has entries T[i, j, k] where A is n x m, B is m x p.
    # Output C[i, k] = sum_j A[i, j] * B[j, k]
    # In the trace convention: W index is k*n + i, U index is i*m + j, V index is j*p + k
    # So T[U_idx, V_idx, W_idx] = 1 when i*m+j = U_idx, j*p+k = V_idx, k*n+i = W_idx
    # for some (i, j, k).

    # Rank-21 decomposition for 2x2 * 2x6:
    # Think of it as two 2x2 * 2x3 blocks.
    # For each 2x2 * 2x3, we can potentially use rank 10 (known: 2x2*2x3 has rank 10).
    # Two such blocks would give rank 20, but we need to be careful about sharing.
    # Actually, 2x2*2x3 has known rank 10. So 2x2*2x6 = two 2x2*2x3 blocks, rank 20.
    # But can we do better? Let's check if rank 20 is achievable.
    # Actually, let me try to construct rank 20 first.

    # Let me use a different approach: try to find a valid decomposition.
    # Start with the schoolbook rank-24 and try to reduce.

    # Schoolbook: 24 terms, one for each (i,j,k).
    # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1

    # Let's try to find rank-20 by merging terms.
    # For fixed i, we have m*p = 12 terms. For i=0: j=0,1; k=0,1,2,3,4,5 -> 12 terms
    # For i=1: j=0,1; k=0,1,2,3,4,5 -> 12 terms

    # Alternative: use the fact that for 2x2 * 2x6, we can view it as:
    # C[0, :] = A[0,0]*B[0,:] + A[0,1]*B[1,:]
    # C[1, :] = A[1,0]*B[0,:] + A[1,1]*B[1,:]

    # Each row of C is a linear combination of two rows of B.
    # C[0,:] has 6 elements, C[1,:] has 6 elements.

    # Let me try a rank-20 construction:
    # Split the 6 columns of B into two groups of 3.
    # For group 1 (k=0,1,2): we need to compute A * B[:, 0:3]
    # For group 2 (k=3,4,5): we need to compute A * B[:, 3:6]

    # For 2x2 * 2x3, the known minimum rank is 10.
    # So total rank = 10 + 10 = 20.

    # Let me construct a rank-10 decomposition for 2x2 * 2x3.
    # This is a known result. Let me derive it.

    # For 2x2 * 2x3:
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01, b02], [b10, b11, b12]]
    # C = A*B, C[0,:] = [a00*b00+a01*b10, a00*b01+a01*b11, a00*b02+a01*b12]
    # C[1,:] = [a10*b00+a11*b10, a10*b01+a11*b11, a10*b02+a11*b12]

    # I'll use a known rank-10 decomposition for 2x2*2x3.
    # Actually, let me just try to verify if rank 20 works by construction.

    # Let me use a simpler approach: try rank 20 and verify.
    # If it doesn't work, fall back to rank 21.

    # Actually, let me try to construct a valid rank-20 decomposition.
    # I'll use the following strategy: for each half (3 columns), use a rank-10 block.

    # For 2x2 * 2x3, here's a rank-10 decomposition:
    # Variables: a00, a01, a10, a11, b00, b01, b02, b10, b11, b12
    # Products needed: 6 products for C[0,:], 6 for C[1,:], total 12 scalar products
    # but we share the A entries.

    # Let me just try rank 20 with a specific construction and verify.

    # Rank-20 construction:
    # Columns 0-2: use 10 ranks
    # Columns 3-5: use 10 ranks

    # For columns 0-2 (k=0,1,2), I need to compute:
    # C[0,0] = a00*b00 + a01*b10
    # C[0,1] = a00*b01 + a01*b11
    # C[0,2] = a00*b02 + a01*b12
    # C[1,0] = a10*b00 + a11*b10
    # C[1,1] = a10*b01 + a11*b11
    # C[1,2] = a10*b02 + a11*b12

    # Let me use the following rank-10 for 2x2*2x3:
    # I'll index the 10 rank terms.

    # Actually, let me just try to verify a rank-20 decomposition by building it
    # and checking against the tensor.

    # Let me try a different approach: use known results.
    # The tensor rank of 2x2 * 2x6 matrix multiplication.
    # Lower bound: max(n*m, m*p, p*n) = max(4, 12, 12) = 12
    # Upper bound: schoolbook = 24
    # Known: for 2x2 * 2x3, rank is 10 (this is a known result).
    # So 2x2 * 2x6 should have rank at most 20.

    # Let me construct rank-20 and verify.

    # For the first 3 columns (k=0,1,2), I'll use these 10 rank terms:
    # Term 0: a00 * b00 -> contributes to C[0,0]
    # Term 1: a01 * b10 -> contributes to C[0,0]
    # Term 2: a00 * b01 -> contributes to C[0,1]
    # Term 3: a01 * b11 -> contributes to C[0,1]
    # Term 4: a00 * b02 -> contributes to C[0,2]
    # Term 5: a01 * b12 -> contributes to C[0,2]
    # Term 6: a10 * b00 -> contributes to C[1,0]
    # Term 7: a11 * b10 -> contributes to C[1,0]
    # Term 8: a10 * b01 -> contributes to C[1,1]
    # Term 9: a11 * b11 -> contributes to C[1,1]
    # But we're missing a10*b02 and a11*b12!

    # So we need 12 terms for the schoolbook of 2x2*2x3, but rank 10 means we can
    # share some products. Let me think about this differently.

    # The key insight for rank reduction is that we can compute combinations.
    # For example, (a00+a10)*(b00+b10) = a00*b00 + a00*b10 + a10*b00 + a10*b10
    # This gives us 4 products in 1 rank term, but we need to subtract the unwanted ones.

    # This is getting complex. Let me just try rank 21 with a Strassen-like approach.

    # Actually, let me try to verify if a rank-20 decomposition exists by using
    # a more systematic approach. For now, let me construct a rank-21 decomposition
    # that I can verify, and then try to optimize.

    # Let me use a different strategy: I'll construct the decomposition and then
    # verify it by checking that for all (i,j,k), the sum over c of
    # U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1, and for all other
    # (U_idx, V_idx, W_idx) combinations, the sum is 0.

    # For rank-21, let me try the following:
    # Use Strassen's trick for the 2x2 part, combined with the 6 columns.

    # Strassen for 2x2*2x2 uses 7 multiplications.
    # For 2x2*2x6, we can view it as 3 separate 2x2*2x2 multiplications (one for each
    # pair of columns), giving 3*7 = 21 rank terms.

    # This is the approach! Let me construct it.

    # For columns (0,1), (2,3), (4,5), use Strassen's 7-multiplication algorithm.

    # Strassen's algorithm for 2x2 * 2x2:
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b01-b11)
    # M4 = a11*(b10-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b01)
    # M7 = (a01-a11)*(b10+b11)
    #
    # C00 = M1 - M4 + M6
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M7

    # For our problem, A is 2x2, B is 2x6.
    # Split B into three 2x2 blocks: B1 = B[:, 0:2], B2 = B[:, 2:4], B3 = B[:, 4:6]
    # C = A * B = [A*B1, A*B2, A*B3]

    # For each block, use Strassen's 7 multiplications. Total: 21.

    # Let me construct this.

    col = 0
    for block in range(3):
        # Block has columns 2*block and 2*block+1
        k0 = 2 * block
        k1 = 2 * block + 1

        # M1 = (a00+a11)*(b00+b11)
        # In our indexing:
        # a00 is A[0,0] -> U index 0*m+0 = 0
        # a11 is A[1,1] -> U index 1*m+1 = 3
        # b00 is B[0,k0] -> V index 0*p+k0 = 0*6+k0 = k0
        # b11 is B[1,k1] -> V index 1*p+k1 = 6+k1

        # For M1: U has a00 and a11, V has b00 and b11
        # U[0, col] += 1, U[3, col] += 1
        # V[k0, col] += 1, V[6+k1, col] += 1
        # W contributions:
        # C00 += M1 -> W[k0*n+0, col] += 1 = W[k0*2+0, col] = W[2*k0, col]
        # C11 += M1 -> W[k1*n+1, col] += 1 = W[k1*2+1, col] = W[2*k1+1, col]

        U[0, col] += 1
        U[3, col] += 1
        V[k0, col] += 1
        V[6 + k1, col] += 1
        W[2 * k0, col] += 1
        W[2 * k1 + 1, col] += 1
        col += 1

        # M2 = (a10+a11)*b00
        # a10 -> U index 1*m+0 = 2
        # a11 -> U index 3
        # b00 -> V index k0

        U[2, col] += 1
        U[3, col] += 1
        V[k0, col] += 1
        # C10 += M2 -> W[k0*n+1, col] = W[2*k0+1, col]
        W[2 * k0 + 1, col] += 1
        col += 1

        # M3 = a00*(b01-b11)
        # a00 -> U index 0
        # b01 -> V index k1
        # b11 -> V index 6+k1

        U[0, col] += 1
        V[k1, col] += 1
        V[6 + k1, col] -= 1
        # C01 += M3 -> W[k1*n+0, col] = W[2*k1, col]
        W[2 * k1, col] += 1
        col += 1

        # M4 = a11*(b10-b00)
        # a11 -> U index 3
        # b10 -> V index 6+k0
        # b00 -> V index k0

        U[3, col] += 1
        V[6 + k0, col] += 1
        V[k0, col] -= 1
        # C00 -= M4 -> W[2*k0, col] -= 1
        # C10 += M4 -> W[2*k0+1, col] += 1
        W[2 * k0, col] -= 1
        W[2 * k0 + 1, col] += 1
        col += 1

        # M5 = (a00+a01)*b11
        # a00 -> U index 0
        # a01 -> U index 1
        # b11 -> V index 6+k1

        U[0, col] += 1
        U[1, col] += 1
        V[6 + k1, col] += 1
        # C01 += M5 -> W[2*k1, col] += 1
        W[2 * k1, col] += 1
        col += 1

        # M6 = (a10-a00)*(b00+b01)
        # a10 -> U index 2
        # a00 -> U index 0
        # b00 -> V index k0
        # b01 -> V index k1

        U[2, col] += 1
        U[0, col] -= 1
        V[k0, col] += 1
        V[k1, col] += 1
        # C00 += M6 -> W[2*k0, col] += 1
        W[2 * k0, col] += 1
        col += 1

        # M7 = (a01-a11)*(b10+b11)
        # a01 -> U index 1
        # a11 -> U index 3
        # b10 -> V index 6+k0
        # b11 -> V index 6+k1

        U[1, col] += 1
        U[3, col] -= 1
        V[6 + k0, col] += 1
        V[6 + k1, col] += 1
        # C11 += M7 -> W[2*k1+1, col] += 1
        W[2 * k1 + 1, col] += 1
        col += 1

    # Now verify the decomposition.
    # For each (i, j, k), check that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # For other combinations, the sum should be 0.

    # Let's verify for all valid (i, j, k):
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                total = 0
                for c in range(rank):
                    total += U[u_idx, c] * V[v_idx, c] * W[w_idx, c]
                if total != 1:
                    # This decomposition is incorrect. Fall back to schoolbook.
                    U = np.zeros((U_size, n * m * p), dtype=np.int64)
                    V = np.zeros((V_size, n * m * p), dtype=np.int64)
                    W = np.zeros((W_size, n * m * p), dtype=np.int64)
                    c = 0
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                U[ii * m + jj, c] = 1
                                V[jj * p + kk, c] = 1
                                W[kk * n + ii, c] = 1
                                c += 1
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
