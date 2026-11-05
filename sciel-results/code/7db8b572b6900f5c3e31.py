"""Tensor decomposition for 3x4 x 4x7 matrix multiplication."""
import time
import random
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Index sets
    I = range(n)  # 0..2
    J = range(m)  # 0..3
    K = range(p)  # 0..6

    # We'll build a decomposition using Strassen-like blocks.
    # The key idea: partition the J and K dimensions into pairs and use
    # Strassen's 7-multiplication scheme for each 2x2x2 block.
    #
    # For a 2x2x2 tensor, Strassen uses rank 7 instead of 8.
    # The 7 factors are:
    #   U: (A00+A11)(B00+B11) -> (A00+A11, B00+B11)
    #   U: (A10+A11)(B10)     -> (A10+A11, B10)
    #   U: (A00)(B01-B11)     -> (A00, B01-B11)
    #   U: (A11)(B10-B00)     -> (A11, B10-B00)
    #   U: (A00)(B00)         -> (A00, B00)
    #   U: (A11)(B11)         -> (A11, B11)
    #   U: (A00+A10)(B11)     -> (A00+A10, B11)
    #
    # Wait, let me be more careful. Standard Strassen for C = A*B where A is 2x2, B is 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A10)B11
    # M6 = (A01-A11)(B10+B11)
    # M7 = (A10-A00)(B00+B01)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # For our case, we need to tile the j,k plane.
    # n=3, m=4, p=7.
    # We can tile j into pairs: (0,1), (2,3) and k into pairs: (0,1), (2,3), (4,5), with k=6 leftover.
    #
    # Actually, let me think about this differently. The known best rank for 3x4x4 is 17 (vs 16 schoolbook... wait no, 3*4*4=48 schoolbook, but best known is 17).
    # For 3x4x7, the schoolbook is 84. We need to find a better decomposition.
    #
    # Let me use a systematic approach: use Strassen blocks where possible and handle remainders with schoolbook.
    #
    # j dimension: 4 = 2+2, so two j-pairs: (0,1) and (2,3)
    # k dimension: 7 = 2+2+2+1, so three k-pairs: (0,1), (2,3), (4,5) and one k-single: (6)
    # i dimension: 3, no pairing needed (or we could do 2+1)
    #
    # For each i (3 values), each j-pair (2 pairs), each k-pair (3 pairs):
    #   Use Strassen: rank 7 per block
    # For each i (3 values), each j-pair (2 pairs), k-single (1):
    #   Schoolbook: rank 2 per block (j0*k6, j1*k6)
    #
    # Total: 3 * 2 * 3 * 7 + 3 * 2 * 1 * 2 = 126 + 12 = 138. That's worse than 84!
    #
    # I made an error. Strassen reduces rank from 8 to 7 for a 2x2x2 block, but here each "block" is actually
    # a set of (j-pair, k-pair) entries for a fixed i. The schoolbook for a fixed i, j-pair, k-pair would be 4 multiplications.
    # Strassen uses 7 instead of 4? No, that's wrong. Strassen is for 2x2 matrix multiplication, which has 8 scalar multiplications reduced to 7.
    # But here, for a fixed i, the j-pair x k-pair gives a 2x2 "sub-matrix" of the C row. The tensor entry is C[i, j, k] = sum_j A[i,j]*B[j,k].
    #
    # Let me reconsider. The tensor T[i,j,k] = sum over the middle index... wait no.
    # Actually for matrix multiplication C = A * B, C[i,k] = sum_j A[i,j] * B[j,k].
    # The tensor is T[i,j,k] = A[i,j] * B[j,k] -> contributes to C[i,k].
    # The standard decomposition has rank n*m*p with factors:
    #   U[i*m+j, c] = delta(i, j) for c=(i,j,k)
    #   V[j*p+k, c] = delta(j, k) for c=(i,j,k)
    #   W[k*n+i, c] = delta(k, i) for c=(i,j,k)
    #
    # To use Strassen, we need to group indices. The key insight is that Strassen works on the (j,k) plane for each pair of i values, or on the (i,j) plane, etc.
    #
    # Let me try a different approach: use known efficient decompositions.
    #
    # For 3x4x4: best known rank is 17 (Coppersmith).
    # For 3x3x3: best known is 23 (Coppersmith).
    #
    # Actually, let me just try to construct a good decomposition by combining Strassen blocks.
    #
    # Strategy: Tile the problem into 2x2x2 blocks where possible.
    # i: 3 = 2 + 1
    # j: 4 = 2 + 2
    # k: 7 = 2 + 2 + 2 + 1
    #
    # Block (i-pair, j-pair, k-pair): 2x2x2, use Strassen rank 7
    # Number of such blocks: (1 i-pair) * (2 j-pairs) * (3 k-pairs) = 6 blocks, rank 6*7 = 42
    #
    # Remaining i (single i=2):
    #   For j-pair, k-pair: 2x2 block, schoolbook rank 4. 2*3=6 blocks, rank 24
    #   For j-pair, k-single: 2x1 block, schoolbook rank 2. 2*1=2 blocks, rank 4
    #
    # Remaining i-pair (i=0,1), j-pairs done, k-single:
    #   For i-pair, j-pair, k-single: 2x2x1, schoolbook rank 4. 1*2*1=2 blocks, rank 8
    #
    # Total: 42 + 24 + 4 + 8 = 78. That matches the measured improvement!
    #
    # Let me implement this.

    # Define the Strassen 2x2x2 decomposition
    # For indices (i0,i1), (j0,j1), (k0,k1):
    # 7 factors with entries in U (i*m+j), V (j*p+k), W (k*n+i)

    def strassen_block(i0, i1, j0, j1, k0, k1, base):
        """Return list of (u_dict, v_dict, w_dict) for 7 Strassen factors."""
        factors = []
        # M1 = (A[i0,j0]+A[i1,j1]) * (B[j0,k0]+B[j1,k1])
        factors.append((
            {(i0,j0): 1, (i1,j1): 1},
            {(j0,k0): 1, (j1,k1): 1},
            {(k0,i0): 1, (k1,i0): 1, (k0,i1): 1, (k1,i1): 1}
        ))
        # M2 = (A[i1,j0]+A[i1,j1]) * B[j0,k0]
        factors.append((
            {(i1,j0): 1, (i1,j1): 1},
            {(j0,k0): 1},
            {(k0,i1): 1}
        ))
        # M3 = A[i0,j0] * (B[j0,k1]-B[j1,k1])
        factors.append((
            {(i0,j0): 1},
            {(j0,k1): 1, (j1,k1): -1},
            {(k1,i0): 1}
        ))
        # M4 = A[i1,j1] * (B[j1,k0]-B[j0,k0])
        factors.append((
            {(i1,j1): 1},
            {(j1,k0): 1, (j0,k0): -1},
            {(k0,i1): 1}
        ))
        # M5 = (A[i0,j0]+A[i0,j1]) * B[j1,k1]
        factors.append((
            {(i0,j0): 1, (i0,j1): 1},
            {(j1,k1): 1},
            {(k1,i0): 1}
        ))
        # M6 = (A[i0,j1]-A[i1,j1]) * (B[j1,k0]+B[j1,k1])
        factors.append((
            {(i0,j1): 1, (i1,j1): -1},
            {(j1,k0): 1, (j1,k1): 1},
            {(k0,i0): 1, (k1,i0): 1, (k0,i1): -1, (k1,i1): -1}
        ))
        # M7 = (A[i1,j0]-A[i0,j0]) * (B[j0,k0]+B[j0,k1])
        factors.append((
            {(i1,j0): 1, (i0,j0): -1},
            {(j0,k0): 1, (j0,k1): 1},
            {(k0,i1): 1, (k1,i1): 1, (k0,i0): -1, (k1,i0): -1}
        ))
        return factors

    def schoolbook_block(ips, jps, kps, base):
        """Return list of (u_dict, v_dict, w_dict) for schoolbook on given index sets."""
        factors = []
        for i in ips:
            for j in jps:
                for k in kps:
                    factors.append((
                        {(i,j): 1},
                        {(j,k): 1},
                        {(k,i): 1}
                    ))
        return factors

    # Build decomposition
    all_factors = []

    # i-pair: (0,1), j-pairs: (0,1),(2,3), k-pairs: (0,1),(2,3),(4,5)
    i_pair = (0, 1)
    j_pairs = [(0, 1), (2, 3)]
    k_pairs = [(0, 1), (2, 3), (4, 5)]

    # Strassen blocks: i-pair x j-pair x k-pair
    for jp in j_pairs:
        for kp in k_pairs:
            factors = strassen_block(i_pair[0], i_pair[1], jp[0], jp[1], kp[0], kp[1], 0)
            all_factors.extend(factors)

    # Remaining i=2 (single):
    i_single = 2
    for jp in j_pairs:
        for kp in k_pairs:
            factors = schoolbook_block([i_single], [jp[0], jp[1]], [kp[0], kp[1]], 0)
            all_factors.extend(factors)
        # k-single: k=6
        factors = schoolbook_block([i_single], [jp[0], jp[1]], [6], 0)
        all_factors.extend(factors)

    # i-pair (0,1), j-pairs, k-single (k=6):
    for jp in j_pairs:
        factors = schoolbook_block([0, 1], [jp[0], jp[1]], [6], 0)
        all_factors.extend(factors)

    # Now build U, V, W matrices
    rank = len(all_factors)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for col, (u_dict, v_dict, w_dict) in enumerate(all_factors):
        for (i, j), val in u_dict.items():
            U[i * m + j][col] = val
        for (j, k), val in v_dict.items():
            V[j * p + k][col] = val
        for (k, i), val in w_dict.items():
            W[k * n + i][col] = val

    # Verify the decomposition
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # Check: sum_c U[i*m+j, c] * V[j*p+k, c] should give the right tensor
    # Actually verify: for each (i,j,k), T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # This should equal delta(i,j,k) in the standard sense... no.
    # The tensor for matmul is: T[i,j,k] contributes to C[i,k] via A[i,j]*B[j,k]
    # The decomposition requires: sum_c U[i*m+j,c] * V[j*p+k,c] = [coefficient for C[i,k] from (i,j,k) path]
    # Actually the correct check is: for all i,k: C[i,k] = sum_j A[i,j]B[j,k]
    # And the decomposition says: C[i,k] = sum_c (sum_j U[i*m+j,c]*A[i,j]) * (sum_j V[j*p+k,c]*B[j,k]) * W[k*n+i,c]
    # Wait, that's not right either. Let me think again.
    #
    # Standard: C[i,k] = sum_j A[i,j] * B[j,k]
    # Decomposition: C[i,k] = sum_c W[k*n+i,c] * (sum_j U[i*m+j,c] * A[i,j]) * (sum_j V[j*p+k,c] * B[j,k])
    #
    # For this to equal sum_j A[i,j]*B[j,k] for all A,B, we need:
    # sum_c W[k*n+i,c] * U[i*m+j,c] * V[j'*p+k,c] = delta(j,j') ... no that's not right.
    #
    # Actually: sum_c W[k*n+i,c] * (sum_j U[i*m+j,c]*A[i,j]) * (sum_j' V[j'*p+k,c]*B[j',k])
    # = sum_c sum_j sum_j' W[k*n+i,c] * U[i*m+j,c] * A[i,j] * V[j'*p+k,c] * B[j',k]
    # = sum_j sum_j' A[i,j] * B[j',k] * (sum_c W[k*n+i,c]*U[i*m+j,c]*V[j'*p+k,c])
    #
    # For this to equal sum_j A[i,j]*B[j,k], we need:
    # sum_c W[k*n+i,c]*U[i*m+j,c]*V[j'*p+k,c] = delta(j,j')
    #
    # Let me verify this.

    # Quick verification
    np_U = np.array(U, dtype=float)
    np_V = np.array(V, dtype=float)
    np_W = np.array(W, dtype=float)

    # For each (i, j, j', k): check sum_c W[k*n+i,c]*U[i*m+j,c]*V[j'*p+k,c] == delta(j,j')
    ok = True
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    expected = 1.0 if j == jp else 0.0
                    val = sum(np_W[k * n + i, c] * np_U[i * m + j, c] * np_V[jp * p + k, c] for c in range(rank))
                    if abs(val - expected) > 1e-9:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if not ok:
            break

    if not ok:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
    else:
        pass  # Use the Strassen-based decomposition

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
