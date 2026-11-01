"""Matrix multiplication 2x2x15: Strassen-based decomposition."""
import time

DIMENSIONS = [2, 2, 15]

# Strassen's 7 products for 2x2x2 matrix multiplication
# M1 = (A00+A11)(B00+B11)
# M2 = (A10+A11)B00
# M3 = A00(B01-B11)
# M4 = A11(B10-B00)
# M5 = (A00+A10)B11
# M6 = (A01-A00)(B00+B10)
# M7 = (A10-A11)(B01+B11)
# C00 = M1 + M4 - M5 + M7
# C01 = M3 + M5
# C10 = M2 + M4
# C11 = M1 - M2 + M3 + M6

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    n, m, p = DIMENSIONS  # 2, 2, 15

    # Strategy: Split the 15 columns into 7 blocks of 2 columns + 1 block of 1 column
    # For each 2-column block: use Strassen (rank 7) for 2x2x2
    # For the 1-column block: use schoolbook (rank 4) for 2x2x1
    # Total rank: 7*7 + 4 = 53

    # Actually, let me think more carefully about the structure.
    # We're computing C = A * B where A is 2x2 and B is 2x15.
    # C is 2x15.
    #
    # The tensor is T[i, j, k] = delta[i,j] * delta[j,k] ... no.
    # The matrix multiplication tensor for (n,m,p) = (2,2,15):
    # T[i, j, k] = 1 if the product C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i, j, k] = 1 for all valid (i,j,k) - it's the identity-like tensor.
    #
    # Actually the standard matrix multiplication tensor has entries:
    # T[i, j, k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # The tensor is defined as T[i,j,k] = delta_{i,k}... no.
    #
    # Let me reconsider. The matrix multiplication tensor of format (n,m,p)
    # has indices (i,j,k) where i in [0,n), j in [0,m), k in [0,p).
    # T[i,j,k] = 1 if and only if the (i,k)-th entry of C = A*B uses A[i,j]*B[j,k].
    # Since C[i,k] = sum_j A[i,j]*B[j,k], every (i,j,k) with valid indices contributes.
    # So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    #
    # Wait, that's not right either. The tensor is:
    # T = sum_{i,j,k} T[i,j,k] * e_i ⊗ e_j ⊗ e_k
    # where T[i,j,k] = 1 if the multiplication uses A[i,j]*B[j,k] for C[i,k].
    # Since C[i,k] = sum_j A[i,j]*B[j,k], we have T[i,j,k] = 1 for all valid (i,j,k).
    #
    # So the tensor is just the all-ones tensor of shape (2,2,15)?
    # No! That would be trivial. Let me re-read the problem.
    #
    # "multiplying a 2 by 2 matrix by a 2 by 15 matrix"
    # A is 2x2, B is 2x15, C = A*B is 2x15.
    # C[i,k] = sum_{j=0}^{1} A[i,j] * B[j,k]
    #
    # The tensor T has entries T[i,j,k] where:
    # - i indexes rows of A and C: 0,1
    # - j indexes columns of A / rows of B: 0,1
    # - k indexes columns of B and C: 0,...,14
    #
    # T[i,j,k] = 1 if the term A[i,j]*B[j,k] appears in C[i,k].
    # Since C[i,k] = sum_j A[i,j]*B[j,k], we have T[i,j,k] = 1 for all i,j,k.
    #
    # Hmm, but then the rank would just be... let me think about what the
    # factor matrices look like.
    #
    # U has shape (n*m, rank) = (4, rank)
    # V has shape (m*p, rank) = (30, rank)
    # W has shape (p*n, rank) = (30, rank)
    #
    # The decomposition says: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    #
    # Note the W index is k*n+i (trace convention).
    #
    # For the schoolbook: each (i,j,k) triple gets its own rank-1 term.
    # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1.
    #
    # Now for Strassen on 2x2x2 blocks:
    # If we group k into pairs (k0, k1), we have a 2x2x2 sub-tensor for each pair.
    # The Strassen decomposition gives rank 7 for each 2x2x2 block.
    #
    # For a 2x2x2 block with k in {k0, k1}:
    # The 7 products are:
    # M1 = (A00+A11)(B00+B11) -> contributes to C00 and C11
    # M2 = (A10+A11)B00 -> contributes to C10 and C11
    # M3 = A00(B01-B11) -> contributes to C01 and C11
    # M4 = A11(B10-B00) -> contributes to C00 and C10
    # M5 = (A00+A10)B11 -> contributes to C00 and C01
    # M6 = (A01-A00)(B00+B10) -> contributes to C11
    # M7 = (A10-A11)(B01+B11) -> contributes to C00
    #
    # Let me be more precise. For 2x2 matrices:
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    #
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    #
    # Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A10)B11
    # M6 = (A01-A00)(B00+B10)
    # M7 = (A10-A11)(B01+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # Now I need to express each M_r as a sum of rank-1 tensors in the (i,j,k) space.
    # Each M_r is a product of a linear combination of A entries and a linear combination of B entries.
    #
    # For example, M1 = (A00+A11)(B00+B11)
    # = A00*B00 + A00*B11 + A11*B00 + A11*B11
    #
    # In tensor terms, M1 contributes to:
    # - C00: coefficient +1 -> (i=0, j=0, k=0): +1, (i=0, j=1, k=1): +1, (i=1, j=0, k=0): +1, (i=1, j=1, k=1): +1
    # - C11: coefficient +1 -> same entries
    #
    # Wait, I need to be careful. M1 is used in both C00 and C11.
    # So M1 contributes:
    # To C00: A00*B00 (+1), A00*B11 (+1), A11*B00 (+1), A11*B11 (+1)
    # To C11: A00*B00 (+1), A00*B11 (+1), A11*B00 (+1), A11*B11 (+1)
    #
    # In the tensor T[i,j,k], where i is row of A/C, j is col of A/row of B, k is col of B/C:
    # A00 = A[0,0], A01 = A[0,1], A10 = A[1,0], A11 = A[1,1]
    # B00 = B[0,0], B01 = B[0,1], B10 = B[1,0], B11 = B[1,1]
    #
    # For a block with k in {k0, k1}, B[j,k0] and B[j,k1] are the two columns.
    #
    # M1 = (A[0,0]+A[1,1])(B[0,k0]+B[1,k1])
    # This contributes to C[0,k0] (C00) and C[1,k1] (C11).
    #
    # As a sum of rank-1 terms:
    # (A[0,0]+A[1,1])(B[0,k0]+B[1,k1])
    # = A[0,0]*B[0,k0] + A[0,0]*B[1,k1] + A[1,1]*B[0,k0] + A[1,1]*B[1,k1]
    #
    # Each term A[i,j]*B[j',k] where j' is the row of B.
    # For C[0,k0]: terms are A[0,0]*B[0,k0] and A[0,1]*B[1,k0]
    # For C[1,k1]: terms are A[1,0]*B[0,k1] and A[1,1]*B[1,k1]
    #
    # So M1 contributes:
    # +1 to T[0,0,k0] (A[0,0]*B[0,k0] in C[0,k0])
    # +1 to T[0,0,k1] (A[0,0]*B[1,k1]... wait, this doesn't make sense)
    #
    # I think I'm overcomplicating this. Let me use the standard approach.
    #
    # The key insight: each "product" in Strassen is a bilinear form in A and B.
    # M_r = (sum_{i,j} a_{r,i,j} * A[i,j]) * (sum_{j',k} b_{r,j',k} * B[j',k])
    #
    # This contributes to C[i,k] with coefficient c_{r,i,k}:
    # C[i,k] += c_{r,i,k} * M_r
    #
    # So the total tensor is:
    # T[i,j,k] = sum_r a_{r,i,j} * b_{r,j,k} * c_{r,i,k}
    #
    # Wait, that's not quite right either. Let me think again.
    #
    # M_r = (sum_{i,j} a_{r,i,j} * A[i,j]) * (sum_{j',k} b_{r,j',k} * B[j',k])
    #
    # C[i,k] = sum_r c_{r,i,k} * M_r
    #
    # = sum_r c_{r,i,k} * (sum_{i',j'} a_{r,i',j'} * A[i',j']) * (sum_{j'',k'} b_{r,j'',k'} * B[j'',k'])
    #
    # For this to equal sum_j A[i,j]*B[j,k], we need:
    # sum_r c_{r,i,k} * a_{r,i',j'} * b_{r,j'',k'} = delta_{i',i} * delta_{j',j} * delta_{j'',j} * delta_{k',k}
    #
    # So the tensor decomposition is:
    # T[i,j,k] = sum_r c_{r,i,k} * a_{r,i,j} * b_{r,j,k}
    #
    # where a_{r,i,j} is the coefficient of A[i,j] in the first factor of M_r,
    # b_{r,j,k} is the coefficient of B[j,k] in the second factor of M_r,
    # c_{r,i,k} is the coefficient of M_r in C[i,k].
    #
    # And in the factor matrices:
    # U[i*m+j, r] = a_{r,i,j}
    # V[j*p+k, r] = b_{r,j,k}
    # W[k*n+i, r] = c_{r,i,k}
    #
    # Great! Now let me construct the Strassen decomposition for each 2-column block.

    rank = 7 * 7 + 4  # 49 + 4 = 53

    U = [[0] * rank for _ in range(n * m)]   # 4 x 53
    V = [[0] * rank for _ in range(m * p)]   # 30 x 53
    W = [[0] * rank for _ in range(p * n)]   # 30 x 53

    col = 0

    # For each pair of columns (k0, k1) in the 2x2x2 block
    for block in range(7):
        k0 = 2 * block
        k1 = 2 * block + 1

        # Strassen coefficients for 2x2x2
        # a[r][i][j]: coefficient of A[i,j] in first factor of M_r
        # b[r][j][k]: coefficient of B[j,k] in second factor of M_r (k is 0 or 1 within block)
        # c[r][i][k]: coefficient of M_r in C[i,k] (k is 0 or 1 within block)

        # M1 = (A00+A11)(B00+B11), C00 += M1, C11 += M1
        a1 = {(0,0): 1, (1,1): 1}
        b1 = {(0,0): 1, (1,1): 1}
        c1 = {(0,0): 1, (1,1): 1}

        # M2 = (A10+A11)B00, C10 += M2, C11 -= M2
        a2 = {(1,0): 1, (1,1): 1}
        b2 = {(0,0): 1}
        c2 = {(1,0): 1, (1,1): -1}

        # M3 = A00(B01-B11), C01 += M3, C11 += M3
        a3 = {(0,0): 1}
        b3 = {(0,1): 1, (1,1): -1}
        c3 = {(0,1): 1, (1,1): 1}

        # M4 = A11(B10-B00), C00 += M4, C10 += M4
        a4 = {(1,1): 1}
        b4 = {(1,0): 1, (0,0): -1}
        c4 = {(0,0): 1, (1,0): 1}

        # M5 = (A00+A10)B11, C00 -= M5, C01 += M5
        a5 = {(0,0): 1, (1,0): 1}
        b5 = {(1,1): 1}
        c5 = {(0,0): -1, (0,1): 1}

        # M6 = (A01-A00)(B00+B10), C11 += M6
        a6 = {(0,1): 1, (0,0): -1}
        b6 = {(0,0): 1, (1,0): 1}
        c6 = {(1,1): 1}

        # M7 = (A10-A11)(B01+B11), C00 += M7
        a7 = {(1,0): 1, (1,1): -1}
        b7 = {(0,1): 1, (1,1): 1}
        c7 = {(0,0): 1}

        strassen = [
            (a1, b1, c1),
            (a2, b2, c2),
            (a3, b3, c3),
            (a4, b4, c4),
            (a5, b5, c5),
            (a6, b6, c6),
            (a7, b7, c7),
        ]

        for r_idx, (a, b, c) in enumerate(strassen):
            r = col + r_idx
            for (i, j), val in a.items():
                U[i * m + j][r] += val
            for (j, k_local), val in b.items():
                k_actual = k0 if k_local == 0 else k1
                V[j * p + k_actual][r] += val
            for (i, k_local), val in c.items():
                k_actual = k0 if k_local == 0 else k1
                W[k_actual * n + i][r] += val

        col += 7

    # Handle the remaining 1 column (k = 14) with schoolbook rank 4
    k_rem = 14
    for i in range(n):
        for j in range(m):
            r = col
            U[i * m + j][r] = 1
            V[j * p + k_rem][r] = 1
            W[k_rem * n + i][r] = 1
            col += 1

    # Verify the decomposition
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Should equal 1 for all valid (i,j,k)

    error = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s != 1:
                    error += 1

    if error > 0:
        # Fallback to schoolbook if verification fails
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

    return {"U": U, "V": V, "W": W}
