"""Strassen-based decomposition for 2x2 * 2x15 matrix multiplication tensor."""
import time
import numpy as np

DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    # Schoolbook fallback: rank = n*m*p = 60
    rank_schoolbook = n * m * p
    U_fb = np.zeros((n * m, rank_schoolbook))
    V_fb = np.zeros((m * p, rank_schoolbook))
    W_fb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 1
                V_fb[j * p + k, col] = 1
                W_fb[k * n + i, col] = 1
                col += 1
    best_result = {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    best_rank = rank_schoolbook

    # Strassen decomposition for 2x2 * 2x15
    # Split the 15 columns of B into blocks of size 2 (last block padded to 2)
    # 15 = 7*2 + 1, so we have 7 blocks: 6 full blocks of 2 cols + 1 block of 1 col (padded)
    # Actually, let's think more carefully.
    #
    # The tensor T[i,j,k] for i in {0,1}, j in {0,1}, k in {0..14}
    # We want: sum_c U[i*2+j, c] * V[j*15+k, c] * W[k*2+i, c] = 1 for all i,j,k
    #
    # Strassen for 2x2 * 2x2: rank 7 instead of 8
    # The 7 products are:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B10-B11)
    # M4 = A11*(B01-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B10)
    # M7 = (A01-A11)*(B01+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # For 2x2 * 2xp where p is even, we can group columns into p/2 blocks of 2
    # and apply Strassen to each block.
    # For p=15 (odd), we handle 14 columns as 7 blocks of 2, and 1 column separately.
    #
    # Actually, a cleaner approach: 
    # For 2x2 * 2xp, the rank is known to be 2p+2 for p>=2 (from Bini et al.)
    # Wait, that's not right either. Let me think about what's achievable.
    #
    # Actually, the known result: rank(2x2xp) for p >= 2:
    # - Schoolbook: 4p
    # - Strassen-based: For p even, 7*(p/2) = 3.5p
    # - For p odd: 7*((p-1)/2) + 4 = 3.5p - 0.5 + 4 = 3.5p + 3.5
    #   Wait, the last single column needs rank 4 (schoolbook for 2x2*2x1)
    #   So for p=15: 7*7 + 4 = 49 + 4 = 53
    #
    # This matches the observation that Strassen reduced rank from 60 to 53!
    #
    # Let me construct this explicitly.
    #
    # For each 2-column block (columns 2b, 2b+1 for b=0..6):
    # Apply Strassen with 7 rank-1 terms.
    #
    # For the last single column (column 14):
    # Use schoolbook with 4 rank-1 terms.
    #
    # Total rank: 7*7 + 4 = 53

    R = 53
    U = np.zeros((n * m, R))  # 4 x 53
    V = np.zeros((m * p, R))  # 30 x 53
    W = np.zeros((p * n, R))  # 30 x 53

    col_offset = 0

    # Process 7 blocks of 2 columns each (columns 0-13)
    for b in range(7):
        k0 = 2 * b      # first column in block
        k1 = 2 * b + 1  # second column in block

        # Strassen 7 products for this block:
        # M1 = (A00+A11)*(B0,k0 + B1,k1)
        # M2 = (A10+A11)*B0,k0
        # M3 = A00*(B1,k0 - B1,k1)
        # M4 = A11*(B0,k1 - B0,k0)
        # M5 = (A00+A01)*B1,k1
        # M6 = (A10-A00)*(B0,k0 + B1,k0)
        # M7 = (A01-A11)*(B0,k1 + B1,k1)
        #
        # Output:
        # C0,k0 = M1 + M4 - M5 + M7
        # C0,k1 = M3 + M5
        # C1,k0 = M2 + M4
        # C1,k1 = M1 - M2 + M3 + M6
        #
        # For each Strassen product, we create one rank-1 term.
        # The U vector (for A indices) is the "left factor"
        # The V vector (for B indices) is the "right factor"  
        # The W vector (for C indices) distributes the result.

        # M1: (A00+A11) * (B0,k0 + B1,k1)
        # U: A00=1, A11=1, A01=0, A10=0 => U[i*2+j]: U[0]=1(A00), U[3]=1(A11), U[1]=0(A01), U[2]=0(A10)
        c = col_offset
        U[0, c] = 1  # A[0,0]
        U[3, c] = 1  # A[1,1]
        V[0 * p + k0, c] = 1  # B[0,k0]
        V[1 * p + k1, c] = 1  # B[1,k1]
        # C0,k0 gets +1, C1,k1 gets +1
        W[k0 * n + 0, c] = 1  # C[0,k0]
        W[k1 * n + 1, c] = 1  # C[1,k1]
        col_offset += 1

        # M2: (A10+A11) * B0,k0
        c = col_offset
        U[2, c] = 1  # A[1,0]
        U[3, c] = 1  # A[1,1]
        V[0 * p + k0, c] = 1  # B[0,k0]
        # C1,k0 gets +1
        W[k0 * n + 1, c] = 1  # C[1,k0]
        col_offset += 1

        # M3: A00 * (B1,k0 - B1,k1)
        c = col_offset
        U[0, c] = 1  # A[0,0]
        V[1 * p + k0, c] = 1  # B[1,k0]
        V[1 * p + k1, c] = -1  # B[1,k1]
        # C0,k1 gets +1
        W[k1 * n + 0, c] = 1  # C[0,k1]
        col_offset += 1

        # M4: A11 * (B0,k1 - B0,k0)
        c = col_offset
        U[3, c] = 1  # A[1,1]
        V[0 * p + k1, c] = 1  # B[0,k1]
        V[0 * p + k0, c] = -1  # B[0,k0]
        # C0,k0 gets +1, C1,k0 gets +1
        W[k0 * n + 0, c] = 1  # C[0,k0]
        W[k0 * n + 1, c] = 1  # C[1,k0]
        col_offset += 1

        # M5: (A00+A01) * B1,k1
        c = col_offset
        U[0, c] = 1  # A[0,0]
        U[1, c] = 1  # A[0,1]
        V[1 * p + k1, c] = 1  # B[1,k1]
        # C0,k0 gets -1, C0,k1 gets +1
        W[k0 * n + 0, c] = -1  # C[0,k0]
        W[k1 * n + 0, c] = 1  # C[0,k1]
        col_offset += 1

        # M6: (A10-A00) * (B0,k0 + B1,k0)
        c = col_offset
        U[2, c] = 1  # A[1,0]
        U[0, c] = -1  # A[0,0]
        V[0 * p + k0, c] = 1  # B[0,k0]
        V[1 * p + k0, c] = 1  # B[1,k0]
        # C1,k1 gets +1
        W[k1 * n + 1, c] = 1  # C[1,k1]
        col_offset += 1

        # M7: (A01-A11) * (B0,k1 + B1,k1)
        c = col_offset
        U[1, c] = 1  # A[0,1]
        U[3, c] = -1  # A[1,1]
        V[0 * p + k1, c] = 1  # B[0,k1]
        V[1 * p + k1, c] = 1  # B[1,k1]
        # C0,k0 gets +1
        W[k0 * n + 0, c] = 1  # C[0,k0]
        col_offset += 1

    # Now handle the last column (k=14) with schoolbook (rank 4)
    k_last = 14
    for i in range(n):
        for j in range(m):
            c = col_offset
            U[i * m + j, c] = 1  # A[i,j]
            V[j * p + k_last, c] = 1  # B[j,k_last]
            W[k_last * n + i, c] = 1  # C[i,k_last]
            col_offset += 1

    assert col_offset == R, f"Expected {R} columns, got {col_offset}"

    # Verify the decomposition
    # For all i in {0,1}, j in {0,1}, k in {0..14}:
    # sum_c U[i*2+j, c] * V[j*15+k, c] * W[k*2+i, c] should equal 1

    errors = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for c in range(R):
                    total += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(total - 1.0) > 1e-10:
                    errors.append((i, j, k, total))

    if len(errors) == 0:
        # Verification passed
        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        best_rank = R
    else:
        # If verification failed, try to debug or fallback
        # For now, print first few errors and fallback
        if len(errors) > 0:
            pass  # Don't print in production
        best_result = {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
        best_rank = rank_schoolbook

    if time.time() > deadline - 5:
        return best_result

    return best_result
# EVOLVE-BLOCK-END
