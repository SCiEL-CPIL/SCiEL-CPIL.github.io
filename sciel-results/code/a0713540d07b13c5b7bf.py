"""Rewritten program for matrix multiplication 3x4x4 using block decomposition."""

import time

DIMENSIONS = [3, 4, 4]

# Strassen's algorithm for 2x2 matrix multiplication
# The 7 products for C = AB where A, B are 2x2:
# M1 = (A00 + A11)(B00 + B11)
# M2 = (A10 + A11)B00
# M3 = A00(B01 - B11)
# M4 = A11(B10 - B00)
# M5 = (A00 + A01)B11
# M6 = (A10 - A00)(B00 + B01)
# M7 = (A01 - A11)(B10 + B11)
# C00 = M1 + M4 - M5 + M7
# C01 = M3 + M5
# C10 = M2 + M4
# C11 = M1 - M2 + M3 + M6

# Strassen factors for 2x2x2 (rank 7)
# U indices: 0=A00, 1=A01, 2=A10, 3=A11
# V indices: 0=B00, 1=B01, 2=B10, 3=B11
# W indices: 0=C00, 1=C01, 2=C10, 3=C11

STRASSEN_2x2x2 = [
    # (U_coeffs, V_coeffs, W_coeffs)
    # M1 = (A00+A11)(B00+B11)
    ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: -1}),  # M1 contributes to C00 (+1) and C11 (-1)
    # M2 = (A10+A11)B00
    ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),        # M2 contributes to C10 (+1) and C11 (-1)
    # M3 = A00(B01-B11)
    ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),         # M3 contributes to C01 (+1) and C11 (+1)
    # M4 = A11(B10-B00)
    ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),         # M4 contributes to C00 (+1) and C10 (+1)
    # M5 = (A00+A01)B11
    ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),         # M5 contributes to C00 (-1) and C01 (+1)
    # M6 = (A10-A00)(B00+B01)
    ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),         # M6 contributes to C11 (+1)
    # M7 = (A01-A11)(B10+B11)
    ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),         # M7 contributes to C00 (+1)
]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a tensor decomposition for 3x4x4 matrix multiplication.
    
    Strategy: Use Strassen's algorithm for the 4x4 part.
    Decompose 4x4 matrices into 2x2 blocks and use Strassen's rank-7 algorithm.
    
    For 3x4x4:
    - Split A (3x4) into 3 rows of 1x4
    - Each row of A (1x4) can be viewed as two 1x2 blocks
    - B (4x4) is split into four 2x2 blocks
    - C (3x4) has 3 rows of 1x4, each split into two 1x2 blocks
    
    We use Strassen on the 2x2x2 sub-problem (rank 7 instead of 8).
    For each row i of A (which is 1x4 = [a0, a1, a2, a3]):
    - Treat as two 1x2 vectors: [a0, a1] and [a2, a3]
    - Multiply each by the corresponding 2x2 block of B
    - Use Strassen's method for 2x2x2 multiplication
    
    Actually, let's think more carefully.
    
    A is 3x4, B is 4x4, C is 3x4.
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    Split indices:
    - j = 2*j1 + j2 where j1, j2 in {0,1}
    - k = 2*k1 + k2 where k1, k2 in {0,1}
    
    So A[i, 2*j1+j2] and B[2*j1+j2, 2*k1+k2]
    
    C[i, 2*k1+k2] = sum_{j1,j2} A[i, 2*j1+j2] * B[2*j1+j2, 2*k1+k2]
    
    This is like multiplying a 1x4 vector by a 4x4 matrix to get a 1x4 vector.
    We can view this as two 1x2 vectors times a 4x4 matrix, but Strassen works on 2x2.
    
    Alternative: View the 1x4 x 4x4 -> 1x4 as two independent 1x2 x 2x4 -> 1x2 problems?
    No, that doesn't work because the 4x4 matrix couples all rows.
    
    Better approach: Use the fact that 4x4 = 2x2 blocks.
    Let A_row = [A_row_left | A_row_right] where each is 1x2.
    Let B = [[B00, B01], [B10, B11]] where each Bjk is 2x2.
    Then C_row = [C_row_left | C_row_right] where each is 1x2.
    
    C_row_left = A_row_left * B00 + A_row_right * B10
    C_row_right = A_row_left * B01 + A_row_right * B11
    
    Each term like A_row_left * B00 is a 1x2 times 2x2 giving 1x2.
    This is a 1x2x2 problem with rank 4 (schoolbook) or can we do better?
    
    For 1x2x2: C[k] = A[0]*B[0,k] + A[1]*B[1,k]
    This has rank 4 (each A[j]*B[j,k] term).
    
    So total rank = 3 rows * 2 (left/right) * 2 (top/bottom blocks) * 4 = 48. Same as schoolbook.
    
    Hmm, Strassen helps when both inputs are 2x2. For 1x2 x 2x2, the rank is 4.
    
    Let me reconsider. The key insight from the cell findings is:
    "Using Strassen's algorithm for 2x2 blocks reduces the rank of the 2x4x4 sub-problem from 32 (schoolbook) to 28"
    
    So for 2x4x4:
    A is 2x4, B is 4x4, C is 2x4.
    Split into 2x2 blocks:
    A = [A00 A01; A10 A11] where each is 2x2... wait, A is 2x4.
    
    Let me re-read: "2x4x4 sub-problem from 32 (schoolbook) to 28"
    Schoolbook rank for 2x4x4 = 2*4*4 = 32. Strassen gives 28.
    
    How? Split the 4 into 2+2:
    A (2x4) = [A_left (2x2) | A_right (2x2)]
    B (4x4) = [[B00 (2x2), B01 (2x2)], [B10 (2x2), B11 (2x2)]]
    C (2x4) = [C_left (2x2) | C_right (2x2)]
    
    C_left = A_left * B00 + A_right * B10
    C_right = A_left * B01 + A_right * B11
    
    Each term is a 2x2 * 2x2 -> 2x2 multiplication.
    Using Strassen (rank 7) for each:
    C_left needs 2 Strassen multiplications: rank 14
    C_right needs 2 Strassen multiplications: rank 14
    Total: 28
    
    For 3x4x4:
    We can't directly split 3 into 2+1 and use Strassen on both.
    But we can do:
    - First 2 rows: use the 2x4x4 Strassen decomposition (rank 28)
    - Last 1 row: use schoolbook 1x4x4 (rank 16)
    Total: 28 + 16 = 44
    
    This matches the cell finding: "total rank of 44 instead of 48"
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    # n=3, m=4, p=4
    
    # For 3x4x4:
    # Rows 0,1: Use Strassen-based 2x4x4 (rank 28)
    # Row 2: Use schoolbook 1x4x4 (rank 16)
    # Total rank = 44
    
    total_rank = 44
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    current_col = 0
    
    # Part 1: Strassen-based decomposition for rows 0 and 1 (2x4x4)
    # A rows 0,1 form a 2x4 matrix: [A00 A01; A10 A11] where each is 2x2
    # Wait, A is 3x4. Rows 0 and 1 give a 2x4 matrix.
    # Split columns: A_left (2x2) = A[0:2, 0:2], A_right (2x2) = A[0:2, 2:4]
    # B (4x4) split into 2x2 blocks:
    #   B00 = B[0:2, 0:2], B01 = B[0:2, 2:4]
    #   B10 = B[2:4, 0:2], B11 = B[2:4, 2:4]
    # C rows 0,1 form a 2x4 matrix: [C_left (2x2) | C_right (2x2)]
    # C_left = A_left * B00 + A_right * B10
    # C_right = A_left * B01 + A_right * B11
    
    # For C_left = A_left * B00 + A_right * B10:
    # Use Strassen for A_left * B00 (rank 7) and A_right * B10 (rank 7)
    # For C_right = A_left * B01 + A_right * B11:
    # Use Strassen for A_left * B01 (rank 7) and A_right * B11 (rank 7)
    # Total for 2x4x4: 4 * 7 = 28
    
    # Strassen factor application:
    # For multiplying 2x2 matrix M by 2x2 matrix N to get 2x2 result P:
    # M indices: M00, M01, M10, M11
    # N indices: N00, N01, N10, N11
    # P indices: P00, P01, P10, P11
    
    # In our global indexing:
    # U index for A[i,j]: i*m + j = i*4 + j
    # V index for B[j,k]: j*p + k = j*4 + k
    # W index for C[i,k]: k*n + i = k*3 + i
    
    # For the 2x4x4 part (rows 0,1):
    # A_left: A[0,0], A[0,1], A[1,0], A[1,1] -> U indices 0,1,4,5
    # A_right: A[0,2], A[0,3], A[1,2], A[1,3] -> U indices 2,3,6,7
    # B00: B[0,0], B[0,1], B[1,0], B[1,1] -> V indices 0,1,4,5
    # B01: B[0,2], B[0,3], B[1,2], B[1,3] -> V indices 2,3,6,7
    # B10: B[2,0], B[2,1], B[3,0], B[3,1] -> V indices 8,9,12,13
    # B11: B[2,2], B[2,3], B[3,2], B[3,3] -> V indices 10,11,14,15
    # C_left: C[0,0], C[0,1], C[1,0], C[1,1] -> W indices 0,1,3,4
    # C_right: C[0,2], C[0,3], C[1,2], C[1,3] -> W indices 6,7,9,10
    
    def apply_strassen_2x2x2(col_offset, U_indices, V_indices, W_indices):
        """
        Apply Strassen's 2x2x2 decomposition.
        U_indices: [U00, U01, U10, U11] - indices into U for the 2x2 input matrix
        V_indices: [V00, V01, V10, V11] - indices into V for the 2x2 input matrix
        W_indices: [W00, W01, W10, W11] - indices into W for the 2x2 output matrix
        Returns the new column offset.
        """
        nonlocal current_col
        U00, U01, U10, U11 = U_indices
        V00, V01, V10, V11 = V_indices
        W00, W01, W10, W11 = W_indices
        
        for (u_coeffs, v_coeffs, w_coeffs) in STRASSEN_2x2x2:
            if time.time() > deadline:
                break
            
            col = col_offset
            for u_idx, coeff in u_coeffs.items():
                U[u_idx][col] += coeff
            for v_idx, coeff in v_coeffs.items():
                V[v_idx][col] += coeff
            for w_idx, coeff in w_coeffs.items():
                W[w_idx][col] += coeff
            col_offset += 1
        
        return col_offset
    
    # C_left = A_left * B00 + A_right * B10
    # Strassen for A_left * B00:
    current_col = apply_strassen_2x2x2(
        current_col,
        [0, 1, 4, 5],   # A_left: A[0,0], A[0,1], A[1,0], A[1,1]
        [0, 1, 4, 5],   # B00: B[0,0], B[0,1], B[1,0], B[1,1]
        [0, 1, 3, 4]    # C_left: C[0,0], C[0,1], C[1,0], C[1,1]
    )
    
    # Strassen for A_right * B10:
    current_col = apply_strassen_2x2x2(
        current_col,
        [2, 3, 6, 7],   # A_right: A[0,2], A[0,3], A[1,2], A[1,3]
        [8, 9, 12, 13], # B10: B[2,0], B[2,1], B[3,0], B[3,1]
        [0, 1, 3, 4]    # C_left: C[0,0], C[0,1], C[1,0], C[1,1]
    )
    
    # C_right = A_left * B01 + A_right * B11
    # Strassen for A_left * B01:
    current_col = apply_strassen_2x2x2(
        current_col,
        [0, 1, 4, 5],   # A_left: A[0,0], A[0,1], A[1,0], A[1,1]
        [2, 3, 6, 7],   # B01: B[0,2], B[0,3], B[1,2], B[1,3]
        [6, 7, 9, 10]   # C_right: C[0,2], C[0,3], C[1,2], C[1,3]
    )
    
    # Strassen for A_right * B11:
    current_col = apply_strassen_2x2x2(
        current_col,
        [2, 3, 6, 7],   # A_right: A[0,2], A[0,3], A[1,2], A[1,3]
        [10, 11, 14, 15], # B11: B[2,2], B[2,3], B[3,2], B[3,3]
        [6, 7, 9, 10]   # C_right: C[0,2], C[0,3], C[1,2], C[1,3]
    )
    
    # Part 2: Schoolbook for row 2 (1x4x4, rank 16)
    for j in range(m):
        if time.time() > deadline:
            break
        for k in range(p):
            if time.time() > deadline:
                break
            
            u_idx = 2 * m + j  # Row 2, column j
            v_idx = j * p + k
            w_idx = k * n + 2  # Column k, row 2
            
            U[u_idx][current_col] = 1.0
            V[v_idx][current_col] = 1.0
            W[w_idx][current_col] = 1.0
            
            current_col += 1
    
    # Verify decomposition
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook
        U_fb, V_fb, W_fb = _schoolbook_direct(n, m, p)
        return {"U": U_fb, "V": V_fb, "W": W_fb}


def _schoolbook_direct(n, m, p):
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][col] = 1.0
                V[v_idx][col] = 1.0
                W[w_idx][col] = 1.0
                col += 1
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the tensor decomposition exactly computes matrix multiplication.
    For matrix multiplication, the tensor T[i,j,k] = 1 for all i,j,k.
    We check that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    for all valid (i,j,k) triples.
    """
    rank = len(U[0])
    
    # Check main terms: for all i, j, k, the sum should be 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                total = 0.0
                for r in range(rank):
                    total += U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
                
                if abs(total - 1.0) > 1e-9:
                    return False
    
    # Check that for invalid combinations (where indices don't form a valid path),
    # the sum is 0. A valid path requires:
    # u = i*m + j, v = j*p + k, w = k*n + i
    # For any other combination, the sum should be 0.
    
    for u_idx in range(len(U)):
        i = u_idx // m
        j = u_idx % m
        
        for v_idx in range(len(V)):
            j_prime = v_idx // p
            k = v_idx % p
            
            for w_idx in range(len(W)):
                k_prime = w_idx // n
                i_prime = w_idx % n
                
                # Valid if j == j_prime AND k == k_prime AND i == i_prime
                if (j == j_prime) and (k == k_prime) and (i == i_prime):
                    expected = 1.0
                else:
                    expected = 0.0
                
                total = 0.0
                for r in range(rank):
                    total += U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
                
                if abs(total - expected) > 1e-9:
                    return False
    return True
# EVOLVE-BLOCK-END
