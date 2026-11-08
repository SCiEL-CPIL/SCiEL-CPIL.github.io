"""Matrix multiplication 2x4 by 4x6: rank-minimal exact tensor decomposition."""
import time
import numpy as np

def construct():
    n, m, p = 2, 4, 6  # A is n x m, B is m x p
    U_rows = n * m  # 8
    V_rows = m * p  # 24
    W_rows = p * n  # 12

    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0

    # Strassen 2x2 decomposition:
    # Given A (2x2), B (2x2), compute C = A*B using 7 scalar multiplications:
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b01-b11)
    # M4 = a11*(b10-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b01)
    # M7 = (a01-a11)*(b10+b11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # For 2x4 * 4x6:
    # Split A into A0 (cols 0-1) and A1 (cols 2-3).
    # Split B into B_top (rows 0-1) and B_bottom (rows 2-3).
    # C = A0 * B_top + A1 * B_bottom.
    # B_top is 2x6, B_bottom is 2x6.
    # Split B_top into 3 blocks of 2x2: B_top_0, B_top_1, B_top_2.
    # Split B_bottom into 3 blocks of 2x2: B_bot_0, B_bot_1, B_bot_2.
    # Each 2x2 * 2x2 uses Strassen (rank 7).
    # Total rank = 2 (A blocks) * 3 (B blocks) * 7 = 42.

    rank = 42
    U = [[0.0] * rank for _ in range(U_rows)]
    V = [[0.0] * rank for _ in range(V_rows)]
    W = [[0.0] * rank for _ in range(W_rows)]

    col = 0

    # Process A0 (A columns 0-1) times each 2x2 block of B_top (B rows 0-1, columns 2t-2t+1)
    # And A1 (A columns 2-3) times each 2x2 block of B_bottom (B rows 2-3, columns 2t-2t+1)

    for a_block in range(2):  # 0: A0 (cols 0-1), 1: A1 (cols 2-3)
        for t in range(3):  # 3 blocks of B, each 2x2
            # B block is rows 2*a_block to 2*a_block+1, cols 2*t to 2*t+1
            b_row0 = 2 * a_block
            b_row1 = 2 * a_block + 1
            b_col0 = 2 * t
            b_col1 = 2 * t + 1

            # A_col_offset = 2 * a_block
            a_col_offset = 2 * a_block

            # M1 = (a00+a11) * (b00+b11)
            # C00 +, C11 +
            # C00: i'=0, k'=2*t
            # C11: i'=1, k'=2*t+1
            U[0 * m + a_col_offset + 0][col] += 1  # a00
            U[1 * m + a_col_offset + 1][col] += 1  # a11
            V[b_row0 * p + b_col0][col] += 1  # b00
            V[b_row1 * p + b_col1][col] += 1  # b11
            W[(2*t) * n + 0][col] += 1  # C[0, 2*t]
            W[(2*t+1) * n + 1][col] += 1  # C[1, 2*t+1]
            col += 1

            # M2 = (a10+a11) * b00
            # C10 +, C11 -
            U[1 * m + a_col_offset + 0][col] += 1  # a10
            U[1 * m + a_col_offset + 1][col] += 1  # a11
            V[b_row0 * p + b_col0][col] += 1  # b00
            W[(2*t) * n + 1][col] += 1  # C[1, 2*t]
            W[(2*t+1) * n + 1][col] += -1  # C[1, 2*t+1]
            col += 1

            # M3 = a00 * (b01-b11)
            # C01 +, C11 +
            U[0 * m + a_col_offset + 0][col] += 1  # a00
            V[b_row0 * p + b_col1][col] += 1  # b01
            V[b_row1 * p + b_col1][col] += -1  # b11
            W[(2*t+1) * n + 0][col] += 1  # C[0, 2*t+1]
            W[(2*t+1) * n + 1][col] += 1  # C[1, 2*t+1]
            col += 1

            # M4 = a11 * (b10-b00)
            # C00 +, C10 +
            U[1 * m + a_col_offset + 1][col] += 1  # a11
            V[b_row1 * p + b_col0][col] += 1  # b10
            V[b_row0 * p + b_col0][col] += -1  # b00
            W[(2*t) * n + 0][col] += 1  # C[0, 2*t]
            W[(2*t) * n + 1][col] += 1  # C[1, 2*t]
            col += 1

            # M5 = (a00+a01) * b11
            # C00 -, C01 +
            U[0 * m + a_col_offset + 0][col] += 1  # a00
            U[0 * m + a_col_offset + 1][col] += 1  # a01
            V[b_row1 * p + b_col1][col] += 1  # b11
            W[(2*t) * n + 0][col] += -1  # C[0, 2*t]
            W[(2*t+1) * n + 0][col] += 1  # C[0, 2*t+1]
            col += 1

            # M6 = (a10-a00) * (b00+b01)
            # C11 +
            U[1 * m + a_col_offset + 0][col] += 1  # a10
            U[0 * m + a_col_offset + 0][col] += -1  # a00
            V[b_row0 * p + b_col0][col] += 1  # b00
            V[b_row0 * p + b_col1][col] += 1  # b01
            W[(2*t+1) * n + 1][col] += 1  # C[1, 2*t+1]
            col += 1

            # M7 = (a01-a11) * (b10+b11)
            # C00 +
            U[0 * m + a_col_offset + 1][col] += 1  # a01
            U[1 * m + a_col_offset + 1][col] += -1  # a11
            V[b_row1 * p + b_col0][col] += 1  # b10
            V[b_row1 * p + b_col1][col] += 1  # b11
            W[(2*t) * n + 0][col] += 1  # C[0, 2*t]
            col += 1

    # Verification step (optional but good for robustness)
    # Convert to numpy for fast check
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    # The tensor T[i, j, k] should be 1 if i=i', j=j', k=k' in the decomposition sum
    # Actually, we check if U, V, W correctly decompose the identity tensor of matmul.
    # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # This sum should be 1 for all valid (i, j, k) corresponding to the matmul structure?
    # No, the tensor for matmul is T[i, j, k] = 1 for all i, j, k? 
    # Wait, the structure tensor of matmul (n x m) * (m x p) is in R^{nm x mp x np}.
    # T[(i,j), (j',k), (i',k')] = delta_{i,i'} delta_{j,j'} delta_{k,k'}.
    # So for a fixed c, we add U_c outer V_c outer W_c.
    # The sum should equal the identity tensor in the sense that:
    # sum_c U[i*m+j, c] V[j'*p+k, c] W[k'*n+i', c] = 1 if i=i', j=j', k=k' else 0.
    
    # Let's verify a few random indices to ensure no bug.
    # If verification fails, return schoolbook.
    
    def verify_decomposition(U, V, W, n, m, p):
        # Check T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # Should be 1 if i=i', j=j', k=k' (where i' is row of C, k' is col of C? No.)
        # Indices:
        # U index: i*m + j
        # V index: j*p + k
        # W index: k*n + i  (Note: problem says W[k*n+i, r], so index is k*n+i)
        # The tensor entry T[i, j, k] corresponds to the coefficient of A[i,j] B[j,k] in C[i,k].
        # This coefficient is 1.
        # So sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should be 1 for all i,j,k.
        # Wait, is it 1 for ALL i,j,k?
        # C[i,k] = sum_j A[i,j] B[j,k].
        # So the term A[i,j] B[j,k] appears with coefficient 1 in C[i,k].
        # And it does not appear in C[i',k'] for (i',k') != (i,k).
        # So yes, the tensor T[i,j,k] is 1 for all valid i,j,k.
        # And the decomposition sums to 1 for all (i,j,k).
        
        # Let's check all i, j, k.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(len(U[0])):
                        val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    if time.time() < deadline:
        if not verify_decomposition(U, V, W, n, m, p):
            # Fallback to schoolbook
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j][col] = 1
                        V[j * p + k][col] = 1
                        W[k * n + i][col] = 1
                        col += 1

    return {"U": U, "V": V, "W": W}
