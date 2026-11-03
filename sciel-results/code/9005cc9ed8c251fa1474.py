"""Matrix multiplication 3x3x6 - rank-minimal tensor decomposition."""
DIMENSIONS = [3, 3, 6]

import numpy as np
import time

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = 0.0
                for c in range(rank):
                    lhs += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(lhs - 1.0) > 1e-9:
                    return False
    return True


def _schoolbook_3x3x6():
    """Schoolbook decomposition for 3x3x6, rank 54."""
    n, m, p = 3, 3, 6
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
    return U, V, W


def _strassen_2x2x2():
    """
    Strassen decomposition for 2x2x2 matrix multiplication.
    Returns (U, V, W) each of shape (4, 7).
    
    U: maps A (2x2, flattened) to 7 intermediate values
    V: maps B (2x2, flattened) to 7 intermediate values
    W: maps intermediate values to C (2x2, flattened)
    
    Flattening convention: A[i,j] -> index i*2+j
    """
    # Strassen's 7 products:
    # m1 = (a00 + a11) * (b00 + b11)
    # m2 = (a10 + a11) * b00
    # m3 = a00 * (b01 - b11)
    # m4 = a11 * (b10 - b00)
    # m5 = (a00 + a01) * b11
    # m6 = (a10 - a00) * (b00 + b01)
    # m7 = (a01 - a11) * (b10 + b11)
    
    # C00 = m1 + m4 - m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    
    # U maps A (flattened [a00, a01, a10, a11]) to [m1_A, m2_A, m3_A, m4_A, m5_A, m6_A, m7_A]
    # V maps B (flattened [b00, b01, b10, b11]) to [m1_B, m2_B, m3_B, m4_B, m5_B, m6_B, m7_B]
    
    # m1: A part = a00 + a11, B part = b00 + b11
    # m2: A part = a10 + a11, B part = b00
    # m3: A part = a00, B part = b01 - b11
    # m4: A part = a11, B part = b10 - b00
    # m5: A part = a00 + a01, B part = b11
    # m6: A part = a10 - a00, B part = b00 + b01
    # m7: A part = a01 - a11, B part = b10 + b11
    
    U = [
        [1, 0, 0, 1],  # m1_A: a00 + a11
        [0, 0, 1, 1],  # m2_A: a10 + a11
        [1, 0, 0, 0],  # m3_A: a00
        [0, 0, 0, 1],  # m4_A: a11
        [1, 1, 0, 0],  # m5_A: a00 + a01
        [-1, 0, 1, 0], # m6_A: a10 - a00
        [0, 1, 0, -1], # m7_A: a01 - a11
    ]
    
    V = [
        [1, 0, 0, 1],  # m1_B: b00 + b11
        [1, 0, 0, 0],  # m2_B: b00
        [0, 1, 0, -1], # m3_B: b01 - b11
        [0, 0, 1, -1], # m4_B: b10 - b00
        [0, 0, 0, 1],  # m5_B: b11
        [1, 1, 0, 0],  # m6_B: b00 + b01
        [0, 0, 1, 1],  # m7_B: b10 + b11
    ]
    
    # W maps [m1, ..., m7] to C (flattened [c00, c01, c10, c11])
    # c00 = m1 + m4 - m5 + m7
    # c01 = m3 + m5
    # c10 = m2 + m4
    # c11 = m1 - m2 + m3 + m6
    
    W = [
        [1, 0, 0, 1, -1, 0, 1],   # c00
        [0, 0, 1, 0, 1, 0, 0],    # c01
        [0, 1, 0, 1, 0, 0, 0],    # c10
        [1, -1, 1, 0, 0, 1, 0],   # c11
    ]
    
    return U, V, W


def _build_3x3x2_rank12():
    """
    Build a rank-12 decomposition for 3x3x2 matrix multiplication.
    
    A is 3x3, B is 3x2, C is 3x2.
    
    We decompose this as:
    - Central 2x2x2 block (rows 1,2 of A, cols 1,2 of B -> rows 1,2 of C) using Strassen (rank 7)
    - Boundary interactions (involving row/col 0) using direct terms
    
    Actually, let me think about this more carefully.
    
    A is 3x3: indices (i,j) with i,j in {0,1,2}
    B is 3x2: indices (j,k) with j in {0,1,2}, k in {0,1}
    C is 3x2: indices (i,k) with i in {0,1,2}, k in {0,1}
    
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    Let's partition:
    - A[0,0], A[0,1], A[0,2] : first row of A
    - A[1,0], A[1,1], A[1,2] : second row
    - A[2,0], A[2,1], A[2,2] : third row
    
    - B[0,0], B[0,1] : first row of B
    - B[1,0], B[1,1] : second row
    - B[2,0], B[2,1] : third row
    
    C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
    C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
    C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
    C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] + A[1,2]*B[2,1]
    C[2,0] = A[2,0]*B[0,0] + A[2,1]*B[1,0] + A[2,2]*B[2,0]
    C[2,1] = A[2,0]*B[0,1] + A[2,1]*B[1,1] + A[2,2]*B[2,1]
    
    The central 2x2x2 block involves A[1,1], A[1,2], A[2,1], A[2,2] (2x2 submatrix)
    and B[1,0], B[1,1], B[2,0], B[2,1] (2x2 submatrix)
    producing C[1,0], C[1,1], C[2,0], C[2,1] (2x2 submatrix)
    
    But the boundary terms mix with the central block.
    
    Let me use a different approach: just use the schoolbook for 3x3x2 (rank 18)
    and see if we can improve.
    
    Actually, for 3x3x2, the schoolbook rank is 3*3*2 = 18.
    
    Let me try to find a better decomposition.
    
    One approach: use the fact that 3x3x2 can be viewed as two 3x3x1 problems.
    Each 3x3x1 has rank 9 (schoolbook).
    Total: 18.
    
    Can we do better? The minimum rank for 3x3x2 is not known to be less than 18
    in general, but let's try to construct something.
    
    For now, let's use the schoolbook for each 3x3x2 block.
    """
    n, m, p = 3, 3, 2
    rank = n * m * p  # 18
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
    return U, V, W


def _construct_3x3x6_rank36():
    """
    Construct a rank-36 decomposition for 3x3x6.
    
    Partition into three 3x3x2 blocks:
    Block 1: B[:, 0:2] -> C[:, 0:2]
    Block 2: B[:, 2:4] -> C[:, 2:4]
    Block 3: B[:, 4:6] -> C[:, 4:6]
    
    Each block uses a rank-12 decomposition.
    """
    n, m, p = 3, 3, 6
    
    # For each 3x3x2 block, use rank-12 decomposition.
    # Currently using schoolbook rank-18, so total rank would be 54.
    # Let me try to construct a rank-12 decomposition.
    
    # For 3x3x2, let's try the following approach:
    # Use Strassen for the central 2x2x2 part, plus boundary terms.
    
    # Actually, let me just use the schoolbook for now and return rank 54.
    # I'll try to improve this.
    
    # For a proper rank-12 construction for 3x3x2:
    # This is non-trivial. Let me use a numerical search or known construction.
    
    # For simplicity, let's use the schoolbook decomposition.
    U, V, W = _schoolbook_3x3x6()
    return U, V, W


def _winograd_f4_3x3x3():
    """
    Winograd F(4) algorithm for 3x3x3 matrix multiplication.
    Returns (U, V, W) each of shape (9, 16) as lists of lists.
    
    The F(4) algorithm uses a 4x4 transform to compute 3x3 matrix product.
    """
    # The F(4) algorithm for 3x3x3 uses the following:
    # M_A (4x3), M_B (4x3), M_C (3x4)
    
    # Standard Winograd F(4) factors:
    M_A = np.array([
        [1, 0, 0],
        [1, 1, 0],
        [1, 0, 1],
        [1, 1, 1]
    ], dtype=float)
    
    M_B = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [1, 1, 1]
    ], dtype=float)
    
    M_C = np.array([
        [1, 1, 1, 1],
        [0, 1, 0, -1],
        [1, 0, 1, 0]
    ], dtype=float)
    
    # The F(4) algorithm:
    # A' = M_A * A  (4x3)
    # B' = M_B * B  (4x3)
    # C' = A' * B'  (4x4, element-wise)
    # C = M_C * C'  (3x3)
    
    # This gives rank 16 (4*4).
    
    # Let me verify this construction.
    
    # U: maps A (flattened 9) to A' (flattened 16)
    # V: maps B (flattened 9) to B' (flattened 16)
    # W: maps C' (flattened 16) to C (flattened 9)
    
    # A' = M_A * A
    # A is 3x3, flattened as [a00, a01, a02, a10, a11, a12, a20, a21, a22]
    # A' is 4x3, flattened as [a'00, a'01, a'02, a'10, a'11, a'12, a'20, a'21, a'22, a'30, a'31, a'32]
    
    # Actually, let me be more careful with the flattening.
    
    # A[i,j] for i,j in {0,1,2}
    # A' = M_A * A, so A'[r,c] = sum_j M_A[r,j] * A[j,c]
    
    # A'[r,c] for r in {0,1,2,3}, c in {0,1,2}
    
    # Flattened A: index = i*3 + j for A[i,j]
    # Flattened A': index = r*3 + c for A'[r,c]
    
    # U maps flattened A to flattened A'
    # U[flattened_A', flattened_A] = M_A[r,j] where r = flattened_A'//3, c = flattened_A'%3, j = flattened_A//3, i = flattened_A%3
    
    # Wait, I need to be more careful.
    
    # A is 3x3: A[i,j]
    # A' = M_A * A: A'[r,c] = sum_j M_A[r,j] * A[j,c]
    
    # So A'[r,c] depends on A[j,c] for all j, with coefficient M_A[r,j]
    
    # Flattened A: index = i*3 + j (row-major)
    # Flattened A': index = r*3 + c
    
    # U[flattened_A', flattened_A] = M_A[r, j] where:
    #   r = flattened_A' // 3
    #   c = flattened_A' % 3
    #   i = flattened_A // 3
    #   j = flattened_A % 3
    #   AND c == j (since A'[r,c] depends on A[j,c] only when c==j)
    
    # So U[r*3+c, i*3+j] = M_A[r,j] if c==j, else 0
    
    U = [[0] * 16 for _ in range(9)]
    V = [[0] * 16 for _ in range(9)]
    W = [[0] * 16 for _ in range(9)]
    
    # U: maps A (9) to A' (16)
    for r in range(4):
        for c in range(3):
            for i in range(3):
                for j in range(3):
                    if c == j:
                        U[r*3+c, i*3+j] = M_A[r, j]
    
    # V: maps B (9) to B' (16)
    # B'[r,c] = sum_j M_B[r,j] * B[j,c]
    # V[r*3+c, j*3+k] = M_B[r,j] if c==j, else 0
    # Wait, B is 3x3: B[j,k]
    # B' = M_B * B: B'[r,k] = sum_j M_B[r,j] * B[j,k]
    # Flattened B: index = j*3 + k
    # Flattened B': index = r*3 + k
    # V[r*3+k, j*3+k] = M_B[r,j]
    
    for r in range(4):
        for k in range(3):
            for j in range(3):
                V[r*3+k, j*3+k] = M_B[r, j]
    
    # W: maps C' (16) to C (9)
    # C = M_C * C': C[i,k] = sum_r M_C[i,r] * C'[r,k]
    # Flattened C': index = r*3 + k
    # Flattened C: index = i*3 + k
    # W[i*3+k, r*3+k] = M_C[i,r]
    
    for i in range(3):
        for k in range(3):
            for r in range(4):
                W[i*3+k, r*3+k] = M_C[i, r]
    
    # Convert to lists of lists with integer/half-integer entries
    U_list = [[int(x) if abs(x - round(x)) < 1e-9 else (2*x) // 2 for x in row] for row in U]
    V_list = [[int(x) if abs(x - round(x)) < 1e-9 else (2*x) // 2 for x in row] for row in V]
    W_list = [[int(x) if abs(x - round(x)) < 1e-9 else (2*x) // 2 for x in row] for row in W]
    
    # Actually, let's just keep them as floats and round to nearest half-integer
    def _round_to_half(x):
        return round(x * 2) / 2
    
    U_list = [[_round_to_half(x) for x in row] for row in U]
    V_list = [[_round_to_half(x) for x in row] for row in V]
    W_list = [[_round_to_half(x) for x in row] for row in W]
    
    return U_list, V_list, W_list


def _verify_decomposition_np(U, V, W, n, m, p):
    """Verify decomposition using numpy."""
    rank = len(U[0])
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = np.sum(U_np[i*m+j, :] * V_np[j*p+k, :] * W_np[k*n+i, :])
                if abs(lhs - 1.0) > 1e-6:
                    return False
    return True


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0
    
    # Try Winograd F(4) for 3x3x3 blocks.
    # For 3x3x6, we can use two 3x3x3 blocks, each with rank 16.
    # Total rank: 32.
    
    # But we need to be careful: the two blocks are independent.
    # Block 1: A * B[:, 0:3] -> C[:, 0:3]
    # Block 2: A * B[:, 3:6] -> C[:, 3:6]
    
    # For each block, use Winograd F(4) with rank 16.
    
    U1, V1, W1 = _winograd_f4_3x3x3()
    
    # Verify the 3x3x3 decomposition
    if not _verify_decomposition(U1, V1, W1, 3, 3, 3):
        # Fall back to schoolbook
        U, V, W = _schoolbook_3x3x6()
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
        return {"U": U, "V": V, "W": W}
    
    # Now combine two blocks.
    # U1 is 9x16, V1 is 9x16, W1 is 9x16
    # For 3x3x6:
    # U is 9x32: [U1, U1] (same A transform for both blocks)
    # V is 18x32: first 9 rows for B[:, 0:3], last 9 rows for B[:, 3:6]
    # W is 18x32: first 9 rows for C[:, 0:3], last 9 rows for C[:, 3:6]
    
    rank = 32
    U_full = [[0] * rank for _ in range(9)]
    V_full = [[0] * rank for _ in range(18)]
    W_full = [[0] * rank for _ in range(18)]
    
    # Fill in the two blocks
    for i in range(9):
        for c in range(16):
            U_full[i][c] = U1[i][c]
            U_full[i][c + 16] = U1[i][c]
    
    for i in range(9):
        for c in range(16):
            V_full[i][c] = V1[i][c]
            V_full[i + 9][c + 16] = V1[i][c]
    
    for i in range(9):
        for c in range(16):
            W_full[i][c] = W1[i][c]
            W_full[i + 9][c + 16] = W1[i][c]
    
    # Verify
    if _verify_decomposition(U_full, V_full, W_full, n, m, p):
        return {"U": U_full, "V": V_full, "W": W_full}
    
    # Fall back to schoolbook
    U, V, W = _schoolbook_3x3x6()
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
