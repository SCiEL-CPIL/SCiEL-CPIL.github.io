import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    _start = time.time()
    _deadline = _start + 118

    n, m, p = DIMENSIONS
    rank = 36

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def idx_u(i, j):
        return i * m + j

    def idx_v(j, k):
        return j * p + k

    def idx_w(k, i):
        return k * n + i

    current_col = 0

    def add_strassen(a_block, b_block, c_block):
        """
        a_block: list of 4 (i, j) indices for A (2x2)
        b_block: list of 4 (j, k) indices for B (2x2)
        c_block: list of 4 (k, i) indices for C (2x2)
        """
        nonlocal current_col
        
        # a_block order: (0,0), (0,1), (1,0), (1,1)
        # b_block order: (0,0), (0,1), (1,0), (1,1)
        # c_block order: (0,0), (1,0), (0,1), (1,1) -> W[k*n+i]
        
        a00, a01, a10, a11 = a_block
        b00, b01, b10, b11 = b_block
        c00, c10, c01, c11 = c_block

        # M1
        U[idx_u(*a00)][current_col] += 1
        U[idx_u(*a11)][current_col] += 1
        V[idx_v(*b00)][current_col] += 1
        V[idx_v(*b11)][current_col] += 1
        W[idx_w(*c00)][current_col] += 1
        W[idx_w(*c11)][current_col] += 1
        current_col += 1

        # M2
        U[idx_u(*a10)][current_col] += 1
        U[idx_u(*a11)][current_col] += 1
        V[idx_v(*b00)][current_col] += 1
        W[idx_w(*c10)][current_col] += 1
        W[idx_w(*c11)][current_col] -= 1
        current_col += 1

        # M3
        U[idx_u(*a00)][current_col] += 1
        V[idx_v(*b01)][current_col] += 1
        V[idx_v(*b11)][current_col] -= 1
        W[idx_w(*c01)][current_col] += 1
        W[idx_w(*c11)][current_col] += 1
        current_col += 1

        # M4
        U[idx_u(*a11)][current_col] += 1
        V[idx_v(*b10)][current_col] += 1
        V[idx_v(*b00)][current_col] -= 1
        W[idx_w(*c00)][current_col] += 1
        W[idx_w(*c10)][current_col] += 1
        current_col += 1

        # M5
        U[idx_u(*a00)][current_col] += 1
        U[idx_u(*a01)][current_col] += 1
        V[idx_v(*b11)][current_col] += 1
        W[idx_w(*c00)][current_col] -= 1
        W[idx_w(*c01)][current_col] += 1
        current_col += 1

        # M6
        U[idx_u(*a10)][current_col] += 1
        U[idx_u(*a00)][current_col] -= 1
        V[idx_v(*b00)][current_col] += 1
        V[idx_v(*b01)][current_col] += 1
        W[idx_w(*c11)][current_col] += 1
        current_col += 1

        # M7
        U[idx_u(*a01)][current_col] += 1
        U[idx_u(*a11)][current_col] -= 1
        V[idx_v(*b10)][current_col] += 1
        V[idx_v(*b11)][current_col] += 1
        W[idx_w(*c00)][current_col] += 1
        current_col += 1

    # Define blocks
    # A is 2x4. Split into A_L (2x2) and A_R (2x2)
    A_L = [(0,0), (0,1), (1,0), (1,1)]
    A_R = [(0,2), (0,3), (1,2), (1,3)]
    
    # B is 4x5. Split into B_00, B_01, B_02 (row 0-1) and B_10, B_11, B_12 (row 2-3)
    # B_00 is 2x2 (cols 0-1), B_01 is 2x2 (cols 2-3), B_02 is 2x1 (col 4)
    # B_10 is 2x2 (cols 0-1), B_11 is 2x2 (cols 2-3), B_12 is 2x1 (col 4)
    
    B_00 = [(0,0), (0,1), (1,0), (1,1)]
    B_01 = [(0,2), (0,3), (1,2), (1,3)]
    B_02 = [(0,4), (1,4)] # 2x1
    
    B_10 = [(2,0), (2,1), (3,0), (3,1)]
    B_11 = [(2,2), (2,3), (3,2), (3,3)]
    B_12 = [(2,4), (3,4)] # 2x1
    
    # C is 2x5.
    # C_00 is 2x2 (cols 0-1), C_01 is 2x2 (cols 2-3), C_02 is 2x1 (col 4)
    # Note W index is k*n + i. k is col of C (0..4), i is row of C (0..1)
    
    C_00 = [(0,0), (1,0), (0,1), (1,1)] # c00, c10, c01, c11 in local 2x2 sense?
    # In strassen_block, c_block was [(0,0), (1,0), (0,1), (1,1)] corresponding to w[0], w[2], w[1], w[3]
    # w[0] is C[k0,i0], w[1] is C[k1,i0], w[2] is C[k0,i1], w[3] is C[k1,i1]
    # So C_00 global: k=0,1; i=0,1
    C_00_global = [(0,0), (1,0), (0,1), (1,1)]
    C_01_global = [(2,0), (3,0), (2,1), (3,1)]
    C_02_global = [(4,0), (4,1)] # 2x1

    # Block 1: A_L * B_00 -> C_00
    add_strassen(A_L, B_00, C_00_global)
    
    # Block 2: A_R * B_10 -> C_00
    add_strassen(A_R, B_10, C_00_global)

    # Block 3: A_L * B_01 -> C_01
    add_strassen(A_L, B_01, C_01_global)

    # Block 4: A_R * B_11 -> C_01
    add_strassen(A_R, B_11, C_01_global)

    # Block 5: A_L * B_02 -> C_02 (2x2 * 2x1)
    # A_L is 2x2, B_02 is 2x1.
    # C_02 is 2x1.
    # C[0,4] = A[0,0]*B[0,4] + A[0,1]*B[1,4]
    # C[1,4] = A[1,0]*B[0,4] + A[1,1]*B[1,4]
    
    # Term 1: A[0,0]*B[0,4]
    U[idx_u(0,0)][current_col] += 1
    V[idx_v(0,4)][current_col] += 1
    W[idx_w(4,0)][current_col] += 1
    current_col += 1

    # Term 2: A[0,1]*B[1,4]
    U[idx_u(0,1)][current_col] += 1
    V[idx_v(1,4)][current_col] += 1
    W[idx_w(4,0)][current_col] += 1
    current_col += 1

    # Term 3: A[1,0]*B[0,4]
    U[idx_u(1,0)][current_col] += 1
    V[idx_v(0,4)][current_col] += 1
    W[idx_w(4,1)][current_col] += 1
    current_col += 1

    # Term 4: A[1,1]*B[1,4]
    U[idx_u(1,1)][current_col] += 1
    V[idx_v(1,4)][current_col] += 1
    W[idx_w(4,1)][current_col] += 1
    current_col += 1

    # Block 6: A_R * B_12 -> C_02
    # Term 1: A[0,2]*B[2,4]
    U[idx_u(0,2)][current_col] += 1
    V[idx_v(2,4)][current_col] += 1
    W[idx_w(4,0)][current_col] += 1
    current_col += 1

    # Term 2: A[0,3]*B[3,4]
    U[idx_u(0,3)][current_col] += 1
    V[idx_v(3,4)][current_col] += 1
    W[idx_w(4,0)][current_col] += 1
    current_col += 1

    # Term 3: A[1,2]*B[2,4]
    U[idx_u(1,2)][current_col] += 1
    V[idx_v(2,4)][current_col] += 1
    W[idx_w(4,1)][current_col] += 1
    current_col += 1

    # Term 4: A[1,3]*B[3,4]
    U[idx_u(1,3)][current_col] += 1
    V[idx_v(3,4)][current_col] += 1
    W[idx_w(4,1)][current_col] += 1
    current_col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
