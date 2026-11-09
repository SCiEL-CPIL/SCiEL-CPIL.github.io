import time

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-44 tensor decomposition for 3x4x4 matrix multiplication
    using Strassen's algorithm for 2x2x2 sub-blocks.
    """
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    rank = 44
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    def add_strassen(a_r0, a_c0, b_r0, b_c0, c_r0, c_c0, start_col):
        """
        Adds a Strassen decomposition for a 2x2x2 block.
        A block: rows a_r0, a_r0+1; cols a_c0, a_c0+1
        B block: rows b_r0, b_r0+1; cols b_c0, b_c0+1
        C block: rows c_r0, c_r0+1; cols c_c0, c_c0+1
        """
        if time.time() > deadline:
            return
            
        # Local indices in 2x2 blocks: 0,1 for row, 0,1 for col
        # A indices: (a_r0+dr, a_c0+dc) -> idx = (a_r0+dr)*m + (a_c0+dc)
        # B indices: (b_r0+dr, b_c0+dc) -> idx = (b_r0+dr)*p + (b_c0+dc)
        # C indices: (c_c0+dc, c_r0+dr) -> idx = (c_c0+dc)*n + (c_r0+dr)
        
        # Strassen multiplications M1..M7
        # M1: (A11+A22)(B11+B22)
        # M2: (A21+A22)B11
        # M3: A11(B12-B22)
        # M4: A22(B21-B11)
        # M5: (A11+A12)B22
        # M6: (A21-A11)(B11+B12)
        # M7: (A12-A22)(B21+B22)
        
        # Coefficients for A (U)
        # 1, 2, 3, 4, 5, 6, 7
        a_coeffs = [
            {0: 1, 3: 1},      # M1: A11, A22
            {2: 1, 3: 1},      # M2: A21, A22
            {0: 1},            # M3: A11
            {3: 1},            # M4: A22
            {0: 1, 1: 1},      # M5: A11, A12
            {2: 1, 0: -1},     # M6: A21, -A11
            {1: 1, 3: -1}      # M7: A12, -A22
        ]
        
        # Coefficients for B (V)
        b_coeffs = [
            {0: 1, 3: 1},      # M1: B11, B22
            {0: 1},            # M2: B11
            {1: 1, 3: -1},     # M3: B12, -B22
            {2: 1, 0: -1},     # M4: B21, -B11
            {3: 1},            # M5: B22
            {0: 1, 1: 1},      # M6: B11, B12
            {2: 1, 3: 1}       # M7: B21, B22
        ]
        
        # Coefficients for C (W)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        c_coeffs = [
            {0: 1, 3: 1, 1: -1, 7: 1},   # C11 (local idx 0)
            {3: 1, 5: 1},                 # C12 (local idx 1)
            {2: 1, 4: 1},                 # C21 (local idx 2)
            {1: 1, 2: -1, 3: 1, 6: 1}    # C22 (local idx 3)
        ]
        
        for r in range(7):
            col = start_col + r
            if col >= rank:
                break
                
            # Fill U
            for local_idx, coeff in a_coeffs[r].items():
                dr, dc = divmod(local_idx, 2)
                u_idx = (a_r0 + dr) * m + (a_c0 + dc)
                U[u_idx][col] += coeff
                
            # Fill V
            for local_idx, coeff in b_coeffs[r].items():
                dr, dc = divmod(local_idx, 2)
                v_idx = (b_r0 + dr) * p + (b_c0 + dc)
                V[v_idx][col] += coeff
                
            # Fill W
            # Local C indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
            # Global C index: c_r0+dr, c_c0+dc -> (c_c0+dc)*n + (c_r0+dr)
            for local_idx, coeff in c_coeffs[r].items():
                dr, dc = divmod(local_idx, 2)
                w_idx = (c_c0 + dc) * n + (c_r0 + dr)
                W[w_idx][col] += coeff

    # 1. A11 B11 -> C11
    # A11: r0=0, c0=0. B11: r0=0, c0=0. C11: r0=0, c0=0
    add_strassen(0, 0, 0, 0, 0, 0, 0)
    
    # 2. A12 B21 -> C11
    # A12: r0=0, c0=2. B21: r0=2, c0=0. C11: r0=0, c0=0
    # Note: A12 is 2x2, B21 is 2x2.
    # A12 rows 0,1 cols 2,3.
    # B21 rows 2,3 cols 0,1.
    # Result adds to C11 (rows 0,1 cols 0,1).
    add_strassen(0, 2, 2, 0, 0, 0, 7)
    
    # 3. A11 B12 -> C12
    # A11: r0=0, c0=0. B12: r0=0, c0=2. C12: r0=0, c0=2
    add_strassen(0, 0, 0, 2, 0, 2, 14)
    
    # 4. A12 B22 -> C12
    # A12: r0=0, c0=2. B22: r0=2, c0=2. C12: r0=0, c0=2
    add_strassen(0, 2, 2, 2, 0, 2, 21)
    
    # 5. Bottom part: A_bot (1x4) * B (4x4) -> C_bot (1x4)
    # A_bot is row 2, cols 0,1,2,3.
    # B is all rows 0,1,2,3 and cols 0,1,2,3.
    # C_bot is row 2, cols 0,1,2,3.
    # Schoolbook: 16 terms.
    start_col = 28
    for j in range(4):
        for k in range(4):
            col = start_col + j * 4 + k
            if col >= rank:
                break
            u_idx = 2 * m + j
            v_idx = j * p + k
            w_idx = k * n + 2
            
            U[u_idx][col] = 1.0
            V[v_idx][col] = 1.0
            W[w_idx][col] = 1.0

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
