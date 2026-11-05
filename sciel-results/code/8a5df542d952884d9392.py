import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # Rank calculation:
    # Part 1: A1 (2x4) * B (4x8)
    # Split m into 2x2. Two blocks of 2x2x8.
    # Split p into 4x4. Two blocks of 2x2x4.
    # Split p into 2x2. Two blocks of 2x2x2.
    # 2 * 2 * 2 = 8 blocks of 2x2x2.
    # Each 2x2x2 uses Strassen (rank 7).
    # Rank 1 = 8 * 7 = 56.
    # Part 2: A2 (1x4) * B (4x8)
    # 1x4x8 is 32 scalar mults.
    # Rank 2 = 32.
    # Total Rank = 88.
    
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen coefficients for 2x2x2
    # M1: (A00+A11)(B00+B11) -> C00:1, C11:1
    # M2: (A10+A11)B00 -> C10:1, C11:-1
    # M3: A00(B10-B11) -> C01:1, C11:1
    # M4: A11(B01-B00) -> C00:1, C10:1
    # M5: (A00+A01)B11 -> C00:-1, C01:1
    # M6: (A10-A00)(B00+B10) -> C11:1
    # M7: (A01-A11)(B01+B11) -> C00:1
    
    # Define the 7 terms as (U_coeffs, V_coeffs, W_coeffs)
    # U coeffs for A00, A01, A10, A11
    # V coeffs for B00, B01, B10, B11
    # W coeffs for C00, C01, C10, C11
    
    strassen_terms = [
        # M1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        # M3
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 1]),
        # M4
        ([0, 0, 0, 1], [0, 1, -1, 0], [1, 0, 1, 0]),
        # M5
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # M6
        ([-1, 0, 1, 0], [1, 0, 1, 0], [0, 0, 0, 1]),
        # M7
        ([0, 1, 0, -1], [0, 1, 0, 1], [1, 0, 0, 0]),
    ]
    
    # Iterate over the 8 blocks for A1 * B
    # m_block in [0, 1] -> j offset 0, 2
    # p_block1 in [0, 1] -> k offset 0, 4
    # p_block2 in [0, 1] -> k offset 0, 2 (within p_block1)
    
    for m_block in range(2):
        j_off = m_block * 2
        for p_block1 in range(2):
            k_off_base = p_block1 * 4
            for p_block2 in range(2):
                k_off = k_off_base + p_block2 * 2
                
                # For each of the 7 Strassen terms
                for u_c, v_c, w_c in strassen_terms:
                    # Add to U
                    # A indices: i in {0, 1}, j in {j_off, j_off+1}
                    # A00 -> i=0, j=j_off
                    # A01 -> i=0, j=j_off+1
                    # A10 -> i=1, j=j_off
                    # A11 -> i=1, j=j_off+1
                    U[0 * m + j_off][col] += u_c[0]
                    U[0 * m + j_off + 1][col] += u_c[1]
                    U[1 * m + j_off][col] += u_c[2]
                    U[1 * m + j_off + 1][col] += u_c[3]
                    
                    # Add to V
                    # B indices: j in {j_off, j_off+1}, k in {k_off, k_off+1}
                    # B00 -> j=j_off, k=k_off
                    # B01 -> j=j_off, k=k_off+1
                    # B10 -> j=j_off+1, k=k_off
                    # B11 -> j=j_off+1, k=k_off+1
                    V[j_off * p + k_off][col] += v_c[0]
                    V[j_off * p + k_off + 1][col] += v_c[1]
                    V[(j_off + 1) * p + k_off][col] += v_c[2]
                    V[(j_off + 1) * p + k_off + 1][col] += v_c[3]
                    
                    # Add to W
                    # C indices: i in {0, 1}, k in {k_off, k_off+1}
                    # C00 -> i=0, k=k_off
                    # C01 -> i=0, k=k_off+1
                    # C10 -> i=1, k=k_off
                    # C11 -> i=1, k=k_off+1
                    W[k_off * n + 0][col] += w_c[0]
                    W[(k_off + 1) * n + 0][col] += w_c[1]
                    W[k_off * n + 1][col] += w_c[2]
                    W[(k_off + 1) * n + 1][col] += w_c[3]
                    
                    col += 1
    
    # Part 2: A2 * B (1x4 * 4x8)
    # i = 2
    # j in 0..3, k in 0..7
    for j in range(4):
        for k in range(8):
            U[2 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 2][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
