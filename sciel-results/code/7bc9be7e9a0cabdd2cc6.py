import time

DIMENSIONS = [2, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    rank = 21
    
    # Laderman's decomposition for 4x4x4
    # We embed 2x4x4 into 4x4x4
    # A_pad is 4x4, top 2x4 is A, bottom 2x4 is 0
    # B is 4x4
    # C_pad is 4x4, top 2x4 is C
    
    # Laderman coefficients (16x21)
    # U, V, W
    # I will use a known set of coefficients.
    
    # Note: The specific coefficients for Laderman are complex.
    # I will use a placeholder if I can't recall them exactly.
    # However, I can use the Strassen 28-rank solution which is known to be correct.
    # But 21 is better.
    
    # Let's try to use a simpler Rank 24 solution if Laderman is too risky.
    # Or just stick to 28.
    
    # Actually, I will implement the Rank 28 solution as it is verified in the prompt.
    # The prompt says "Keep this design".
    # So I should NOT switch to Laderman if it's a different design family.
    # "Do NOT switch to a different design family."
    # The current design is "Block Strassen".
    # Laderman is a different algorithm.
    
    # So I must stay with Block Strassen or similar.
    # Can I improve Block Strassen?
    # Maybe by using a better $2 \times 2 \times 2$ decomposition? No, Strassen is optimal for $2 \times 2 \times 2$.
    
    # So I will stick to Rank 28.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})
    ]
    
    blocks = [
        (0, 0, 0, 0),
        (2, 2, 0, 0),
        (0, 0, 2, 2),
        (2, 2, 2, 2)
    ]
    
    col_idx = 0
    for a_off, b_row_off, b_col_off, c_off in blocks:
        for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
            for local_idx, val in u_coeffs.items():
                g_idx = (local_idx // 2) * m + a_off + (local_idx % 2)
                U[g_idx][col_idx] = val
            for local_idx, val in v_coeffs.items():
                g_idx = (b_row_off + local_idx // 2) * p + b_col_off + (local_idx % 2)
                V[g_idx][col_idx] = val
            for local_idx, val in w_coeffs.items():
                g_idx = (c_off + local_idx % 2) * n + (local_idx // 2)
                W[g_idx][col_idx] = val
            col_idx += 1
            
    return {"U": U, "V": V, "W": W}
