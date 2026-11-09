"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin before 150s kill
    
    # 4x4 x 4x8 = two 4x4 x 4x4 multiplications
    # Use Strassen recursively for 4x4x4: rank 49
    # Total rank: 2 * 49 = 98 (vs schoolbook 128)
    n, m, p = DIMENSIONS  # n=4, m=4, p=8
    
    rank = 98
    U = [[0] * rank for _ in range(n * m)]  # 16 x 98
    V = [[0] * rank for _ in range(m * p)]  # 32 x 98
    W = [[0] * rank for _ in range(p * n)]  # 32 x 98
    
    def build_4x4x4_strassen():
        """Build rank-49 Strassen decomposition for 4x4x4."""
        r = 49
        U_local = [[0] * r for _ in range(16)]
        V_local = [[0] * r for _ in range(16)]
        W_local = [[0] * r for _ in range(16)]
        
        # Inner Strassen 2x2x2: 7 scalar products
        inner_strassen = [
            # M1 = (a00+a11)(b00+b11) -> c00+=M1, c11+=M1
            ([((0,0),1), ((1,1),1)], [((0,0),1), ((1,1),1)], [((0,0),1), ((1,1),1)]),
            # M2 = (a10+a11)b00 -> c10+=M2, c11-=M2
            ([((1,0),1), ((1,1),1)], [((0,0),1)], [((1,0),1), ((1,1),-1)]),
            # M3 = a00(b10-b11) -> c01+=M3, c11+=M3
            ([((0,0),1)], [((1,0),1), ((1,1),-1)], [((0,1),1), ((1,1),1)]),
            # M4 = a11(b01-b10) -> c00+=M4, c10+=M4
            ([((1,1),1)], [((0,1),1), ((1,0),-1)], [((0,0),1), ((1,0),1)]),
            # M5 = (a00+a10)b11 -> c00-=M5, c01+=M5
            ([((0,0),1), ((1,0),1)], [((1,1),1)], [((0,0),-1), ((0,1),1)]),
            # M6 = (a10-a00)(b00+b10) -> c11+=M6
            ([((1,0),1), ((0,0),-1)], [((0,0),1), ((1,0),1)], [((1,1),1)]),
            # M7 = (a11-a01)(b01+b11) -> c00+=M7
            ([((1,1),1), ((0,1),-1)], [((0,1),1), ((1,1),1)], [((0,0),1)]),
        ]
        
        # Macro Strassen for 4x4x4
        macros = [
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
            ({(1,0):1, (1,1):1}, {(0,0):1}),
            ({(0,0):1}, {(1,0):1, (1,1):-1}),
            ({(1,1):1}, {(0,1):1, (1,0):-1}),
            ({(0,0):1, (1,0):1}, {(1,1):1}),
            ({(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
            ({(1,1):1, (0,1):-1}, {(0,1):1, (1,1):1}),
        ]
        
        macro_output_blocks = [
            [(0,0,1), (1,1,1)],
            [(1,0,1), (1,1,-1)],
            [(0,1,1), (1,1,1)],
            [(0,0,1), (1,0,1)],
            [(0,0,-1), (0,1,1)],
            [(1,1,1)],
            [(0,0,1)],
        ]
        
        col_idx = 0
        for macro_idx, (A_coeffs, B_coeffs) in enumerate(macros):
            out_blocks = macro_output_blocks[macro_idx]
            for a_terms, b_terms, c_terms in inner_strassen:
                for (bi, bj), alpha in A_coeffs.items():
                    for (di, dj), ca in a_terms:
                        row = bi * 2 + di
                        col_a = bj * 2 + dj
                        U_local[row * 4 + col_a][col_idx] += alpha * ca
                
                for (bi_b, bj_b), beta in B_coeffs.items():
                    for (dj_b, dk), cb in b_terms:
                        row_b = bi_b * 2 + dj_b
                        col_b = bj_b * 2 + dk
                        V_local[row_b * 4 + col_b][col_idx] += beta * cb
                
                for (out_bi, out_bj, sign) in out_blocks:
                    for (di, dk), cc in c_terms:
                        row_c = out_bi * 2 + di
                        col_c = out_bj * 2 + dk
                        W_local[col_c * 4 + row_c][col_idx] += sign * cc
                
                col_idx += 1
        
        assert col_idx == 49
        return U_local, V_local, W_local
    
    U4, V4, W4 = build_4x4x4_strassen()
    
    for i in range(16):
        for c in range(49):
            U[i][c] = U4[i][c]
            U[i][49 + c] = U4[i][c]
    
    for j in range(4):
        for k in range(4):
            for c in range(49):
                V[j * 8 + k][c] = V4[j * 4 + k][c]
                V[j * 8 + 4 + k][49 + c] = V4[j * 4 + k][c]
    
    for k in range(4):
        for i in range(4):
            for c in range(49):
                W[k * 4 + i][c] = W4[k * 4 + i][c]
                W[(4 + k) * 4 + i][49 + c] = W4[k * 4 + i][c]
    
    if time.time() > deadline:
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
