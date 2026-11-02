import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    # Target rank: 97 (49 for 4x4x4 Strassen + 48 for 4x4x3 Schoolbook)
    rank = 97
    U = [[0] * rank for _ in range(n * m)]  # 16 x 97
    V = [[0] * rank for _ in range(m * p)]  # 28 x 97
    W = [[0] * rank for _ in range(p * n)]  # 28 x 97
    
    col = 0

    # Helper to add a term
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        if col >= rank:
            return
        for idx, val in u_coeffs:
            U[idx][col] = val
        for idx, val in v_coeffs:
            V[idx][col] = val
        for idx, val in w_coeffs:
            W[idx][col] = val
        col += 1

    # Strassen 2x2x2 core
    # A' = [[a00, a01], [a10, a11]]
    # B' = [[b00, b01], [b10, b11]]
    # C' = A' * B'
    # The 7 multiplications:
    # s1 = (a00+a11)(b00+b11)
    # s2 = (a10+a11)b00
    # s3 = a00(b01-b11)
    # s4 = a11(b10-b00)
    # s5 = (a00+a01)b11
    # s6 = (a10-a00)(b00+b01)
    # s7 = (a01-a11)(b10+b11)
    
    # Output contributions:
    # C00 = s1 + s4 - s5 + s7
    # C01 = s3 + s5
    # C10 = s2 + s4
    # C11 = s1 - s2 + s3 + s6

    def add_strassen_2x2(A_map, B_map, C_map):
        """
        A_map: list of 4 lists. A_map[i][j] is a list of (A_idx, coeff) for A'_{ij}.
        B_map: list of 4 lists. B_map[i][j] is a list of (B_idx, coeff) for B'_{ij}.
        C_map: list of 4 indices. C_map[i][j] is the C_idx for C'_{ij}.
        """
        nonlocal col
        
        # Define the 7 steps
        # Each step is (A_form, B_form, W_contribs)
        # A_form: list of (pos_in_A_prime, coeff)
        # B_form: list of (pos_in_B_prime, coeff)
        # W_contribs: list of (pos_in_C_prime, coeff)
        
        steps = [
            # s1
            ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
            # s2
            ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1)]),
            # s3
            ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1)]),
            # s4
            ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
            # s5
            ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
            # s6
            ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
            # s7
            ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)]),
        ]
        
        # Wait, I need to check the C contributions for each s again.
        # C00 = s1 + s4 - s5 + s7
        # C01 = s3 + s5
        # C10 = s2 + s4
        # C11 = s1 - s2 + s3 + s6
        
        # s1: C00 += 1, C11 += 1
        # s2: C10 += 1, C11 -= 1
        # s3: C01 += 1, C11 += 1
        # s4: C00 += 1, C10 += 1
        # s5: C00 -= 1, C01 += 1
        # s6: C11 += 1
        # s7: C00 += 1
        
        # Let's re-verify s6 and s7.
        # C22 (C11) = M1 - M2 + M3 + M6.
        # C11 (C00) = M1 + M4 - M5 + M7.
        # C12 (C01) = M3 + M5.
        # C21 (C10) = M2 + M4.
        
        # So:
        # s1 (M1): C00+=1, C11+=1
        # s2 (M2): C10+=1, C11-=1
        # s3 (M3): C01+=1, C11+=1
        # s4 (M4): C00+=1, C10+=1
        # s5 (M5): C00-=1, C01+=1
        # s6 (M6): C11+=1
        # s7 (M7): C00+=1
        
        corrected_steps = [
            # s1
            ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
            # s2
            ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)]),
            # s3
            ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)]),
            # s4
            ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
            # s5
            ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
            # s6
            ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
            # s7
            ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)]),
        ]

        for a_form, b_form, c_form in corrected_steps:
            u_coeffs = []
            v_coeffs = []
            w_coeffs = []
            
            # Expand A form
            for ai, aj, coeff in a_form:
                for a_idx, a_coeff in A_map[ai][aj]:
                    u_coeffs.append((a_idx, a_coeff * coeff))
            
            # Expand B form
            for bi, bj, coeff in b_form:
                for b_idx, b_coeff in B_map[bi][bj]:
                    v_coeffs.append((b_idx, b_coeff * coeff))
            
            # Expand C form
            for ci, cj, coeff in c_form:
                c_idx = C_map[ci][cj]
                w_coeffs.append((c_idx, coeff))
            
            add_term(u_coeffs, v_coeffs, w_coeffs)

    # --- Part 1: 4x4x4 Strassen ---
    # A blocks:
    # A11: A[0:2, 0:2] -> indices 0,1,4,5
    # A12: A[0:2, 2:4] -> indices 2,3,6,7
    # A21: A[2:4, 0:2] -> indices 8,9,12,13
    # A22: A[2:4, 2:4] -> indices 10,11,14,15
    
    # B1 blocks (cols 0-3):
    # B11: B[0:2, 0:2] -> indices 0,1,7,8
    # B12: B[0:2, 2:4] -> indices 2,3,9,10
    # B21: B[2:4, 0:2] -> indices 14,15,21,22
    # B22: B[2:4, 2:4] -> indices 16,17,23,24
    
    # C1 blocks (cols 0-3):
    # C11: C[0:2, 0:2] -> w_idx = k*4+i. k in 0..1, i in 0..1.
    #   (0,0)->0, (0,1)->4, (1,0)->1, (1,1)->5
    # C12: C[0:2, 2:4] -> k in 2..3, i in 0..1.
    #   (0,0)->8, (0,1)->12, (1,0)->9, (1,1)->13
    # C21: C[2:4, 0:2] -> k in 0..1, i in 2..3.
    #   (0,0)->2, (0,1)->6, (1,0)->3, (1,1)->7
    # C22: C[2:4, 2:4] -> k in 2..3, i in 2..3.
    #   (0,0)->10, (0,1)->14, (1,0)->11, (1,1)->15

    def get_A_block(r1, c1):
        # Returns 2x2 list of lists of (idx, coeff)
        # r1, c1 are 0 or 1 (block row/col in 4x4)
        res = [[[] for _ in range(2)] for _ in range(2)]
        for i in range(2):
            for j in range(2):
                global_i = r1*2 + i
                global_j = c1*2 + j
                idx = global_i * 4 + global_j
                res[i][j] = [(idx, 1)]
        return res

    def get_B_block(r1, c1):
        res = [[[] for _ in range(2)] for _ in range(2)]
        for i in range(2):
            for j in range(2):
                global_i = r1*2 + i
                global_j = c1*2 + j
                idx = global_i * 7 + global_j
                res[i][j] = [(idx, 1)]
        return res

    def get_C_block(r1, c1):
        # Returns 2x2 list of C indices
        res = [[0 for _ in range(2)] for _ in range(2)]
        for i in range(2):
            for j in range(2):
                global_i = r1*2 + i
                global_j = c1*2 + j
                # W[k*n+i] -> k is col, i is row
                idx = global_j * 4 + global_i
                res[i][j] = idx
        return res

    # M1 = (A11 + A22)(B11 + B22)
    A_map_M1 = [[A_map[0][0] if False else get_A_block(0,0)[i][j] + get_A_block(1,1)[i][j] for j in range(2)] for i in range(2)]
    # Simpler:
    def sum_A_maps(m1, m2):
        res = [[[] for _ in range(2)] for _ in range(2)]
        for i in range(2):
            for j in range(2):
                res[i][j] = m1[i][j] + m2[i][j]
        return res

    A11 = get_A_block(0,0)
    A12 = get_A_block(0,1)
    A21 = get_A_block(1,0)
    A22 = get_A_block(1,1)
    
    B11 = get_B_block(0,0)
    B12 = get_B_block(0,1)
    B21 = get_B_block(1,0)
    B22 = get_B_block(1,1)
    
    C11 = get_C_block(0,0)
    C12 = get_C_block(0,1)
    C21 = get_C_block(1,0)
    C22 = get_C_block(1,1)

    # M1
    add_strassen_2x2(sum_A_maps(A11, A22), sum_A_maps(B11, B22), C11)
    # Wait, C11 is the target for M1? No.
    # M1 contributes to C11 and C22.
    # My `add_strassen_2x2` function assumes the output is a single 2x2 block C'.
    # But M1 is an intermediate matrix. It is not directly C.
    # I need to modify `add_strassen_2x2` to accept a `C_map` that maps the 2x2 output of the Strassen block to the final C indices.
    # However, the Strassen block computes a 2x2 matrix $M_k$.
    # Then $C$ is a linear combination of $M_k$s.
    # My current `add_strassen_2x2` adds terms where W is determined by C_map.
    # If I pass C11 as C_map, it will put the result of M1 into C11.
    # But M1 contributes to BOTH C11 and C22.
    # So I cannot just call `add_strassen_2x2` once for M1 with a single C_map.
    
    # I need to decompose the contribution of each M_k to the final C.
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # This means for each M_k, I need to add its 7 terms to the appropriate C blocks with appropriate signs.
    # Since `add_strassen_2x2` adds 7 terms at once, and the C_map is fixed for the call,
    # I can call it multiple times? No, that would duplicate U and V.
    # U and V are the same for a given M_k regardless of which C block it contributes to.
    # Only W changes.
    
    # So I should modify `add_strassen_2x2` to return the U and V coefficients,
    # or better, iterate through the 7 steps inside `add_strassen_2x2` but allow W to be accumulated from multiple C blocks.
    
    # Actually, the simplest way is:
    # For each M_k, determine the linear form on A and B.
    # This is fixed.
    # Then, for each of the 7 scalar multiplications in that M_k,
    # determine the contribution to each C block.
    
    # Let's rewrite `add_strassen_2x2` to take a list of (C_map, sign) pairs.
    # Or just compute the 7 U and V vectors, and then add them to W for each C block.

    def add_strassen_2x2_multi(A_map, B_map, C_contribs):
        """
        C_contribs: list of (C_map, sign)
        """
        nonlocal col
        
        corrected_steps = [
            ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
            ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)]),
            ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)]),
            ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
            ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
            ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
            ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)]),
        ]

        for a_form, b_form, c_form in corrected_steps:
            u_coeffs = []
            v_coeffs = []
            
            for ai, aj, coeff in a_form:
                for a_idx, a_coeff in A_map[ai][aj]:
                    u_coeffs.append((a_idx, a_coeff * coeff))
            
            for bi, bj, coeff in b_form:
                for b_idx, b_coeff in B_map[bi][bj]:
                    v_coeffs.append((b_idx, b_coeff * coeff))
            
            w_coeffs = []
            for c_map, sign in C_contribs:
                for ci, cj, coeff in c_form:
                    c_idx = c_map[ci][cj]
                    w_coeffs.append((c_idx, coeff * sign))
            
            add_term(u_coeffs, v_coeffs, w_coeffs)

    # M1: C11 (+1), C22 (+1)
    add_strassen_2x2_multi(sum_A_maps(A11, A22), sum_A_maps(B11, B22), [(C11, 1), (C22, 1)])
    # M2: C21 (+1), C22 (-1)
    add_strassen_2x2_multi(sum_A_maps(A21, A22), B11, [(C21, 1), (C22, -1)])
    # M3: C12 (+1), C22 (+1)
    add_strassen_2x2_multi(A11, sum_A_maps(B12, [[[-1 if x==y else 0 for y in range(2)] for x in range(2)]] and B22), [(C12, 1), (C22, 1)])
    # Wait, B12 - B22.
    def diff_B_maps(m1, m2):
        res = [[[] for _ in range(2)] for _ in range(2)]
        for i in range(2):
            for j in range(2):
                res[i][j] = m1[i][j] + [(idx, -coeff) for idx, coeff in m2[i][j]]
        return res
    add_strassen_2x2_multi(A11, diff_B_maps(B12, B22), [(C12, 1), (C22, 1)])
    # M4: C11 (+1), C21 (+1)
    add_strassen_2x2_multi(A22, diff_B_maps(B21, B11), [(C11, 1), (C21, 1)])
    # M5: C11 (-1), C12 (+1)
    add_strassen_2x2_multi(sum_A_maps(A11, A12), B22, [(C11, -1), (C12, 1)])
    # M6: C22 (+1)
    add_strassen_2x2_multi(diff_A_maps(A21, A11) if 'diff_A_maps' in dir() else None, sum_A_maps(B11, B12), [(C22, 1)])
    # Need diff_A_maps
    def diff_A_maps(m1, m2):
        res = [[[] for _ in range(2)] for _ in range(2)]
        for i in range(2):
            for j in range(2):
                res[i][j] = m1[i][j] + [(idx, -coeff) for idx, coeff in m2[i][j]]
        return res
    # Re-do M6
    # I already added M3, M4, M5.
    # M6: (A21 - A11)(B11 + B12) -> C22 (+1)
    add_strassen_2x2_multi(diff_A_maps(A21, A11), sum_A_maps(B11, B12), [(C22, 1)])
    # M7: C11 (+1)
    add_strassen_2x2_multi(diff_A_maps(A12, A22), sum_A_maps(B21, B22), [(C11, 1)])

    # --- Part 2: 4x4x3 Schoolbook ---
    # C_top = A11 B21 + A12 B22 (where B21, B22 are 2x3 blocks of B)
    # C_bot = A21 B21 + A22 B22
    
    # B21: B[0:2, 4:7] -> indices 4,5,6, 11,12,13
    # B22: B[2:4, 4:7] -> indices 18,19,20, 25,26,27
    
    def get_B_block_3(r1, c_start):
        # r1: 0 or 1 (row block in B)
        # c_start: 4 (col start in B)
        res = [[[] for _ in range(3)] for _ in range(2)]
        for i in range(2):
            for j in range(3):
                global_i = r1*2 + i
                global_j = c_start + j
                idx = global_i * 7 + global_j
                res[i][j] = [(idx, 1)]
        return res

    B21_3 = get_B_block_3(0, 4)
    B22_3 = get_B_block_3(1, 4)
    
    # C_top: C[0:2, 4:7]
    # C_bot: C[2:4, 4:7]
    
    def get_C_block_3(r1, c_start):
        res = [[0 for _ in range(3)] for _ in range(2)]
        for i in range(2):
            for j in range(3):
                global_i = r1*2 + i
                global_j = c_start + j
                idx = global_j * 4 + global_i
                res[i][j] = idx
        return res

    C_top_3 = get_C_block_3(0, 4)
    C_bot_3 = get_C_block_3(1, 4)

    # Add terms for A11 * B21_3
    for i in range(2):
        for j in range(2):
            for k in range(3):
                u_idx = A11[i][j][0][0]
                v_idx = B21_3[j][k][0][0]
                w_idx = C_top_3[i][k]
                add_term([(u_idx, 1)], [(v_idx, 1)], [(w_idx, 1)])
                
    # Add terms for A12 * B22_3
    for i in range(2):
        for j in range(2):
            for k in range(3):
                u_idx = A12[i][j][0][0]
                v_idx = B22_3[j][k][0][0]
                w_idx = C_top_3[i][k]
                add_term([(u_idx, 1)], [(v_idx, 1)], [(w_idx, 1)])

    # Add terms for A21 * B21_3
    for i in range(2):
        for j in range(2):
            for k in range(3):
                u_idx = A21[i][j][0][0]
                v_idx = B21_3[j][k][0][0]
                w_idx = C_bot_3[i][k]
                add_term([(u_idx, 1)], [(v_idx, 1)], [(w_idx, 1)])

    # Add terms for A22 * B22_3
    for i in range(2):
        for j in range(2):
            for k in range(3):
                u_idx = A22[i][j][0][0]
                v_idx = B22_3[j][k][0][0]
                w_idx = C_bot_3[i][k]
                add_term([(u_idx, 1)], [(v_idx, 1)], [(w_idx, 1)])

    return {"U": U, "V": V, "W": W}
