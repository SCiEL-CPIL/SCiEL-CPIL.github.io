import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 49
    # Initialize U, V, W as lists of lists of zeros
    # U: (n*m, rank) -> (16, 49)
    # V: (m*p, rank) -> (16, 49)
    # W: (p*n, rank) -> (16, 49)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Indices for 4x4 matrix
    # A: i, j in 0..3
    # B: j, k in 0..3
    # C: k, i in 0..3 (Note: W index is k*n + i)
    
    # Strassen 2x2 coefficients
    # u_s[i], v_s[j], w_s[k] for s in 0..6
    # i, j, k are local indices 0,1 for 2x2 blocks
    # u_s is coeff for A_block[i]
    # v_s is coeff for B_block[j]
    # w_s is coeff for C_block[k]
    
    # Standard Strassen for 2x2:
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B10-B11)
    # M3 = A11(B01-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    # u_s[i] for i in 0,1 (rows of A block)
    # v_s[j] for j in 0,1 (cols of B block)
    # Note: A is 2x2, B is 2x2.
    # A indices: (0,0), (0,1), (1,0), (1,1)
    # B indices: (0,0), (0,1), (1,0), (1,1)
    
    # Let's define the 7 terms for the INNER 2x2 multiplication.
    # Each term s has:
    # u_s: vector of 4 coeffs for A_local
    # v_s: vector of 4 coeffs for B_local
    # w_s: vector of 4 coeffs for C_local
    
    inner_u = []
    inner_v = []
    inner_w = []
    
    # s=0: (A00+A11)(B00+B11)
    # u: A00=1, A11=1
    # v: B00=1, B11=1
    # w: C00=1, C11=1
    inner_u.append([1, 0, 0, 1]) # A00, A01, A10, A11
    inner_v.append([1, 0, 0, 1]) # B00, B01, B10, B11
    inner_w.append([1, 0, 0, 1]) # C00, C01, C10, C11
    
    # s=1: (A10+A11)B00
    # u: A10=1, A11=1
    # v: B00=1
    # w: C10=1, C11=-1
    inner_u.append([0, 0, 1, 1])
    inner_v.append([1, 0, 0, 0])
    inner_w.append([0, 0, 1, -1])
    
    # s=2: A00(B10-B11)
    # u: A00=1
    # v: B10=1, B11=-1
    # w: C01=1, C11=1
    inner_u.append([1, 0, 0, 0])
    inner_v.append([0, 0, 1, -1])
    inner_w.append([0, 1, 0, 1])
    
    # s=3: A11(B01-B00)
    # u: A11=1
    # v: B01=1, B00=-1
    # w: C00=1, C10=1
    inner_u.append([0, 0, 0, 1])
    inner_v.append([-1, 1, 0, 0])
    inner_w.append([1, 0, 1, 0])
    
    # s=4: (A00+A01)B11
    # u: A00=1, A01=1
    # v: B11=1
    # w: C00=-1, C01=1
    inner_u.append([1, 1, 0, 0])
    inner_v.append([0, 0, 0, 1])
    inner_w.append([-1, 1, 0, 0])
    
    # s=5: (A10-A00)(B00+B01)
    # u: A10=1, A00=-1
    # v: B00=1, B01=1
    # w: C11=1
    inner_u.append([-1, 0, 1, 0])
    inner_v.append([1, 1, 0, 0])
    inner_w.append([0, 0, 0, 1])
    
    # s=6: (A01-A11)(B10+B11)
    # u: A01=1, A11=-1
    # v: B10=1, B11=1
    # w: C00=1
    inner_u.append([0, 1, 0, -1])
    inner_v.append([0, 0, 1, 1])
    inner_w.append([1, 0, 0, 0])
    
    # Now define the OUTER Strassen for 4x4 blocks.
    # Blocks are 2x2.
    # A blocks: A00, A01, A10, A11
    # B blocks: B00, B01, B10, B11
    # C blocks: C00, C01, C10, C11
    
    # Outer terms k=0..6
    # u_k: coeffs for A blocks (4 blocks)
    # v_k: coeffs for B blocks (4 blocks)
    # w_k: coeffs for C blocks (4 blocks)
    
    outer_u = []
    outer_v = []
    outer_w = []
    
    # k=0: (A00+A11)(B00+B11)
    outer_u.append([1, 0, 0, 1])
    outer_v.append([1, 0, 0, 1])
    outer_w.append([1, 0, 0, 1]) # C00, C01, C10, C11
    
    # k=1: (A10+A11)B00
    outer_u.append([0, 0, 1, 1])
    outer_v.append([1, 0, 0, 0])
    outer_w.append([0, 0, 1, -1])
    
    # k=2: A00(B10-B11)
    outer_u.append([1, 0, 0, 0])
    outer_v.append([0, 0, 1, -1])
    outer_w.append([0, 1, 0, 1])
    
    # k=3: A11(B01-B00)
    outer_u.append([0, 0, 0, 1])
    outer_v.append([-1, 1, 0, 0])
    outer_w.append([1, 0, 1, 0])
    
    # k=4: (A00+A01)B11
    outer_u.append([1, 1, 0, 0])
    outer_v.append([0, 0, 0, 1])
    outer_w.append([-1, 1, 0, 0])
    
    # k=5: (A10-A00)(B00+B01)
    outer_u.append([-1, 0, 1, 0])
    outer_v.append([1, 1, 0, 0])
    outer_w.append([0, 0, 0, 1])
    
    # k=6: (A01-A11)(B10+B11)
    outer_u.append([0, 1, 0, -1])
    outer_v.append([0, 0, 1, 1])
    outer_w.append([1, 0, 0, 0])
    
    # Now combine.
    # For each outer term k, and each inner term s:
    # The term corresponds to:
    # U_global = sum over blocks I,J of outer_u[k][block_index] * inner_u[s][local_index]
    # V_global = sum over blocks J,K of outer_v[k][block_index] * inner_v[s][local_index]
    # W_global = sum over blocks K,I of outer_w[k][block_index] * inner_w[s][local_index]
    
    # Block index mapping:
    # A blocks: 00, 01, 10, 11 -> indices 0, 1, 2, 3
    # Local indices in block: 00, 01, 10, 11 -> indices 0, 1, 2, 3
    
    # Global index for A: i = I*2 + i_local, j = J*2 + j_local
    # Global index for B: j = J*2 + j_local, k = K*2 + k_local
    # Global index for C: k = K*2 + k_local, i = I*2 + i_local
    
    rank_idx = 0
    for k in range(7):
        for s in range(7):
            # Compute U coefficients for this rank term
            # U is 16x1. Flatten (I, J, i_l, j_l) -> I*4 + J*4 + i_l*2 + j_l?
            # No, U index is i*m + j. i, j are 0..3.
            # i = I*2 + i_l, j = J*2 + j_l.
            
            u_vals = [0] * 16
            v_vals = [0] * 16
            w_vals = [0] * 16
            
            for I in range(2):
                for J in range(2):
                    # Block index for A: I*2 + J
                    a_block_idx = I * 2 + J
                    coeff_a = outer_u[k][a_block_idx]
                    if coeff_a == 0:
                        continue
                    for i_l in range(2):
                        for j_l in range(2):
                            # Local index for A block: i_l*2 + j_l
                            local_idx = i_l * 2 + j_l
                            coeff_u = inner_u[s][local_idx]
                            if coeff_u == 0:
                                continue
                            # Global i, j
                            i_g = I * 2 + i_l
                            j_g = J * 2 + j_l
                            u_idx = i_g * 4 + j_g
                            u_vals[u_idx] += coeff_a * coeff_u
                            
            for J in range(2):
                for K in range(2):
                    # Block index for B: J*2 + K
                    b_block_idx = J * 2 + K
                    coeff_b = outer_v[k][b_block_idx]
                    if coeff_b == 0:
                        continue
                    for j_l in range(2):
                        for k_l in range(2):
                            # Local index for B block: j_l*2 + k_l
                            local_idx = j_l * 2 + k_l
                            coeff_v = inner_v[s][local_idx]
                            if coeff_v == 0:
                                continue
                            # Global j, k
                            j_g = J * 2 + j_l
                            k_g = K * 2 + k_l
                            v_idx = j_g * 4 + k_g
                            v_vals[v_idx] += coeff_b * coeff_v
                            
            for K in range(2):
                for I in range(2):
                    # Block index for C: K*2 + I
                    c_block_idx = K * 2 + I
                    coeff_c = outer_w[k][c_block_idx]
                    if coeff_c == 0:
                        continue
                    for k_l in range(2):
                        for i_l in range(2):
                            # Local index for C block: k_l*2 + i_l
                            local_idx = k_l * 2 + i_l
                            coeff_w = inner_w[s][local_idx]
                            if coeff_w == 0:
                                continue
                            # Global k, i
                            k_g = K * 2 + k_l
                            i_g = I * 2 + i_l
                            w_idx = k_g * 4 + i_g
                            w_vals[w_idx] += coeff_c * coeff_w
                            
            # Assign to U, V, W
            for idx in range(16):
                U[idx][rank_idx] = u_vals[idx]
                V[idx][rank_idx] = v_vals[idx]
                W[idx][rank_idx] = w_vals[idx]
                
            rank_idx += 1
            
    return {"U": U, "V": V, "W": W}
