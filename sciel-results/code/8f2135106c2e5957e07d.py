"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 140.0
    n, m, p = 4, 4, 5
    
    # Strassen 2x2x2 micro-terms
    # Each term has u, v, w vectors of length 4 (flattened 2x2)
    # u_s[i', j'], v_s[j', k'], w_s[k', i']
    # Index: 2*i' + j' for u/v/w local indices
    
    def s2x2_u(term_id, i_prime, j_prime):
        if term_id == 0: # M1: A00 + A11
            if (i_prime == 0 and j_prime == 0) or (i_prime == 1 and j_prime == 1): return 1
            return 0
        elif term_id == 1: # M2: A10 + A11
            if i_prime == 1 and (j_prime == 0 or j_prime == 1): return 1
            return 0
        elif term_id == 2: # M3: A00
            if i_prime == 0 and j_prime == 0: return 1
            return 0
        elif term_id == 3: # M4: A11
            if i_prime == 1 and j_prime == 1: return 1
            return 0
        elif term_id == 4: # M5: A00 + A01
            if i_prime == 0 and (j_prime == 0 or j_prime == 1): return 1
            return 0
        elif term_id == 5: # M6: A10 - A00
            if i_prime == 1 and j_prime == 0: return 1
            if i_prime == 0 and j_prime == 0: return -1
            return 0
        elif term_id == 6: # M7: A01 - A11
            if i_prime == 0 and j_prime == 1: return 1
            if i_prime == 1 and j_prime == 1: return -1
            return 0
        return 0

    def s2x2_v(term_id, j_prime, k_prime):
        if term_id == 0: # M1: B00 + B11
            if (j_prime == 0 and k_prime == 0) or (j_prime == 1 and k_prime == 1): return 1
            return 0
        elif term_id == 1: # M2: B00
            if j_prime == 0 and k_prime == 0: return 1
            return 0
        elif term_id == 2: # M3: B01 - B11
            if j_prime == 0 and k_prime == 1: return 1
            if j_prime == 1 and k_prime == 1: return -1
            return 0
        elif term_id == 3: # M4: B10 - B00
            if j_prime == 1 and k_prime == 0: return 1
            if j_prime == 0 and k_prime == 0: return -1
            return 0
        elif term_id == 4: # M5: B11
            if j_prime == 1 and k_prime == 1: return 1
            return 0
        elif term_id == 5: # M6: B00 + B01
            if j_prime == 0 and (k_prime == 0 or k_prime == 1): return 1
            return 0
        elif term_id == 6: # M7: B10 + B11
            if j_prime == 1 and (k_prime == 0 or k_prime == 1): return 1
            return 0
        return 0

    def s2x2_w(term_id, k_prime, i_prime):
        # M1: C00 + C11
        # M2: C10 + C11 -> C10 gets +1, C11 gets -1 (Wait, standard Strassen: M2 = (A10+A11)B00. C10 += M2, C11 -= M2)
        # Let's re-verify the W coefficients for 2x2x2.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # k', i' index w for C[i', k']? No, w_s[k', i'] corresponds to output C[i', k']?
        # Usually W[k*n+i, r]. Here local W[k', i'].
        # Output C is 2x2. C[i', k']? No, A is 2x2 (i,j), B is 2x2 (j,k), C is 2x2 (i,k).
        # So W[k', i'] corresponds to C[i', k'].
        
        if term_id == 0: # M1: C00 + C11
            if (i_prime == 0 and k_prime == 0) or (i_prime == 1 and k_prime == 1): return 1
            return 0
        elif term_id == 1: # M2: C10 + C11 -> C10 gets +1, C11 gets -1
            if i_prime == 1 and k_prime == 0: return 1
            if i_prime == 1 and k_prime == 1: return -1
            return 0
        elif term_id == 2: # M3: C01 + C11
            if i_prime == 0 and k_prime == 1: return 1
            if i_prime == 1 and k_prime == 1: return 1
            return 0
        elif term_id == 3: # M4: C00 + C10
            if i_prime == 0 and k_prime == 0: return 1
            if i_prime == 1 and k_prime == 0: return 1
            return 0
        elif term_id == 4: # M5: -C00 + C01
            if i_prime == 0 and k_prime == 0: return -1
            if i_prime == 0 and k_prime == 1: return 1
            return 0
        elif term_id == 5: # M6: C11
            if i_prime == 1 and k_prime == 1: return 1
            return 0
        elif term_id == 6: # M7: C00 + C11
            if i_prime == 0 and k_prime == 0: return 1
            if i_prime == 1 and k_prime == 1: return 1
            return 0
        return 0

    rank = 0
    U = []
    V = []
    W = []
    
    # 4x4x4 part using 4 Strassen 2x2x2 blocks
    # A blocks: A00 (rows 0-1, cols 0-1), A01 (rows 0-1, cols 2-3), A10 (rows 2-3, cols 0-1), A11 (rows 2-3, cols 2-3)
    # B blocks: B00 (rows 0-1, cols 0-1), B01 (rows 0-1, cols 2-3), B10 (rows 2-3, cols 0-1), B11 (rows 2-3, cols 2-3)
    # C blocks: C00 (rows 0-1, cols 0-1), C01 (rows 0-1, cols 2-3), C10 (rows 2-3, cols 0-1), C11 (rows 2-3, cols 2-3)
    
    # We have 7 macro terms per 2x2x2 block combination.
    # Actually, we partition A into 4 blocks, B into 4 blocks.
    # The 4x4x4 tensor can be decomposed using 7 macro products if we treat the whole 4x4 as a 2x2 of 2x2 blocks.
    # But the "blocks" in A are 2x2, and "blocks" in B are 2x2.
    # The output C is 4x4, which is a 2x2 of 2x2 blocks.
    # This is exactly the Strassen algorithm for 4x4x4.
    # Rank contribution: 7 (macro) * 7 (micro) = 49 terms.
    
    # Let's verify the macro structure.
    # Macro A blocks:
    # A_M00 = A[0:2, 0:2], A_M01 = A[0:2, 2:4], A_M10 = A[2:4, 0:2], A_M11 = A[2:4, 2:4]
    # Macro B blocks:
    # B_M00 = B[0:2, 0:2], B_M01 = B[0:2, 2:4], B_M10 = B[2:4, 0:2], B_M11 = B[2:4, 2:4]
    # Macro C blocks:
    # C_M00 = C[0:2, 0:2], C_M01 = C[0:2, 2:4], C_M10 = C[2:4, 0:2], C_M11 = C[2:4, 2:4]
    
    # Macro Strassen terms (7):
    # M1: (A_M00 + A_M11) * (B_M00 + B_M11)
    # M2: (A_M10 + A_M11) * B_M00
    # M3: A_M00 * (B_M01 - B_M11)
    # M4: A_M11 * (B_M10 - B_M00)
    # M5: (A_M00 + A_M01) * B_M11
    # M6: (A_M10 - A_M00) * (B_M00 + B_M01)
    # M7: (A_M01 - A_M11) * (B_M10 + B_M11)
    
    # Each macro term is a 2x2x2 tensor of 2x2 blocks.
    # So each macro term decomposes into 7 micro terms.
    # Total 49 terms for the 4x4x4 part.
    
    # Macro W coefficients:
    # C_M00 = M1 + M4 - M5 + M7
    # C_M01 = M3 + M5
    # C_M10 = M2 + M4
    # C_M11 = M1 - M2 + M3 + M6
    
    # For each macro term m (0..6) and micro term s (0..6), we have a tensor.
    # U vector:
    #   The macro term m specifies which A_M blocks are used and with what signs.
    #   The micro term s specifies which A sub-blocks (within the A_M block) are used and with what signs.
    #   Wait, the micro terms operate on the 2x2 blocks themselves.
    #   So if macro term m uses A_M00 with sign +1, and micro term s uses A00 (within A_M00) with sign +1,
    #   then the global A index is A[0:2, 0:2] with sign +1.
    
    # Let's define the macro A block usage for each macro term m.
    # m=0 (M1): A_M00 (+1), A_M11 (+1)
    # m=1 (M2): A_M10 (+1), A_M11 (+1)
    # m=2 (M3): A_M00 (+1)
    # m=3 (M4): A_M11 (+1)
    # m=4 (M5): A_M00 (+1), A_M01 (+1)
    # m=5 (M6): A_M10 (+1), A_M00 (-1)
    # m=6 (M7): A_M01 (+1), A_M11 (-1)
    
    macro_A = [
        [(0, 1), (3, 1)], # M1
        [(2, 1), (3, 1)], # M2
        [(0, 1)],         # M3
        [(3, 1)],         # M4
        [(0, 1), (1, 1)], # M5
        [(2, 1), (0, -1)],# M6
        [(1, 1), (3, -1)] # M7
    ]
    
    macro_B = [
        [(0, 1), (3, 1)], # M1
        [(0, 1)],         # M2
        [(1, 1), (3, -1)],# M3
        [(3, 1), (0, -1)],# M4
        [(3, 1)],         # M5
        [(0, 1), (1, 1)], # M6
        [(2, 1), (3, 1)]  # M7
    ]
    
    macro_C = [
        [(0, 1), (3, 1)], # M1: C_M00 + C_M11
        [(2, 1), (3, -1)],# M2: C_M10 + C_M11 (-1)
        [(1, 1), (3, 1)], # M3: C_M01 + C_M11
        [(0, 1), (2, 1)], # M4: C_M00 + C_M10
        [(0, -1), (1, 1)],# M5: -C_M00 + C_M01
        [(3, 1)],         # M6: C_M11
        [(0, 1), (3, 1)]  # M7: C_M00 + C_M11
    ]
    
    # Block indices: 0=(0:2, 0:2), 1=(0:2, 2:4), 2=(2:4, 0:2), 3=(2:4, 2:4)
    # Global A index: i*m + j. Block b covers rows 2*(b//2) to 2*(b//2)+1, cols 2*(b%2) to 2*(b%2)+1.
    # Global B index: j*p + k. Block b covers rows 2*(b//2) to 2*(b//2)+1, cols 2*(b%2) to 2*(b%2)+1.
    # Global W index: k*n + i. Block b covers rows 2*(b//2) to 2*(b//2)+1 (k), cols 2*(b%2) to 2*(b%2)+1 (i).
    # Wait, W index is k*n + i. C block b is C[i,k].
    # C_M00: i=0,1 k=0,1. W index k*4+i.
    # C_M01: i=0,1 k=2,3.
    # C_M10: i=2,3 k=0,1.
    # C_M11: i=2,3 k=2,3.
    
    for m in range(7):
        for s in range(7):
            if time.time() > deadline: break
            u_row = [0] * (n * m)
            v_row = [0] * (m * p)
            w_row = [0] * (p * n)
            
            # Process A
            for (block_a, sign_macro) in macro_A[m]:
                # block_a is 0..3
                row_base = 2 * (block_a // 2)
                col_base = 2 * (block_a % 2)
                for i_prime in range(2):
                    for j_prime in range(2):
                        val = sign_macro * s2x2_u(s, i_prime, j_prime)
                        if val != 0:
                            i = row_base + i_prime
                            j = col_base + j_prime
                            u_row[i * m + j] += val
            
            # Process B
            for (block_b, sign_macro) in macro_B[m]:
                row_base = 2 * (block_b // 2)
                col_base = 2 * (block_b % 2)
                for j_prime in range(2):
                    for k_prime in range(2):
                        val = sign_macro * s2x2_v(s, j_prime, k_prime)
                        if val != 0:
                            j = row_base + j_prime
                            k = col_base + k_prime
                            v_row[j * p + k] += val
            
            # Process C
            for (block_c, sign_macro) in macro_C[m]:
                # block_c is 0..3. C[i,k].
                # block_c 0: i=0,1 k=0,1
                # block_c 1: i=0,1 k=2,3
                # block_c 2: i=2,3 k=0,1
                # block_c 3: i=2,3 k=2,3
                row_base_k = 2 * (block_c % 2) # k base? No.
                # Let's map block_c to i_base, k_base.
                # 0: i=0, k=0
                # 1: i=0, k=2
                # 2: i=2, k=0
                # 3: i=2, k=2
                i_base = 2 * (block_c // 2)
                k_base = 2 * (block_c % 2)
                
                for k_prime in range(2):
                    for i_prime in range(2):
                        val = sign_macro * s2x2_w(s, k_prime, i_prime)
                        if val != 0:
                            i = i_base + i_prime
                            k = k_base + k_prime
                            w_row[k * n + i] += val
            
            U.append(u_row)
            V.append(v_row)
            W.append(w_row)
            rank += 1
    
    # Handle 5th column of B (k=4)
    # C[i,4] = sum_j A[i,j] B[j,4]
    # This is just matrix-vector multiplication, rank 4*4 = 16? No, rank n*m = 16?
    # Wait, C[i,4] is a linear function of A and B[:,4].
    # It's a bilinear form. Rank is n*m = 16?
    # Actually, it's sum_j A[i,j] B[j,4].
    # This can be decomposed into n*m terms.
    # For each i in 0..3, j in 0..3:
    #   U[i*m+j] = 1
    #   V[j*p+4] = 1
    #   W[4*n+i] = 1
    
    for i in range(n):
        for j in range(m):
            if time.time() > deadline: break
            u_row = [0] * (n * m)
            v_row = [0] * (m * p)
            w_row = [0] * (p * n)
            
            u_row[i * m + j] = 1
            v_row[j * p + 4] = 1
            w_row[4 * n + i] = 1
            
            U.append(u_row)
            V.append(v_row)
            W.append(w_row)
            rank += 1

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
