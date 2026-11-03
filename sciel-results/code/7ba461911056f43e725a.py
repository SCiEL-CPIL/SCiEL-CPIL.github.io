<<<<<<< SEARCH
# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Use standard schoolbook decomposition for guaranteed correctness
    # Rank = n * m * p
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Index in U: i*m + j
                u_idx = i * m + j
                # Index in V: j*p + k
                v_idx = j * p + k
                # Index in W: k*n + i
                w_idx = k * n + i
                
                U[u_idx][r_idx] = 1
                V[v_idx][r_idx] = 1
                W[w_idx][r_idx] = 1
                
                r_idx += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
=======
# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=7
    # Block-structured decomposition:
    # Split A (2x4) into [A_left | A_right], each 2x2
    # Split B (4x7) into [B_top; B_bottom], each 2x7
    # C = A_left * B_top + A_right * B_bottom
    #
    # For each 2x2 * 2x7 product, split the 7 columns into 2+2+3:
    #   2x2 * 2x2 uses Strassen (7 mults)
    #   2x2 * 2x2 uses Strassen (7 mults)  
    #   2x2 * 2x3 uses schoolbook (6 mults)
    # Total per 2x2*2x7: 7+7+6 = 20
    # Two such blocks: 40 total rank
    #
    # This gives rank 40 < 56 schoolbook.
    
    U_list = []  # list of (u_idx, coeff) per rank term
    V_list = []  # list of (v_idx, coeff) per rank term
    W_list = []  # list of (w_idx, coeff) per rank term
    
    def add_term(u_coeff, v_coeff, w_coeff):
        """Add a rank-1 term. u_coeff: dict {u_idx: val}, v_coeff: dict {v_idx: val}, w_coeff: dict {w_idx: val}"""
        U_list.append(u_coeff)
        V_list.append(v_coeff)
        W_list.append(w_coeff)
    
    # Strassen for 2x2 * 2x2:
    # P = [[P11, P12], [P21, P22]], Q = [[Q11, Q12], [Q21, Q22]]
    # 7 multiplications M1..M7, each a product of a linear combo of P entries and a linear combo of Q entries
    # Result R = P*Q where:
    #   R11 = M1 + M4 - M5 + M7
    #   R12 = M3 + M5
    #   R21 = M2 + M4
    #   R22 = M1 - M2 + M3 + M6
    #
    # u_idx for A_left: A_left[0][0]=0, A_left[0][1]=1, A_left[1][0]=2, A_left[1][1]=3
    # u_idx for A_right: A_right[0][0]=4, A_right[0][1]=5, A_right[1][0]=6, A_right[1][1]=7
    #
    # v_idx for B_top: B_top[0][0]=0, B_top[0][1]=1, B_top[0][2]=2, B_top[0][3]=3, B_top[0][4]=4, B_top[0][5]=5, B_top[0][6]=6
    #                   B_top[1][0]=7, B_top[1][1]=8, B_top[1][2]=9, B_top[1][3]=10, B_top[1][4]=11, B_top[1][5]=12, B_top[1][6]=13
    # v_idx for B_bottom: B_bottom[0][0]=14, ..., B_bottom[0][6]=20
    #                     B_bottom[1][0]=21, ..., B_bottom[1][6]=27
    #
    # w_idx: W[k*n+i, r] where k=0..6, i=0..1, so w_idx = k*2 + i
    #   w_idx 0: (k=0,i=0), 1: (k=0,i=1), 2: (k=1,i=0), 3: (k=1,i=1), ..., 12: (k=6,i=0), 13: (k=6,i=1)
    
    # Helper: u_idx for A block (left or right), row i (0,1), col j (0,1)
    def a_u_idx(block, i, j):
        # block: 0 for left, 1 for right
        return block * 4 + i * 2 + j
    
    # Helper: v_idx for B block (top or bottom), row i (0,1), col k (0..6)
    def b_v_idx(block, i, k):
        # block: 0 for top, 1 for bottom
        return block * 14 + i * 7 + k
    
    # Helper: w_idx for output, col k (0..6), row i (0,1)
    def c_w_idx(k, i):
        return k * 2 + i
    
    # Strassen decomposition for one 2x2 * 2x2 sub-product
    # P is at A block `a_block`, rows 0,1 cols 0,1
    # Q is at B block `b_block`, rows 0,1 cols `k0` and `k0+1`
    def strassen_2x2_2x2(a_block, b_block, k0):
        # P11 = A[a_block][0][0], P12 = A[a_block][0][1], P21 = A[a_block][1][0], P22 = A[a_block][1][1]
        # Q11 = B[b_block][0][k0], Q12 = B[b_block][0][k0+1], Q21 = B[b_block][1][k0], Q22 = B[b_block][1][k0+1]
        
        # M1 = (P11 + P22) * (Q11 + Q22)
        add_term(
            {a_u_idx(a_block, 0, 0): 1, a_u_idx(a_block, 1, 1): 1},
            {b_v_idx(b_block, 0, k0): 1, b_v_idx(b_block, 1, k0+1): 1},
            {c_w_idx(k0, 0): 1, c_w_idx(k0+1, 1): 1}  # contributes to R11 and R22
        )
        # Wait, this is wrong. Each Mj is a scalar, and it contributes to specific output entries with specific coefficients.
        # I need to track which output entries each Mj contributes to.
        # Let me redo this more carefully.
        pass
    
    # Let me restart with a cleaner approach. I'll build the full tensor decomposition directly.
    
    # Actually, let me just use the known Strassen decomposition and track contributions properly.
    
    # For a 2x2 * 2x2 multiplication, the 7 rank-1 terms and their contributions to the 4 output entries:
    # Let P = [[a, b], [c, d]], Q = [[e, f], [g, h]]
    # M1 = (a+d)(e+h) -> R11 += 1, R22 += 1
    # M2 = (c+d)e    -> R21 += 1
    # M3 = a(g-h)    -> R12 += 1, R22 += 1
    # M4 = d(f-e)    -> R11 += 1, R21 += 1
    # M5 = (a+b)h    -> R11 += -1, R12 += 1
    # M6 = (c-a)(e+f) -> R22 += 1
    # M7 = (b-d)(g+h) -> R11 += 1, R22 += -1
    
    # Verify: R11 = M1 + M4 - M5 + M7 = (a+d)(e+h) + d(f-e) - (a+b)h + (b-d)(g+h)
    #   = ae+ah+de+dh + df-de - ah-bh + bg+bh-dg-dh
    #   = ae + df + bg + dh - bg - dg + bg... let me recompute
    #   = ae+ah+de+dh+df-de-ah-bh+bg+bh-dg-dh
    #   = ae+df+bg-dg+dh-dh+de-de = ae+df+bg-dg... hmm this doesn't look right.
    #   
    # Let me use the standard Strassen formulas:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(e)
    # M3 = (a)(g-h)
    # M4 = (d)(f-e)
    # M5 = (a+b)(h)
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    #
    # R11 = M1 + M4 - M5 + M7
    # R12 = M3 + M5
    # R21 = M2 + M4
    # R22 = M1 - M2 + M3 + M6
    
    # Let me verify R11:
    # M1 = ae+ah+de+dh
    # M4 = df-de
    # M5 = ah+bh
    # M7 = bg+bh-dg-dh
    # R11 = (ae+ah+de+dh) + (df-de) - (ah+bh) + (bg+bh-dg-dh)
    #     = ae+ah+de+dh+df-de-ah-bh+bg+bh-dg-dh
    #     = ae+df+bg-dg+dh-dh+de-de = ae+df+bg-dg
    # That's not ae+dh (which is R11 for matrix mult). Something is wrong.
    #
    # Let me look up the correct Strassen formulas.
    # Standard Strassen:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(g-h)
    # M4 = d(f-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    #
    # p11 = M1 + M4 - M5 + M7
    # p12 = M3 + M5
    # p21 = M2 + M4
    # p22 = M1 - M2 + M3 + M6
    #
    # Let me re-verify p11 more carefully:
    # M1 = (a+d)(e+h) = ae+ah+de+dh
    # M4 = d(f-e) = df-de
    # M5 = (a+b)h = ah+bh
    # M7 = (b-d)(g+h) = bg+bh-dg-dh
    #
    # p11 = M1 + M4 - M5 + M7
    #     = (ae+ah+de+dh) + (df-de) - (ah+bh) + (bg+bh-dg-dh)
    #     = ae+ah+de+dh+df-de-ah-bh+bg+bh-dg-dh
    #     = ae + (ah-ah) + (de-de) + (dh-dh) + df + (-bh+bh) + bg - dg
    #     = ae + df + bg - dg
    #
    # But the correct p11 = ae + dh. So there's an error in my Strassen formulas.
    #
    # Let me use a verified set. From Wikipedia:
    # m1 = (a + d)(e + h)
    # m2 = (c + d)(e)
    # m3 = (a)(g − h)
    # m4 = (d)(f − e)
    # m5 = (a + b)(h)
    # m6 = (c − a)(e + f)
    # m7 = (b − d)(g + h)
    #
    # p11 = m1 + m4 − m5 + m7
    # p12 = m3 + m5
    # p21 = m2 + m4
    # p22 = m1 − m2 + m3 + m6
    #
    # Hmm, let me try different formulas. Maybe I have the indices wrong.
    #
    # Actually, I think the issue is that I should just verify numerically. But for now,
    # let me use a simpler approach: just use schoolbook but with a different block structure
    # that might allow some sharing.
    #
    # Actually, for the purpose of this task, let me just implement a correct decomposition.
    # The simplest improvement is to note that we can't easily share between the two 2x2*2x7 blocks.
    #
    # Let me just implement Strassen correctly by testing it numerically first.
    
    import numpy as np
    
    # Test Strassen
    def strassen_test():
        a, b, c, d = 1, 2, 3, 4
        e, f, g, h = 5, 6, 7, 8
        P = np.array([[a, b], [c, d]])
        Q = np.array([[e, f], [g, h]])
        expected = P @ Q
        
        m1 = (a + d) * (e + h)
        m2 = (c + d) * e
        m3 = a * (g - h)
        m4 = d * (f - e)
        m5 = (a + b) * h
        m6 = (c - a) * (e + f)
        m7 = (b - d) * (g + h)
        
        p11 = m1 + m4 - m5 + m7
        p12 = m3 + m5
        p21 = m2 + m4
        p22 = m1 - m2 + m3 + m6
        
        result = np.array([[p11, p12], [p21, p22]])
        return np.allclose(result, expected)
    
    # For now, let me just use schoolbook with rank 56 to ensure correctness,
    # and then optimize if I have time.
    
    # Actually, let me just implement the schoolbook correctly and return it.
    # The improvement will come from finding a better decomposition.
    
    # Let me try a different approach: use the fact that we can decompose the tensor
    # in a way that exploits the rectangular structure.
    
    # For 2x4x7, one approach is to split the middle dimension 4 into 2+2,
    # giving two 2x2x7 sub-tensors. Each 2x2x7 has rank at most 14 (schoolbook).
    # Total rank: 28. But we can do better on each 2x2x7.
    
    # For 2x2x7, split the 7 into 2+2+2+1:
    # 2x2x2 has rank 7 (Strassen), 2x2x2 has rank 7, 2x2x2 has rank 7, 2x2x1 has rank 2.
    # Total per 2x2x7: 7+7+7+2 = 23
    # Two such: 46. Worse than 56? No, 46 < 56.
    
    # Or split 7 into 3+4:
    # 2x2x3: schoolbook rank 6
    # 2x2x4: split into 2+2, each 2x2x2 rank 7, total 14
    # Total per 2x2x7: 6+14 = 20
    # Two such: 40
    
    # Let me implement the 2x2x2 Strassen decomposition correctly.
    
    # I'll build the full decomposition by explicitly listing all rank-1 terms.
    
    # Clear the lists
    U_list.clear()
    V_list.clear()
    W_list.clear()
    
    # For each A block (left=0, right=1) and each B block (top=0, bottom=1):
    for a_block in range(2):
        for b_block in range(2):
            # We're computing A_block (2x2) * B_block (2x7)
            # Split B_block's 7 columns into 3+4:
            #   First 3 columns: schoolbook 2x2x3 (6 mults)
            #   Next 4 columns: split into 2+2, each Strassen 2x2x2 (7 mults each)
            
            # Schoolbook for 2x2 * 2x3 (columns k0, k0+1, k0+2)
            k0 = 0
            for i in range(2):
                for j in range(2):
                    for k in range(k0, k0+3):
                        u_idx = a_u_idx(a_block, i, j)
                        v_idx = b_v_idx(b_block, j, k)
                        w_idx = c_w_idx(k, i)
                        add_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
            
            # Strassen for 2x2 * 2x2 on columns k0+3, k0+4
            k1 = k0 + 3  # = 3
            # P = A_block, Q = B_block[:, k1:k1+2]
            # a=P11, b=P12, c=P21, d=P22
            # e=Q11, f=Q12, g=Q21, h=Q22
            # where Q11=B_block[0][k1], Q12=B_block[0][k1+1], Q21=B_block[1][k1], Q22=B_block[1][k1+1]
            
            a_u = a_u_idx(a_block, 0, 0)  # P11
            b_u = a_u_idx(a_block, 0, 1)  # P12
            c_u = a_u_idx(a_block, 1, 0)  # P21
            d_u = a_u_idx(a_block, 1, 1)  # P22
            
            e_v = b_v_idx(b_block, 0, k1)      # Q11
            f_v = b_v_idx(b_block, 0, k1 + 1)  # Q12
            g_v = b_v_idx(b_block, 1, k1)      # Q21
            h_v = b_v_idx(b_block, 1, k1 + 1)  # Q22
            
            # M1 = (P11 + P22) * (Q11 + Q22)
            # Contributes: R11 += 1, R22 += 1
            add_term(
                {a_u: 1, d_u: 1},
                {e_v: 1, h_v: 1},
                {c_w_idx(k1, 0): 1, c_w_idx(k1+1, 1): 1}
            )
            
            # M2 = (P21 + P22) * Q11
            # Contributes: R21 += 1
            add_term(
                {c_u: 1, d_u: 1},
                {e_v: 1},
                {c_w_idx(k1, 1): 1}
            )
            
            # M3 = P11 * (Q21 - Q22)
            # Contributes: R12 += 1, R22 += 1
            add_term(
                {a_u: 1},
                {g_v: 1, h_v: -1},
                {c_w_idx(k1+1, 0): 1, c_w_idx(k1+1, 1): 1}
            )
            
            # M4 = P22 * (Q12 - Q11)
            # Contributes: R11 += 1, R21 += 1
            add_term(
                {d_u: 1},
                {f_v: 1, e_v: -1},
                {c_w_idx(k1, 0): 1, c_w_idx(k1, 1): 1}
            )
            
            # M5 = (P11 + P12) * Q22
            # Contributes: R11 += -1, R12 += 1
            add_term(
                {a_u: 1, b_u: 1},
                {h_v: 1},
                {c_w_idx(k1, 0): -1, c_w_idx(k1+1, 0): 1}
            )
            
            # M6 = (P21 - P11) * (Q11 + Q12)
            # Contributes: R22 += 1
            add_term(
                {c_u: 1, a_u: -1},
                {e_v: 1, f_v: 1},
                {c_w_idx(k1+1, 1): 1}
            )
            
            # M7 = (P12 - P22) * (Q21 + Q22)
            # Contributes: R11 += 1, R22 += -1
            add_term(
                {b_u: 1, d_u: -1},
                {g_v: 1, h_v: 1},
                {c_w_idx(k1, 0): 1, c_w_idx(k1+1, 1): -1}
            )
            
            # Strassen for 2x2 * 2x2 on columns k1+2, k1+3
            k2 = k1 + 2  # = 5
            e_v2 = b_v_idx(b_block, 0, k2)      # Q11
            f_v2 = b_v_idx(b_block, 0, k2 + 1)  # Q12
            g_v2 = b_v_idx(b_block, 1, k2)      # Q21
            h_v2 = b_v_idx(b_block, 1, k2 + 1)  # Q22
            
            # M1
            add_term(
                {a_u: 1, d_u: 1},
                {e_v2: 1, h_v2: 1},
                {c_w_idx(k2, 0): 1, c_w_idx(k2+1, 1): 1}
            )
            
            # M2
            add_term(
                {c_u: 1, d_u: 1},
                {e_v2: 1},
                {c_w_idx(k2, 1): 1}
            )
            
            # M3
            add_term(
                {a_u: 1},
                {g_v2: 1, h_v2: -1},
                {c_w_idx(k2+1, 0): 1, c_w_idx(k2+1, 1): 1}
            )
            
            # M4
            add_term(
                {d_u: 1},
                {f_v2: 1, e_v2: -1},
                {c_w_idx(k2, 0): 1, c_w_idx(k2, 1): 1}
            )
            
            # M5
            add_term(
                {a_u: 1, b_u: 1},
                {h_v2: 1},
                {c_w_idx(k2, 0): -1, c_w_idx(k2+1, 0): 1}
            )
            
            # M6
            add_term(
                {c_u: 1, a_u: -1},
                {e_v2: 1, f_v2: 1},
                {c_w_idx(k2+1, 1): 1}
            )
            
            # M7
            add_term(
                {b_u: 1, d_u: -1},
                {g_v2: 1, h_v2: 1},
                {c_w_idx(k2, 0): 1, c_w_idx(k2+1, 1): -1}
            )
    
    # Convert lists to matrices
    rank = len(U_list)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        for u_idx, val in U_list[r].items():
            U[u_idx][r] = val
        for v_idx, val in V_list[r].items():
            V[v_idx][r] = val
        for w_idx, val in W_list[r].items():
            W[w_idx][r] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
>>>>>>> REPLACE

Wait, I need to verify that my Strassen decomposition is actually correct. Let me add a verification step and fix any issues. Let me also make sure the index mappings are correct.

Let me reconsider the approach and add verification:

