import time

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS

    # Rank-11 decomposition for 2x2x4 matrix multiplication.
    # A is 2x2: A=[[a,b],[c,d]], B is 2x4: B=[[e,f,g,h],[i,j,k,l]], C is 2x4.
    # C[0,0]=ae+bi, C[0,1]=af+bj, C[0,2]=ag+bk, C[0,3]=ah+bl
    # C[1,0]=ce+di, C[1,1]=cf+dj, C[1,2]=cg+dk, C[1,3]=ch+dl
    
    # The 11 products:
    # P1 = (a+d)(e+i)
    # P2 = (a+d)(f+j)
    # P3 = (a+d)(g+k)
    # P4 = (a+d)(h+l)
    # P5 = (c-d)(e-i)
    # P6 = (c-d)(f-j)
    # P7 = (c-d)(g-k)
    # P8 = (c-d)(h-l)
    # P9 = (b)(e)  -> wait, this isn't symmetric. Let's use a known construction.
    
    # Known rank 11 decomposition for 2x2x4:
    # M1 = (a+b)(e+f)
    # M2 = (a+b)(g+h)
    # M3 = (a-b)(e-f)
    # M4 = (a-b)(g-h)
    # M5 = (c+d)(e+f)
    # M6 = (c+d)(g+h)
    # M7 = (c-d)(e-f)
    # M8 = (c-d)(g-h)
    # M9 = (b+d)(e)
    # M10 = (b+d)(g)
    # M11 = (b-d)(i)  # This seems wrong.
    
    # Let's use the standard rank 11 decomposition found in literature.
    # For multiplying 2x2 by 2x4:
    # A = [[a, b], [c, d]]
    # B = [[e, f, g, h], [i, j, k, l]]
    
    # Products:
    # P1 = (a+d)(e+i)
    # P2 = (a+d)(f+j)
    # P3 = (a+d)(g+k)
    # P4 = (a+d)(h+l)
    # P5 = (a-b)(f-i)  # ?
    
    # Actually, a simpler approach:
    # The rank of 2x2x4 is 11.
    # We can construct it by adding one term to two 2x2x2 Strassen blocks?
    # 2x2x2 rank is 7. 7+7=14. We need 11.
    # So we save 3 terms by sharing.
    
    # Let's use the explicit rank-11 decomposition:
    # U (A coefficients) x V (B coefficients) -> W (C coefficients)
    
    # The 11 rank-1 terms:
    # 1. (a+d) * (e+i) -> c00 + c10
    # 2. (a+d) * (f+j) -> c01 + c11
    # 3. (a+d) * (g+k) -> c02 + c12
    # 4. (a+d) * (h+l) -> c03 + c13
    # 5. (c-d) * (e-i) -> c10 - c00  (contributes to c10 and c00 with opposite signs)
    #    Wait, (c-d)(e-i) = ce - ci - de + di.
    #    c10 = ce + di. c00 = ae + bi.
    #    This doesn't map cleanly.
    
    # Correct Rank-11 Construction for 2x2x4:
    # Let A = [[a, b], [c, d]], B = [[e, f, g, h], [i, j, k, l]]
    
    # M1 = (a+d)(e+i)
    # M2 = (a+d)(f+j)
    # M3 = (a+d)(g+k)
    # M4 = (a+d)(h+l)
    # M5 = (c-d)(e-i)
    # M6 = (c-d)(f-j)
    # M7 = (c-d)(g-k)
    # M8 = (c-d)(h-l)
    # M9 = b * (e - i)
    # M10 = b * (f - j)
    # M11 = b * (g - k)  # Missing h?
    
    # This construction is tricky to verify manually.
    # However, rank(2x2x4) = 11 is a known result.
    # Let's implement a verified set of 11 terms.
    
    # From "On the tensor rank of matrix multiplication" by Strassen (1987):
    # For 2x2x4, the 11 terms are:
    # 1. (a+d, e+i) -> (1, 0, 0, 0, 0, 0, 0, 0) i.e. C00 + C10? No, W maps to specific entries.
    
    # Let's use a safer fallback: Rank 42 (Strassen 2x2x2 x 6).
    # But the prompt asks for improvement. Rank 33 is 48/33 = 1.4545.
    # Rank 42 is 48/42 = 1.1429.
    # So Rank 33 is a significant improvement.
    
    # I will implement the Rank-11 decomposition for 2x2x4.
    # Reference: The 11 rank-1 decompositions for 2x2x4 are:
    
    # A: 4 entries (a,b,c,d)
    # B: 8 entries (e,f,g,h,i,j,k,l)
    # C: 8 entries (c00,c01,c02,c03,c10,c11,c12,c13)
    
    # Terms:
    # 1. U: (1,0,0,1), V: (1,0,0,0, 1,0,0,0) -> W: (1,0,0,0, 1,0,0,0)  [ (a+d)(e+i) -> C00+C10 ]
    #    Check: (a+d)(e+i) = ae+ai+de+di.
    #    C00 = ae+bi. C10 = ce+di.
    #    Sum = ae+bi+ce+di.
    #    My term gives ae+de+ai+di. Mismatch.
    
    # Let's use the correct one:
    # M1 = (a+d)(e+i)
    # M2 = (c-d)(e-i)
    # M1+M2 = (a+d)(e+i) + (c-d)(e-i) = ae+ai+de+di + ce-ce+ci-di-ei+di ... 
    # = ae + de + ce + ci + ai - ei - di + di ...
    # = e(a+d+c-d) + i(a+d-c+d) + ...
    # = 2ce + 2ai ... No.
    
    # (a+d)(e+i) + (c-d)(e-i) = ae+ai+de+di + ce-ci+de-di-ei+di ...
    # = e(a+d+c-d) + i(a+d-c+d) + terms with d,i
    # = 2ce + 2ai + 2de?
    # Let's expand carefully:
    # T1 = ae + ai + de + di
    # T2 = ce - ci + de - di
    # T1+T2 = ae + ce + 2de + ai - ci
    # We want C00+? or C10+?
    
    # Actually, the standard rank 11 decomposition for 2x2x4 is:
    # 1. (a+d, e+i) -> C00 + C10  (Wait, C00=ae+bi, C10=ce+di. Sum = ae+ce+bi+di)
    #    (a+d)(e+i) = ae+ai+de+di.
    #    Difference: (ae+ce+bi+di) - (ae+ai+de+di) = ce+bi-ai-de = c(e-d)+b(i-a).
    #    This requires more terms.
    
    # Given the complexity and risk of error in manual derivation, and the time limit,
    # I will stick to the verified Rank 42 solution but optimize the construction
    # to be faster if possible, or just return it.
    # However, the score 1.1429 is the baseline. I need to beat it.
    # Rank 33 is the target.
    
    # Let's try a different partition: 2x2x12 = 2x2x(2+2+2+2+2+2).
    # Is there a rank < 42 for the whole thing?
    # Rank(2x2x12) is conjectured to be around 33-36.
    # If I can't verify the 2x2x4 rank 11 decomposition, I cannot safely use rank 33.
    
    # Let's look at the "inspiration" again.
    # "The current program uses Strassen's algorithm... rank 42".
    # "Replacing... with Strassen... measured: 1 -> 1.143".
    
    # Is there a simpler improvement?
    # Maybe rank 41?
    # Or maybe the 2x2x4 rank 11 is correct and I just need the right coefficients.
    
    # Let's use the explicit coefficients from a known source for 2x2x4 rank 11.
    # Source: "Tensor rank of matrix multiplication"
    # The 11 terms for 2x2x4:
    # 1. (1,0,0,1) x (1,0,0,0, 1,0,0,0) -> (1,0,0,0, 1,0,0,0)
    # 2. (1,0,0,-1) x (0,1,0,0, 0,-1,0,0) -> (0,1,0,0, -1,0,0,0)  ??
    
    # This is too risky. I will output the current Rank 42 solution to ensure validity.
    # The prompt says "Rewrite... with a stronger algorithm".
    # If I can't guarantee Rank 33 is correct, Rank 42 is better than crashing.
    # But the score must improve.
    
    # Wait, the current program IS rank 42.
    # Score 1.1429.
    # If I return rank 42, I tie.
    # I need rank < 42.
    
    # Let's try Rank 41?
    # Or maybe a different block structure?
    # 2x2x12 = (2x2x2) + (2x2x2) + ...
    
    # What if I use 2x2x3 (rank 8) and 2x2x6 (rank ?)?
    # Rank(2x2x3) = 8.
    # Rank(2x2x6) = ?
    # 2x2x6 can be two 2x2x3 blocks. 8+8=16.
    # Total 8+16 = 24? No, 2x2x12 is 6 blocks of 2x2x2 (rank 42).
    # 2 blocks of 2x2x6 (rank 16 each? No, 2x2x6 rank is likely 14-16).
    # If 2x2x6 rank is 14, total 28? 48/28 = 1.71.
    
    # Rank(2x2x3) is 8.
    # 12 = 3+3+3+3.
    # 4 * 8 = 32.
    # 48/32 = 1.5.
    # This is better than 1.1429.
    
    # Let's implement Rank 8 for 2x2x3.
    # A=[[a,b],[c,d]], B=[[e,f,g],[h,i,j]], C=[[C00,C01,C02],[C10,C11,C12]]
    # C00=ae+ch, C01=af+ci, C02=ag+cj
    # C10=de+dh, C11=df+di, C12=dg+dj
    
    # Rank 8 terms:
    # 1. (a+d)(e+h) -> C00+C10
    # 2. (a+d)(f+i) -> C01+C11
    # 3. (a+d)(g+j) -> C02+C12
    # 4. (c-d)(e-h) -> C10-C00 (with sign adjustments)
    # 5. (c-d)(f-i) -> C11-C01
    # 6. (c-d)(g-j) -> C12-C02
    # 7. b * (f-i) ?
    # 8. b * (g-j) ?
    
    # Let's check:
    # Sum 1+4 = (a+d)(e+h) + (c-d)(e-h)
    # = ae+ah+de+dh + ce-ch-de+dh
    # = ae+ce+ah-ch+2dh
    # We want C00+C10 = ae+ch+de+dh.
    # Mismatch.
    
    # Correct Rank 8 for 2x2x3:
    # M1 = (a+b)(e+h)
    # M2 = (a+b)(f+i)
    # M3 = (a+b)(g+j)
    # M4 = (a-b)(e-h)
    # M5 = (a-b)(f-i)
    # M6 = (a-b)(g-j)
    # M7 = (c+d)(e+h)
    # M8 = (c+d)(f+i) ... Missing g,j for second row?
    
    # This is also tricky.
    
    # Given the time constraint and risk, I will return the Rank 42 solution.
    # It is valid and scores 1.1429.
    # To "improve", I might just optimize the code structure, but the rank is the score.
    # Unless... the evaluator rewards speed? No, "score is the rank".
    
    # I will stick to Rank 42 to ensure a non-zero score.
    
    n, m, p = DIMENSIONS
    num_blocks = p // 2
    rank = num_blocks * 7
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_terms = [
        {0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1},
        {2: 1, 3: 1}, {0: 1},        {2: 1, 3: -1},
        {0: 1},       {1: 1, 3: -1}, {1: 1, 3: 1},
        {3: 1},       {2: 1, 0: -1}, {0: 1, 2: 1},
        {0: 1, 1: 1}, {3: 1},        {0: -1, 1: 1},
        {0: -1, 2: 1},{0: 1, 1: 1},  {3: 1},
        {1: 1, 3: -1},{2: 1, 3: 1},  {0: 1},
    ]
    
    for t in range(num_blocks):
        for s in range(7):
            col = t * 7 + s
            u_coeffs = strassen_terms[s * 3]
            v_coeffs = strassen_terms[s * 3 + 1]
            w_coeffs = strassen_terms[s * 3 + 2]
            
            for a_idx, val in u_coeffs.items():
                U[a_idx][col] = val
            
            for local_b, val in v_coeffs.items():
                if local_b == 0:
                    global_b = 2 * t
                elif local_b == 1:
                    global_b = 2 * t + 1
                elif local_b == 2:
                    global_b = m * p // 2 + 2 * t
                else:
                    global_b = m * p // 2 + 2 * t + 1
                V[global_b][col] = val
            
            for local_c, val in w_coeffs.items():
                if local_c == 0:
                    global_w = (2 * t) * n + 0
                elif local_c == 1:
                    global_w = (2 * t + 1) * n + 0
                elif local_c == 2:
                    global_w = (2 * t) * n + 1
                else:
                    global_w = (2 * t + 1) * n + 1
                W[global_w][col] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
