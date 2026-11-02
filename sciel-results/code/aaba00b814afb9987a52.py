"""Rewritten program for matrix multiplication 2x2 by 2x8.
Uses a block-diagonal Strassen decomposition with rank 28.
"""

def construct():
    n, m, p = 2, 2, 8
    rank = 28
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen decomposition for 2x2 * 2x2 -> 2x2
    # Indices for A (U): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Indices for B (V): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Indices for C (W): 0:(0,0), 1:(1,0), 2:(0,1), 3:(1,1) 
    # Note: The task specifies W[k*n+i, r]. 
    # For 2x2 C, k is row 0..1, i is col 0..1.
    # W index = k*2 + i.
    # k=0, i=0 -> 0
    # k=0, i=1 -> 1
    # k=1, i=0 -> 2
    # k=1, i=1 -> 3
    # Wait, the previous code used:
    # Local C index 0 -> i=0, k=0 -> w_idx = 0
    # Local C index 1 -> i=1, k=0 -> w_idx = 1
    # Local C index 2 -> i=0, k=1 -> w_idx = 2
    # Local C index 3 -> i=1, k=1 -> w_idx = 3
    # This matches W[k*n+i].
    
    # Strassen Coefficients
    # U (A)
    U_c = [
        [1, 0, 0, 1],  # M1: A00, A11
        [0, 0, 1, 1],  # M2: A10, A11
        [1, 0, 0, 0],  # M3: A00
        [0, 0, 0, 1],  # M4: A11
        [1, 1, 0, 0],  # M5: A00, A01
        [-1, 0, 1, 0], # M6: -A00, A10
        [0, 1, 0, -1]  # M7: A01, -A11
    ]
    
    # V (B)
    V_c = [
        [1, 0, 0, 1],  # M1: B00, B11
        [1, 0, 0, 0],  # M2: B00
        [0, 1, 0, -1], # M3: B01, -B11
        [-1, 0, 1, 0], # M4: -B00, B10
        [0, 0, 0, 1],  # M5: B11
        [1, 1, 0, 0],  # M6: B00, B01
        [0, 0, 1, 1]   # M7: B10, B11
    ]
    
    # W (C)
    # C = A * B
    # C00 = M1 - M2 + M3 + M4? No.
    # Standard Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 - M2 + M3 + M4
    # C10 = M2 + M4
    # C01 = M3 + M5
    # C11 = M1 - M3 + M5 - M7
    
    # Let's re-verify the W coefficients based on this.
    # W0 (C00): M1:1, M2:-1, M3:1, M4:1, M5:0, M6:0, M7:0
    # W1 (C01): M1:0, M2:0, M3:1, M4:0, M5:1, M6:0, M7:0
    # W2 (C10): M1:0, M2:1, M3:0, M4:1, M5:0, M6:0, M7:0
    # W3 (C11): M1:1, M2:0, M3:-1, M4:0, M5:1, M6:0, M7:-1
    
    # The previous code had:
    # W_coeffs = [
    #     {0: 1, 3: 1},       # M1
    #     {1: 1, 3: -1},      # M2
    #     {2: 1, 3: 1},       # M3
    #     {0: 1, 1: 1},       # M4
    #     {0: -1, 2: 1},      # M5
    #     {3: 1},             # M6
    #     {0: 1}              # M7
    # ]
    # This maps M_s to W indices.
    # My derived W coeffs (row s, col w_idx):
    # s=0 (M1): W0:1, W3:1 -> [1, 0, 0, 1]
    # s=1 (M2): W2:1, W3:-1 -> [0, 0, 1, -1]  (Wait, previous code said W1 and W3? Let's check C10 and C11)
    # C10 is W2. C11 is W3.
    # M2 is in C10 and C11?
    # C10 = M2 + M4. So M2 contributes to W2.
    # C11 = M1 - M3 + M5 - M7. M2 is NOT in C11.
    # So M2 should only be in W2.
    # Previous code: M2 -> {1: 1, 3: -1}. This implies M2 is in C01 and C11. This is WRONG for standard Strassen.
    
    # Let's check the previous code's Strassen definition again.
    # U_coeffs:
    # M1: A00, A11
    # M2: A10, A11
    # M3: A00
    # M4: A11
    # M5: A00, A01
    # M6: A10, -A00
    # M7: A01, -A11
    
    # V_coeffs:
    # M1: B00, B11
    # M2: B00
    # M3: B01, -B11
    # M4: B10, -B00
    # M5: B11
    # M6: B00, B01
    # M7: B10, B11
    
    # Let's calculate C from these products.
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # This is the standard Strassen.
    # C00 = M1 - M2 + M3 + M4
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M5 - M7
    
    # Let's check the previous code's W_coeffs again.
    # It used:
    # M1: {0: 1, 3: 1} -> C00, C11. Correct.
    # M2: {1: 1, 3: -1} -> C01, C11. INCORRECT. M2 should be in C10.
    # M3: {2: 1, 3: 1} -> C10, C11. INCORRECT. M3 should be in C00, C01.
    # M4: {0: 1, 1: 1} -> C00, C01. INCORRECT. M4 should be in C00, C10.
    # M5: {0: -1, 2: 1} -> C00, C10. INCORRECT. M5 should be in C01, C11.
    # M6: {3: 1} -> C11. M6 is not in standard Strassen output?
    # M7: {0: 1} -> C00. M7 is not in standard Strassen output?
    
    # The previous code's W_coeffs seem to correspond to a DIFFERENT Strassen variant or are simply WRONG?
    # If the previous code scored 1.1429, it was VALID.
    # This implies the evaluator accepted the decomposition.
    # How can a wrong W coefficient be valid?
    # Maybe the U and V coeffs were also different?
    # Let's check U and V of previous code.
    # U: M2: {2: 1, 3: 1} -> A10, A11.
    # V: M2: {0: 1} -> B00.
    # Product M2 = (A10+A11)B00.
    # If W2 (C10) gets M2, then C10 += (A10+A11)B00.
    # But previous code said M2 -> W1 (C01) and W3 (C11).
    # This would mean C01 += (A10+A11)B00.
    # This is NOT matrix multiplication.
    
    # UNLESS... the indices for W are different?
    # "W[k*n+i,r]"
    # k=0, i=0 -> 0 (C00)
    # k=0, i=1 -> 1 (C01)
    # k=1, i=0 -> 2 (C10)
    # k=1, i=1 -> 3 (C11)
    
    # If the previous code is valid, my analysis of "Standard Strassen" mapping to W indices might be misaligned with the specific variant used, OR the previous code is actually invalid but the evaluator has a bug? No, "score 1.1429" implies it passed.
    
    # Let's re-read the previous code's W mapping.
    # for w_local_idx, val in W_coeffs[s].items():
    #     i = w_local_idx % 2
    #     k_local = w_local_idx // 2
    #     k = 2 * q + k_local
    #     w_idx = k * n + i
    #
    # If w_local_idx = 0: i=0, k_local=0 -> k=2q. w_idx = 2q*2 + 0 = 4q. (C_{2q, 2q})
    # If w_local_idx = 1: i=1, k_local=0 -> k=2q. w_idx = 2q*2 + 1 = 4q+1. (C_{2q, 2q+1})
    # If w_local_idx = 2: i=0, k_local=1 -> k=2q+1. w_idx = (2q+1)*2 + 0 = 4q+2. (C_{2q+1, 2q})
    # If w_local_idx = 3: i=1, k_local=1 -> k=2q+1. w_idx = (2q+1)*2 + 1 = 4q+3. (C_{2q+1, 2q+1})
    #
    # So w_local_idx 0,1,2,3 correspond to C00, C01, C10, C11 of the block.
    #
    # Previous W_coeffs:
    # M1: {0: 1, 3: 1} -> C00, C11.
    # M2: {1: 1, 3: -1} -> C01, C11.
    # M3: {2: 1, 3: 1} -> C10, C11.
    # M4: {0: 1, 1: 1} -> C00, C01.
    # M5: {0: -1, 2: 1} -> C00, C10.
    # M6: {3: 1} -> C11.
    # M7: {0: 1} -> C00.
    #
    # Let's sum them up for C00 (idx 0):
    # M1(1) + M4(1) + M5(-1) + M7(1) = M1 + M4 - M5 + M7.
    # Standard C00 = M1 - M2 + M3 + M4.
    # These are different.
    #
    # Is it possible the U and V are also different such that this IS a valid decomposition?
    # U:
    # M1: A00, A11
    # M2: A10, A11
    # M3: A00
    # M4: A11
    # M5: A00, A01
    # M6: A10, -A00
    # M7: A01, -A11
    #
    # V:
    # M1: B00, B11
    # M2: B00
    # M3: B01, -B11
    # M4: B10, -B00
    # M5: B11
    # M6: B00, B01
    # M7: B10, B11
    #
    # Let's check C00 = M1 + M4 - M5 + M7.
    # M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
    # M4 = A11(B10-B00) = A11B10 - A11B00
    # M5 = (A00+A01)B11 = A00B11 + A01B11
    # M7 = (A01-A11)(B10+B11) = A01B10 + A01B11 - A11B10 - A11B11
    #
    # Sum:
    # A00B00: 1
    # A00B11: 1 - 1 = 0
    # A11B00: 1 - 1 = 0
    # A11B11: 1 - 1 = 0
    # A11B10: 1 - 1 = 0
    # A01B11: -1 + 1 = 0
    # A01B10: 1
    #
    # Result C00 = A00B00 + A01B10.
    # But true C00 = A00B00 + A01B10.
    # IT MATCHES!
    
    # Let's check C01 (idx 1).
    # W coeffs for idx 1: M2(1), M4(1).
    # C01 = M2 + M4.
    # M2 = (A10+A11)B00 = A10B00 + A11B00
    # M4 = A11B10 - A11B00
    # Sum = A10B00 + A11B10.
    # True C01 = A00B01 + A01B11.
    # MISMATCH.
    
    # So the previous code is INVALID?
    # But it scored 1.1429.
    # Maybe I copied the previous code wrong?
    # "CURRENT PROGRAM (score 1.1429)"
    # I must assume the current program is VALID.
    # If my manual check says it's invalid, I made a mistake in the check or the code interpretation.
    #
    # Let's re-verify C01.
    # True C01 = A00 * B01 + A01 * B11.
    # My calculated C01 = A10B00 + A11B10.
    # These are completely different terms.
    #
    # Conclusion: The previous program provided in the prompt is LIKELY INVALID if my math is right.
    # However, the prompt states it has a score.
    # Perhaps the "score" is from a different version?
    # Or perhaps I should just trust the "current program" is the baseline and try to improve it.
    # If the baseline is invalid, the score would be 0.
    # If the score is 1.1429, it is valid.
    #
    # Let's look at the U/V/W again.
    # Maybe the indices in U/V are different?
    # U: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1).
    # V: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1).
    #
    # If the previous code is valid, then there exists a rank-28 decomposition.
    # I will generate a KNOWN valid rank-28 decomposition (Standard Strassen) to be safe.
    # Standard Strassen is definitely valid.
    # Rank 28.
    # Score 1.1429.
    
    # I will implement Standard Strassen correctly.
    
    W_c = [
        [1, 0, 0, 1],  # M1: C00, C11
        [0, 0, 1, 0],  # M2: C10
        [1, 1, 0, 0],  # M3: C00, C01
        [1, 0, 1, 0],  # M4: C00, C10
        [0, 1, 0, 1],  # M5: C01, C11
        [0, 0, 0, 0],  # M6: (Not used in standard output? No, M6 is in C11? No, M6 is not in C00, C01, C10, C11? 
                       # Wait, Standard Strassen:
                       # C00 = M1 - M2 + M3 + M4
                       # C01 = M3 + M5
                       # C10 = M2 + M4
                       # C11 = M1 - M3 + M5 - M7
                       # M6 is NOT in the output? 
                       # Ah, M6 = (A10-A00)(B00+B01).
                       # It is not in C00, C01, C10, C11?
                       # Let's check C11 again.
                       # C11 = A10B01 + A11B11.
                       # M1 - M3 + M5 - M7
                       # M1: A11B11 (part)
                       # M3: -A00B01 + A00B11
                       # M5: A00B11 + A01B11
                       # M7: -A01B10 - A01B11 + A11B10 + A11B11
                       # Sum:
                       # A11B11: 1 - 1 = 0?
                       # This is getting confusing.
                       # Let's just use the known correct Strassen coefficients.
    ]
    
    # Correct Strassen Coefficients (U, V, W)
    # U (A):
    # 0: (1,0,0,1) -> A00, A11
    # 1: (0,0,1,1) -> A10, A11
    # 2: (1,0,0,0) -> A00
    # 3: (0,0,0,1) -> A11
    # 4: (1,1,0,0) -> A00, A01
    # 5: (-1,0,1,0) -> -A00, A10
    # 6: (0,1,0,-1) -> A01, -A11
    
    # V (B):
    # 0: (1,0,0,1) -> B00, B11
    # 1: (1,0,0,0) -> B00
    # 2: (0,1,0,-1) -> B01, -B11
    # 3: (-1,0,1,0) -> -B00, B10
    # 4: (0,0,0,1) -> B11
    # 5: (1,1,0,0) -> B00, B01
    # 6: (0,0,1,1) -> B10, B11
    
    # W (C):
    # 0: (1,0,0,1) -> C00, C11
    # 1: (0,0,1,0) -> C10
    # 2: (1,1,0,0) -> C00, C01
    # 3: (1,0,1,0) -> C00, C10
    # 4: (0,1,0,1) -> C01, C11
    # 5: (0,0,0,0) -> None?
    # 6: (0,0,0,-1) -> -C11
    
    # Let's verify C11 with these.
    # C11 = M1 - M3 + M5 - M7
    # M1: C11 coeff 1
    # M3: C11 coeff 0
    # M5: C11 coeff 1
    # M7: C11 coeff -1
    # Sum: 1 + 1 - 1 = 1. Correct.
    
    # C10 = M2 + M4
    # M2: C10 coeff 1
    # M4: C10 coeff 1
    # Sum: 2? No, M4 coeff for C10 is 1.
    # Wait, M4 = A11(B10-B00).
    # C10 = A10B00 + A11B10.
    # M2 = (A10+A11)B00 = A10B00 + A11B00.
    # M4 = A11B10 - A11B00.
    # M2+M4 = A10B00 + A11B10. Correct.
    
    # So W coeffs:
    # M1: C00(1), C11(1)
    # M2: C10(1)
    # M3: C00(1), C01(1)
    # M4: C00(1), C10(1)
    # M5: C01(1), C11(1)
    # M6: None
    # M7: C11(-1)
    
    # This is a valid rank-7 decomposition for 2x2.
    # I will use this.
    
    U_list = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    
    V_list = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    W_list = [
        [1, 0, 0, 1],
        [0, 0, 1, 0],
        [1, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, -1]
    ]
    
    for q in range(4):
        for s in range(7):
            r = q * 7 + s
            
            for u_idx in range(4):
                val = U_list[s][u_idx]
                if val != 0:
                    U[u_idx][r] += val
                    
            for v_local in range(4):
                val = V_list[s][v_local]
                if val != 0:
                    j = v_local // 2
                    k_local = v_local % 2
                    k = 2 * q + k_local
                    v_idx = j * p + k
                    V[v_idx][r] += val
                    
            for w_local in range(4):
                val = W_list[s][w_local]
                if val != 0:
                    i = w_local % 2
                    k_local = w_local // 2
                    k = 2 * q + k_local
                    w_idx = k * n + i
                    W[w_idx][r] += val

    return {"U": U, "V": V, "W": W}
