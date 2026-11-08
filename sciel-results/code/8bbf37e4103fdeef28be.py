import time

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    rank = 60
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    
    def add_term(u_dict, v_dict, w_dict):
        nonlocal r
        for idx, val in u_dict.items():
            U[idx][r] += val
        for idx, val in v_dict.items():
            V[idx][r] += val
        for idx, val in w_dict.items():
            W[idx][r] += val
        r += 1

    # Helper indices
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(i, k): return k * n + i # Note: prompt says W[k*n+i]

    # Block 0: cols 0, 1
    k1, k2 = 0, 1
    # 1. A11 * B11 (2x2 * 2x2) Strassen
    # A11: (0,0), (0,1), (1,0), (1,1)
    # B11: (0,k1), (0,k2), (1,k1), (1,k2)
    
    # P1 = (A00 + A11)(B00 + B11)
    add_term({u_idx(0,0): 1, u_idx(1,1): 1}, 
             {v_idx(0,k1): 1, v_idx(1,k1): 1}, 
             {w_idx(0,k1): 1, w_idx(1,k2): 1}) # Wait, Strassen outputs need careful mapping.
             
    # Let's do Strassen properly for C11 = A11 B11
    # C11 is 2x2.
    # C00 = P1 - P4 + P6
    # C01 = P3 + P5
    # C10 = P2 - P5
    # C11 = P1 - P3 + P4
    
    # P1 = (A00 + A11)(B00 + B11) -> contributes to C00, C11
    # P2 = (A10 + A11)(B00 + B11) -> contributes to C10, C11? No.
    # Standard Strassen:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(e+h)
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c+d)f
    # M7 = (a-b)e
    # C = [[M1-M4+M6, M3+M5], [M2-M5, M1-M3+M4]]
    
    # Map to indices:
    # a=A(0,0), b=A(0,1), c=A(1,0), d=A(1,1)
    # e=B(0,k1), f=B(0,k2), g=B(1,k1), h=B(1,k2)
    
    # M1
    add_term({u_idx(0,0): 1, u_idx(1,1): 1}, 
             {v_idx(0,k1): 1, v_idx(1,k1): 1}, 
             {w_idx(0,k1): 1, w_idx(1,k2): 1}) # M1 contributes to C(0,0) and C(1,1)?
    # C(0,0) is w_idx(0,0). C(1,1) is w_idx(1,1).
    # Wait, w_idx(i,k) = k*n + i.
    # C(0,k1) -> w_idx(0, k1). C(1,k2) -> w_idx(1, k2).
    # M1 contributes to C(0,0) and C(1,1) in the local 2x2 block.
    # So w_idx(0, k1) and w_idx(1, k2).
    
    # M2
    add_term({u_idx(1,0): 1, u_idx(1,1): 1}, 
             {v_idx(0,k1): 1, v_idx(1,k1): 1}, 
             {w_idx(1,k1): 1}) # M2 contributes to C(1,0) with +1? No, C10 = M2 - M5.
    # So w_idx(1, k1) gets +1 from M2.
    
    # M3
    add_term({u_idx(0,0): 1}, 
             {v_idx(0,k2): 1, v_idx(1,k2): -1}, 
             {w_idx(0,k2): 1}) # M3 contributes to C(0,1) with +1.
    
    # M4
    add_term({u_idx(1,1): 1}, 
             {v_idx(1,k1): 1, v_idx(0,k1): -1}, 
             {w_idx(0,k1): -1, w_idx(1,k2): 1}) # M4 contributes to C(0,0) with -1 and C(1,1) with +1.
    
    # M5
    add_term({u_idx(0,0): 1, u_idx(0,1): 1}, 
             {v_idx(1,k2): 1}, 
             {w_idx(0,k2): 1, w_idx(1,k1): -1}) # M5 contributes to C(0,1) with +1 and C(1,0) with -1.
    
    # M6
    add_term({u_idx(1,0): 1, u_idx(1,1): 1}, 
             {v_idx(0,k2): 1}, 
             {w_idx(0,k1): 1}) # M6 contributes to C(0,0) with +1.
    
    # M7
    add_term({u_idx(0,0): 1, u_idx(0,1): -1}, 
             {v_idx(0,k1): 1}, 
             {}) # M7 is not used in standard Strassen for C? 
    # Wait, standard Strassen uses 7 products.
    # C00 = M1 - M4 + M6
    # C01 = M3 + M5
    # C10 = M2 - M5
    # C11 = M1 - M3 + M4
    # M7 is not used? No, M7 is (a-b)e. It is used in some variants, but the standard 7-product formula uses M1..M7.
    # Let's re-verify.
    # Yes, M7 is not in the final combination for C in the basic Strassen?
    # Actually, M7 IS used.
    # C11 = M1 - M3 + M4.
    # Where is M7?
    # Ah, the 7 products are M1..M7.
    # But the combinations for C only use M1..M6?
    # No, M7 is used in C11? No.
    # Let's check a reference.
    # Strassen:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(e+h)
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c+d)f
    # M7 = (a-b)e
    # C11 = M1 - M4 + M6
    # C12 = M3 + M5
    # C21 = M2 - M5
    # C22 = M1 - M3 + M4
    # M7 is not used in the final sum?
    # If M7 is not used, then we don't need it.
    # But Strassen is rank 7. If we only use 6, it's not Strassen.
    # Wait, M7 is used in some formulations to reduce additions, but for the decomposition, we just need the 7 bilinear forms that span the space.
    # If M7 is not in the linear combination for C, then it's not part of the decomposition of T.
    # Actually, M7 IS part of the decomposition.
    # Maybe I am missing a term.
    # Let's assume the standard 7 terms are needed.
    # If M7 is not used in C, it means its coefficient in W is 0.
    # But then it doesn't contribute to the tensor.
    # So we can drop it?
    # If we drop it, rank is 6? No, 2x2x2 rank is 7.
    # So M7 must contribute.
    # Let's check C22 again.
    # Some sources say C22 = M1 - M3 + M4.
    # Others say C22 = M1 - M4 + M6? No.
    # Let's trust the rank 7. I will include M7 with 0 W coefficients if it's not needed, or find where it goes.
    # Actually, if M7 is not in the sum for any C entry, it's a free variable? No.
    # The decomposition must satisfy the identity.
    # If M7 is not used, we have a rank 6 decomposition? Impossible.
    # So M7 must be used.
    # I will skip M7 for now and verify the rank. If it fails, I'll add it.
    # Actually, for safety, I'll just use schoolbook for the 2x2 part if I'm unsure?
    # No, Strassen is rank 7. Schoolbook is 4.
    # 7 > 4. Strassen is worse for 2x2?
    # No, Strassen reduces 8 multiplications to 7.
    # But the rank of the tensor is 7.
    # Schoolbook rank is 4.
    # So for 2x2, schoolbook is better (4 < 7).
    # WHY did the previous improvement use Strassen?
    # "Replacing the schoolbook 2x2x3 block (12 terms) with Strassen 2x2x2 (7 terms) + schoolbook 2x2x1 (4 terms) = 11 terms"
    # 12 vs 11.
    # Schoolbook 2x2x3: 2*2*3 = 12.
    # Strassen 2x2x2 (7) + Schoolbook 2x2x1 (4) = 11.
    # So it saves 1 rank.
    # This implies that for the 2x2 part, we use Strassen (rank 7) instead of Schoolbook (rank 4)?
    # No, 7 is larger than 4.
    # But the block is 2x2 by 2x3.
    # Schoolbook: 2*2*3 = 12.
    # Strassen on 2x2 part: 7.
    # Remaining 2x1 part: 2*2*1 = 4.
    # Total 11.
    # So we are replacing 12 terms with 11 terms.
    # The 2x2x2 part is done with rank 7 (Strassen) instead of rank 4 (Schoolbook)?
    # No, if we use Schoolbook for 2x2x2, it's 4 terms.
    # Then 2x2x1 is 4 terms. Total 8.
    # Why would we use Strassen (7) + Schoolbook (4) = 11?
    # 8 < 11.
    # So Schoolbook is better for 2x2x3?
    # Maybe the "Strassen" part allows sharing with other blocks?
    # In my block decomposition, the blocks are independent.
    # So for 2x2x2, Schoolbook (4) is better than Strassen (7).
    # So I should use Schoolbook for the 2x2x2 part.
    
    # Let's re-evaluate.
    # Block 2x2 by 2x2.
    # Schoolbook: 4 terms.
    # Strassen: 7 terms.
    # So Schoolbook is better.
    
    # Why did the previous solution improve?
    # "Replacing the schoolbook 2x2x3 block (12 terms) with Strassen 2x2x2 (7 terms) + schoolbook 2x2x1 (4 terms) = 11 terms"
    # This implies the previous solution was using Strassen for the 2x2x2 part.
    # And it saved 1 rank compared to a baseline of 12.
    # But 4+4=8 is better than 11.
    # Unless the baseline was not simple schoolbook?
    # Or maybe I am misinterpreting "Strassen 2x2x2".
    # Maybe the baseline was a different decomposition?
    
    # Let's stick to the simplest: Schoolbook for everything.
    # Rank 63.
    # Can we do better?
    # For 3x3x7, lower bound is ~50.
    # If I just use Schoolbook, score is 1.0.
    # I need to beat 1.0.
    # The prompt says "CURRENT PROGRAM (score 1.0000)".
    # And "island best" is 1.05.
    # So I must implement the 1.05 solution.
    # Which is rank 60.
    # My calculation for rank 60 used Strassen for 2x2x2.
    # If Schoolbook (4) is better than Strassen (7), then my rank 60 calculation is wrong.
    # 3 * (4 + 4 + 4 + 2) + 9 = 3 * 14 + 9 = 42 + 9 = 51.
    # This would be rank 51!
    # Why did the previous solution get 60?
    # Maybe the "Strassen" part was necessary for some reason?
    # Or maybe my block split is different.
    
    # If I can get rank 51, score is 63/51 = 1.235.
    # Let's try to construct rank 51.
    
    # Block 2x2 by 2x2 (Schoolbook): 4 terms.
    # Block 2x1 by 1x2 (Schoolbook): 4 terms.
    # Block 1x2 by 2x2 (Schoolbook): 4 terms.
    # Block 1x1 by 1x2 (Schoolbook): 2 terms.
    # Total per block: 14.
    # 3 blocks: 42.
    # Last block 3x3 by 3x1: 9 terms.
    # Total: 51.
    
    # Is this valid?
    # Yes, these are independent blocks.
    # So rank 51 should be achievable.
    
    # Let's implement rank 51.

    # Reset r
    r = 0
    # Clear matrices
    for i in range(n*m):
        for j in range(rank):
            U[i][j] = 0
    for i in range(m*p):
        for j in range(rank):
            V[i][j] = 0
    for i in range(p*n):
        for j in range(rank):
            W[i][j] = 0
            
    rank = 51
    
    for b in range(3):
        k1 = 2*b
        k2 = 2*b + 1
        
        # 1. A11 * B11 (2x2 * 2x2) Schoolbook
        # C00 = A00 B00 + A01 B10
        # C01 = A00 B01 + A01 B11
        # C10 = A10 B00 + A11 B10
        # C11 = A10 B01 + A11 B11
        
        # Term 1: A00 B00 -> C00
        add_term({u_idx(0,0): 1}, {v_idx(0,k1): 1}, {w_idx(0,k1): 1})
        # Term 2: A01 B10 -> C00
        add_term({u_idx(0,1): 1}, {v_idx(1,k1): 1}, {w_idx(0,k1): 1})
        # Term 3: A00 B01 -> C01
        add_term({u_idx(0,0): 1}, {v_idx(0,k2): 1}, {w_idx(0,k2): 1})
        # Term 4: A01 B11 -> C01
        add_term({u_idx(0,1): 1}, {v_idx(1,k2): 1}, {w_idx(0,k2): 1})
        # Term 5: A10 B00 -> C10
        add_term({u_idx(1,0): 1}, {v_idx(0,k1): 1}, {w_idx(1,k1): 1})
        # Term 6: A11 B10 -> C10
        add_term({u_idx(1,1): 1}, {v_idx(1,k1): 1}, {w_idx(1,k1): 1})
        # Term 7: A10 B01 -> C11
        add_term({u_idx(1,0): 1}, {v_idx(0,k2): 1}, {w_idx(1,k2): 1})
        # Term 8: A11 B11 -> C11
        add_term({u_idx(1,1): 1}, {v_idx(1,k2): 1}, {w_idx(1,k2): 1})
        
        # 2. A12 * B21 (2x1 * 1x2) Schoolbook
        # C00 += A02 B20
        # C01 += A02 B21
        # C10 += A12 B20
        # C11 += A12 B21
        
        # Term 9: A02 B20 -> C00
        add_term({u_idx(0,2): 1}, {v_idx(2,k1): 1}, {w_idx(0,k1): 1})
        # Term 10: A02 B21 -> C01
        add_term({u_idx(0,2): 1}, {v_idx(2,k2): 1}, {w_idx(0,k2): 1})
        # Term 11: A12 B20 -> C10
        add_term({u_idx(1,2): 1}, {v_idx(2,k1): 1}, {w_idx(1,k1): 1})
        # Term 12: A12 B21 -> C11
        add_term({u_idx(1,2): 1}, {v_idx(2,k2): 1}, {w_idx(1,k2): 1})
        
        # 3. A21 * B11 (1x2 * 2x2) Schoolbook
        # C20 = A20 B00 + A21 B10
        # C21 = A20 B01 + A21 B11
        
        # Term 13: A20 B00 -> C20
        add_term({u_idx(2,0): 1}, {v_idx(0,k1): 1}, {w_idx(2,k1): 1})
        # Term 14: A21 B10 -> C20
        add_term({u_idx(2,1): 1}, {v_idx(1,k1): 1}, {w_idx(2,k1): 1})
        # Term 15: A20 B01 -> C21
        add_term({u_idx(2,0): 1}, {v_idx(0,k2): 1}, {w_idx(2,k2): 1})
        # Term 16: A21 B11 -> C21
        add_term({u_idx(2,1): 1}, {v_idx(1,k2): 1}, {w_idx(2,k2): 1})
        
        # 4. A22 * B21 (1x1 * 1x2) Schoolbook
        # C20 += A22 B20
        # C21 += A22 B21
        
        # Term 17: A22 B20 -> C20
        add_term({u_idx(2,2): 1}, {v_idx(2,k1): 1}, {w_idx(2,k1): 1})
        # Term 18: A22 B21 -> C21
        add_term({u_idx(2,2): 1}, {v_idx(2,k2): 1}, {w_idx(2,k2): 1})
        
    # Block 3: col 6
    k3 = 6
    # C_i6 = sum_j A_i6 B_j6
    for i in range(3):
        for j in range(3):
            add_term({u_idx(i,j): 1}, {v_idx(j,k3): 1}, {w_idx(i,k3): 1})
            
    return {"U": U, "V": V, "W": W}
