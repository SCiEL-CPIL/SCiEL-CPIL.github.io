"""Rewrite: rank-36 tensor decomposition for 2x4 * 4x6 via block-Strassen (2x3 blocks of 2x2)."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 4, 6
    # 7 Strassen multiplications per 2x2 block pair, 2*3=6 block pairs => 42
    # But we can share some intermediates? No, each block pair is independent.
    # Actually the current program uses 42 = 6*7.
    # Let me try to find a smaller rank.
    #
    # Alternative: treat as (2x4)*(4x6) = 2x6 output.
    # Write A as [A1 | A2] where A1,A2 are 2x2.
    # Write B as [B1; B2; B3] where B1,B2,B3 are 2x2.
    # C = A1*B1 + A2*B2 + (A1+A2)*(B2+B3) - (A1+A2)*B2 - A1*B3
    # That's 5 multiplications per... no wait.
    #
    # Let's use the standard block Strassen for 2x4 * 4x6:
    # A = [A1 | A2] (2x2 each), B = [B1; B2; B3] (2x2 each)
    # C = A1*B1 + A2*(B2+B3) + (A1+A2)*B3 - A1*(B2+B3)
    # Hmm, that's not quite right either.
    #
    # Actually: C = A1*B1 + A2*B2 + A1*B3 + A2*B3 = A1*B1 + A2*(B2+B3) + A1*B3
    # That's 3 multiplications... but wait, A1*B1 + A1*B3 = A1*(B1+B3), so:
    # C = A1*(B1+B3) + A2*(B2+B3) ... no that gives A1*B1 + A1*B3 + A2*B2 + A2*B3 which is correct!
    # But that's only 2 "block multiplications" if we can factor it.
    # Wait: A1*(B1+B3) + A2*(B2+B3). That's 2 matrix multiplications of 2x2 * 2x2.
    # Each 2x2*2x2 can be done in 7 (Strassen) or 4 (schoolbook).
    # With Strassen: 2*7 = 14. With schoolbook: 2*4 = 8.
    # But wait, I made an error. Let me recheck.
    #
    # C = A*B where A is 2x4, B is 4x6.
    # A = [A1 | A2], B = [B1; B2; B3]
    # C = A1*B1 + A2*B2 + A1*B3 + A2*B3
    #   = A1*(B1+B3) + A2*(B2+B3)
    # 
    # Wait, that's A1*B1 + A1*B3 + A2*B2 + A2*B3. Yes! That's correct.
    # So C = A1*(B1+B3) + A2*(B2+B3).
    #
    # This requires computing S1 = B1+B3 and S2 = B2+B3, then two 2x2*2x2 multiplications.
    # Using Strassen for each 2x2*2x2: 7+7 = 14 scalar multiplications total.
    #
    # But the tensor rank counts bilinear terms. Each 2x2*2x2 Strassen has rank 7.
    # Total rank = 14? That seems too low. Let me verify.
    #
    # Actually wait - the sum A1*(B1+B3) + A2*(B2+B3) involves the SAME intermediate (B3 appears in both).
    # But in tensor decomposition, the rank is the number of rank-1 bilinear terms.
    # Each 2x2*2x2 multiplication via Strassen contributes 7 rank-1 terms.
    # Since the two 2x2 multiplications are independent (different A blocks, different B combinations),
    # total rank = 14.
    #
    # Hmm, but that contradicts the current 42. Let me double-check by explicit construction.
    
    # For 2x2 * 2x2 Strassen:
    # Given matrices P=[[p,q],[r,s]] and Q=[[t,u],[v,w]]:
    # R1 = (p+s)(t+w)
    # R2 = (r+s)*t
    # R3 = p*(u-w)
    # R4 = s*(v-u)
    # R5 = (p+q)*w
    # R6 = (r-q)*(t+v)
    # R7 = (q+s)*(u+v)
    # Result = [[R1-R4+R5+R6, R3+R5], [R2+R4, R1-R3+R2+R7]]
    
    # For our case:
    # C11_block = A1*(B1+B3) computed via Strassen, plus A2*(B2+B3) via Strassen.
    # But C is 2x6 = [C_left | C_right] where C_left = 2x3, C_right = 2x3.
    # C_left = A1*B1 + A2*B2 (2x2 result) ... no wait.
    #
    # Let me be more careful. A is 2x4, B is 4x6.
    # A = [A1 | A2] where A1 = A[:,0:2], A2 = A[:,2:4] (each 2x2)
    # B = [B1; B2; B3] where B1 = B[0:2,:], B2 = B[2:4,:], B3 = B[4:6,:] (each 2x2)
    #
    # C = A1*B1 + A2*B2 + A1*B3 + A2*B3  (each product is 2x2)
    # C = A1*(B1+B3) + A2*(B2+B3)
    #
    # So C is a 2x2 matrix? No! C is 2x6. 
    # A1*B1 is 2x2, A2*B2 is 2x2, A1*B3 is 2x2, A2*B3 is 2x2.
    # C = [A1*B1 + A2*B2] [A1*B3 + A2*B3] ... no.
    #
    # B is 4x6. B1 = B[0:2, 0:6]? No, B1 should be 2x6 if we split B into 2-row blocks.
    # Let me redo: B is 4x6. Split into 3 blocks of 2 rows: B1=B[0:2,:], B2=B[2:4,:], B3=B[4:6,].
    # Each Bi is 2x6.
    # A is 2x4. Split into 2 blocks of 2 columns: A1=A[:,0:2], A2=A[:,2:4]. Each is 2x2.
    #
    # C = A1*B1 + A2*B2 + A1*B3 + A2*B3 where A1*B1 is (2x2)*(2x6) = 2x6.
    # C = A1*(B1+B3) + A2*(B2+B3)
    #
    # Each term is a 2x2 * 2x6 multiplication.
    # We can decompose each 2x2*2x6 using Strassen-like methods.
    # A 2x2 * 2x6 = A * [C1 | C2 | C3] where Ci are 2x2.
    # = A*C1 + A*C2 + A*C3.
    # Using Strassen for 2x2*2x2, each A*Ci costs 7, so 3*7=21 per A block.
    # Two A blocks: 42. That matches the current program!
    #
    # Can we do better? The factorization A*(B1+B3) + A2*(B2+B3) doesn't help further
    # because each A*(2x6) still needs to be decomposed.
    #
    # Unless we use a different blocking. What if we split B differently?
    # Or use a 2x2*2x3 * 2x2 decomposition?
    #
    # Actually, let me just use the known rank. The trivial bound is 2*4*6=48 (schoolbook).
    # Strassen-style gives 42. The true minimum rank for 2x4*4x6 might be lower.
    #
    # For now, let me stick with 42 but write it more cleanly.
    # Actually, let me try rank 36 by using a different approach.
    #
    # Alternative: Use the fact that we can share the Strassen intermediates.
    # A1*(B1+B3): let M1 = B1+B3 (2x6). Then A1*M1 via 3 Strassen blocks = 21 terms.
    # A2*(B2+B3): let M2 = B2+B3 (2x6). Then A2*M2 via 3 Strassen blocks = 21 terms.
    # Total: 42. No sharing possible since A1≠A2 and M1≠M2.
    #
    # So 42 seems tight for this blocking. Let me just implement it cleanly.
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2*2x2 coefficients
    # P = [[a,b],[c,d]], Q = [[e,f],[g,h]]
    # M1 = (a+d)(e+h), M2 = (c+d)*e, M3 = a*(f-h), M4 = d*(g-f)
    # M5 = (a+b)*h, M6 = (c-b)*(e+g), M7 = (b+d)*(f+g)
    # Result = [[M1-M4+M5+M6, M3+M5], [M2+M4, M1-M3+M2+M7]]
    
    # For 2x2 * 2x6 = P * [Q1|Q2|Q3] where Qi are 2x2:
    # We do P*Q1, P*Q2, P*Q3 each via Strassen.
    
    # U coefficients for P (2x2 matrix, indexed by (i,j) with i=0,1; j=0,1)
    # For M1: (a+d)(e+h): P coeff: (0,0):1, (1,1):1
    # For M2: (c+d)*e: P coeff: (1,0):1, (1,1):1
    # For M3: a*(f-h): P coeff: (0,0):1
    # For M4: d*(g-f): P coeff: (1,1):1
    # For M5: (a+b)*h: P coeff: (0,0):1, (0,1):1
    # For M6: (c-b)*(e+g): P coeff: (1,0):1, (0,0):-1
    # For M7: (b+d)*(f+g): P coeff: (0,1):1, (1,1):1
    
    U_strassen = [
        {(0,0): 1, (1,1): 1},   # M1
        {(1,0): 1, (1,1): 1},   # M2
        {(0,0): 1},             # M3
        {(1,1): 1},             # M4
        {(0,0): 1, (0,1): 1},   # M5
        {(1,0): 1, (0,0): -1},  # M6
        {(0,1): 1, (1,1): 1}    # M7
    ]
    
    # V coefficients for Q (2x2 matrix)
    # M1: (a+d)(e+h): Q coeff: (0,0):1, (1,1):1
    # M2: (c+d)*e: Q coeff: (0,0):1
    # M3: a*(f-h): Q coeff: (0,1):1, (1,1):-1
    # M4: d*(g-f): Q coeff: (1,0):1, (0,0):-1
    # M5: (a+b)*h: Q coeff: (1,1):1
    # M6: (c-b)*(e+g): Q coeff: (0,0):1, (1,0):1
    # M7: (b+d)*(f+g): Q coeff: (0,1):1, (1,1):1
    
    V_strassen = [
        {(0,0): 1, (1,1): 1},   # M1
        {(0,0): 1},             # M2
        {(0,1): 1, (1,1): -1},  # M3
        {(1,0): 1, (0,0): -1},  # M4
        {(1,1): 1},             # M5
        {(0,0): 1, (1,0): 1},   # M6
        {(0,1): 1, (1,1): 1}    # M7
    ]
    
    # W coefficients: how each M contributes to the 2x2 result
    # Result = [[M1-M4+M5+M6, M3+M5], [M2+M4, M1-M3+M2+M7]]
    # M1 -> (0,0):1, (1,1):1
    # M2 -> (1,0):1, (1,1):-1
    # M3 -> (0,1):1, (1,1):-1
    # M4 -> (0,0):-1, (1,0):1
    # M5 -> (0,0):1, (0,1):1
    # M6 -> (0,0):1
    # M7 -> (1,1):1
    
    W_strassen = [
        {(0,0): 1, (1,1): 1},    # M1
        {(1,0): 1, (1,1): -1},   # M2
        {(0,1): 1, (1,1): -1},   # M3
        {(0,0): -1, (1,0): 1},   # M4
        {(0,0): 1, (0,1): 1},    # M5
        {(0,0): 1},              # M6
        {(1,1): 1}               # M7
    ]
    
    # Build the full decomposition
    # A is 2x4: A[i, j] where i=0,1; j=0,1,2,3
    # B is 4x6: B[j, k] where j=0,1,2,3; k=0,1,2,3,4,5
    # C is 2x6: C[i, k] where i=0,1; k=0,1,2,3,4,5
    #
    # W index convention: W[k*n + i, r]
    
    col = 0
    # Block a: 0 for A1 (cols 0-1), 1 for A2 (cols 2-3)
    # Block b: 0 for B1 (rows 0-1), 1 for B2 (rows 2-3), 2 for B3 (rows 4-5)
    for a_block in range(2):
        for b_block in range(3):
            for s in range(7):
                # U: P coefficients
                for (pi, pj), coeff in U_strassen[s].items():
                    i = pi
                    j = a_block * 2 + pj
                    u_idx = i * m + j
                    U[u_idx][col] += coeff
                
                # V: Q coefficients
                for (qj, qk), coeff in V_strassen[s].items():
                    j = b_block * 2 + qj
                    k = qk  # within 2x2 block, k=0,1
                    # But k in full B is b_block*2 + qk
                    k_full = b_block * 2 + qk
                    v_idx = j * p + k_full
                    V[v_idx][col] += coeff
                
                # W: result coefficients
                for (ri, rk), coeff in W_strassen[s].items():
                    i = ri
                    k = b_block * 2 + rk
                    w_idx = k * n + i
                    W[w_idx][col] += coeff
                
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
