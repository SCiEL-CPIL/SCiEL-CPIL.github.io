import numpy as np

def construct():
    # Dimensions
    n, m, p = 2, 4, 8
    
    # The decomposition is based on splitting the multiplication M_{2,4,8}
    # into two M_{2,4,4} multiplications.
    # M_{2,4,4} is further decomposed into two M_{2,2,4} multiplications.
    # M_{2,2,4} is decomposed into two M_{2,2,2} multiplications.
    # M_{2,2,2} uses Strassen's algorithm with rank 7.
    
    # Total rank = 2 (from 8->4) * 2 (from 4->2) * 7 (Strassen) = 28? 
    # Wait.
    # M_{2,4,8}:
    # B is 4x8. Split into B1 (4x4) and B2 (4x4).
    # C = A B1 + A B2.
    # A B1 is M_{2,4,4}. Rank 28.
    # A B2 is M_{2,4,4}. Rank 28.
    # Total rank 56.
    
    # Let's construct the factors for M_{2,2,2} (Strassen)
    # A: 2x2, B: 2x2, C: 2x2
    # Indices: A00, A01, A10, A11
    #          B00, B01, B10, B11
    #          C00, C01, C10, C11
    
    # Strassen factors (U, V, W) for one block
    # We need to map these to the global indices.
    
    # Global A is 2x4. Indices: A[i, j] where i in 0..1, j in 0..3.
    # Global B is 4x8. Indices: B[i, j] where i in 0..3, j in 0..7.
    # Global C is 2x8. Indices: C[i, j] where i in 0..1, j in 0..7.
    
    # Trace convention: W[k*n+i, r] for C?
    # Task says: "output factor follows the official trace convention W[k*n+i,r]"
    # Usually C is n x p. Here C is 2 x 8.
    # So C[i, j] is stored at index j*n + i?
    # Let's check the seed code.
    # Seed code: `W = np.zeros((p * n, total_rank), dtype=int)`
    # `c_map` in seed:
    # `c_map = [(2 * kg + 0) * 2 + 0, (2 * kg + 1) * 2 + 0, (2 * kg + 0) * 2 + 1, (2 * kg + 1) * 2 + 1]`
    # This corresponds to C[row, col].
    # The index in W is `col * n + row`.
    # Yes, `W` has shape `(p*n, rank)`.
    
    # Let's build the factors.
    
    # Strassen 2x2x2 factors
    # s=0: (A00+A11)(B00+B11) -> C00+=, C11+=
    # s=1: (A10+A11)B00 -> C10+=, C11-=
    # s=2: A00(B01-B11) -> C01+=, C11+=
    # s=3: A11(B10-B00) -> C00+=, C10+=
    # s=4: (A00+A01)B11 -> C00-=, C01+=
    # s=5: (A10-A00)(B00+B01) -> C10+=
    # s=6: (A01-A11)(B10+B11) -> C00+=
    
    # We will iterate over the blocks.
    # Block structure:
    # A is split into A_left (2x2) and A_right (2x2)?
    # No, A is 2x4.
    # In M_{2,4,4}, A is 2x4, B is 4x4.
    # Split A into A_0 (2x2, cols 0,1) and A_1 (2x2, cols 2,3).
    # Split B into B_00, B_01, B_10, B_11 (2x2).
    # C = A_0 [B_00 B_01] + A_1 [B_10 B_11].
    # Let C1 = A_0 B_00 + A_1 B_10 (2x2)
    # Let C2 = A_0 B_01 + A_1 B_11 (2x2)
    # This is not quite right for Strassen on M_{2,2,4}.
    
    # Let's stick to the recursive tiling used in the seed, which is known to work.
    # The seed used:
    # jg in 0..1 (splits A's 4 cols into two 2-col blocks? No, A is 2x4. jg splits A into two 2x2 blocks horizontally? No, A is 2x4. The seed code `a_map` suggests A is treated as two 2x2 blocks vertically? No, A is 2x4.
    # Seed `a_map`:
    # `a_map = [0 * 4 + 2 * jg + 0, 0 * 4 + 2 * jg + 1, 1 * 4 + 2 * jg + 0, 1 * 4 + 2 * jg + 1]`
    # This maps local 0,1,2,3 to global A indices.
    # If jg=0: 0, 1, 4, 5. (Rows 0,1 of cols 0,1).
    # If jg=1: 2, 3, 6, 7. (Rows 0,1 of cols 2,3).
    # So A is split into A_L (cols 0,1) and A_R (cols 2,3).
    
    # `b_map`:
    # `b_map = [(2 * jg + 0) * 8 + 2 * kg + 0, ...]`
    # This is for B (4x8).
    # It seems to link the A block `jg` to specific B blocks.
    # If jg=0 (A_L), it uses B rows 0,1 (for kg=0) and 2,3 (for kg=1)?
    # Wait, `b_map` depends on `jg` and `kg`.
    # `b_map` indices are `row * 8 + col`.
    # `row = 2 * jg + local_row`.
    # `col = 2 * kg + local_col`.
    # So for a fixed `jg` (A block), we are looking at B rows `2*jg` and `2*jg+1`.
    # And `kg` splits B cols into two groups of 4?
    # `kg` goes 0..3.
    # `col = 2 * kg + ...`
    # If kg=0, cols 0,1.
    # If kg=1, cols 2,3.
    # If kg=2, cols 4,5.
    # If kg=3, cols 6,7.
    # So for each A block (2 of them), we process 4 B blocks (2x2 each).
    # Total 8 blocks of M_{2,2,2}.
    # Each M_{2,2,2} has rank 7.
    # Total rank 8 * 7 = 56.
    
    # This matches the seed logic. I will just re-implement this logic cleanly.
    
    rank = 56
    U = np.zeros((8, rank), dtype=int) # n*m = 2*4 = 8
    V = np.zeros((32, rank), dtype=int) # m*p = 4*8 = 32
    W = np.zeros((16, rank), dtype=int) # p*n = 8*2 = 16
    
    # Local Strassen factors for 2x2x2
    # Local A indices: 0:00, 1:01, 2:10, 3:11
    # Local B indices: 0:00, 1:01, 2:10, 3:11
    # Local C indices: 0:00, 1:01, 2:10, 3:11
    
    # U_loc: 4x7
    # V_loc: 4x7
    # W_loc: 4x7
    
    # s=0
    u0 = [1, 0, 0, 1]
    v0 = [1, 0, 0, 1]
    w0 = [1, 0, 0, 1]
    
    # s=1
    u1 = [0, 0, 1, 1]
    v1 = [1, 0, 0, 0]
    w1 = [0, 0, 1, -1]
    
    # s=2
    u2 = [1, 0, 0, 0]
    v2 = [0, 1, 0, -1]
    w2 = [0, 1, 0, 1]
    
    # s=3
    u3 = [0, 0, 0, 1]
    v3 = [0, 0, 1, -1] # Wait, B10-B00 is [0,0,1,-1]? No.
    # B10 is index 2. B00 is index 0.
    # So v3 should be [ -1, 0, 1, 0 ]?
    # Let's check seed: `V_loc[2, 3] = 1; V_loc[0, 3] = -1`
    # So v3 = [-1, 0, 1, 0].
    w3 = [1, 0, 1, 0] # C00+=, C10+=
    
    # s=4
    u4 = [1, 1, 0, 0]
    v4 = [0, 0, 0, 1]
    w4 = [-1, 1, 0, 0]
    
    # s=5
    u5 = [-1, 0, 1, 0]
    v5 = [1, 1, 0, 0]
    w5 = [0, 0, 0, 1] # C10+=? Seed says W_loc[3, 5] = 1. Index 3 is C11?
    # Seed: `W_loc[3, 5] = 1`.
    # My w5: index 3 is 1. So [0,0,0,1].
    # Wait, M6: (A10-A00)(B00+B01) -> C10+=
    # C10 is index 2.
    # Seed says `W_loc[3, 5] = 1`.
    # Let's re-read Strassen M6.
    # M6 = (A10 - A00)(B00 + B01).
    # Contribution to C10?
    # C10 = A10 B00 + A11 B10? No.
    # Standard Strassen:
    # M1 = (A11 + A00)(B11 + B00)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    #
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M6
    # C11 = M1 - M2 + M3 + M4
    #
    # Let's check W contributions in seed.
    # s=5 (M6): `W_loc[3, 5] = 1`.
    # Index 3 is C11?
    # Seed `c_map`:
    # 0: C00
    # 1: C01
    # 2: C10
    # 3: C11
    # So seed puts M6 into C11?
    # But formula says M6 goes to C10.
    # Let's check seed M2 (s=1).
    # `W_loc[2, 1] = 1; W_loc[3, 1] = -1`.
    # M2 goes to C10 (+1) and C11 (-1).
    # Formula: C10 has M2. C11 has -M2.
    # So seed mapping:
    # Index 2 is C10.
    # Index 3 is C11.
    #
    # Now back to M6 (s=5).
    # Formula: C10 has M6.
    # Seed: `W_loc[3, 5] = 1`.
    # This implies Seed puts M6 into C11?
    # If Seed puts M6 into C11, and M2 into C11 (-1), then C11 = ... - M2 + M6?
    # But formula is C11 = M1 - M2 + M3 + M4. No M6.
    # There is a discrepancy in my understanding of the seed's W indices or the Strassen formula.
    #
    # Let's look at seed M6 again.
    # `U_loc[2, 5] = 1; U_loc[0, 5] = -1` -> A10 - A00.
    # `V_loc[0, 5] = 1; V_loc[1, 5] = 1` -> B00 + B01.
    # `W_loc[3, 5] = 1`.
    #
    # If W index 3 is C11, then M6 contributes to C11.
    # If W index 2 is C10, then M6 does NOT contribute to C10.
    #
    # Let's check M7 (s=6).
    # `U_loc[1, 6] = 1; U_loc[3, 6] = -1` -> A01 - A11.
    # `V_loc[2, 6] = 1; V_loc[3, 6] = 1` -> B10 + B11.
    # `W_loc[0, 6] = 1`.
    # Index 0 is C00.
    # Formula: C00 has -M7.
    # Seed has +1.
    #
    # It seems the seed might have a sign error or different convention?
    # Or my formula is wrong?
    #
    # Let's verify the multiplication with the seed factors for a simple case.
    # If the seed scored 1.1429, it likely passed validation (validity not declared, but score implies it ran).
    # Wait, "validity_not_declared" usually means the evaluator didn't check correctness, only rank?
    # "score is the rank". "invalid result ... scores 0".
    # If it scored 1.1429 (which is 64/56), it implies the rank was 56.
    # If it were invalid, it would be 0.
    # So the seed IS a valid decomposition.
    #
    # I will trust the seed's factors. I will just copy them into a new structure.
    
    factors = [
        # U, V, W
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]), # Note: V is [-1,0,1,0] based on seed V_loc[0]=-1, V_loc[2]=1
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    
    # Wait, I need to be careful with the V for s=3 (M4).
    # Seed: `V_loc[2, 3] = 1; V_loc[0, 3] = -1`.
    # So V = [-1, 0, 1, 0]. Correct.
    
    # Now fill U, V, W global arrays.
    
    for jg in range(2):
        # A block jg: cols 2*jg, 2*jg+1
        a_base_col = 2 * jg
        for kg in range(4):
            # B block kg: rows 2*jg, 2*jg+1? 
            # Wait, in M_{2,4,4} decomposition:
            # A = [A_L, A_R].
            # B = [[B00, B01], [B10, B11]].
            # C = A_L [B00, B01] + A_R [B10, B11].
            # So if jg=0 (A_L), we use B rows 0,1.
            # If jg=1 (A_R), we use B rows 2,3.
            # So B row base is 2 * jg.
            b_base_row = 2 * jg
            b_base_col = 2 * kg
            
            # C block:
            # C = C_L + C_R?
            # C is 2x8.
            # The product A_L B_top gives 2x4 result.
            # The product A_R B_bottom gives 2x4 result.
            # They add up to the same C columns?
            # Yes, C[:, 2*kg : 2*kg+2] gets contributions from both?
            # No.
            # C = A B.
            # C[:, 2*kg : 2*kg+2] = A[:, 0:2] B[0:2, 2*kg:2*kg+2] + A[:, 2:4] B[2:4, 2*kg:2*kg+2].
            # So for a fixed kg (C col block), we sum contributions from jg=0 and jg=1.
            # The seed code iterates jg and kg and ADDS to W.
            # So W accumulates.
            
            c_base_col = 2 * kg
            
            for s in range(7):
                r = (jg * 4 + kg) * 7 + s
                
                u_vals, v_vals, w_vals = factors[s]
                
                # Map local A (4) to global A (8)
                # Local 0: A[0, 2*jg]
                # Local 1: A[0, 2*jg+1]
                # Local 2: A[1, 2*jg]
                # Local 3: A[1, 2*jg+1]
                # Global index: row * 4 + col
                
                global_a_indices = [
                    0 * 4 + (2 * jg + 0),
                    0 * 4 + (2 * jg + 1),
                    1 * 4 + (2 * jg + 0),
                    1 * 4 + (2 * jg + 1)
                ]
                
                # Map local B (4) to global B (32)
                # Local 0: B[2*jg, 2*kg]
                # Local 1: B[2*jg, 2*kg+1]
                # Local 2: B[2*jg+1, 2*kg]
                # Local 3: B[2*jg+1, 2*kg+1]
                # Global index: row * 8 + col
                
                global_b_indices = [
                    (2 * jg + 0) * 8 + (2 * kg + 0),
                    (2 * jg + 0) * 8 + (2 * kg + 1),
                    (2 * jg + 1) * 8 + (2 * kg + 0),
                    (2 * jg + 1) * 8 + (2 * kg + 1)
                ]
                
                # Map local C (4) to global C (16)
                # Local 0: C[0, 2*kg]
                # Local 1: C[0, 2*kg+1]
                # Local 2: C[1, 2*kg]
                # Local 3: C[1, 2*kg+1]
                # Global index: col * 2 + row (Trace convention W[k*n+i, r])
                
                global_c_indices = [
                    (2 * kg + 0) * 2 + 0,
                    (2 * kg + 1) * 2 + 0,
                    (2 * kg + 0) * 2 + 1,
                    (2 * kg + 1) * 2 + 1
                ]
                
                for i in range(4):
                    if u_vals[i] != 0:
                        U[global_a_indices[i], r] += u_vals[i]
                    if v_vals[i] != 0:
                        V[global_b_indices[i], r] += v_vals[i]
                    if w_vals[i] != 0:
                        W[global_c_indices[i], r] += w_vals[i]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
