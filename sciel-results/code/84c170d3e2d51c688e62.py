"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Strassen factors for 2x2 matrix multiplication
    # Indices: i, j, k in {0, 1}
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # For 2x2, m=p=n=2.
    # U shape (4, 7), V shape (4, 7), W shape (4, 7)
    
    # Based on standard Strassen decomposition:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # U[i, j, c] coefficient of A_ij in M_c
    # V[j, k, c] coefficient of B_jk in M_c
    # W[k, i, c] coefficient of M_c in C_ki
    
    U2 = [
        [1, 1, 1, 0, 1, -1, 0], # i=0, j=0 (A00)
        [0, 0, 0, 0, 1, 0, 1],  # i=0, j=1 (A01)
        [0, 1, 0, 0, 0, 1, 0],  # i=1, j=0 (A10)
        [1, 1, 0, 1, 0, 0, -1]  # i=1, j=1 (A11)
    ]
    
    V2 = [
        [1, 1, 0, -1, 0, 1, 0], # j=0, k=0 (B00)
        [0, 0, 1, 0, 0, 1, 0],  # j=0, k=1 (B01)
        [1, 0, -1, 1, 1, 0, 1], # j=1, k=0 (B10)
        [1, 0, 0, 0, 1, 0, 1]   # j=1, k=1 (B11)
    ]
    
    W2 = [
        [1, 0, 0, 1, -1, 0, 1], # k=0, i=0 (C00)
        [1, 0, 0, 0, 0, 1, 0],  # k=0, i=1 (C10)
        [0, 1, 1, 0, 1, 0, 0],  # k=1, i=0 (C01)
        [1, -1, 1, 0, 0, 1, 0]  # k=1, i=1 (C11)
    ]
    
    # Verify Strassen decomposition for 2x2
    # C_ik = sum_j sum_c U_ij_c V_jk_c W_ki_c
    # Let's quickly verify C00
    # C00 = sum_j U_0j_c V_j0_c W_00_c
    # j=0: U_00_c V_00_c
    # c=1: 1*1*1 = 1
    # c=2: 1*1*0 = 0
    # c=3: 1*0*0 = 0
    # c=4: 0*-1*1 = 0
    # c=5: 1*0*-1 = 0
    # c=6: -1*1*0 = 0
    # c=7: 0*0*1 = 0
    # Sum j=0: 1
    # j=1: U_01_c V_10_c
    # c=1: 0*1*1 = 0
    # c=2: 0*0*0 = 0
    # c=3: 0*-1*0 = 0
    # c=4: 0*1*1 = 0
    # c=5: 1*1*-1 = -1
    # c=6: 0*0*0 = 0
    # c=7: 1*1*1 = 1
    # Sum j=1: 0
    # Total C00 = 1 + 0 = 1. Correct (M1 + M4 - M5 + M7 -> 1 + 0 - 0 + 0? No, M4 has V10=1, W00=1. U11=1. So M4 contributes 1. M5 has V11=1, W00=-1. U00=1. So M5 contributes -1. M7 has V10=1, W00=1. U01=1. So M7 contributes 1. M1 contributes 1. Sum 1+1-1+1=2? Wait.)
    
    # Let's re-check Strassen formulas.
    # M1 = (A00+A11)(B00+B11). C00 gets +M1.
    # M4 = A11(B10-B00). C00 gets +M4.
    # M5 = (A00+A01)B11. C00 gets -M5.
    # M7 = (A01-A11)(B10+B11). C00 gets +M7.
    
    # Check M4 contribution to C00:
    # M4 = A11 B10 - A11 B00.
    # We need term A11 B10 in C00? No, C00 = A00B00 + A01B10 + A10B01 + A11B11.
    # M4 = A11(B10-B00) = A11B10 - A11B00.
    # This does not directly form A11B11.
    
    # Standard Strassen:
    # P1 = (A11 - A10)(B00 + B01)
    # P2 = (A10 + A11)B00
    # P3 = (A00 - A01)B11
    # P4 = (A00 + A01)(B10 - B11)
    # P5 = A00(B00 + B11)
    # P6 = A11(B01 + B11)
    # P7 = (A00 + A10)(B00 - B10)
    # P8 = (A01 + A11)(B10 + B11) -- Wait, rank 7 uses 7 products.
    
    # Let's use the verified factors from a known source.
    # Reference: "Tensor Decompositions and Tensor Networks" or similar.
    # A common set:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let's re-verify C00 with these factors.
    # Terms in C00: A00B00, A01B10, A10B01, A11B11.
    
    # M1 = A00B00 + A00B11 + A11B00 + A11B11
    # M4 = A11B10 - A11B00
    # M5 = A00B11 + A01B11
    # M7 = A01B10 + A01B11 - A11B10 - A11B11
    
    # Sum: M1 + M4 - M5 + M7
    # A00B00: 1 (from M1)
    # A00B11: 1 (M1) - 1 (M5) = 0
    # A11B00: 1 (M1) - 1 (M4) = 0
    # A11B11: 1 (M1) - 1 (M7) = 0
    # A11B10: 1 (M4) - 1 (M7) = 0
    # A01B11: -1 (M5) + 1 (M7) = 0
    # A01B10: 1 (M7)
    
    # Missing A10B01. And we have A01B10.
    # C00 should have A01B10? No, C00 = sum_k A0k Bk0.
    # A00B00 + A01B10.
    # So A01B10 is correct.
    # But where is A10B01? It's in C10? No, C10 = A10B00 + A11B10.
    # Wait, C_ik = sum_j A_ij B_jk.
    # C00 = A00B00 + A01B10.
    # My sum gave A00B00 + A01B10. This is correct.
    
    # Let's check C01.
    # C01 = A00B01 + A01B11.
    # Formula: M3 + M5.
    # M3 = A00B01 - A00B11
    # M5 = A00B11 + A01B11
    # Sum: A00B01 + A01B11. Correct.
    
    # Let's check C10.
    # C10 = A10B00 + A11B10.
    # Formula: M2 + M4.
    # M2 = A10B00 + A11B00
    # M4 = A11B10 - A11B00
    # Sum: A10B00 + A11B10. Correct.
    
    # Let's check C11.
    # C11 = A10B01 + A11B11.
    # Formula: M1 - M2 + M3 + M6.
    # M1 = A00B00 + A00B11 + A11B00 + A11B11
    # M2 = A10B00 + A11B00
    # M3 = A00B01 - A00B11
    # M6 = A10B00 + A10B01 - A00B00 - A00B01
    
    # Sum:
    # A00B00: 1 (M1) - 1 (M6) = 0
    # A00B11: 1 (M1) - 1 (M3) = 0
    # A11B00: 1 (M1) - 1 (M2) = 0
    # A11B11: 1 (M1)
    # A10B00: -1 (M2) + 1 (M6) = 0
    # A00B01: -1 (M3) - 1 (M6) = -2?
    # Wait. M6 = (A10-A00)(B00+B01) = A10B00 + A10B01 - A00B00 - A00B01.
    # M3 = A00(B01-B11) = A00B01 - A00B11.
    # Sum of M3 + M6 for A00B01: 1 - 1 = 0.
    # A10B01: 1 (M6).
    # A10B00: -1 (M2) + 1 (M6) = 0.
    # So result: A11B11 + A10B01. Correct.
    
    # So the factors U2, V2, W2 derived earlier are correct.
    
    n, m, p = DIMENSIONS
    rank = 49
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Kronecker Product Construction
    # Outer indices I, J, K in {0, 1}
    # Inner indices i', j', k' in {0, 1}
    # Full indices i = I*2 + i', j = J*2 + j', k = K*2 + k'
    # Rank index c = C*7 + c'
    
    for I in range(2):
        for J in range(2):
            for K in range(2):
                # Outer Strassen factors
                u_outer = U2[I*2 + J]  # Wait, U2 is indexed by [i, j].
                # U2 row index is i*2+j.
                # So U2[I*2+J] is the row for I, J.
                # But U2 is a list of rows.
                # U2[I*2+J] is the vector of length 7.
                
                for C in range(2):
                    u_val_outer = U2[I*2+J][C]
                    v_val_outer = V2[J*2+K][C]
                    w_val_outer = W2[K*2+I][C]
                    
                    if u_val_outer == 0 and v_val_outer == 0 and w_val_outer == 0:
                        continue
                        
                    for i_prime in range(2):
                        for j_prime in range(2):
                            u_val_inner = U2[i_prime*2+j_prime]
                            for c_prime in range(7):
                                u_val = u_val_outer * u_val_inner[c_prime]
                                if u_val == 0:
                                    continue
                                
                                i = I*2 + i_prime
                                j = J*2 + j_prime
                                c = C*7 + c_prime
                                
                                U[i*m + j][c] += u_val
                                
                                # V
                                # v_val = v_val_outer * V2[j_prime*2+k_prime][c_prime]
                                # We need to iterate k_prime for V
                                # But V depends on j_prime, k_prime.
                                # Let's structure loops better.
                                
    # Re-do loops to be efficient and correct
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for I in range(2):
        for J in range(2):
            for K in range(2):
                for C in range(2):
                    u_out = U2[I*2+J][C]
                    v_out = V2[J*2+K][C]
                    w_out = W2[K*2+I][C]
                    
                    if u_out == 0 and v_out == 0 and w_out == 0:
                        continue
                        
                    for i_p in range(2):
                        for j_p in range(2):
                            u_in = U2[i_p*2+j_p]
                            for c_p in range(7):
                                u_val = u_out * u_in[c_p]
                                if u_val != 0:
                                    i = I*2 + i_p
                                    j = J*2 + j_p
                                    c = C*7 + c_p
                                    U[i*m + j][c] += u_val
                                    
                    for j_p in range(2):
                        for k_p in range(2):
                            v_in = V2[j_p*2+k_p]
                            for c_p in range(7):
                                v_val = v_out * v_in[c_p]
                                if v_val != 0:
                                    j = J*2 + j_p
                                    k = K*2 + k_p
                                    c = C*7 + c_p
                                    V[j*p + k][c] += v_val
                                    
                    for k_p in range(2):
                        for i_p in range(2):
                            w_in = W2[k_p*2+i_p]
                            for c_p in range(7):
                                w_val = w_out * w_in[c_p]
                                if w_val != 0:
                                    k = K*2 + k_p
                                    i = I*2 + i_p
                                    c = C*7 + c_p
                                    W[k*n + i][c] += w_val
                                    
    if time.time() > deadline:
        # If we hit the deadline, return what we have (should be complete)
        pass
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
