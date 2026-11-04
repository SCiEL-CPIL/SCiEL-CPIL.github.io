import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Fallback: Schoolbook
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    fallback = {"U": U_school, "V": V_school, "W": W_school}
    
    # Try Kronecker Product: 3x3 Schoolbook (27) x 2x2 Strassen (7) = 189
    try:
        # 3x3 Schoolbook
        n3, m3, p3 = 3, 3, 3
        R3 = n3 * m3 * p3
        U3 = np.zeros((n3 * m3, R3))
        V3 = np.zeros((m3 * p3, R3))
        W3 = np.zeros((p3 * n3, R3))
        c = 0
        for i in range(n3):
            for j in range(m3):
                for k in range(p3):
                    U3[i * m3 + j, c] = 1
                    V3[j * p3 + k, c] = 1
                    W3[k * n3 + i, c] = 1
                    c += 1

        # 2x2 Strassen
        n2, m2, p2 = 2, 2, 2
        R2 = 7
        U2 = np.zeros((n2 * m2, R2))
        V2 = np.zeros((m2 * p2, R2))
        W2 = np.zeros((p2 * n2, R2))
        
        # M1
        U2[0, 0] = 1; U2[3, 0] = 1
        V2[0, 0] = 1; V2[3, 0] = 1
        W2[0, 0] = 1; W2[3, 0] = 1
        # M2
        U2[2, 1] = 1; U2[3, 1] = 1
        V2[0, 1] = 1
        W2[2, 1] = 1; W2[3, 1] = -1
        # M3
        U2[0, 2] = 1
        V2[2, 2] = 1; V2[3, 2] = -1
        W2[1, 2] = 1; W2[3, 2] = 1
        # M4
        U2[3, 3] = 1
        V2[1, 3] = 1; V2[0, 3] = -1
        W2[0, 3] = 1; W2[2, 3] = 1
        # M5
        U2[0, 4] = 1; U2[2, 4] = 1
        V2[3, 4] = 1
        W2[0, 4] = -1; W2[1, 4] = 1
        # M6
        U2[3, 5] = 1; U2[0, 5] = -1
        V2[0, 5] = 1; V2[2, 5] = 1
        W2[2, 5] = 1; W2[3, 5] = 1 # Check this one
        # M7
        U2[1, 6] = 1; U2[3, 6] = -1
        V2[3, 6] = 1; V2[1, 6] = 1
        W2[0, 6] = -1; W2[3, 6] = 1
        
        # Verify 2x2 Strassen
        # C_ik = sum_j sum_r U_2[i,j,r] V_2[j,k,r] W_2[k,i,r]
        # Check C_00
        # This verification is complex to do inline quickly without numpy arrays for the whole tensor.
        # I will construct the 6x6 and verify against a random 6x6 multiplication.
        
        R_total = R3 * R2
        U_tot = np.zeros((n * m, R_total))
        V_tot = np.zeros((m * p, R_total))
        W_tot = np.zeros((p * n, R_total))
        
        for i3 in range(n3):
            for j3 in range(m3):
                for k3 in range(p3):
                    for i2 in range(n2):
                        for j2 in range(m2):
                            for k2 in range(p2):
                                # Indices in 6x6
                                i = i3 * 2 + i2
                                j = j3 * 2 + j2
                                k = k3 * 2 + k2
                                
                                # Rank index
                                r3 = i3 * 9 + j3 * 3 + k3
                                r2 = 0 # Loop over r2? No, r2 is the rank index of the 2x2 part.
                                # The Kronecker product sums over r2? No, the rank is the product.
                                # The term is U3[i3,j3,r3] * U2[i2,j2,r2]
                                # So we iterate r2 from 0 to 6.
                                pass
        
        # Correct Construction Loop
        for r3 in range(R3):
            for r2 in range(R2):
                r_tot = r3 * R2 + r2
                
                # U part
                # U3 is non-zero only if r3 corresponds to specific i3, j3, k3?
                # In schoolbook, U3[i3, j3, r3] is 1 only if r3 = i3*9 + j3*3 + k3?
                # No, in my U3 construction, U3[i3*m3+j3, r3] is 1 for the specific k3?
                # My U3 construction:
                # for i, j, k: U3[i*m+j, c] = 1.
                # So for a fixed r3, there is exactly one (i3, j3, k3) such that U3 is non-zero?
                # No, U3[i3*m3+j3, r3] is 1.
                # It is 1 for the specific i3, j3 associated with r3?
                # r3 = i3*9 + j3*3 + k3.
                # So for a given r3, i3, j3, k3 are fixed.
                # U3[i3*m3+j3, r3] = 1.
                # U3 is 0 elsewhere for this column r3.
                
                # So for each r3, we know i3, j3, k3.
                i3 = r3 // 9
                j3 = (r3 % 9) // 3
                k3 = r3 % 3
                
                # Now for this r3, we need to fill U_tot, V_tot, W_tot for all i2, j2, k2, r2.
                # U_tot row is (i3*2+i2, j3*2+j2).
                # U_tot[row, r_tot] = U3[i3, j3, r3] * U2[i2, j2, r2]
                # Since U3[i3, j3, r3] = 1:
                for i2 in range(2):
                    for j2 in range(2):
                        row_u = (i3 * 2 + i2) * 6 + (j3 * 2 + j2)
                        for r2 in range(7):
                            val = U2[i2 * 2 + j2, r2]
                            if val != 0:
                                U_tot[row_u, r3 * 7 + r2] += val
                                
                # V_tot row is (j3*2+j2, k3*2+k2).
                # V_tot[row, r_tot] = V3[j3, k3, r3] * V2[j2, k2, r2]
                # V3[j3, k3, r3] = 1.
                for j2 in range(2):
                    for k2 in range(2):
                        row_v = (j3 * 2 + j2) * 6 + (k3 * 2 + k2)
                        for r2 in range(7):
                            val = V2[j2 * 2 + k2, r2]
                            if val != 0:
                                V_tot[row_v, r3 * 7 + r2] += val
                                
                # W_tot row is (k3*2+k2, i3*2+i2).
                # W_tot[row, r_tot] = W3[k3, i3, r3] * W2[k2, i2, r2]
                # W3[k3, i3, r3] = 1.
                for k2 in range(2):
                    for i2 in range(2):
                        row_w = (k3 * 2 + k2) * 6 + (i3 * 2 + i2)
                        for r2 in range(7):
                            val = W2[k2 * 2 + i2, r2]
                            if val != 0:
                                W_tot[row_w, r3 * 7 + r2] += val
                                
        # Verify
        A = np.random.rand(6, 6)
        B = np.random.rand(6, 6)
        C_expected = A @ B
        
        C_calc = np.zeros((6, 6))
        for i in range(6):
            for j in range(6):
                for k in range(6):
                    for r in range(189):
                        C_calc[i, k] += U_tot[i * 6 + j, r] * V_tot[j * 6 + k, r] * W_tot[k * 6 + i, r]
                        
        if not np.allclose(C_expected, C_calc, atol=1e-9):
            return fallback
            
        return {
            "U": U_tot.tolist(),
            "V": V_tot.tolist(),
            "W": W_tot.tolist()
        }
    except Exception:
        return fallback
# EVOLVE-BLOCK-END
