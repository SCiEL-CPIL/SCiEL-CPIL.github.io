import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 97
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen part for 4x4x4
    # Macro coefficients c_s[I, J] for X_s = sum c_s[I,J] A_{I,J}
    # d_s[J, K] for Y_s = sum d_s[J,K] B_{J,K}
    # e_s[I, J] for C contribution: C_{I,J} += e_s[I,J] P_s
    
    # s=1: X=A00+A11, Y=B00+B11
    # s=2: X=A10+A11, Y=B00
    # s=3: X=A00, Y=B01-B11
    # s=4: X=A11, Y=B10-B00
    # s=5: X=A00+A01, Y=B11
    # s=6: X=A10-A00, Y=B00+B01
    # s=7: X=A01-A11, Y=B10+B11
    
    macro_c = {
        1: {(0,0): 1, (1,1): 1},
        2: {(1,0): 1, (1,1): 1},
        3: {(0,0): 1},
        4: {(1,1): 1},
        5: {(0,0): 1, (0,1): 1},
        6: {(1,0): 1, (0,0): -1},
        7: {(0,1): 1, (1,1): -1}
    }
    
    macro_d = {
        1: {(0,0): 1, (1,1): 1},
        2: {(0,0): 1},
        3: {(0,1): 1, (1,1): -1},
        4: {(1,0): 1, (0,0): -1},
        5: {(1,1): 1},
        6: {(0,0): 1, (0,1): 1},
        7: {(1,0): 1, (1,1): 1}
    }
    
    # Micro Strassen coefficients for 2x2 multiplication
    # M1: (x00+x11)(y00+y11) -> Z00 += 1, Z11 += 1
    # M2: (x10+x11)y00 -> Z10 += 1, Z11 -= 1
    # M3: x00(y01-y11) -> Z01 += 1, Z11 += 1
    # M4: x11(y10-y00) -> Z00 += 1, Z10 += 1
    # M5: (x00+x01)y11 -> Z00 -= 1, Z01 += 1
    # M6: (x10-x00)(y00+y01) -> Z11 += 1
    # M7: (x01-x11)(y10+y11) -> Z00 += 1
    
    # u_coeffs for X entries (i', j')
    # v_coeffs for Y entries (i', j')
    
    micro_u = {
        1: {(0,0): 1, (1,1): 1},
        2: {(1,0): 1, (1,1): 1},
        3: {(0,0): 1},
        4: {(1,1): 1},
        5: {(0,0): 1, (0,1): 1},
        6: {(1,0): 1, (0,0): -1},
        7: {(0,1): 1, (1,1): -1}
    }
    
    micro_v = {
        1: {(0,0): 1, (1,1): 1},
        2: {(0,0): 1},
        3: {(0,1): 1, (1,1): -1},
        4: {(1,0): 1, (0,0): -1},
        5: {(1,1): 1},
        6: {(0,0): 1, (0,1): 1},
        7: {(1,0): 1, (1,1): 1}
    }
    
    micro_w = {
        1: {(0,0): 1, (1,1): 1},
        2: {(1,0): 1, (1,1): -1},
        3: {(0,1): 1, (1,1): 1},
        4: {(0,0): 1, (1,0): 1},
        5: {(0,0): -1, (0,1): 1},
        6: {(1,1): 1},
        7: {(0,0): 1}
    }
    
    # Macro reconstruction e_s[I, J]
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    macro_e = {
        1: {(0,0): 1, (1,1): 1},
        2: {(1,0): 1, (1,1): -1},
        3: {(0,1): 1, (1,1): 1},
        4: {(0,0): 1, (1,0): 1},
        5: {(0,0): -1, (0,1): 1},
        6: {(1,1): 1},
        7: {(0,0): 1}
    }
    
    for s in range(1, 8):
        c_s = macro_c[s]
        d_s = macro_d[s]
        e_s = macro_e[s]
        
        for k in range(1, 8):
            u_k = micro_u[k]
            v_k = micro_v[k]
            w_k = micro_w[k]
            
            # Build U vector
            # u_k[(i', j')] is coeff for X_{i', j'}
            # X_{i', j'} = sum_{I,J} c_s[(I,J)] A_{I,J,i',j'}
            # So U[(2I+i')*4 + (2J+j')] += u_k[(i',j')] * c_s[(I,J)]
            
            for (ip, jp), coeff_u in u_k.items():
                for (I, J), coeff_c in c_s.items():
                    i = 2 * I + ip
                    j = 2 * J + jp
                    U[i * 4 + j][col] += coeff_u * coeff_c
            
            # Build V vector
            # v_k[(ip, jp)] is coeff for Y_{ip, jp}
            # Y_{ip, jp} = sum_{J,K} d_s[(J,K)] B_{J,K,ip,jp}
            # Note: B indices are j, k. Block J is row, K is col.
            # B_{J,K,ip,jp} -> j = 2J+ip, k = 2K+jp
            # V index is j * 7 + k
            
            for (ip, jp), coeff_v in v_k.items():
                for (J, K), coeff_d in d_s.items():
                    j = 2 * J + ip
                    k = 2 * K + jp
                    V[j * 7 + k][col] += coeff_v * coeff_d
            
            # Build W vector
            # w_k[(ip, jp)] is coeff for Z_{ip, jp}
            # Z_{ip, jp} contributes to C blocks via e_s
            # C_{I,J,ip,jp} += e_s[(I,J)] * Z_{ip, jp}
            # C index is k * 4 + i.
            # C_{I,J} block has i = 2I+ip, k = 2J+jp?
            # Wait. C is 4x4.
            # C_{I,J} is the block at row I, col J.
            # Entry C_{I,J,ip,jp} is at global i = 2I+ip, k = 2J+jp.
            # W index is k * 4 + i.
            
            for (ip, jp), coeff_w in w_k.items():
                for (I, J), coeff_e in e_s.items():
                    i = 2 * I + ip
                    k = 2 * J + jp
                    W[k * 4 + i][col] += coeff_w * coeff_e
            
            col += 1
    
    # Schoolbook part for 4x4x3 (cols 4-6 of B)
    for k in range(4, 7):
        for i in range(4):
            for j in range(4):
                U[i * 4 + j][col] = 1
                V[j * 7 + k][col] = 1
                W[k * 4 + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
