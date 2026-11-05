def construct():
    n, m, p = 2, 3, 4
    rank = 22
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a rank-1 component
    # u_coeffs: dict {(i,j): val}
    # v_coeffs: dict {(k,l): val}
    # w_coeffs: dict {(i,l): val}
    def add_component(u_coeffs, v_coeffs, w_coeffs, col):
        for (i, j), val in u_coeffs.items():
            U[i * m + j][col] += val
        for (k, l), val in v_coeffs.items():
            V[k * p + l][col] += val
        for (i, l), val in w_coeffs.items():
            W[l * n + i][col] += val
            
    col = 0
    
    # Part 1: C_L1 (cols 0,1) using Strassen on A_L (2x2) and B_T1 (2x2)
    # A_L uses A[0,0], A[0,1], A[1,0], A[1,1]
    # B_T1 uses B[0,0], B[0,1], B[1,0], B[1,1]
    
    # M1 = (A00+A11)(B00+B11)
    add_component({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, col); col+=1
    # M2 = (A10+A11)B00
    add_component({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1}, col); col+=1
    # M3 = A00(B01-B11)
    add_component({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1}, col); col+=1
    # M4 = A11(B10-B00)
    add_component({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):-1, (1,0):1}, col); col+=1
    # M5 = (A00+A01)B11
    add_component({(0,0):1, (0,1):1}, {(1,1):1}, {(0,1):1}, col); col+=1
    # M6 = (A10-A00)(B00+B01)
    add_component({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(0,0):1}, col); col+=1
    # M7 = (A01-A11)(B10+B11)
    add_component({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(1,1):1}, col); col+=1
    
    # Part 2: C_L2 (cols 2,3) using Strassen on A_L (2x2) and B_T2 (2x2)
    # A_L is same.
    # B_T2 uses B[0,2], B[0,3], B[1,2], B[1,3]
    
    # M1' = (A00+A11)(B02+B12)
    add_component({(0,0):1, (1,1):1}, {(0,2):1, (1,2):1}, {(0,2):1, (1,2):1}, col); col+=1
    # M2' = (A10+A11)B02
    add_component({(1,0):1, (1,1):1}, {(0,2):1}, {(1,2):1}, col); col+=1
    # M3' = A00(B03-B13)
    add_component({(0,0):1}, {(0,3):1, (1,3):-1}, {(0,3):1}, col); col+=1
    # M4' = A11(B12-B02)
    add_component({(1,1):1}, {(1,2):1, (0,2):-1}, {(0,2):-1, (1,2):1}, col); col+=1
    # M5' = (A00+A01)B13
    add_component({(0,0):1, (0,1):1}, {(1,3):1}, {(0,3):1}, col); col+=1
    # M6' = (A10-A00)(B02+B03)
    add_component({(1,0):1, (0,0):-1}, {(0,2):1, (0,3):1}, {(0,2):1}, col); col+=1
    # M7' = (A01-A11)(B12+B13)
    add_component({(0,1):1, (1,1):-1}, {(1,2):1, (1,3):1}, {(1,2):1}, col); col+=1

    # Part 3: C_R (all cols) using Outer Product A_R (2x1) and B_B (1x4)
    # A_R uses A[0,2], A[1,2]
    # B_B uses B[2,0], B[2,1], B[2,2], B[2,3]
    
    # C00 += A02 * B20
    add_component({(0,2):1}, {(2,0):1}, {(0,0):1}, col); col+=1
    # C01 += A02 * B21
    add_component({(0,2):1}, {(2,1):1}, {(0,1):1}, col); col+=1
    # C02 += A02 * B22
    add_component({(0,2):1}, {(2,2):1}, {(0,2):1}, col); col+=1
    # C03 += A02 * B23
    add_component({(0,2):1}, {(2,3):1}, {(0,3):1}, col); col+=1
    # C10 += A12 * B20
    add_component({(1,2):1}, {(2,0):1}, {(1,0):1}, col); col+=1
    # C11 += A12 * B21
    add_component({(1,2):1}, {(2,1):1}, {(1,1):1}, col); col+=1
    # C12 += A12 * B22
    add_component({(1,2):1}, {(2,2):1}, {(1,2):1}, col); col+=1
    # C13 += A12 * B23
    add_component({(1,2):1}, {(2,3):1}, {(1,3):1}, col); col+=1

    return {"U": U, "V": V, "W": W}
