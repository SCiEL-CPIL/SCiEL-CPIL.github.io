"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # Laderman's algorithm for 3x3 matrix multiplication (rank 23)
    # Each step: (A_coeffs, B_coeffs, C_coeffs)
    # A_coeffs: dict {(i,j): coeff} for A[i,j]
    # B_coeffs: dict {(j,k): coeff} for B[j,k]
    # C_coeffs: dict {(i,k): coeff} for C[i,k]
    
    laderman_steps = [
        # Step 1: M1 = (A00+A11-A22)(B00-B11+B22)
        {
            'A': {(0,0):1, (1,1):1, (2,2):-1},
            'B': {(0,0):1, (1,1):-1, (2,2):1},
            'C': {(0,0):1, (1,1):-1, (2,2):-1, (1,0):-1, (0,1):-1, (2,1):-1, (1,2):-1, (0,2):-1, (2,0):-1}
        },
        # Step 2: M2 = A00(B01-B11)
        {
            'A': {(0,0):1},
            'B': {(0,1):1, (1,1):-1},
            'C': {(0,1):1, (1,1):1}
        },
        # Step 3: M3 = A11(B12-B02)
        {
            'A': {(1,1):1},
            'B': {(1,2):1, (0,2):-1},
            'C': {(1,2):1, (0,2):1}
        },
        # Step 4: M4 = A22(B20-B10)
        {
            'A': {(2,2):1},
            'B': {(2,0):1, (1,0):-1},
            'C': {(2,0):1, (1,0):1}
        },
        # Step 5: M5 = A01(B10+B20)
        {
            'A': {(0,1):1},
            'B': {(1,0):1, (2,0):1},
            'C': {(0,0):-1, (1,0):-1}
        },
        # Step 6: M6 = A12(B01+B11)
        {
            'A': {(1,2):1},
            'B': {(0,1):1, (1,1):1},
            'C': {(1,1):-1, (0,1):-1}
        },
        # Step 7: M7 = A20(B21+B01)
        {
            'A': {(2,0):1},
            'B': {(2,1):1, (0,1):1},
            'C': {(2,1):-1, (1,1):-1}
        },
        # Step 8: M8 = A02(B12+B22)
        {
            'A': {(0,2):1},
            'B': {(1,2):1, (2,2):1},
            'C': {(0,2):-1, (1,2):-1}
        },
        # Step 9: M9 = A10(B02+B12)
        {
            'A': {(1,0):1},
            'B': {(0,2):1, (1,2):1},
            'C': {(1,2):-1, (0,2):-1}
        },
        # Step 10: M10 = A21(B20+B00)
        {
            'A': {(2,1):1},
            'B': {(2,0):1, (0,0):1},
            'C': {(2,0):-1, (1,0):-1}
        },
        # Step 11: M11 = A00B22
        {
            'A': {(0,0):1},
            'B': {(2,2):1},
            'C': {(0,2):1, (2,0):1}
        },
        # Step 12: M12 = A11B00
        {
            'A': {(1,1):1},
            'B': {(0,0):1},
            'C': {(1,0):1, (0,1):1}
        },
        # Step 13: M13 = A22B11
        {
            'A': {(2,2):1},
            'B': {(1,1):1},
            'C': {(2,1):1, (1,2):1}
        },
        # Step 14: M14 = A01B12
        {
            'A': {(0,1):1},
            'B': {(1,2):1},
            'C': {(0,2):1, (1,1):1}
        },
        # Step 15: M15 = A12B20
        {
            'A': {(1,2):1},
            'B': {(2,0):1},
            'C': {(1,0):1, (2,1):1}
        },
        # Step 16: M16 = A20B01
        {
            'A': {(2,0):1},
            'B': {(0,1):1},
            'C': {(2,1):1, (0,2):1}
        },
        # Step 17: M17 = A02B21
        {
            'A': {(0,2):1},
            'B': {(2,1):1},
            'C': {(0,1):1, (2,0):1}
        },
        # Step 18: M18 = A10B10
        {
            'A': {(1,0):1},
            'B': {(1,0):1},
            'C': {(1,0):1}
        },
        # Step 19: M19 = A21B11
        {
            'A': {(2,1):1},
            'B': {(1,1):1},
            'C': {(2,1):1}
        },
        # Step 20: M20 = A02B02
        {
            'A': {(0,2):1},
            'B': {(0,2):1},
            'C': {(0,2):1}
        },
        # Step 21: M21 = A10B00
        {
            'A': {(1,0):1},
            'B': {(0,0):1},
            'C': {(1,0):1}
        },
        # Step 22: M22 = A21B22
        {
            'A': {(2,1):1},
            'B': {(2,2):1},
            'C': {(2,1):1}
        },
        # Step 23: M23 = A02B11
        {
            'A': {(0,2):1},
            'B': {(1,1):1},
            'C': {(0,1):1}
        },
    ]
    
    rank = len(laderman_steps)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, step in enumerate(laderman_steps):
        A_coeffs = step['A']
        B_coeffs = step['B']
        C_coeffs = step['C']
        
        for u in range(3):
            for v in range(3):
                for w in range(3):
                    # U part
                    for (a, b), coeff_a in A_coeffs.items():
                        i = a * 3 + u
                        j = b * 3 + w
                        U[i * 6 + j][col] += coeff_a
                    
                    # V part
                    for (b, g), coeff_b in B_coeffs.items():
                        j = b * 3 + w
                        k = g * 3 + v
                        V[j * 6 + k][col] += coeff_b
                    
                    # W part
                    for (a, g), coeff_c in C_coeffs.items():
                        k = g * 3 + v
                        i = a * 3 + u
                        W[k * 6 + i][col] += coeff_c
                        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
