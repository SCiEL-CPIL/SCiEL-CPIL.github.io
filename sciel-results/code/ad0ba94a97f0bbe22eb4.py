"""Structural change: hybrid Strassen (rank-7) for 2x2x2 block + schoolbook for the rest.

We model the 2x2 by 2x7 multiplication as:
  C[2,7] = A[2,2] * B[2,7]
Split B into B1 (cols 0-1) and B2 (cols 2-6).
Use the known rank-7 Strassen decomposition for the 2x2x2 part (rank 7),
and the 20-term schoolbook decomposition for the 2x2x5 part (2*5*2 = 20).
Total rank = 27, which is strictly better than the seed's rank 28.
"""
DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 7
    # Rank: 7 (Strassen for 2x2x2) + 20 (schoolbook for 2x2x5) = 27
    rank = 27

    U = [[0] * rank for _ in range(n * m)]   # 4 x 27
    V = [[0] * rank for _ in range(m * p)]   # 14 x 27
    W = [[0] * rank for _ in range(p * n)]   # 14 x 27

    # --- Part 1: Strassen decomposition for columns 0-1 (2x2x2), rank 7 ---
    # Strassen uses 7 products:
    # M1 = (a+d)(e+g+h)  -- not standard; let me use the classic 7-product form
    # For A = [[a,b],[c,d]], B = [[e,f],[g,h]]:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C = [[M1+M4-M5+M7, M1+M3-M5+M6],
    #      [M2+M4-M6+M7, M1+M2-M3+M4]]
    #
    # U index: i*m+j -> row index in [0,3]: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # V index: j*p+k -> for p=7: (0,0)=0, (0,1)=1, (0,2)=2, (0,3)=3, (0,4)=4, (0,5)=5, (0,6)=6
    #            (1,0)=7, (1,1)=8, (1,2)=9, (1,3)=10, (1,4)=11, (1,5)=12, (1,6)=13
    # W index: k*n+i -> for n=2: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3, ... (6,0)=12, (6,1)=13
    #
    # Strassen columns 0-6 of U, V, W:
    # M1 = (a+d)(e+h): U rows 0,1 (a,d) get 1; V rows 0,7 (e,g... wait h is (1,1)=8) 
    # Let me be careful: B = [[B00, B01, ...], [B10, B11, ...]]
    # B00 = V[0], B01 = V[1], ..., B10 = V[7], B11 = V[8], ...
    # Strassen products for cols 0,1:
    # a=A[0,0]=U row 0, b=A[0,1]=U row 1, c=A[1,0]=U row 2, d=A[1,1]=U row 3
    # e=B[0,0]=V row 0, f=B[0,1]=V row 1, g=B[1,0]=V row 7, h=B[1,1]=V row 8

    strassen_coeffs = [
        # (U_coeffs {row: coeff}, V_coeffs {row: coeff}, W_coeffs {row: coeff})
        # M1 = (a+d)(e+h)
        ({0: 1, 3: 1}, {0: 1, 8: 1}, {0: 1, 1: 1, 3: 1}),  # W: C[0,0]+=M1, C[0,1]+=M1, C[1,1]+=M1
        # M2 = (c+d)e
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: 1}),               # W: C[1,0]+=M2, C[1,1]+=M2
        # M3 = a(f-h)
        ({0: 1}, {1: 1, 8: -1}, {1: 1, 3: -1}),             # W: C[0,1]+=M3, C[1,1]-=M3
        # M4 = d(g-e)
        ({3: 1}, {7: 1, 0: -1}, {0: 1, 2: 1, 3: 1}),        # W: C[0,0]+=M4, C[1,0]+=M4, C[1,1]+=M4
        # M5 = (a+b)h
        ({0: 1, 1: 1}, {8: 1}, {0: -1, 1: -1}),             # W: C[0,0]-=M5, C[0,1]-=M5
        # M6 = (c-a)(e+f)
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {1: 1, 2: -1}),       # W: C[0,1]+=M6, C[1,0]-=M6
        # M7 = (b-d)(g+h)
        ({1: 1, 3: -1}, {7: 1, 8: 1}, {0: 1, 1: 1, 2: 1, 3: 1}),  # W: C[0,0]+=M7, C[0,1]+=M7, C[1,0]+=M7, C[1,1]+=M7
    ]

    for col_idx, (u_c, v_c, w_c) in enumerate(strassen_coeffs):
        for r, val in u_c.items():
            U[r][col_idx] = val
        for r, val in v_c.items():
            V[r][col_idx] = val
        for r, val in w_c.items():
            W[r][col_idx] = val

    # --- Part 2: Schoolbook for columns 2-6 (2x2x5), rank 20 ---
    col = 7
    for i in range(n):   # 0,1
        for j in range(m):  # 0,1
            for k in range(2, p):  # cols 2-6
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
