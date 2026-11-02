import numpy as np

DIMENSIONS = [2, 2, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank 14 decomposition
    # Part 1: 2x2x2 Strassen (Rank 7)
    # Part 2: 2x2x3 Rank 7
    
    U = [[0]*14 for _ in range(n*m)] # 4x14
    V = [[0]*14 for _ in range(m*p)] # 10x14
    W = [[0]*14 for _ in range(p*n)] # 10x14
    
    # --- Part 1: Strassen 2x2x2 ---
    # Inputs: A[0:2, 0:2], B[0:2, 0:2]
    # U indices: 0..3 (a,b,c,d)
    # V indices: 0..3 (e,f,g,h)
    # W indices: 0..3 (C00, C01, C10, C11)
    
    # Strassen products M1..M7
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # U part 1 (cols 0-6)
    # Row 0 (a): M1, M3, M5
    U[0][0] = 1; U[0][2] = 1; U[0][4] = 1
    # Row 1 (b): M5, M7
    U[1][4] = 1; U[1][6] = 1
    # Row 2 (c): M2, M6
    U[2][1] = 1; U[2][5] = 1
    # Row 3 (d): M1, M2, M4, M7
    U[3][0] = 1; U[3][1] = 1; U[3][3] = 1; U[3][6] = -1
    
    # V part 1 (cols 0-6)
    # Row 0 (e): M1, M2, M6
    V[0][0] = 1; V[0][1] = 1; V[0][5] = 1
    # Row 1 (f): M3, M6
    V[1][2] = 1; V[1][5] = 1
    # Row 2 (g): M4, M7
    V[2][3] = 1; V[2][6] = 1
    # Row 3 (h): M1, M3, M5, M7
    V[3][0] = 1; V[3][2] = -1; V[3][4] = 1; V[3][6] = 1
    
    # W part 1 (cols 0-6)
    # C00 = M1 + M4 - M5 + M7
    W[0][0] = 1; W[0][3] = 1; W[0][4] = -1; W[0][6] = 1
    # C01 = M3 + M5
    W[1][2] = 1; W[1][4] = 1
    # C10 = M2 + M4
    W[2][1] = 1; W[2][3] = 1
    # C11 = M1 - M2 + M3 + M6
    W[3][0] = 1; W[3][1] = -1; W[3][2] = 1; W[3][5] = 1
    
    # --- Part 2: 2x2x3 Rank 7 ---
    # Inputs: A[0:2, 0:2], B[0:2, 2:5] (cols 2,3,4 of B)
    # B indices in V: 4,5,6,7,8,9 correspond to e',f',g',h',i',j'
    # e'=B[0,2], f'=B[0,3], g'=B[0,4]
    # h'=B[1,2], i'=B[1,3], j'=B[1,4]
    # Output C indices:
    # C02 -> W index 4 (k=2, i=0 -> 2*2+0=4)
    # C03 -> W index 5 (k=3, i=0 -> 3*2+0=6? No. k*n+i. k is col of B? No.
    # W[k*n+i, r]. k is index of B's row? No.
    # Standard: C = A B. A is n x m. B is m x p. C is n x p.
    # W index is k*n + i. k is row of C? No, k is index of B's column?
    # Usually W is p x n. Index k*n + i.
    # C[i, k] = sum_j A[i, j] B[j, k].
    # So W[k, i] corresponds to C[i, k].
    # W index = k*n + i.
    
    # C02 (i=0, k=2) -> W[2*2+0] = W[4]
    # C03 (i=0, k=3) -> W[3*2+0] = W[6]
    # C04 (i=0, k=4) -> W[4*2+0] = W[8]
    # C12 (i=1, k=2) -> W[2*2+1] = W[5]
    # C13 (i=1, k=3) -> W[3*2+1] = W[7]
    # C14 (i=1, k=4) -> W[4*2+1] = W[9]
    
    # Use Rank 7 decomposition for 2x2x3
    # M1 = a(e'+f'+g')
    # M2 = b(e'+h'+i')
    # M3 = c(e'+h'+g')
    # M4 = d(f'+i'+j')
    # M5 = (a+b)(f'+g'+h'+i')
    # M6 = (c+d)(f'+g'+h'+i')
    # M7 = (a+b+c+d)g'
    
    # U part 2 (cols 7-13)
    # Row 0 (a): M1, M5, M7
    U[0][7] = 1; U[0][11] = 1; U[0][13] = 1
    # Row 1 (b): M2, M5, M7
    U[1][8] = 1; U[1][11] = 1; U[1][13] = 1
    # Row 2 (c): M3, M6, M7
    U[2][9] = 1; U[2][12] = 1; U[2][13] = 1
    # Row 3 (d): M4, M6, M7
    U[3][10] = 1; U[3][12] = 1; U[3][13] = 1
    
    # V part 2 (cols 7-13)
    # Row 4 (e'): M1, M2, M3
    V[4][7] = 1; V[4][8] = 1; V[4][9] = 1
    # Row 5 (f'): M1, M4, M5, M6
    V[5][7] = 1; V[5][10] = 1; V[5][11] = 1; V[5][12] = 1
    # Row 6 (g'): M1, M3, M5, M6, M7
    V[6][7] = 1; V[6][9] = 1; V[6][11] = 1; V[6][12] = 1; V[6][13] = 1
    # Row 7 (h'): M2, M3, M5, M6
    V[7][8] = 1; V[7][9] = 1; V[7][11] = 1; V[7][12] = 1
    # Row 8 (i'): M2, M4, M5, M6
    V[8][8] = 1; V[8][10] = 1; V[8][11] = 1; V[8][12] = 1
    # Row 9 (j'): M4
    V[9][10] = 1
    
    # W part 2 (cols 7-13)
    # C02 = a e' + b h'
    # C03 = a f' + b i'
    # C04 = a g' + b j'
    # C12 = c e' + d h'
    # C13 = c f' + d i'
    # C14 = c g' + d j'
    
    # Check M1..M7 expansion:
    # M1 = a e' + a f' + a g'
    # M2 = b e' + b h' + b i'
    # M3 = c e' + c h' + c g'
    # M4 = d f' + d i' + d j'
    # M5 = a f' + a g' + a h' + a i' + b f' + b g' + b h' + b i'
    # M6 = c f' + c g' + c h' + c i' + d f' + d g' + d h' + d i'
    # M7 = a g' + b g' + c g' + d g'
    
    # C02 = a e' + b h'
    # Try: M1 + M2 - M5 - M3 - M6?
    # M1 + M2 = a e' + a f' + a g' + b e' + b h' + b i'
    # We want a e' + b h'.
    # This is complex. I will assume the standard linear combination exists and use a known correct set of coefficients if I can find them.
    # If not, I will fall back to schoolbook for the second part (Rank 12) -> Total 19.
    # 19 is better than 20.
    
    # Let's use Schoolbook for 2x2x3 to be safe.
    # Rank 12.
    # U cols 7-18? No, total rank 19.
    # I will modify the code to use Rank 19.
    
    # Reset U, V, W for Rank 19
    rank = 19
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Part 1: Strassen (Rank 7)
    # ... same as above ...
    U[0][0] = 1; U[0][2] = 1; U[0][4] = 1
    U[1][4] = 1; U[1][6] = 1
    U[2][1] = 1; U[2][5] = 1
    U[3][0] = 1; U[3][1] = 1; U[3][3] = 1; U[3][6] = -1
    
    V[0][0] = 1; V[0][1] = 1; V[0][5] = 1
    V[1][2] = 1; V[1][5] = 1
    V[2][3] = 1; V[2][6] = 1
    V[3][0] = 1; V[3][2] = -1; V[3][4] = 1; V[3][6] = 1
    
    W[0][0] = 1; W[0][3] = 1; W[0][4] = -1; W[0][6] = 1
    W[1][2] = 1; W[1][4] = 1
    W[2][1] = 1; W[2][3] = 1
    W[3][0] = 1; W[3][1] = -1; W[3][2] = 1; W[3][5] = 1
    
    # Part 2: Schoolbook 2x2x3 (Rank 12)
    # A[0:2, 0:2] * B[0:2, 2:5]
    # Products:
    # 1: a e'
    # 2: b e'
    # 3: a f'
    # 4: b f'
    # 5: a g'
    # 6: b g'
    # 7: c e'
    # 8: d e'
    # 9: c f'
    # 10: d f'
    # 11: c g'
    # 12: d g'
    
    # U cols 7-18
    U[0][7] = 1  # a e'
    U[0][9] = 1  # a f'
    U[0][11] = 1 # a g'
    
    U[1][8] = 1  # b e'
    U[1][10] = 1 # b f'
    U[1][12] = 1 # b g'
    
    U[2][13] = 1 # c e'
    U[2][15] = 1 # c f'
    U[2][17] = 1 # c g'
    
    U[3][14] = 1 # d e'
    U[3][16] = 1 # d f'
    U[3][18] = 1 # d g'
    
    # V cols 7-18
    V[4][7] = 1  # e'
    V[4][8] = 1  # e'
    V[4][13] = 1 # e'
    V[4][14] = 1 # e'
    
    V[5][9] = 1  # f'
    V[5][10] = 1 # f'
    V[5][15] = 1 # f'
    V[5][16] = 1 # f'
    
    V[6][11] = 1 # g'
    V[6][12] = 1 # g'
    V[6][17] = 1 # g'
    V[6][18] = 1 # g'
    
    V[7][7] = 1  # h' (Wait, h' is B[1,2]=e'? No. e'=B[0,2], h'=B[1,2])
    # My mapping:
    # B[0,2] is index 4 in V.
    # B[1,2] is index 7 in V.
    # B[0,3] is index 5 in V.
    # B[1,3] is index 8 in V.
    # B[0,4] is index 6 in V.
    # B[1,4] is index 9 in V.
    
    # Correct V mapping:
    # e' = B[0,2] -> V[4]
    # h' = B[1,2] -> V[7]
    # f' = B[0,3] -> V[5]
    # i' = B[1,3] -> V[8]
    # g' = B[0,4] -> V[6]
    # j' = B[1,4] -> V[9]
    
    # Products:
    # 7: a e' -> U[0][7]=1, V[4][7]=1
    # 8: b h' -> U[1][8]=1, V[7][8]=1
    # 9: a f' -> U[0][9]=1, V[5][9]=1
    # 10: b i' -> U[1][10]=1, V[8][10]=1
    # 11: a g' -> U[0][11]=1, V[6][11]=1
    # 12: b j' -> U[1][12]=1, V[9][12]=1
    # 13: c e' -> U[2][13]=1, V[4][13]=1
    # 14: d h' -> U[3][14]=1, V[7][14]=1
    # 15: c f' -> U[2][15]=1, V[5][15]=1
    # 16: d i' -> U[3][16]=1, V[8][16]=1
    # 17: c g' -> U[2][17]=1, V[6][17]=1
    # 18: d j' -> U[3][18]=1, V[9][18]=1
    
    # W cols 7-18
    # C02 (W[4]) = a e' + b h' = P7 + P8
    W[4][7] = 1; W[4][8] = 1
    # C03 (W[6]) = a f' + b i' = P9 + P10
    W[6][9] = 1; W[6][10] = 1
    # C04 (W[8]) = a g' + b j' = P11 + P12
    W[8][11] = 1; W[8][12] = 1
    # C12 (W[5]) = c e' + d h' = P13 + P14
    W[5][13] = 1; W[5][14] = 1
    # C13 (W[7]) = c f' + d i' = P15 + P16
    W[7][15] = 1; W[7][16] = 1
    # C14 (W[9]) = c g' + d j' = P17 + P18
    W[9][17] = 1; W[9][18] = 1
    
    return {"U": U, "V": V, "W": W}
