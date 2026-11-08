import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank 105
    rank = 105
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(i, j, k, u_val, v_val, w_val):
        nonlocal col
        if col >= rank:
            return
        U[i*m + j][col] += u_val
        V[j*p + k][col] += v_val
        W[k*n + i][col] += w_val
        col += 1

    # Part 1: 4x4x4 using Strassen
    # Strassen factors for 4x4x4
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # The 7 terms for 2x2x2 Strassen
    # Let A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need to map the 2x2x2 terms to the 4x4x4 indices.
    # The 2x2x2 terms are defined for a single 2x2 block multiplication.
    # We will define a function that adds the 7 terms for a 2x2x2 multiplication
    # where A is at (i0, j0), B is at (j0, k0), C is at (k0, i0) in the 4x4 grid.
    # But the A and B blocks are linear combinations.
    
    # Let's define the 7 terms for 2x2x2 explicitly.
    # Term 1: U=(1,0,0,1), V=(1,0,0,1), W=(1,1,1,-1) ? No.
    # The W vector for a term is the contribution to C.
    # For M1, the contribution to C is:
    # C11 += 1, C12 += 0, C21 += 0, C22 += 1? No.
    # M1 contributes to C11 with 1, C22 with 1.
    # So W for M1 term is (1, 0, 0, 1) in C indices (C11, C12, C21, C22)?
    # C indices: 00, 01, 10, 11.
    # W1 = [1, 0, 0, 1]
    # W2 = [0, 0, 1, -1] (M2: C21+=1, C22-=1)
    # W3 = [0, 1, 0, 1] (M3: C12+=1, C22+=1)
    # W4 = [1, 0, 1, 0] (M4: C11+=1, C21+=1)
    # W5 = [-1, 1, 0, 0] (M5: C11-=1, C12+=1)
    # W6 = [0, 0, 0, 1] (M6: C22+=1)
    # W7 = [1, 0, 0, 0] (M7: C11+=1)
    
    # Let's verify:
    # C11 = W1[0] + W4[0] - W5[0] + W7[0] = 1 + 1 - (-1) + 1 = 4? No.
    # My W vectors are wrong.
    
    # Let's re-derive.
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For each M_l, we have 7 rank-1 terms.
    # Let the 7 terms for M_l be (U_l, V_l, W_l).
    # Then the total term is sum_l sum_r U_l,r V_l,r W_l,r.
    # But W_l,r is the contribution to the 2x2 output of M_l.
    # Then we need to map the 2x2 output of M_l to the 4x4 output C.
    
    # This is getting complex. Let's just hardcode the 49 terms for 4x4x4 Strassen.
    # Or better, write a small script to generate them.
    
    pass
