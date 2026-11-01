import numpy as np

DIMENSIONS = [2, 2, 13]

def construct():
    n, m, p = DIMENSIONS
    # We will use Strassen for pairs of columns and schoolbook for the last one.
    # Strassen rank for 2x2x2 is 7.
    # Schoolbook rank for 2x2x1 is 4.
    # p=13: 6 pairs (12 cols) + 1 single.
    # Rank = 6 * 7 + 4 = 46.
    
    rank = 6 * 7 + 4
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen decomposition for 2x2 matrix multiplication
    # C = A * B
    # A = [[a, b], [c, d]]
    # B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (d-b)(g+h)
    # C11 = M1 - M4 + M5 - M7
    # C12 = M3 + M4 + M7
    # C21 = M2 + M5 - M1 + M6
    # C22 = M1 - M3 + M2 - M6
    
    # Indices for A: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Indices for B: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) relative to block
    # Indices for C: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) relative to block
    
    # Strassen U, V, W vectors for the 7 multiplications
    # U is length 4, V is length 4, W is length 4
    # For each multiplication r, we have U_r, V_r, W_r
    
    strassen_U = [
        [1, 0, 0, 1], # a+d
        [0, 0, 1, 1], # c+d
        [1, 0, 0, 0], # a
        [0, 0, 0, 1], # d
        [1, 1, 0, 0], # a+b
        [0, 0, 1, -1],# c-a
        [0, 0, 0, 1]  # d-b ... wait, d-b is [0,0,0,1] - [0,1,0,0]? No.
    ]
    # Let's write them explicitly
    # A indices: 0=a, 1=b, 2=c, 3=d
    U_s = [
        [1, 0, 0, 1], # 1: a+d
        [0, 0, 1, 1], # 2: c+d
        [1, 0, 0, 0], # 3: a
        [0, 0, 0, 1], # 4: d
        [1, 1, 0, 0], # 5: a+b
        [0, 0, 1, -1],# 6: c-a
        [0, -1, 0, 1] # 7: d-b
    ]
    
    # B indices: 0=e, 1=f, 2=g, 3=h
    V_s = [
        [1, 0, 0, 1], # 1: e+h
        [1, 0, 0, 0], # 2: e
        [0, 1, 0, -1],# 3: f-h
        [0, -1, 1, 0],# 4: g-e
        [0, 0, 0, 1], # 5: h
        [1, 1, 0, 0], # 6: e+f
        [0, 0, 1, 1]  # 7: g+h
    ]
    
    # C indices: 0=c11, 1=c12, 2=c21, 3=c22
    W_s = [
        [1, 0, 0, 0],   # M1
        [0, 0, 1, 0],   # M2
        [0, 1, 0, 0],   # M3
        [0, 1, 0, 0],   # M4 ... wait
    ]
    # C11 = M1 - M4 + M5 - M7
    # C12 = M3 + M4 + M7
    # C21 = M2 + M5 - M1 + M6
    # C22 = M1 - M3 + M2 - M6
    
    W_s = [
        [1, 0, -1, 1],  # M1
        [0, 0, 1, 1],   # M2
        [0, 1, 0, -1],  # M3
        [-1, 1, 0, 0],  # M4
        [1, 0, 1, 0],   # M5
        [0, 0, 1, -1],  # M6
        [-1, 1, 0, 0]   # M7
    ]
    
    # Verify Strassen
    # C11: 1 - (-1) + 1 - (-1) = 4? No.
    # Let's re-evaluate W_s.
    # C11 = M1 - M4 + M5 - M7
    # C12 = M3 + M4 + M7
    # C21 = M2 + M5 - M1 + M6
    # C22 = M1 - M3 + M2 - M6
    
    W_s = [
        [1, 0, -1, 1],  # M1
        [0, 0, 1, 1],   # M2
        [0, 1, 0, -1],  # M3
        [-1, 1, 0, 0],  # M4
        [1, 0, 1, 0],   # M5
        [0, 0, 1, -1],  # M6
        [-1, 1, 0, 0]   # M7
    ]
    # Check C11: M1(1) - M4(-1) + M5(1) - M7(-1) = 1 - (-1) + 1 - (-1) = 4. Incorrect.
    # C11 should be 1.
    # M1 - M4 + M5 - M7.
    # Coeff of M1 in C11 is 1.
    # Coeff of M4 in C11 is -1.
    # Coeff of M5 in C11 is 1.
    # Coeff of M7 in C11 is -1.
    # So W_s[0] (for M1) should have 1 at C11.
    # W_s[3] (for M4) should have -1 at C11.
    # W_s[4] (for M5) should have 1 at C11.
    # W_s[6] (for M7) should have -1 at C11.
    
    # Let's build W_s correctly.
    # W_s[r][c] is coeff of M_r in C_c.
    W_s = [
        [1, 0, -1, 1],  # M1: C11=1, C12=0, C21=-1, C22=1
        [0, 0, 1, 1],   # M2: C11=0, C12=0, C21=1, C22=1
        [0, 1, 0, -1],  # M3: C11=0, C12=1, C21=0, C22=-1
        [-1, 1, 0, 0],  # M4: C11=-1, C12=1, C21=0, C22=0
        [1, 0, 1, 0],   # M5: C11=1, C12=0, C21=1, C22=0
        [0, 0, 1, -1],  # M6: C11=0, C12=0, C21=1, C22=-1
        [-1, 1, 0, 0]   # M7: C11=-1, C12=1, C21=0, C22=0
    ]
    
    # Check C11: 1 - (-1) + 1 - (-1) = 4. Still wrong.
    # Ah, the formula is C11 = M1 - M4 + M5 - M7.
    # So coeff of M1 is 1. Coeff of M4 is -1. Coeff of M5 is 1. Coeff of M7 is -1.
    # My W_s has:
    # M1 at C11: 1
    # M4 at C11: -1
    # M5 at C11: 1
    # M7 at C11: -1
    # Sum: 1 - (-1) + 1 - (-1) = 4.
    # Why? Because W_s[3] is M4. W_s[3][0] is -1.
    # So sum is W_s[0][0]*1 + W_s[3][0]*(-1) ... NO.
    # The sum is $\sum_r U V W$.
    # For C11, we sum $U_{A,r} V_{B,r} W_{r, C11}$.
    # If $U V$ is the product $M_r$, then the contribution is $M_r \cdot W_{r, C11}$.
    # So we need $\sum_r M_r W_{r, C11} = C_{11}$.
    # $C_{11} = 1 \cdot M_1 + (-1) \cdot M_4 + 1 \cdot M_5 + (-1) \cdot M_7$.
    # So $W_{0, C11} = 1$, $W_{3, C11} = -1$, $W_{4, C11} = 1$, $W_{6, C11} = -1$.
    # This matches my W_s.
    # So why did I think it was 4?
    # $1 \cdot M_1 + (-1) M_4 + 1 M_5 + (-1) M_7$.
    # This is exactly the formula.
    # Okay, the W_s is correct.

    # Now populate U, V, W for the full problem.
    
    # Block 0: k=0,1
    # A indices: 0,1,2,3
    # B indices for k=0,1:
    # B is 2x13.
    # j=0, k=0 -> index 0*13+0 = 0
    # j=0, k=1 -> index 0*13+1 = 1
    # j=1, k=0 -> index 1*13+0 = 13
    # j=1, k=1 -> index 1*13+1 = 14
    # C indices for k=0,1:
    # k=0, i=0 -> 0*2+0 = 0
    # k=0, i=1 -> 0*2+1 = 1
    # k=1, i=0 -> 1*2+0 = 2
    # k=1, i=1 -> 1*2+1 = 3
    
    # Map local B indices (0,1,2,3) to global V indices
    # local 0 (e) -> j=0,k=0 -> 0
    # local 1 (f) -> j=0,k=1 -> 1
    # local 2 (g) -> j=1,k=0 -> 13
    # local 3 (h) -> j=1,k=1 -> 14
    
    # Map local C indices (0,1,2,3) to global W indices
    # local 0 (c11) -> k=0,i=0 -> 0
    # local 1 (c12) -> k=0,i=1 -> 1
    # local 2 (c21) -> k=1,i=0 -> 2
    # local 3 (c22) -> k=1,i=1 -> 3

    for block in range(6):
        base_k = block * 2
        # V indices
        v_map = [base_k, base_k+1, 13+base_k, 13+base_k+1]
        # W indices
        w_map = [base_k*2, base_k*2+1, base_k*2+2, base_k*2+3]
        
        for r in range(7):
            col_r = block * 7 + r
            # U
            for i in range(4):
                U[i][col_r] = U_s[r][i]
            # V
            for j_local in range(4):
                v_idx = v_map[j_local]
                V[v_idx][col_r] = V_s[r][j_local]
            # W
            for c_local in range(4):
                w_idx = w_map[c_local]
                W[w_idx][col_r] = W_s[r][c_local]
                
    # Last column k=12
    # Schoolbook
    # C_{i,12} = A_{i,0}B_{0,12} + A_{i,1}B_{1,12}
    # i=0: C_{0,12} = A_{0,0}B_{0,12} + A_{0,1}B_{1,12}
    # i=1: C_{1,12} = A_{1,0}B_{0,12} + A_{1,1}B_{1,12}
    
    # 4 multiplications
    # M1: A00 * B012
    # M2: A01 * B112
    # M3: A10 * B012
    # M4: A11 * B112
    
    base_col = 6 * 7
    
    # M1
    U[0][base_col] = 1
    V[12][base_col] = 1 # j=0, k=12 -> 0*13+12 = 12
    W[12*2+0][base_col] = 1 # k=12, i=0 -> 24
    
    # M2
    U[1][base_col+1] = 1
    V[13+12][base_col+1] = 1 # j=1, k=12 -> 13+12 = 25
    W[12*2+0][base_col+1] = 1
    
    # M3
    U[2][base_col+2] = 1
    V[12][base_col+2] = 1
    W[12*2+1][base_col+2] = 1 # k=12, i=1 -> 25
    
    # M4
    U[3][base_col+3] = 1
    V[13+12][base_col+3] = 1
    W[12*2+1][base_col+3] = 1

    return {"U": U, "V": V, "W": W}
