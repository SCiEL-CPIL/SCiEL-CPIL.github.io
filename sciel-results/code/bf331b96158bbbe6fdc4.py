"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 35 decomposition using Strassen for 5 independent 2x2 blocks
    rank = 35
    
    U = [[0] * rank for _ in range(n * m)]  # (4, 35)
    V = [[0] * rank for _ in range(m * p)]  # (20, 35)
    W = [[0] * rank for _ in range(p * n)]  # (20, 35)
    
    # Strassen terms for 2x2x2: each term is (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: [a, b, c, d] coefficients for A = [[a,b],[c,d]]
    # v_coeffs: [e, f, g, h] coefficients for B = [[e,f],[g,h]]
    # w_coeffs: [C11, C21, C12, C22] coefficients (matching W index order: k*n+i)
    #
    # For block t (columns 2t, 2t+1 of B):
    #   B indices: e=V[2t], f=V[2t+1], g=V[10+2t], h=V[10+2t+1]
    #   C indices: C11=W[4t], C21=W[4t+1], C12=W[4t+2], C22=W[4t+3]
    #
    # Standard Strassen:
    # P1 = (a+d)(e+h)
    # P2 = (c+d)e
    # P3 = a(g-e)
    # P4 = d(h-f)
    # P5 = (a+b)h
    # P6 = (c-a)(e+f)
    # P7 = (b-d)(g+h)
    #
    # C11 = P1 - P4 + P5
    # C12 = P3 + P5
    # C21 = P2 + P7
    # C22 = P1 - P3 + P6
    
    strassen_terms = [
        # (u_coeffs, v_coeffs, w_coeffs)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),   # P1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, 0]),   # P2
        ([1, 0, 0, 0], [0, -1, 1, 0], [0, 0, 1, 0]),  # P3
        ([0, 0, 0, 1], [0, -1, 0, 1], [-1, 0, 0, 0]), # P4
        ([1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 1, 0]),   # P5
        ([1, 0, -1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),  # P6
        ([0, 1, 0, -1], [0, 0, 1, 1], [0, 1, 0, 0]),  # P7
    ]
    
    col = 0
    for t in range(5):
        for u_c, v_c, w_c in strassen_terms:
            # U: A indices 0:(0,0)=a, 1:(0,1)=b, 2:(1,0)=c, 3:(1,1)=d
            for i in range(4):
                U[i][col] = u_c[i]
            
            # V: B block t indices: e=2t, f=2t+1, g=10+2t, h=10+2t+1
            V[2*t][col] = v_c[0]       # e
            V[2*t+1][col] = v_c[1]     # f
            V[10+2*t][col] = v_c[2]    # g
            V[10+2*t+1][col] = v_c[3]  # h
            
            # W: C block t indices: C11=4t, C21=4t+1, C12=4t+2, C22=4t+3
            W[4*t][col] = w_c[0]       # C11
            W[4*t+1][col] = w_c[1]     # C21
            W[4*t+2][col] = w_c[2]     # C12
            W[4*t+3][col] = w_c[3]     # C22
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
