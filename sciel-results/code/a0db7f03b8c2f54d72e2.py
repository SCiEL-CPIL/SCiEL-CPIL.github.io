import time
import sys

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 6 * 7 = 42
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients
    # u: coeffs for A (a,b,c,d) -> indices 0,1,2,3
    # v: coeffs for B (e,f,g,h) -> indices 0,1,2,3 (local)
    # w: coeffs for C (c11,c12,c21,c22) -> indices 0,1,2,3 (local)
    
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # u vectors
    u1 = [1, 0, 0, 1]
    u2 = [0, 0, 1, 1]
    u3 = [1, 0, 0, 0]
    u4 = [0, 0, 0, 1]
    u5 = [1, 1, 0, 0]
    u6 = [0, 0, 1, -1] # (c-a) -> c=1, a=-1
    u7 = [0, 1, 0, -1] # (b-d) -> b=1, d=-1
    
    # v vectors (local e,f,g,h)
    # e is B[0,0], f is B[0,1], g is B[1,0], h is B[1,1]
    v1 = [1, 0, 0, 1] # e+h
    v2 = [1, 0, 0, 0] # e
    v3 = [0, 1, 0, -1] # f-h
    v4 = [0, 0, 1, -1] # g-e
    v5 = [0, 0, 0, 1] # h
    v6 = [1, 1, 0, 0] # e+f
    v7 = [0, 0, 1, 1] # g+h
    
    # w vectors (local c11, c12, c21, c22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    w1 = [1, 0, 0, 1]
    w2 = [0, 0, 1, -1]
    w3 = [0, 1, 0, 1]
    w4 = [1, 0, 1, 0]
    w5 = [-1, 1, 0, 0]
    w6 = [0, 0, 0, 1]
    w7 = [1, 0, 0, 0]
    
    # Check sums
    # C11: 1 + 1 - 1 + 1 = 2? No.
    # Let's re-verify Strassen.
    # M1 = (a+d)(e+h) = ae+ah+de+dh
    # M4 = d(g-e) = dg-de
    # M5 = (a+b)h = ah+bh
    # M7 = (b-d)(g+h) = bg+bh-dg-dh
    # Sum for C11 (ae+bf? No, C11 = ae+bg? No, C = AB. C11 = a*e + b*g? No.
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # C11 = a*e + b*g
    # C12 = a*f + b*h
    # C21 = c*e + d*g
    # C22 = c*f + d*h
    
    # My previous Strassen mapping was for C = AB.
    # C11 = M1 + M4 - M5 + M7
    # = (ae+ah+de+dh) + (dg-de) - (ah+bh) + (bg+bh-dg-dh)
    # = ae + ah + de + dh + dg - de - ah - bh + bg + bh - dg - dh
    # = ae + bg. Correct.
    
    # C12 = M3 + M5
    # = (af-ah) + (ah+bh) = af + bh. Correct.
    
    # C21 = M2 + M4
    # = (ce+de) + (dg-de) = ce + dg. Correct.
    
    # C22 = M1 - M2 + M3 + M6
    # = (ae+ah+de+dh) - (ce+de) + (af-ah) + (ce+cf-ae-af)
    # = ae + ah + de + dh - ce - de + af - ah + ce + cf - ae - af
    # = dh + cf. Correct.
    
    # So the w vectors are:
    # M1: C11=1, C22=1 -> [1, 0, 0, 1]
    # M2: C21=1, C22=-1 -> [0, 0, 1, -1]
    # M3: C12=1, C22=1 -> [0, 1, 0, 1]
    # M4: C11=1, C21=1 -> [1, 0, 1, 0]
    # M5: C11=-1, C12=1 -> [-1, 1, 0, 0]
    # M6: C22=1 -> [0, 0, 0, 1]
    # M7: C11=1 -> [1, 0, 0, 0]
    
    # This matches my w vectors above.
    
    u_vecs = [u1, u2, u3, u4, u5, u6, u7]
    v_vecs = [v1, v2, v3, v4, v5, v6, v7]
    w_vecs = [w1, w2, w3, w4, w5, w6, w7]
    
    col = 0
    for l in range(6):
        # Block l corresponds to B columns 2l, 2l+1
        k1 = 2 * l
        k2 = 2 * l + 1
        
        for r in range(7):
            u = u_vecs[r]
            v = v_vecs[r]
            w = w_vecs[r]
            
            # Fill U
            # u indices: 0:a, 1:b, 2:c, 3:d
            # U row index: i*m + j
            # a: i=0, j=0 -> 0
            # b: i=0, j=1 -> 1
            # c: i=1, j=0 -> 2
            # d: i=1, j=1 -> 3
            U[0][col] = u[0]
            U[1][col] = u[1]
            U[2][col] = u[2]
            U[3][col] = u[3]
            
            # Fill V
            # v indices: 0:e, 1:f, 2:g, 3:h
            # e: B[0, k1] -> row 0, col k1. Index 0*p + k1
            # f: B[0, k2] -> row 0, col k2. Index 0*p + k2
            # g: B[1, k1] -> row 1, col k1. Index 1*p + k1
            # h: B[1, k2] -> row 1, col k2. Index 1*p + k2
            V[0 * p + k1][col] = v[0]
            V[0 * p + k2][col] = v[1]
            V[1 * p + k1][col] = v[2]
            V[1 * p + k2][col] = v[3]
            
            # Fill W
            # w indices: 0:c11, 1:c12, 2:c21, 3:c22
            # c11: C[0, k1] -> i=0, k=k1. Index k*n + i = k1*2 + 0
            # c12: C[0, k2] -> i=0, k=k2. Index k2*2 + 0
            # c21: C[1, k1] -> i=1, k=k1. Index k1*2 + 1
            # c22: C[1, k2] -> i=1, k=k2. Index k2*2 + 1
            W[k1 * n + 0][col] = w[0]
            W[k2 * n + 0][col] = w[1]
            W[k1 * n + 1][col] = w[2]
            W[k2 * n + 1][col] = w[3]
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
