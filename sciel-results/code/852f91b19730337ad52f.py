import time
import numpy as np

_START = time.time()
TIME_LIMIT = 120.0

def construct():
    n, m, p = 2, 4, 6
    
    # We'll use rank 16: split the middle dimension into two 2-dimensions
    # C = A_left * B_top + A_right * B_bottom
    # where A_left = A[:, 0:2], A_right = A[:, 2:4]
    # B_top = B[0:2, :], B_bottom = B[2:4, :]
    # Each 2x2 * 2x6 multiplication has rank 8
    
    rank = 16
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 16
    V = [[0] * rank for _ in range(m * p)]  # 24 x 16
    W = [[0] * rank for _ in range(p * n)]  # 12 x 16
    
    # Rank 8 decomposition for 2x2 * 2x6
    # A = [[a, b], [c, d]], B = [[e, f, g, h, i, j], [k, l, m, n, o, p]]
    # C = A*B, 2x6
    
    # Products (rank 8):
    # p1 = a*(e+k)
    # p2 = d*(g+m)
    # p3 = (a+d)*(e+k+g+m)
    # p4 = b*(f+l)
    # p5 = c*(h+n)
    # p6 = (b+c)*(f+l+h+n)
    # p7 = (a+c)*(e+g)
    # p8 = (b+d)*(f+h)
    
    # Let me verify this is correct by checking the output:
    # C[0,0] = a*e + b*k = a(e+k) - a*k + b*k = p1 - a*k + b*k... this doesn't work directly
    
    # Let me use a known rank 8 decomposition for 2x2 * 2x4 first, then extend.
    # Actually, R(M_{2,2,4}) = 8, R(M_{2,2,6}) = 10? Let me check.
    # R(M_{2,2,n}) = 2n for n >= 2? No, R(M_{2,2,2}) = 7, R(M_{2,2,3}) = 8, R(M_{2,2,4}) = 8.
    # Actually I'm not sure. Let me just use the safe approach.
    
    # Safe approach: use Strassen for 2x2 * 2x2, giving 7 products per 2x2 block.
    # For 2x2 * 2x6, split the 6 into three 2s: 3 * 7 = 21 products.
    # For the full 2x4 * 4x6: split middle into two 2s, each giving 2x2 * 2x6 = 21 products.
    # Total: 2 * 21 = 42. That's what we have.
    
    # To do better, I need a better decomposition for 2x2 * 2x6.
    # Let me try to find one with rank < 21.
    
    # Alternative: use the fact that R(M_{2,4,6}) <= R(M_{2,4,4}) + R(M_{2,4,2})
    # R(M_{2,4,4}) = ? and R(M_{2,4,2}) = ?
    
    # Let me just return the current rank 42 solution to be safe, since I can't verify a better one.
    
    # Actually, let me try a different approach. Use rank 14 for 2x4 * 4x2 and rank 16 for 2x4 * 4x4.
    # 2x4 * 4x2: A is 2x4, B is 4x2. This is like 2*(2x2 * 2x2) but with shared structure.
    # Actually R(M_{2,4,2}) = R(M_{2,2,4}) by symmetry = 8? Or 14?
    
    # I'm not confident in a better rank. Let me return the current solution.
    
    # Strassen 7-product decomposition for 2x2x2
    U_coeffs = [
        {(0, 0): 1, (1, 1): 1},
        {(1, 0): 1, (1, 1): 1},
        {(0, 0): 1},
        {(1, 1): 1},
        {(0, 0): 1, (0, 1): 1},
        {(1, 0): 1, (0, 0): -1},
        {(0, 1): 1, (1, 1): -1}
    ]
    
    V_coeffs = [
        {(0, 0): 1, (1, 1): 1},
        {(0, 0): 1},
        {(0, 1): 1, (1, 1): -1},
        {(1, 0): 1, (0, 0): -1},
        {(1, 1): 1},
        {(0, 0): 1, (0, 1): 1},
        {(1, 0): 1, (1, 1): 1}
    ]
    
    W_coeffs = [
        {(0, 0): 1, (1, 1): 1},
        {(1, 0): 1, (1, 1): -1},
        {(0, 1): 1, (1, 1): 1},
        {(0, 0): 1, (1, 0): 1},
        {(0, 0): -1, (0, 1): 1},
        {(1, 1): 1},
        {(0, 0): 1}
    ]
    
    col = 0
    for g in range(2):  # 2 blocks in the column direction of A (j = 0,1 and 2,3)
        for h in range(3):  # 3 blocks in the row direction of B (k = 0,1; 2,3; 4,5)
            for s in range(7):
                for (i_loc, j_loc), coeff in U_coeffs[s].items():
                    i = i_loc
                    j = g * 2 + j_loc
                    u_idx = i * m + j
                    U[u_idx][col] += coeff
                
                for (j_loc, k_loc), coeff in V_coeffs[s].items():
                    j = g * 2 + j_loc
                    k = h * 2 + k_loc
                    v_idx = j * p + k
                    V[v_idx][col] += coeff
                
                for (i_loc, k_loc), coeff in W_coeffs[s].items():
                    i = i_loc
                    k = h * 2 + k_loc
                    w_idx = k * n + i
                    W[w_idx][col] += coeff
                
                col += 1
    
    return {"U": U, "V": V, "W": W}
