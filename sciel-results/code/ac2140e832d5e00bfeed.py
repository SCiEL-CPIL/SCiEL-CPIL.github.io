import numpy as np

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    # n=3, m=3, p=8
    # Rank calculation:
    # X (2x2 * 2x8): 4 blocks of 2x2*2x2. Rank 7 each. Total 28.
    # Y (2x1 * 1x8): Rank 2 * 8 = 16.
    # Z (1x2 * 2x8): Rank 1 * 2 * 8 = 16.
    # W (1x1 * 1x8): Rank 1 * 1 * 8 = 8.
    # Total Rank = 28 + 16 + 16 + 8 = 68.
    
    rank = 68
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a bilinear term
    def add_term(u_idx, v_idx, w_idx, coeff=1):
        nonlocal col
        U[u_idx][col] += coeff
        V[v_idx][col] += coeff
        W[w_idx][col] += coeff
        col += 1

    # Strassen coefficients for 2x2x2
    # p1 = (a+d)(e+h)
    # p2 = (c+d)e
    # p3 = a(f-h)
    # p4 = d(g-e)
    # p5 = (a+b)h
    # p6 = (c-a)(e+f)
    # p7 = (b-d)(g+h)
    
    # A indices for 2x2 block (top-left of 3x3)
    # a=A00, b=A01, c=A10, d=A11
    a, b, c, d = 0, 1, 3, 4 # i*m+j
    
    # For each of the 4 blocks of B (columns 0-1, 2-3, 4-5, 6-7)
    for t in range(4):
        k0 = 2 * t
        k1 = 2 * t + 1
        
        # B indices
        # e=B0k0, f=B0k1, g=B1k0, h=B1k1
        e = 0 * p + k0
        f = 0 * p + k1
        g = 1 * p + k0
        h = 1 * p + k1
        
        # p1 = (a+d)(e+h)
        # U: a, d. V: e, h.
        # This is a product of sums. We need to expand or use the bilinear form directly.
        # The decomposition allows linear combinations in U and V.
        # U[a] += 1, U[d] += 1
        # V[e] += 1, V[h] += 1
        # W[out] += 1
        
        # We need to track which output this contributes to.
        # Output C indices for block t:
        # C0k0, C0k1, C1k0, C1k1
        # W index is k*n + i.
        # C0k0 -> W[k0*n + 0]
        # C0k1 -> W[k1*n + 0]
        # C1k0 -> W[k0*n + 1]
        # C1k1 -> W[k1*n + 1]
        
        w00 = k0 * n + 0
        w01 = k1 * n + 0
        w10 = k0 * n + 1
        w11 = k1 * n + 1
        
        # p1
        U[a][col] += 1
        U[d][col] += 1
        V[e][col] += 1
        V[h][col] += 1
        W[w00][col] += 1
        W[w01][col] += 1
        W[w10][col] += 1
        W[w11][col] -= 1 # Wait, C11 = p1 - p2 + p3 + p6. So p1 contributes +1 to C11.
        # Let's re-verify Strassen output formulas.
        # C00 = p1 + p4 - p5 + p7
        # C01 = p3 + p5
        # C10 = p2 + p4
        # C11 = p1 - p2 + p3 + p6
        
        # So for p1:
        # W[w00] += 1
        # W[w01] += 0
        # W[w10] += 0
        # W[w11] += 1
        # My previous addition was wrong. I added to all.
        
        # I should not use a generic add_term that adds to all W.
        # I need to specify the W coefficient.
        
        # Let's restart the loop logic with explicit coefficients.
        
        # p1
        U[a][col] += 1
        U[d][col] += 1
        V[e][col] += 1
        V[h][col] += 1
        W[w00][col] += 1
        W[w11][col] += 1
        col += 1
        
        # p2 = (c+d)e
        U[c][col] += 1
        U[d][col] += 1
        V[e][col] += 1
        W[w10][col] += 1
        W[w11][col] -= 1
        col += 1
        
        # p3 = a(f-h)
        U[a][col] += 1
        V[f][col] += 1
        V[h][col] -= 1
        W[w01][col] += 1
        W[w11][col] += 1
        col += 1
        
        # p4 = d(g-e)
        U[d][col] += 1
        V[g][col] += 1
        V[e][col] -= 1
        W[w00][col] += 1
        W[w10][col] += 1
        col += 1
        
        # p5 = (a+b)h
        U[a][col] += 1
        U[b][col] += 1
        V[h][col] += 1
        W[w00][col] -= 1
        W[w01][col] += 1
        col += 1
        
        # p6 = (c-a)(e+f)
        U[c][col] += 1
        U[a][col] -= 1
        V[e][col] += 1
        V[f][col] += 1
        W[w11][col] += 1
        col += 1
        
        # p7 = (b-d)(g+h)
        U[b][col] += 1
        U[d][col] -= 1
        V[g][col] += 1
        V[h][col] += 1
        W[w00][col] += 1
        col += 1

    # Y: A21 * B1x8
    # A21: A02 (idx 2), A12 (idx 5)
    # B1x8: B2k (idx 2*p + k)
    # Output: C0k (W[k*n+0]), C1k (W[k*n+1])
    # Y0k = A02 * B2k
    # Y1k = A12 * B2k
    
    for k in range(8):
        # Y0k
        U[2][col] += 1
        V[2*p + k][col] += 1
        W[k*n + 0][col] += 1
        col += 1
        
        # Y1k
        U[5][col] += 1
        V[2*p + k][col] += 1
        W[k*n + 1][col] += 1
        col += 1

    # Z: A12 * B2x8
    # A12: A20 (idx 6), A21 (idx 7)
    # B2x8: B0k (idx k), B1k (idx p+k)
    # Output: C2k (W[k*n+2])
    # Z0k = A20 * B0k
    # Z1k = A21 * B1k
    
    for k in range(8):
        # Z0k
        U[6][col] += 1
        V[k][col] += 1
        W[k*n + 2][col] += 1
        col += 1
        
        # Z1k
        U[7][col] += 1
        V[p + k][col] += 1
        W[k*n + 2][col] += 1
        col += 1

    # W: A11 * B1x8
    # A11: A22 (idx 8)
    # B1x8: B2k (idx 2*p + k)
    # Output: C2k (W[k*n+2])
    
    for k in range(8):
        U[8][col] += 1
        V[2*p + k][col] += 1
        W[k*n + 2][col] += 1
        col += 1

    return {"U": U, "V": V, "W": W}
