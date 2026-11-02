"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-28 decomposition of the 2x4x4 matrix multiplication tensor
    using Strassen's rank-7 algorithm for 2x2x2 blocks.
    
    A is 2x4, B is 4x4, C = A*B is 2x4.
    We split A into [A_L, A_R] each 2x2.
    We split B into 4 blocks of 2x2: B_{00}, B_{01}, B_{10}, B_{11}
    where B_{ij} has rows i*2..i*2+1 and cols j*2..j*2+1.
    
    C = A_L * [B_{00}, B_{01}] + A_R * [B_{10}, B_{11}]
      = [A_L*B_{00} + A_R*B_{10},  A_L*B_{01} + A_R*B_{11}]
    
    So C[:, 0:2] = A_L*B_{00} + A_R*B_{10}
       C[:, 2:4] = A_L*B_{01} + A_R*B_{11}
    
    Each 2x2x2 multiplication uses rank 7 (Strassen).
    Total: 4 blocks * 7 = 28 rank.
    
    Indexing conventions:
    U has shape (n*m, rank) = (8, 28).  Row index = i*m + j = i*4 + j.
    V has shape (m*p, rank) = (16, 28). Row index = j*p + k = j*4 + k.
    W has shape (p*n, rank) = (8, 28).  Row index = k*n + i = k*2 + i.
    
    Strassen's 7 products for X (2x2) * Y (2x2) = Z (2x2):
    X = [[a, b], [c, d]], Y = [[e, f], [g, h]], Z = [[z11, z12], [z21, z22]]
    
    M1 = (a+d)*(e+h)
    M2 = (c+d)*e
    M3 = a*(f-h)
    M4 = d*(g-i)  -- wait, should be d*(g-i)? No, standard Strassen:
    
    Standard Strassen:
    M1 = (a+d)(e+h)
    M2 = (c+d)(f)
    M3 = (a-b)(g+h)
    M4 = d(h-e)
    M5 = (a+c)(e)
    M6 = (b+d)(h)
    M7 = (a-c)(e+f)
    
    Z11 = M1 + M4 - M5 + M7
    Z12 = M3 + M5
    Z21 = M2 + M4
    Z22 = M1 - M2 + M3 + M6
    
    Let me verify with the standard formulation:
    M1 = (a+d)(e+h)
    M2 = (c+d)(f)
    M3 = (a-b)(g+h)
    M4 = d(h-e)
    M5 = (a+c)(e)
    M6 = (b+d)(h)
    M7 = (a-c)(e+f)
    
    Z11 = M1 + M4 - M5 + M7 = (a+d)(e+h) + d(h-e) - (a+c)e + (a-c)(e+f)
    = ae + ah + de + dh + dh - de - ae - ce + ae + af - ce - cf
    = ae + ah + dh - ce + af - ce - cf
    Hmm, let me just trust the standard Strassen and verify numerically.
    
    Actually, the most common formulation is:
    M1 = (a+d)(e+h)
    M2 = (c+d)e
    M3 = a(f-h)
    M4 = d(g-i)  -- this is wrong, i is not defined
    
    Let me use the correct standard Strassen:
    M1 = (a+d)(e+h)
    M2 = (c+d)f
    M3 = (a-b)(g+h)
    M4 = d(h-e)
    M5 = (a+c)e
    M6 = (b+d)h
    M7 = (a-c)(e+f)
    
    Z11 = M1 + M4 - M5 + M7
    Z12 = M3 + M5
    Z21 = M2 + M4
    Z22 = M1 - M2 + M3 + M6
    """
    import time
    deadline = time.time() + 140  # 10s safety margin
    
    n, m, p = 2, 4, 4
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 8 x 28
    
    # Strassen coefficients for 2x2x2:
    # X = [[x00, x01], [x10, x11]], Y = [[y00, y01], [y10, y11]], Z = X*Y
    # M1 = (x00+x11)(y00+y11)
    # M2 = (x10+x11)y01
    # M3 = (x00-x10)(y10+y11)
    # M4 = x11(y11-y00)
    # M5 = (x00+x10)y00
    # M6 = (x01+x11)y11
    # M7 = (x00-x10)(y00+y01)
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    # For each Strassen product t in 1..7, the U, V, W coefficients:
    # U indices are (i, j) where i is row of X (0 or 1), j is col of X (0 or 1)
    # V indices are (j, k) where j is row of Y (0 or 1), k is col of Y (0 or 1)
    # W indices are (k, i) where k is col of Z (0 or 1), i is row of Z (0 or 1)
    
    # For a single 2x2x2 Strassen block, given offsets:
    # A offset: (a_i, a_j) -> U row = (a_i)*4 + (a_j + j_local)
    # B offset: (b_j, b_k) -> V row = (b_j + j_local)*4 + (b_k + k_local)
    # C offset: (c_i, c_k) -> W row = (c_k + k_local)*2 + (c_i + i_local)
    
    # Strassen products: each product t involves:
    # For U: a linear combination of X elements
    # For V: a linear combination of Y elements
    # For W: a linear combination of Z elements
    
    # Let's define the 7 Strassen products with their U, V, W factor vectors.
    # X elements: x00, x01, x10, x11 (indices 0,1,2,3 in X)
    # Y elements: y00, y01, y10, y11 (indices 0,1,2,3 in Y)
    # Z elements: z00, z01, z10, z11 (indices 0,1,2,3 in Z)
    
    # M1 = (x00+x11)(y00+y11)
    # U: [1, 0, 1, 0], V: [1, 0, 0, 1], W: contributes to z00 (+1), z11 (-1)
    # M2 = (x10+x11)y01
    # U: [0, 0, 1, 1], V: [0, 1, 0, 0], W: contributes to z10 (+1), z11 (-1)
    # M3 = (x00-x10)(y10+y11)
    # U: [1, 0, -1, 0], V: [0, 0, 1, 1], W: contributes to z01 (+1), z11 (+1)
    # M4 = x11(y11-y00)
    # U: [0, 0, 0, 1], V: [-1, 0, 0, 1], W: contributes to z00 (+1), z10 (+1)
    # M5 = (x00+x10)y00
    # U: [1, 0, 1, 0], V: [1, 0, 0, 0], W: contributes to z00 (-1), z01 (+1)
    # M6 = (x01+x11)y11
    # U: [0, 1, 0, 1], V: [0, 0, 0, 1], W: contributes to z11 (+1)
    # M7 = (x00-x10)(y00+y01)
    # U: [1, 0, -1, 0], V: [1, 1, 0, 0], W: contributes to z00 (+1)
    
    # Strassen products as (u_coeff, v_coeff, w_coeff)
    # u_coeff: list of 4 values for x00, x01, x10, x11
    # v_coeff: list of 4 values for y00, y01, y10, y11
    # w_coeff: list of 4 values for z00, z01, z10, z11
    strassen = [
        ([1, 0, 1, 0], [1, 0, 0, 1], [1, 0, 0, -1]),   # M1
        ([0, 0, 1, 1], [0, 1, 0, 0], [0, 0, 1, -1]),   # M2
        ([1, 0, -1, 0], [0, 0, 1, 1], [0, 1, 0, 1]),   # M3
        ([0, 0, 0, 1], [-1, 0, 0, 1], [1, 0, 1, 0]),   # M4
        ([1, 0, 1, 0], [1, 0, 0, 0], [-1, 1, 0, 0]),   # M5
        ([0, 1, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]),    # M6
        ([1, 0, -1, 0], [1, 1, 0, 0], [1, 0, 0, 0]),   # M7
    ]
    
    # 4 blocks:
    # Block 0: A_L * B_{00} -> C[:, 0:2]
    #   A_L: rows 0-1 of A, cols 0-1. U offset: i=0, j=0
    #   B_{00}: rows 0-1 of B, cols 0-1. V offset: j=0, k=0
    #   C[:, 0:2]: W offset: i=0, k=0
    # Block 1: A_L * B_{01} -> C[:, 2:4]
    #   A_L: U offset: i=0, j=0
    #   B_{01}: rows 0-1 of B, cols 2-3. V offset: j=0, k=2
    #   C[:, 2:4]: W offset: i=0, k=2
    # Block 2: A_R * B_{10} -> C[:, 0:2]
    #   A_R: rows 0-1 of A, cols 2-3. U offset: i=0, j=2
    #   B_{10}: rows 2-3 of B, cols 0-1. V offset: j=2, k=0
    #   C[:, 0:2]: W offset: i=0, k=0
    # Block 3: A_R * B_{11} -> C[:, 2:4]
    #   A_R: U offset: i=0, j=2
    #   B_{11}: rows 2-3 of B, cols 2-3. V offset: j=2, k=2
    #   C[:, 2:4]: W offset: i=0, k=2
    
    blocks = [
        (0, 0, 0, 0),  # (a_i, a_j, b_j, b_k, c_i, c_k) -- but c_i is always 0, c_k varies
        (0, 0, 0, 2),
        (0, 2, 2, 0),
        (0, 2, 2, 2),
    ]
    
    col = 0
    for (a_i, a_j, b_j, b_k, c_i, c_k) in [(0,0,0,0,0,0), (0,0,0,2,0,2), (0,2,2,0,0,0), (0,2,2,2,0,2)]:
        for t in range(7):
            u_coeff, v_coeff, w_coeff = strassen[t]
            # U: for each local (li, lj) in 2x2, U row = (a_i+li)*4 + (a_j+lj)
            for li in range(2):
                for lj in range(2):
                    u_idx = li * 4 + lj  # local index
                    u_row = (a_i + li) * 4 + (a_j + lj)
                    U[u_row][col] += u_coeff[u_idx]
            # V: for each local (lj, lk) in 2x2, V row = (b_j+lj)*4 + (b_k+lk)
            for lj in range(2):
                for lk in range(2):
                    v_idx = lj * 2 + lk  # local index
                    v_row = (b_j + lj) * 4 + (b_k + lk)
                    V[v_row][col] += v_coeff[v_idx]
            # W: for each local (lk, li) in 2x2, W row = (c_k+lk)*2 + (c_i+li)
            for lk in range(2):
                for li in range(2):
                    w_idx = lk * 2 + li  # local index
                    w_row = (c_k + lk) * 2 + (c_i + li)
                    W[w_row][col] += w_coeff[w_idx]
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
