"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 50 decomposition
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # We use 5 products for each 2x2x2x7 block
    # Block 1: A cols 0,1; B rows 0,1
    # Block 2: A cols 2,3; B rows 2,3
    
    # Strassen-like 5 products for 2x2x2xn
    # M1 = (a+d)(x+y)
    # M2 = (c+d)x
    # M4 = d(y-x)
    # M5 = (a+b)y
    # M7 = (b-d)(x+y)
    # C0 = M1 + M4 - M5 + M7
    # C1 = M2 + M4
    
    # Indices for U (A): i*4 + j
    # Indices for V (B): j*7 + k
    # Indices for W (C): k*2 + i (Trace convention W[k*n+i, r])
    
    def add_term(u_idx, v_idx, w_idx, coeff_u, coeff_v, coeff_w):
        # u_idx is list of (index, coeff) in U space
        # v_idx is list of (index, coeff) in V space
        # w_idx is list of (index, coeff) in W space
        pass

    # We need to map the 5 products to the tensor terms.
    # Each product M is a rank-1 term in the tensor.
    # M = u_M \otimes v_M \otimes w_M
    # u_M is in A-space (dim 8)
    # v_M is in B-space (dim 28)
    # w_M is in C-space (dim 14)
    
    # Block 1: A (0,0), (0,1), (1,0), (1,1) -> U indices 0, 1, 4, 5
    # B (0,0..6), (1,0..6) -> V indices 0..6, 7..13
    # C (0,0..6), (1,0..6) -> W indices 0..6, 7..13 (W[k*2+i])
    
    # M1
    # u: (0,0):1, (1,1):1 -> U[0]=1, U[5]=1
    # v: (0,k):1, (1,k):1 for k=0..6 -> V[0..6]=1, V[7..13]=1
    # w: C0 gets +1, C1 gets -1? 
    # C0 = M1 + ...
    # C1 = ...
    # So w_M1: W[0..6]=1, W[7..13]=-1? No, C0 is row 0, C1 is row 1.
    # W[k*2+i]. Row 0 is i=0. Row 1 is i=1.
    # C0 corresponds to i=0. C1 corresponds to i=1.
    # So w_M1: W[k*2+0]=1, W[k*2+1]=-1?
    # Wait, C0 = M1 + M4 - M5 + M7.
    # C1 = M2 + M4.
    # So M1 contributes +1 to C0, 0 to C1.
    # M4 contributes +1 to C0, +1 to C1.
    # M5 contributes -1 to C0, 0 to C1.
    # M7 contributes +1 to C0, 0 to C1.
    # M2 contributes 0 to C0, +1 to C1.
    
    col = 0
    # M1
    U[0][col] += 1; U[5][col] += 1
    for k in range(7):
        V[k][col] += 1; V[7+k][col] += 1
        W[k*2+0][col] += 1
    col += 1
    
    # M2
    U[4][col] += 1; U[5][col] += 1
    for k in range(7):
        V[k][col] += 1
        W[k*2+1][col] += 1
    col += 1
    
    # M4
    U[5][col] += 1
    for k in range(7):
        V[7+k][col] += 1; V[k][col] -= 1
        W[k*2+0][col] += 1; W[k*2+1][col] += 1
    col += 1
    
    # M5
    U[0][col] += 1; U[1][col] += 1
    for k in range(7):
        V[7+k][col] += 1
        W[k*2+0][col] -= 1
    col += 1
    
    # M7
    U[1][col] += 1; U[5][col] -= 1
    for k in range(7):
        V[k][col] += 1; V[7+k][col] += 1
        W[k*2+0][col] += 1
    col += 1
    
    # Block 2: A cols 2,3; B rows 2,3
    # U indices: 2, 3, 6, 7
    # V indices: 14..20, 21..27
    # W indices: same as before (C is shared)
    
    # M1
    U[2][col] += 1; U[7][col] += 1
    for k in range(7):
        V[14+k][col] += 1; V[21+k][col] += 1
        W[k*2+0][col] += 1
    col += 1
    
    # M2
    U[6][col] += 1; U[7][col] += 1
    for k in range(7):
        V[14+k][col] += 1
        W[k*2+1][col] += 1
    col += 1
    
    # M4
    U[7][col] += 1
    for k in range(7):
        V[21+k][col] += 1; V[14+k][col] -= 1
        W[k*2+0][col] += 1; W[k*2+1][col] += 1
    col += 1
    
    # M5
    U[2][col] += 1; U[3][col] += 1
    for k in range(7):
        V[21+k][col] += 1
        W[k*2+0][col] -= 1
    col += 1
    
    # M7
    U[3][col] += 1; U[7][col] -= 1
    for k in range(7):
        V[14+k][col] += 1; V[21+k][col] += 1
        W[k*2+0][col] += 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
