"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=5
    # U shape: (n*m, rank) = (8, rank)
    # V shape: (m*p, rank) = (20, rank)
    # W shape: (p*n, rank) = (10, rank)

    rank = 41

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def u_idx(i, j):
        return i * m + j

    def v_idx(j, k):
        return j * p + k

    def w_idx(k, i):
        return k * n + i

    col = 0

    # Block 1: A_L B_11 (2x2 * 2x2 -> 2x2)
    # A_L: (0,0), (0,1), (1,0), (1,1)
    # B_11: (0,0), (0,1), (1,0), (1,1)
    # Result C_11: (0,0), (0,1), (1,0), (1,1) -> W indices: (k=0,i=0), (k=1,i=0), (k=0,i=1), (k=1,i=1)
    # w_offsets: [(0,0), (1,0), (0,1), (1,1)]
    # Indices in y (A): (0,0), (0,1), (1,0), (1,1)
    # Indices in z (B): (0,0), (0,1), (1,0), (1,1)
    # Indices in w (C): (0,0), (1,0), (0,1), (1,1)
    y1 = [(0,0), (0,1), (1,0), (1,1)]
    z1 = [(0,0), (0,1), (1,0), (1,1)]
    w1 = [(0,0), (1,0), (0,1), (1,1)]
    
    # M1
    U[u_idx(y1[0][0], y1[0][1])][col] = 1
    U[u_idx(y1[3][0], y1[3][1])][col] = 1
    V[v_idx(z1[0][0], z1[0][1])][col] = 1
    V[v_idx(z1[3][0], z1[3][1])][col] = 1
    W[w_idx(w1[0][0], w1[0][1])][col] += 1
    W[w_idx(w1[3][0], w1[3][1])][col] += 1
    col += 1
    
    # M2
    U[u_idx(y1[2][0], y1[2][1])][col] = 1
    U[u_idx(y1[3][0], y1[3][1])][col] = 1
    V[v_idx(z1[0][0], z1[0][1])][col] = 1
    W[w_idx(w1[2][0], w1[2][1])][col] += 1
    W[w_idx(w1[3][0], w1[3][1])][col] -= 1
    col += 1
    
    # M3
    U[u_idx(y1[0][0], y1[0][1])][col] = 1
    V[v_idx(z1[1][0], z1[1][1])][col] = 1
    V[v_idx(z1[3][0], z1[3][1])][col] = -1
    W[w_idx(w1[1][0], w1[1][1])][col] += 1
    W[w_idx(w1[3][0], w1[3][1])][col] += 1
    col += 1
    
    # M4
    U[u_idx(y1[3][0], y1[3][1])][col] = 1
    V[v_idx(z1[2][0], z1[2][1])][col] = 1
    V[v_idx(z1[0][0], z1[0][1])][col] = -1
    W[w_idx(w1[0][0], w1[0][1])][col] += 1
    W[w_idx(w1[2][0], w1[2][1])][col] += 1
    col += 1
    
    # M5
    U[u_idx(y1[0][0], y1[0][1])][col] = 1
    U[u_idx(y1[1][0], y1[1][1])][col] = 1
    V[v_idx(z1[3][0], z1[3][1])][col] = 1
    W[w_idx(w1[0][0], w1[0][1])][col] -= 1
    W[w_idx(w1[1][0], w1[1][1])][col] += 1
    col += 1
    
    # M6
    U[u_idx(y1[2][0], y1[2][1])][col] = 1
    U[u_idx(y1[0][0], y1[0][1])][col] = -1
    V[v_idx(z1[0][0], z1[0][1])][col] = 1
    V[v_idx(z1[1][0], z1[1][1])][col] = 1
    W[w_idx(w1[3][0], w1[3][1])][col] += 1
    col += 1
    
    # M7
    U[u_idx(y1[1][0], y1[1][1])][col] = 1
    U[u_idx(y1[3][0], y1[3][1])][col] = -1
    V[v_idx(z1[2][0], z1[2][1])][col] = 1
    V[v_idx(z1[3][0], z1[3][1])][col] = 1
    W[w_idx(w1[0][0], w1[0][1])][col] += 1
    col += 1

    # Block 2: A_R B_21 (2x2 * 2x2 -> 2x2)
    # A_R: (0,2), (0,3), (1,2), (1,3)
    # B_21: (2,0), (2,1), (3,0), (3,1)
    # Result C_11: same as above
    y2 = [(0,2), (0,3), (1,2), (1,3)]
    z2 = [(2,0), (2,1), (3,0), (3,1)]
    w2 = [(0,0), (1,0), (0,1), (1,1)]
    
    # M1
    U[u_idx(y2[0][0], y2[0][1])][col] = 1
    U[u_idx(y2[3][0], y2[3][1])][col] = 1
    V[v_idx(z2[0][0], z2[0][1])][col] = 1
    V[v_idx(z2[3][0], z2[3][1])][col] = 1
    W[w_idx(w2[0][0], w2[0][1])][col] += 1
    W[w_idx(w2[3][0], w2[3][1])][col] += 1
    col += 1
    
    # M2
    U[u_idx(y2[2][0], y2[2][1])][col] = 1
    U[u_idx(y2[3][0], y2[3][1])][col] = 1
    V[v_idx(z2[0][0], z2[0][1])][col] = 1
    W[w_idx(w2[2][0], w2[2][1])][col] += 1
    W[w_idx(w2[3][0], w2[3][1])][col] -= 1
    col += 1
    
    # M3
    U[u_idx(y2[0][0], y2[0][1])][col] = 1
    V[v_idx(z2[1][0], z2[1][1])][col] = 1
    V[v_idx(z2[3][0], z2[3][1])][col] = -1
    W[w_idx(w2[1][0], w2[1][1])][col] += 1
    W[w_idx(w2[3][0], w2[3][1])][col] += 1
    col += 1
    
    # M4
    U[u_idx(y2[3][0], y2[3][1])][col] = 1
    V[v_idx(z2[2][0], z2[2][1])][col] = 1
    V[v_idx(z2[0][0], z2[0][1])][col] = -1
    W[w_idx(w2[0][0], w2[0][1])][col] += 1
    W[w_idx(w2[2][0], w2[2][1])][col] += 1
    col += 1
    
    # M5
    U[u_idx(y2[0][0], y2[0][1])][col] = 1
    U[u_idx(y2[1][0], y2[1][1])][col] = 1
    V[v_idx(z2[3][0], z2[3][1])][col] = 1
    W[w_idx(w2[0][0], w2[0][1])][col] -= 1
    W[w_idx(w2[1][0], w2[1][1])][col] += 1
    col += 1
    
    # M6
    U[u_idx(y2[2][0], y2[2][1])][col] = 1
    U[u_idx(y2[0][0], y2[0][1])][col] = -1
    V[v_idx(z2[0][0], z2[0][1])][col] = 1
    V[v_idx(z2[1][0], z2[1][1])][col] = 1
    W[w_idx(w2[3][0], w2[3][1])][col] += 1
    col += 1
    
    # M7
    U[u_idx(y2[1][0], y2[1][1])][col] = 1
    U[u_idx(y2[3][0], y2[3][1])][col] = -1
    V[v_idx(z2[2][0], z2[2][1])][col] = 1
    V[v_idx(z2[3][0], z2[3][1])][col] = 1
    W[w_idx(w2[0][0], w2[0][1])][col] += 1
    col += 7

    # Block 3: A_L B_12 (2x2 * 2x2 -> 2x2)
    # A_L: (0,0), (0,1), (1,0), (1,1)
    # B_12: (0,2), (0,3), (1,2), (1,3)
    # Result C_12: (0,2), (0,3), (1,2), (1,3) -> W indices: (k=2,i=0), (k=3,i=0), (k=2,i=1), (k=3,i=1)
    y3 = [(0,0), (0,1), (1,0), (1,1)]
    z3 = [(0,2), (0,3), (1,2), (1,3)]
    w3 = [(2,0), (3,0), (2,1), (3,1)]
    
    # M1
    U[u_idx(y3[0][0], y3[0][1])][col] = 1
    U[u_idx(y3[3][0], y3[3][1])][col] = 1
    V[v_idx(z3[0][0], z3[0][1])][col] = 1
    V[v_idx(z3[3][0], z3[3][1])][col] = 1
    W[w_idx(w3[0][0], w3[0][1])][col] += 1
    W[w_idx(w3[3][0], w3[3][1])][col] += 1
    col += 1
    
    # M2
    U[u_idx(y3[2][0], y3[2][1])][col] = 1
    U[u_idx(y3[3][0], y3[3][1])][col] = 1
    V[v_idx(z3[0][0], z3[0][1])][col] = 1
    W[w_idx(w3[2][0], w3[2][1])][col] += 1
    W[w_idx(w3[3][0], w3[3][1])][col] -= 1
    col += 1
    
    # M3
    U[u_idx(y3[0][0], y3[0][1])][col] = 1
    V[v_idx(z3[1][0], z3[1][1])][col] = 1
    V[v_idx(z3[3][0], z3[3][1])][col] = -1
    W[w_idx(w3[1][0], w3[1][1])][col] += 1
    W[w_idx(w3[3][0], w3[3][1])][col] += 1
    col += 1
    
    # M4
    U[u_idx(y3[3][0], y3[3][1])][col] = 1
    V[v_idx(z3[2][0], z3[2][1])][col] = 1
    V[v_idx(z3[0][0], z3[0][1])][col] = -1
    W[w_idx(w3[0][0], w3[0][1])][col] += 1
    W[w_idx(w3[2][0], w3[2][1])][col] += 1
    col += 1
    
    # M5
    U[u_idx(y3[0][0], y3[0][1])][col] = 1
    U[u_idx(y3[1][0], y3[1][1])][col] = 1
    V[v_idx(z3[3][0], z3[3][1])][col] = 1
    W[w_idx(w3[0][0], w3[0][1])][col] -= 1
    W[w_idx(w3[1][0], w3[1][1])][col] += 1
    col += 1
    
    # M6
    U[u_idx(y3[2][0], y3[2][1])][col] = 1
    U[u_idx(y3[0][0], y3[0][1])][col] = -1
    V[v_idx(z3[0][0], z3[0][1])][col] = 1
    V[v_idx(z3[1][0], z3[1][1])][col] = 1
    W[w_idx(w3[3][0], w3[3][1])][col] += 1
    col += 1
    
    # M7
    U[u_idx(y3[1][0], y3[1][1])][col] = 1
    U[u_idx(y3[3][0], y3[3][1])][col] = -1
    V[v_idx(z3[2][0], z3[2][1])][col] = 1
    V[v_idx(z3[3][0], z3[3][1])][col] = 1
    W[w_idx(w3[0][0], w3[0][1])][col] += 1
    col += 7

    # Block 4: A_R B_22 (2x2 * 2x2 -> 2x2)
    # A_R: (0,2), (0,3), (1,2), (1,3)
    # B_22: (2,2), (2,3), (3,2), (3,3)
    # Result C_12: same as above
    y4 = [(0,2), (0,3), (1,2), (1,3)]
    z4 = [(2,2), (2,3), (3,2), (3,3)]
    w4 = [(2,0), (3,0), (2,1), (3,1)]
    
    # M1
    U[u_idx(y4[0][0], y4[0][1])][col] = 1
    U[u_idx(y4[3][0], y4[3][1])][col] = 1
    V[v_idx(z4[0][0], z4[0][1])][col] = 1
    V[v_idx(z4[3][0], z4[3][1])][col] = 1
    W[w_idx(w4[0][0], w4[0][1])][col] += 1
    W[w_idx(w4[3][0], w4[3][1])][col] += 1
    col += 1
    
    # M2
    U[u_idx(y4[2][0], y4[2][1])][col] = 1
    U[u_idx(y4[3][0], y4[3][1])][col] = 1
    V[v_idx(z4[0][0], z4[0][1])][col] = 1
    W[w_idx(w4[2][0], w4[2][1])][col] += 1
    W[w_idx(w4[3][0], w4[3][1])][col] -= 1
    col += 1
    
    # M3
    U[u_idx(y4[0][0], y4[0][1])][col] = 1
    V[v_idx(z4[1][0], z4[1][1])][col] = 1
    V[v_idx(z4[3][0], z4[3][1])][col] = -1
    W[w_idx(w4[1][0], w4[1][1])][col] += 1
    W[w_idx(w4[3][0], w4[3][1])][col] += 1
    col += 1
    
    # M4
    U[u_idx(y4[3][0], y4[3][1])][col] = 1
    V[v_idx(z4[2][0], z4[2][1])][col] = 1
    V[v_idx(z4[0][0], z4[0][1])][col] = -1
    W[w_idx(w4[0][0], w4[0][1])][col] += 1
    W[w_idx(w4[2][0], w4[2][1])][col] += 1
    col += 1
    
    # M5
    U[u_idx(y4[0][0], y4[0][1])][col] = 1
    U[u_idx(y4[1][0], y4[1][1])][col] = 1
    V[v_idx(z4[3][0], z4[3][1])][col] = 1
    W[w_idx(w4[0][0], w4[0][1])][col] -= 1
    W[w_idx(w4[1][0], w4[1][1])][col] += 1
    col += 1
    
    # M6
    U[u_idx(y4[2][0], y4[2][1])][col] = 1
    U[u_idx(y4[0][0], y4[0][1])][col] = -1
    V[v_idx(z4[0][0], z4[0][1])][col] = 1
    V[v_idx(z4[1][0], z4[1][1])][col] = 1
    W[w_idx(w4[3][0], w4[3][1])][col] += 1
    col += 1
    
    # M7
    U[u_idx(y4[1][0], y4[1][1])][col] = 1
    U[u_idx(y4[3][0], y4[3][1])][col] = -1
    V[v_idx(z4[2][0], z4[2][1])][col] = 1
    V[v_idx(z4[3][0], z4[3][1])][col] = 1
    W[w_idx(w4[0][0], w4[0][1])][col] += 1
    col += 7

    # Block 5: A_L B_15 (2x2 * 2x1 -> 2x1)
    # A_L: (0,0), (0,1), (1,0), (1,1)
    # B_15: (0,4), (1,4)
    # Result C_15: (0,4), (1,4)
    # M1: A00*B04 -> C04
    U[u_idx(0,0)][col] = 1
    V[v_idx(0,4)][col] = 1
    W[w_idx(4,0)][col] += 1
    col += 1
    
    # M2: A01*B14 -> C04
    U[u_idx(0,1)][col] = 1
    V[v_idx(1,4)][col] = 1
    W[w_idx(4,0)][col] += 1
    col += 1
    
    # M3: A10*B04 -> C14
    U[u_idx(1,0)][col] = 1
    V[v_idx(0,4)][col] = 1
    W[w_idx(4,1)][col] += 1
    col += 1
    
    # M4: A11*B14 -> C14
    U[u_idx(1,1)][col] = 1
    V[v_idx(1,4)][col] = 1
    W[w_idx(4,1)][col] += 1
    col += 4

    # Block 6: A_R B_25 (2x2 * 2x1 -> 2x1)
    # A_R: (0,2), (0,3), (1,2), (1,3)
    # B_25: (2,4), (3,4)
    # Result C_15: (0,4), (1,4)
    # M1: A02*B24 -> C04
    U[u_idx(0,2)][col] = 1
    V[v_idx(2,4)][col] = 1
    W[w_idx(4,0)][col] += 1
    col += 1
    
    # M2: A03*B34 -> C04
    U[u_idx(0,3)][col] = 1
    V[v_idx(3,4)][col] = 1
    W[w_idx(4,0)][col] += 1
    col += 1
    
    # M3: A12*B24 -> C14
    U[u_idx(1,2)][col] = 1
    V[v_idx(2,4)][col] = 1
    W[w_idx(4,1)][col] += 1
    col += 1
    
    # M4: A13*B34 -> C14
    U[u_idx(1,3)][col] = 1
    V[v_idx(3,4)][col] = 1
    W[w_idx(4,1)][col] += 1
    col += 4

    # Total rank used: 4*7 + 2*4 = 28 + 8 = 36
    # The rank variable is 41, so we have 5 unused columns (all zeros).
    # This is valid.

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
