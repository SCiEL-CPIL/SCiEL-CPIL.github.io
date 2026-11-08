import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # 145s limit, 5s margin
    
    n, m, p = DIMENSIONS
    # n=2, m=4, p=5
    # U shape: (n*m, rank) = (8, rank)
    # V shape: (m*p, rank) = (20, rank)
    # W shape: (p*n, rank) = (10, rank)
    
    # We use a rank 36 decomposition.
    rank = 36
    
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
    
    def strassen_block(y_indices, z_indices, w_offsets):
        nonlocal col
        if time.time() > deadline:
            raise TimeoutError("Deadline exceeded")
            
        y00, y01, y10, y11 = y_indices
        z00, z01, z10, z11 = z_indices
        
        # M1: (y00 + y11) * (z00 + z11)
        U[u_idx(y00[0], y00[1])][col] += 1
        U[u_idx(y11[0], y11[1])][col] += 1
        V[v_idx(z00[0], z00[1])][col] += 1
        V[v_idx(z11[0], z11[1])][col] += 1
        # W: X00 += M1, X11 += M1
        W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
        col += 1
        
        # M2: (y10 + y11) * z00
        U[u_idx(y10[0], y10[1])][col] += 1
        U[u_idx(y11[0], y11[1])][col] += 1
        V[v_idx(z00[0], z00[1])][col] += 1
        # W: X10 += M2, X11 -= M2
        W[w_idx(w_offsets[2][0], w_offsets[2][1])][col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] -= 1
        col += 1
        
        # M3: y00 * (z01 - z11)
        U[u_idx(y00[0], y00[1])][col] += 1
        V[v_idx(z01[0], z01[1])][col] += 1
        V[v_idx(z11[0], z11[1])][col] -= 1
        # W: X01 += M3, X11 += M3
        W[w_idx(w_offsets[1][0], w_offsets[1][1])][col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
        col += 1
        
        # M4: y11 * (z10 - z00)
        U[u_idx(y11[0], y11[1])][col] += 1
        V[v_idx(z10[0], z10[1])][col] += 1
        V[v_idx(z00[0], z00[1])][col] -= 1
        # W: X00 += M4, X10 += M4
        W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
        W[w_idx(w_offsets[2][0], w_offsets[2][1])][col] += 1
        col += 1
        
        # M5: (y00 + y01) * z11
        U[u_idx(y00[0], y00[1])][col] += 1
        U[u_idx(y01[0], y01[1])][col] += 1
        V[v_idx(z11[0], z11[1])][col] += 1
        # W: X00 -= M5, X01 += M5
        W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] -= 1
        W[w_idx(w_offsets[1][0], w_offsets[1][1])][col] += 1
        col += 1
        
        # M6: (y10 - y00) * (z00 + z01)
        U[u_idx(y10[0], y10[1])][col] += 1
        U[u_idx(y00[0], y00[1])][col] -= 1
        V[v_idx(z00[0], z00[1])][col] += 1
        V[v_idx(z01[0], z01[1])][col] += 1
        # W: X11 += M6
        W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
        col += 1
        
        # M7: (y01 - y11) * (z10 + z11)
        U[u_idx(y01[0], y01[1])][col] += 1
        U[u_idx(y11[0], y11[1])][col] -= 1
        V[v_idx(z10[0], z10[1])][col] += 1
        V[v_idx(z11[0], z11[1])][col] += 1
        # W: X00 += M7
        W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
        col += 1

    # Block 1: A_L B_11 (2x2 * 2x2)
    y1 = [(0,0), (0,1), (1,0), (1,1)]
    z1 = [(0,0), (0,1), (1,0), (1,1)]
    w1 = [(0,0), (1,0), (0,1), (1,1)]
    strassen_block(y1, z1, w1)
    
    # Block 2: A_R B_21 (2x2 * 2x2)
    y2 = [(0,2), (0,3), (1,2), (1,3)]
    z2 = [(2,0), (2,1), (3,0), (3,1)]
    w2 = [(0,0), (1,0), (0,1), (1,1)]
    strassen_block(y2, z2, w2)
    
    # Block 3: A_L B_12 (2x2 * 2x2)
    y3 = [(0,0), (0,1), (1,0), (1,1)]
    z3 = [(0,2), (0,3), (1,2), (1,3)]
    w3 = [(2,0), (3,0), (2,1), (3,1)]
    strassen_block(y3, z3, w3)
    
    # Block 4: A_R B_22 (2x2 * 2x2)
    y4 = [(0,2), (0,3), (1,2), (1,3)]
    z4 = [(2,2), (2,3), (3,2), (3,3)]
    w4 = [(2,0), (3,0), (2,1), (3,1)]
    strassen_block(y4, z4, w4)
    
    # Block 5: A_L B_15 (2x2 * 2x1)
    if time.time() > deadline:
        raise TimeoutError("Deadline exceeded")
        
    # M1: A00*B04 -> C04
    U[u_idx(0,0)][col] += 1
    V[v_idx(0,4)][col] += 1
    W[w_idx(4,0)][col] += 1
    col += 1
    
    # M2: A01*B14 -> C04
    U[u_idx(0,1)][col] += 1
    V[v_idx(1,4)][col] += 1
    W[w_idx(4,0)][col] += 1
    col += 1
    
    # M3: A10*B04 -> C14
    U[u_idx(1,0)][col] += 1
    V[v_idx(0,4)][col] += 1
    W[w_idx(4,1)][col] += 1
    col += 1
    
    # M4: A11*B14 -> C14
    U[u_idx(1,1)][col] += 1
    V[v_idx(1,4)][col] += 1
    W[w_idx(4,1)][col] += 1
    col += 1
    
    # Block 6: A_R B_25 (2x2 * 2x1)
    if time.time() > deadline:
        raise TimeoutError("Deadline exceeded")
        
    # M1: A02*B24 -> C04
    U[u_idx(0,2)][col] += 1
    V[v_idx(2,4)][col] += 1
    W[w_idx(4,0)][col] += 1
    col += 1
    
    # M2: A03*B34 -> C04
    U[u_idx(0,3)][col] += 1
    V[v_idx(3,4)][col] += 1
    W[w_idx(4,0)][col] += 1
    col += 1
    
    # M3: A12*B24 -> C14
    U[u_idx(1,2)][col] += 1
    V[v_idx(2,4)][col] += 1
    W[w_idx(4,1)][col] += 1
    col += 1
    
    # M4: A13*B34 -> C14
    U[u_idx(1,3)][col] += 1
    V[v_idx(3,4)][col] += 1
    W[w_idx(4,1)][col] += 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
