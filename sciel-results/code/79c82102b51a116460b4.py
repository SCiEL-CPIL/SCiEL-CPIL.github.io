import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    n, m, p = DIMENSIONS
    
    # Try to use a lower rank decomposition if possible.
    # Current best known for 2x4x5 is likely around 30-34.
    # We will implement the rank 36 decomposition as it is known to work.
    # To improve, we can try to optimize the 2x2x1 blocks to rank 3 if possible,
    # but without a verified formula, we stick to rank 4 for 2x2x1.
    
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
        
        # M1
        U[u_idx(y00[0], y00[1])][col] += 1
        U[u_idx(y11[0], y11[1])][col] += 1
        V[v_idx(z00[0], z00[1])][col] += 1
        V[v_idx(z11[0], z11[1])][col] += 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
        col += 1
        
        # M2
        U[u_idx(y10[0], y10[1])][col] += 1
        U[u_idx(y11[0], y11[1])][col] += 1
        V[v_idx(z00[0], z00[1])][col] += 1
        W[w_idx(w_offsets[2][0], w_offsets[2][1])][col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] -= 1
        col += 1
        
        # M3
        U[u_idx(y00[0], y00[1])][col] += 1
        V[v_idx(z01[0], z01[1])][col] += 1
        V[v_idx(z11[0], z11[1])][col] -= 1
        W[w_idx(w_offsets[1][0], w_offsets[1][1])][col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
        col += 1
        
        # M4
        U[u_idx(y11[0], y11[1])][col] += 1
        V[v_idx(z10[0], z10[1])][col] += 1
        V[v_idx(z00[0], z00[1])][col] -= 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
        W[w_idx(w_offsets[2][0], w_offsets[2][1])][col] += 1
        col += 1
        
        # M5
        U[u_idx(y00[0], y00[1])][col] += 1
        U[u_idx(y01[0], y01[1])][col] += 1
        V[v_idx(z11[0], z11[1])][col] += 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] -= 1
        W[w_idx(w_offsets[1][0], w_offsets[1][1])][col] += 1
        col += 1
        
        # M6
        U[u_idx(y10[0], y10[1])][col] += 1
        U[u_idx(y00[0], y00[1])][col] -= 1
        V[v_idx(z00[0], z00[1])][col] += 1
        V[v_idx(z01[0], z01[1])][col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
        col += 1
        
        # M7
        U[u_idx(y01[0], y01[1])][col] += 1
        U[u_idx(y11[0], y11[1])][col] -= 1
        V[v_idx(z10[0], z10[1])][col] += 1
        V[v_idx(z11[0], z11[1])][col] += 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
        col += 1

    # Block 1: A_L B_11
    y1 = [(0,0), (0,1), (1,0), (1,1)]
    z1 = [(0,0), (0,1), (1,0), (1,1)]
    w1 = [(0,0), (1,0), (0,1), (1,1)]
    strassen_block(y1, z1, w1)
    
    # Block 2: A_R B_21
    y2 = [(0,2), (0,3), (1,2), (1,3)]
    z2 = [(2,0), (2,1), (3,0), (3,1)]
    w2 = [(0,0), (1,0), (0,1), (1,1)]
    strassen_block(y2, z2, w2)
    
    # Block 3: A_L B_12
    y3 = [(0,0), (0,1), (1,0), (1,1)]
    z3 = [(0,2), (0,3), (1,2), (1,3)]
    w3 = [(2,0), (3,0), (2,1), (3,1)]
    strassen_block(y3, z3, w3)
    
    # Block 4: A_R B_22
    y4 = [(0,2), (0,3), (1,2), (1,3)]
    z4 = [(2,2), (2,3), (3,2), (3,3)]
    w4 = [(2,0), (3,0), (2,1), (3,1)]
    strassen_block(y4, z4, w4)
    
    # Block 5: A_L B_15 (2x2 * 2x1)
    U[u_idx(0,0)][col] += 1
    V[v_idx(0,4)][col] += 1
    W[w_idx(4,0)][col] += 1
    col += 1
    
    U[u_idx(0,1)][col] += 1
    V[v_idx(1,4)][col] += 1
    W[w_idx(4,0)][col] += 1
    col += 1
    
    U[u_idx(1,0)][col] += 1
    V[v_idx(0,4)][col] += 1
    W[w_idx(4,1)][col] += 1
    col += 1
    
    U[u_idx(1,1)][col] += 1
    V[v_idx(1,4)][col] += 1
    W[w_idx(4,1)][col] += 1
    col += 1
    
    # Block 6: A_R B_25 (2x2 * 2x1)
    U[u_idx(0,2)][col] += 1
    V[v_idx(2,4)][col] += 1
    W[w_idx(4,0)][col] += 1
    col += 1
    
    U[u_idx(0,3)][col] += 1
    V[v_idx(3,4)][col] += 1
    W[w_idx(4,0)][col] += 1
    col += 1
    
    U[u_idx(1,2)][col] += 1
    V[v_idx(2,4)][col] += 1
    W[w_idx(4,1)][col] += 1
    col += 1
    
    U[u_idx(1,3)][col] += 1
    V[v_idx(3,4)][col] += 1
    W[w_idx(4,1)][col] += 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
