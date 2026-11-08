"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    
    # U: (n*m, rank) = (8, rank)
    # V: (m*p, rank) = (20, rank)
    # W: (p*n, rank) = (10, rank)
    
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
    
    # Strassen 2x2x2: 7 multiplications
    def strassen_2x2(y, z, w_offsets):
        """
        y: list of 4 (i,j) indices for the 2x2 Y block
        z: list of 4 (j,k) indices for the 2x2 Z block
        w_offsets: list of 4 (k,i) indices for the 2x2 result positions
        """
        nonlocal col
        y00, y01, y10, y11 = y
        z00, z01, z10, z11 = z
        
        # M1 = (y00+y11)(z00+z11)
        U[u_idx(*y00)][col] = 1
        U[u_idx(*y11)][col] = 1
        V[v_idx(*z00)][col] = 1
        V[v_idx(*z11)][col] = 1
        W[w_idx(*w_offsets[0])][col] += 1
        W[w_idx(*w_offsets[3])][col] += 1
        col += 1
        
        # M2 = (y10+y11)z00
        U[u_idx(*y10)][col] = 1
        U[u_idx(*y11)][col] = 1
        V[v_idx(*z00)][col] = 1
        W[w_idx(*w_offsets[2])][col] += 1
        W[w_idx(*w_offsets[3])][col] -= 1
        col += 1
        
        # M3 = y00(z01-z11)
        U[u_idx(*y00)][col] = 1
        V[v_idx(*z01)][col] = 1
        V[v_idx(*z11)][col] = -1
        W[w_idx(*w_offsets[1])][col] += 1
        W[w_idx(*w_offsets[3])][col] += 1
        col += 1
        
        # M4 = y11(z10-z00)
        U[u_idx(*y11)][col] = 1
        V[v_idx(*z10)][col] = 1
        V[v_idx(*z00)][col] = -1
        W[w_idx(*w_offsets[0])][col] += 1
        W[w_idx(*w_offsets[2])][col] += 1
        col += 1
        
        # M5 = (y00+y01)z11
        U[u_idx(*y00)][col] = 1
        U[u_idx(*y01)][col] = 1
        V[v_idx(*z11)][col] = 1
        W[w_idx(*w_offsets[0])][col] -= 1
        W[w_idx(*w_offsets[1])][col] += 1
        col += 1
        
        # M6 = (y10-y00)(z00+z01)
        U[u_idx(*y10)][col] = 1
        U[u_idx(*y00)][col] = -1
        V[v_idx(*z00)][col] = 1
        V[v_idx(*z01)][col] = 1
        W[w_idx(*w_offsets[3])][col] += 1
        col += 1
        
        # M7 = (y01-y11)(z10+z11)
        U[u_idx(*y01)][col] = 1
        U[u_idx(*y11)][col] = -1
        V[v_idx(*z10)][col] = 1
        V[v_idx(*z11)][col] = 1
        W[w_idx(*w_offsets[0])][col] += 1
        col += 1
    
    # Block structure:
    # A = [A_L | A_R] where A_L is 2x2 (cols 0,1), A_R is 2x2 (cols 2,3)
    # B rows 0,1 -> B_T (2x5), rows 2,3 -> B_B (2x5)
    # C = A_L * B_T + A_R * B_B
    
    A_L = [(0,0), (0,1), (1,0), (1,1)]
    A_R = [(0,2), (0,3), (1,2), (1,3)]
    
    # For 2x2 * 2x5: split B into cols 0-1, 2-3, 4
    B_T_01 = [(0,0), (0,1), (1,0), (1,1)]
    C_01 = [(0,0), (1,0), (0,1), (1,1)]
    strassen_2x2(A_L, B_T_01, C_01)
    
    B_T_23 = [(0,2), (0,3), (1,2), (1,3)]
    C_23 = [(2,0), (3,0), (2,1), (3,1)]
    strassen_2x2(A_L, B_T_23, C_23)
    
    # Block 3: A_L * B_T[:,4] -> C[:,4] (2x1 result, trivial rank 4)
    U[u_idx(0,0)][col] = 1
    V[v_idx(0,4)][col] = 1
    W[w_idx(4,0)][col] += 1
    col += 1
    
    U[u_idx(0,1)][col] = 1
    V[v_idx(1,4)][col] = 1
    W[w_idx(4,0)][col] += 1
    col += 1
    
    U[u_idx(1,0)][col] = 1
    V[v_idx(0,4)][col] = 1
    W[w_idx(4,1)][col] += 1
    col += 1
    
    U[u_idx(1,1)][col] = 1
    V[v_idx(1,4)][col] = 1
    W[w_idx(4,1)][col] += 1
    col += 1
    
    # Block 4: A_R * B_B[:,0:2] -> C[:,0:2] (2x2 result)
    B_B_01 = [(2,0), (2,1), (3,0), (3,1)]
    strassen_2x2(A_R, B_B_01, C_01)
    
    # Block 5: A_R * B_B[:,2:4] -> C[:,2:4] (2x2 result)
    B_B_23 = [(2,2), (2,3), (3,2), (3,3)]
    strassen_2x2(A_R, B_B_23, C_23)
    
    # Block 6: A_R * B_B[:,4] -> C[:,4] (2x1 result, trivial rank 4)
    U[u_idx(0,2)][col] = 1
    V[v_idx(2,4)][col] = 1
    W[w_idx(4,0)][col] += 1
    col += 1
    
    U[u_idx(0,3)][col] = 1
    V[v_idx(3,4)][col] = 1
    W[w_idx(4,0)][col] += 1
    col += 1
    
    U[u_idx(1,2)][col] = 1
    V[v_idx(2,4)][col] = 1
    W[w_idx(4,1)][col] += 1
    col += 1
    
    U[u_idx(1,3)][col] = 1
    V[v_idx(3,4)][col] = 1
    W[w_idx(4,1)][col] += 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
