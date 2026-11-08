"""Matrix multiplication 2,4,5 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 5
    # U shape: (n*m, rank) = (8, rank)
    # V shape: (m*p, rank) = (20, rank)
    # W shape: (p*n, rank) = (10, rank)
    
    rank = 32
    
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
    
    # Strategy: Use 4x4 Strassen for the main part, with careful handling
    # of the 5th column of B and 5th row of result
    
    # We decompose A (2x4) * B (4x5) -> C (2x5)
    # Split A into A1 (2x2) and A2 (2x2) along columns
    # Split B into B1 (4x4) and B2 (4x1) along columns
    
    # For A1 * B1 -> C1 (2x2), use 7 Strassen multiplications
    # For A2 * B1 -> C2 (2x2), use 7 Strassen multiplications  
    # For A1 * B2 -> C3 (2x1), use 2 multiplications
    # For A2 * B2 -> C4 (2x1), use 2 multiplications
    # Total: 7 + 7 + 2 + 2 = 18... but we need to be smarter
    
    # Actually, let me use a different approach: treat this as 
    # (2x4)*(4x5) and use a block Strassen-like scheme
    
    # Let me try: Split A into 2 blocks of 2x2, B into one 4x4 and one 4x1
    # Use Strassen for the 2x2 * 4x4 part (treating B's 4 columns as 2x2 blocks)
    
    # A = [A00 A01] where A00, A01 are 2x2
    # B = [B00 B01 B02] where B00, B01 are 4x2, B02 is 4x1
    # Or better: B = [B1 B2] where B1 is 4x4, B2 is 4x1
    
    # C = [C1 C2] where C1 = A*B1 (2x4), C2 = A*B2 (2x1)
    
    # For C1: split into C11 = A00*B11 + A01*B12 (2x2) and C12 = A00*B13 + A01*B14 (2x2)
    # where B1 = [B11 B12; B13 B14] with each B1i being 2x2
    
    # Use 7 Strassen for C11, 7 Strassen for C12 = 14
    # For C2: 4 direct multiplications (or 2x2 * 2x1 = 2 mults per row pair)
    # Total: 14 + 4 = 18? Still more than 32... 
    
    # Let me try the original approach but more carefully
    # 4 blocks of 2x2*2x2 using Strassen = 4*7 = 28
    # Plus 2 blocks of 2x2*2x1 = 2*2 = 4
    # Total = 32
    
    def strassen_block(y, z, w_map):
        """
        y: 4 indices (i,j) for the 2x2 A-block: (y00, y01, y10, y11)
        z: 4 indices (j,k) for the 2x2 B-block: (z00, z01, z10, z11)
        w_map: dict from (local_row, local_col) to (global_k, global_i)
        """
        nonlocal col
        y00, y01, y10, y11 = y
        z00, z01, z10, z11 = z
        
        # M1 = (y00+y11)*(z00+z11)
        U[u_idx(*y00)][col] += 1
        U[u_idx(*y11)][col] += 1
        V[v_idx(*z00)][col] += 1
        V[v_idx(*z11)][col] += 1
        kr, ki = w_map[0,0]
        W[w_idx(kr,ki)][col] += 1
        kr, ki = w_map[1,1]
        W[w_idx(kr,ki)][col] += 1
        col += 1
        
        # M2 = (y10+y11)*z00
        U[u_idx(*y10)][col] += 1
        U[u_idx(*y11)][col] += 1
        V[v_idx(*z00)][col] += 1
        kr, ki = w_map[1,0]
        W[w_idx(kr,ki)][col] += 1
        kr, ki = w_map[1,1]
        W[w_idx(kr,ki)][col] -= 1
        col += 1
        
        # M3 = y00*(z01-z11)
        U[u_idx(*y00)][col] += 1
        V[v_idx(*z01)][col] += 1
        V[v_idx(*z11)][col] -= 1
        kr, ki = w_map[0,1]
        W[w_idx(kr,ki)][col] += 1
        kr, ki = w_map[1,1]
        W[w_idx(kr,ki)][col] += 1
        col += 1
        
        # M4 = y11*(z10-z00)
        U[u_idx(*y11)][col] += 1
        V[v_idx(*z10)][col] += 1
        V[v_idx(*z00)][col] -= 1
        kr, ki = w_map[0,0]
        W[w_idx(kr,ki)][col] += 1
        kr, ki = w_map[1,0]
        W[w_idx(kr,ki)][col] += 1
        col += 1
        
        # M5 = (y00+y01)*z11
        U[u_idx(*y00)][col] += 1
        U[u_idx(*y01)][col] += 1
        V[v_idx(*z11)][col] += 1
        kr, ki = w_map[0,0]
        W[w_idx(kr,ki)][col] -= 1
        kr, ki = w_map[0,1]
        W[w_idx(kr,ki)][col] += 1
        col += 1
        
        # M6 = (y10-y00)*(z00+z01)
        U[u_idx(*y10)][col] += 1
        U[u_idx(*y00)][col] -= 1
        V[v_idx(*z00)][col] += 1
        V[v_idx(*z01)][col] += 1
        kr, ki = w_map[1,1]
        W[w_idx(kr,ki)][col] += 1
        col += 1
        
        # M7 = (y01-y11)*(z10+z11)
        U[u_idx(*y01)][col] += 1
        U[u_idx(*y11)][col] -= 1
        V[v_idx(*z10)][col] += 1
        V[v_idx(*z11)][col] += 1
        kr, ki = w_map[0,0]
        W[w_idx(kr,ki)][col] += 1
        col += 1
    
    # Block 1: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    y1 = [(0,0),(0,1),(1,0),(1,1)]
    z1 = [(0,0),(0,1),(1,0),(1,1)]
    w1 = {(0,0):(0,0), (0,1):(1,0), (1,0):(0,1), (1,1):(1,1)}
    strassen_block(y1, z1, w1)
    
    # Block 2: A[0:2,2:4] * B[2:4,0:2] -> C[0:2,0:2]
    y2 = [(0,2),(0,3),(1,2),(1,3)]
    z2 = [(2,0),(2,1),(3,0),(3,1)]
    w2 = {(0,0):(0,0), (0,1):(1,0), (1,0):(0,1), (1,1):(1,1)}
    strassen_block(y2, z2, w2)
    
    # Block 3: A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
    y3 = [(0,0),(0,1),(1,0),(1,1)]
    z3 = [(0,2),(0,3),(1,2),(1,3)]
    w3 = {(0,0):(2,0), (0,1):(3,0), (1,0):(2,1), (1,1):(3,1)}
    strassen_block(y3, z3, w3)
    
    # Block 4: A[0:2,2:4] * B[2:4,2:4] -> C[0:2,2:4]
    y4 = [(0,2),(0,3),(1,2),(1,3)]
    z4 = [(2,2),(2,3),(3,2),(3,3)]
    w4 = {(0,0):(2,0), (0,1):(3,0), (1,0):(2,1), (1,1):(3,1)}
    strassen_block(y4, z4, w4)
    
    # Block 5: A[0:2,0:2] * B[0:2,4:5] -> C[0:2,4:5] (2x2 * 2x1 = 2 mults)
    U[u_idx(0,0)][col] = 1; V[v_idx(0,4)][col] = 1; W[w_idx(4,0)][col] = 1; col += 1
    U[u_idx(0,1)][col] = 1; V[v_idx(1,4)][col] = 1; W[w_idx(4,0)][col] = 1; col += 1
    
    # Block 6: A[0:2,2:4] * B[2:4,4:5] -> C[0:2,4:5] (2x2 * 2x1 = 2 mults)
    U[u_idx(0,2)][col] = 1; V[v_idx(2,4)][col] = 1; W[w_idx(4,0)][col] = 1; col += 1
    U[u_idx(0,3)][col] = 1; V[v_idx(3,4)][col] = 1; W[w_idx(4,0)][col] = 1; col += 1
    
    # Block 7: A[1:2,0:2] * B[0:2,4:5] -> C[1:2,4:5] (1x2 * 2x1 = 2 mults)
    U[u_idx(1,0)][col] = 1; V[v_idx(0,4)][col] = 1; W[w_idx(4,1)][col] = 1; col += 1
    U[u_idx(1,1)][col] = 1; V[v_idx(1,4)][col] = 1; W[w_idx(4,1)][col] = 1; col += 1
    
    # Block 8: A[1:2,2:4] * B[2:4,4:5] -> C[1:2,4:5] (1x2 * 2x1 = 2 mults)
    U[u_idx(1,2)][col] = 1; V[v_idx(2,4)][col] = 1; W[w_idx(4,1)][col] = 1; col += 1
    U[u_idx(1,3)][col] = 1; V[v_idx(3,4)][col] = 1; W[w_idx(4,1)][col] = 1; col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
