"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 120.0
    n, m, p = DIMENSIONS
    
    # Dimensions
    # U: (n*m, rank) = (8, rank)
    # V: (m*p, rank) = (20, rank)
    # W: (p*n, rank) = (10, rank)
    
    rank = 36
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    def u_idx(i, j):
        return i * m + j

    def v_idx(j, k):
        return j * p + k

    def w_idx(k, i):
        return k * n + i
    
    col = 0

    def add_strassen(y_indices, z_indices, w_offsets):
        nonlocal col
        y00, y01, y10, y11 = y_indices
        z00, z01, z10, z11 = z_indices
        w0, w1, w2, w3 = w_offsets
        
        # M1
        U[u_idx(y00[0], y00[1]), col] += 1
        U[u_idx(y11[0], y11[1]), col] += 1
        V[v_idx(z00[0], z00[1]), col] += 1
        V[v_idx(z11[0], z11[1]), col] += 1
        W[w_idx(w0[0], w0[1]), col] += 1
        W[w_idx(w3[0], w3[1]), col] += 1
        col += 1
        
        # M2
        U[u_idx(y10[0], y10[1]), col] += 1
        U[u_idx(y11[0], y11[1]), col] += 1
        V[v_idx(z00[0], z00[1]), col] += 1
        W[w_idx(w2[0], w2[1]), col] += 1
        W[w_idx(w3[0], w3[1]), col] -= 1
        col += 1
        
        # M3
        U[u_idx(y00[0], y00[1]), col] += 1
        V[v_idx(z01[0], z01[1]), col] += 1
        V[v_idx(z11[0], z11[1]), col] -= 1
        W[w_idx(w1[0], w1[1]), col] += 1
        W[w_idx(w3[0], w3[1]), col] += 1
        col += 1
        
        # M4
        U[u_idx(y11[0], y11[1]), col] += 1
        V[v_idx(z10[0], z10[1]), col] += 1
        V[v_idx(z00[0], z00[1]), col] -= 1
        W[w_idx(w0[0], w0[1]), col] += 1
        W[w_idx(w2[0], w2[1]), col] += 1
        col += 1
        
        # M5
        U[u_idx(y00[0], y00[1]), col] += 1
        U[u_idx(y01[0], y01[1]), col] += 1
        V[v_idx(z11[0], z11[1]), col] += 1
        W[w_idx(w0[0], w0[1]), col] -= 1
        W[w_idx(w1[0], w1[1]), col] += 1
        col += 1
        
        # M6
        U[u_idx(y10[0], y10[1]), col] += 1
        U[u_idx(y00[0], y00[1]), col] -= 1
        V[v_idx(z00[0], z00[1]), col] += 1
        V[v_idx(z01[0], z01[1]), col] += 1
        W[w_idx(w3[0], w3[1]), col] += 1
        col += 1
        
        # M7
        U[u_idx(y01[0], y01[1]), col] += 1
        U[u_idx(y11[0], y11[1]), col] -= 1
        V[v_idx(z10[0], z10[1]), col] += 1
        V[v_idx(z11[0], z11[1]), col] += 1
        W[w_idx(w0[0], w0[1]), col] += 1
        col += 1

    # Block 1
    y1 = [(0,0), (0,1), (1,0), (1,1)]
    z1 = [(0,0), (0,1), (1,0), (1,1)]
    w1 = [(0,0), (1,0), (0,1), (1,1)]
    add_strassen(y1, z1, w1)
    
    # Block 2
    y2 = [(0,2), (0,3), (1,2), (1,3)]
    z2 = [(2,0), (2,1), (3,0), (3,1)]
    w2 = [(0,0), (1,0), (0,1), (1,1)]
    add_strassen(y2, z2, w2)
    
    # Block 3
    y3 = [(0,0), (0,1), (1,0), (1,1)]
    z3 = [(0,2), (0,3), (1,2), (1,3)]
    w3 = [(2,0), (3,0), (2,1), (3,1)]
    add_strassen(y3, z3, w3)
    
    # Block 4
    y4 = [(0,2), (0,3), (1,2), (1,3)]
    z4 = [(2,2), (2,3), (3,2), (3,3)]
    w4 = [(2,0), (3,0), (2,1), (3,1)]
    add_strassen(y4, z4, w4)
    
    # Block 5: Direct 2x2 * 2x1
    # A_L * B_15
    # (0,0)*(0,4) -> (4,0)
    U[u_idx(0,0), col] += 1; V[v_idx(0,4), col] += 1; W[w_idx(4,0), col] += 1; col += 1
    # (0,1)*(1,4) -> (4,0)
    U[u_idx(0,1), col] += 1; V[v_idx(1,4), col] += 1; W[w_idx(4,0), col] += 1; col += 1
    # (1,0)*(0,4) -> (4,1)
    U[u_idx(1,0), col] += 1; V[v_idx(0,4), col] += 1; W[w_idx(4,1), col] += 1; col += 1
    # (1,1)*(1,4) -> (4,1)
    U[u_idx(1,1), col] += 1; V[v_idx(1,4), col] += 1; W[w_idx(4,1), col] += 1; col += 1
    
    # Block 6: Direct 2x2 * 2x1
    # A_R * B_25
    # (0,2)*(2,4) -> (4,0)
    U[u_idx(0,2), col] += 1; V[v_idx(2,4), col] += 1; W[w_idx(4,0), col] += 1; col += 1
    # (0,3)*(3,4) -> (4,0)
    U[u_idx(0,3), col] += 1; V[v_idx(3,4), col] += 1; W[w_idx(4,0), col] += 1; col += 1
    # (1,2)*(2,4) -> (4,1)
    U[u_idx(1,2), col] += 1; V[v_idx(2,4), col] += 1; W[w_idx(4,1), col] += 1; col += 1
    # (1,3)*(3,4) -> (4,1)
    U[u_idx(1,3), col] += 1; V[v_idx(3,4), col] += 1; W[w_idx(4,1), col] += 1; col += 1
    
    # Convert to lists of lists as required by the interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
