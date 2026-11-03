import time
import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    """
    Returns a rank-28 tensor decomposition for 2x4 by 4x4 matrix multiplication
    using Strassen's algorithm on 2x2 blocks.
    """
    n, m, p = DIMENSIONS
    
    # Total rank is 4 blocks * 7 Strassen multiplications = 28
    rank = 28
    
    # Initialize matrices
    # U: (n*m, rank) -> (8, 28)
    # V: (m*p, rank) -> (16, 28)
    # W: (p*n, rank) -> (8, 28)
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Helper to map (i, j) in A to row index in U
    def u_idx(i, j):
        return i * m + j
        
    # Helper to map (j, k) in B to row index in V
    def v_idx(j, k):
        return j * p + k
        
    # Helper to map (k, i) in C to row index in W
    # Task says W[k*n + i, r]
    def w_idx(k, i):
        return k * n + i

    # Define Strassen's 7 products for a 2x2 block
    # X is 2x2, Y is 2x2
    # P1 = (X00 + X11)(Y00 + Y11)
    # P2 = (X10 + X11)Y00
    # P3 = X00(Y01 - Y11)
    # P4 = X11(Y10 - Y00)
    # P5 = (X00 + X01)Y11
    # P6 = (X10 - X00)(Y00 + Y01)
    # P7 = (X01 - X11)(Y10 + Y11)
    
    # Each product corresponds to a rank-1 term.
    # We define the coefficients for U (X part), V (Y part), and W (Z part) for each of the 7 terms.
    # The W part depends on which Z entry the product contributes to.
    
    # Let's create a function that adds a Strassen block to the global matrices.
    # It takes:
    # - x_offsets: list of (i, j) indices in A for X00, X01, X10, X11
    # - y_offsets: list of (j, k) indices in B for Y00, Y01, Y10, Y11
    # - z_offsets: list of (k, i) indices in C for Z00, Z01, Z10, Z11
    # - col_start: starting column index in U, V, W
    
    def add_strassen_block(x_off, y_off, z_off, col_start):
        # x_off: [(i00, j00), (i01, j01), (i10, j10), (i11, j11)]
        # y_off: [(j00, k00), (j01, k01), (j10, k10), (j11, k11)]
        # z_off: [(k00, i00), (k01, i01), (k10, i10), (k11, i11)]
        
        # Extract indices
        x00, x01, x10, x11 = [u_idx(i, j) for i, j in x_off]
        y00, y01, y10, y11 = [v_idx(j, k) for j, k in y_off]
        z00, z01, z10, z11 = [w_idx(k, i) for k, i in z_off]
        
        # 7 terms
        # Term 1: (X00 + X11) * (Y00 + Y11)
        # Contributes to Z00 (+1), Z11 (+1)
        # Wait, Strassen reconstruction:
        # Z00 = P1 + P4 - P5 + P7
        # Z01 = P3 + P5
        # Z10 = P2 + P4
        # Z11 = P1 - P2 + P3 + P6
        
        # We need to distribute the W coefficients for each P.
        
        # P1
        U[x00][col_start] += 1
        U[x11][col_start] += 1
        V[y00][col_start] += 1
        V[y11][col_start] += 1
        W[z00][col_start] += 1
        W[z11][col_start] += 1
        
        # P2
        U[x10][col_start+1] += 1
        U[x11][col_start+1] += 1
        V[y00][col_start+1] += 1
        W[z10][col_start+1] += 1
        W[z11][col_start+1] -= 1
        
        # P3
        U[x00][col_start+2] += 1
        V[y01][col_start+2] += 1
        V[y11][col_start+2] -= 1
        W[z01][col_start+2] += 1
        W[z11][col_start+2] += 1
        
        # P4
        U[x11][col_start+3] += 1
        V[y10][col_start+3] += 1
        V[y00][col_start+3] -= 1
        W[z00][col_start+3] += 1
        W[z10][col_start+3] += 1
        
        # P5
        U[x00][col_start+4] += 1
        U[x01][col_start+4] += 1
        V[y11][col_start+4] += 1
        W[z00][col_start+4] -= 1
        W[z01][col_start+4] += 1
        
        # P6
        U[x10][col_start+5] += 1
        U[x00][col_start+5] -= 1
        V[y00][col_start+5] += 1
        V[y01][col_start+5] += 1
        W[z11][col_start+5] += 1
        
        # P7
        U[x01][col_start+6] += 1
        U[x11][col_start+6] -= 1
        V[y10][col_start+6] += 1
        V[y11][col_start+6] += 1
        W[z00][col_start+6] += 1

    # Block 1: A_L * B_LL -> C_LL
    # A_L: i in {0,1}, j in {0,1}
    # B_LL: j in {0,1}, k in {0,1}
    # C_LL: k in {0,1}, i in {0,1}
    x1 = [(0,0), (0,1), (1,0), (1,1)]
    y1 = [(0,0), (0,1), (1,0), (1,1)]
    z1 = [(0,0), (0,1), (1,0), (1,1)]
    add_strassen_block(x1, y1, z1, 0)
    
    # Block 2: A_R * B_RL -> C_LL
    # A_R: i in {0,1}, j in {2,3}
    # B_RL: j in {2,3}, k in {0,1}
    # C_LL: k in {0,1}, i in {0,1}
    x2 = [(0,2), (0,3), (1,2), (1,3)]
    y2 = [(2,0), (2,1), (3,0), (3,1)]
    z2 = [(0,0), (0,1), (1,0), (1,1)]
    add_strassen_block(x2, y2, z2, 7)
    
    # Block 3: A_L * B_LR -> C_LR
    # A_L: i in {0,1}, j in {0,1}
    # B_LR: j in {0,1}, k in {2,3}
    # C_LR: k in {2,3}, i in {0,1}
    x3 = [(0,0), (0,1), (1,0), (1,1)]
    y3 = [(0,2), (0,3), (1,2), (1,3)]
    z3 = [(2,0), (2,1), (3,0), (3,1)]
    add_strassen_block(x3, y3, z3, 14)
    
    # Block 4: A_R * B_RR -> C_LR
    # A_R: i in {0,1}, j in {2,3}
    # B_RR: j in {2,3}, k in {2,3}
    # C_LR: k in {2,3}, i in {0,1}
    x4 = [(0,2), (0,3), (1,2), (1,3)]
    y4 = [(2,2), (2,3), (3,2), (3,3)]
    z4 = [(2,0), (2,1), (3,0), (3,1)]
    add_strassen_block(x4, y4, z4, 21)
    
    return {"U": U, "V": V, "W": W}

