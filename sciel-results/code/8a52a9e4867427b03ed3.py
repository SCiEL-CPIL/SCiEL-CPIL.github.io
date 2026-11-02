import numpy as np

DIMENSIONS = [2, 2, 15]

def construct():
    n, m, p = DIMENSIONS
    
    # U: (n*m, rank) -> (4, rank)
    # V: (m*p, rank) -> (30, rank)
    # W: (p*n, rank) -> (30, rank)
    
    # Rank calculation:
    # 7 blocks of size 2 cols * rank 7 = 49
    # 1 block of size 1 col * rank 4 = 4
    # Total rank = 53
    rank = 53
    
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    col_offset = 0
    
    # Strassen blocks
    for block in range(7):
        # Columns are 2*block and 2*block+1
        k0 = 2 * block
        k1 = 2 * block + 1
        
        # U indices for A elements
        # A is 2x2. Flat index i*m+j
        u_00 = 0
        u_01 = 1
        u_10 = 2
        u_11 = 3
        
        # V indices for B elements
        # B is 2x15. Flat index j*p+k
        v_0_k0 = 0 * p + k0
        v_1_k0 = 1 * p + k0
        v_0_k1 = 0 * p + k1
        v_1_k1 = 1 * p + k1
        
        # W indices for C elements
        # C is 2x15. Flat index k*n+i (Column major: col k, row i)
        # i is row 0 or 1. k is col.
        w_0_k0 = k0 * n + 0
        w_1_k0 = k0 * n + 1
        w_0_k1 = k1 * n + 0
        w_1_k1 = k1 * n + 1
        
        # Start index for this block's rank
        start = block * 7
        
        # P1 = (A00 + A11)(B0k0 + B1k1)
        U[u_00, start] += 1
        U[u_11, start] += 1
        V[v_0_k0, start] += 1
        V[v_1_k1, start] += 1
        # C0k0 += P1, C1k1 += P1
        W[w_0_k0, start] += 1
        W[w_1_k1, start] += 1
        
        # P2 = (A10 + A11)B0k1
        U[u_10, start+1] += 1
        U[u_11, start+1] += 1
        V[v_0_k1, start+1] += 1
        # C0k1 += P2
        W[w_0_k1, start+1] += 1
        
        # P3 = A00(B1k0 - B1k1)
        U[u_00, start+2] += 1
        V[v_1_k0, start+2] += 1
        V[v_1_k1, start+2] -= 1
        # C1k0 += P3
        W[w_1_k0, start+2] += 1
        
        # P4 = A11(B0k0 - B0k1)
        U[u_11, start+3] += 1
        V[v_0_k0, start+3] += 1
        V[v_0_k1, start+3] -= 1
        # C0k0 += P4, C1k0 += P4
        W[w_0_k0, start+3] += 1
        W[w_1_k0, start+3] += 1
        
        # P5 = (A00 + A01)B1k1
        U[u_00, start+4] += 1
        U[u_01, start+4] += 1
        V[v_1_k1, start+4] += 1
        # C0k0 -= P5, C0k1 += P5
        W[w_0_k0, start+4] -= 1
        W[w_0_k1, start+4] += 1
        
        # P6 = (A10 - A00)(B0k0 + B0k1)
        U[u_10, start+5] += 1
        U[u_00, start+5] -= 1
        V[v_0_k0, start+5] += 1
        V[v_0_k1, start+5] += 1
        # C1k1 += P6
        W[w_1_k1, start+5] += 1
        
        # P7 = (A01 - A11)(B1k0 + B1k1)
        U[u_01, start+6] += 1
        U[u_11, start+6] -= 1
        V[v_1_k0, start+6] += 1
        V[v_1_k1, start+6] += 1
        # C0k0 += P7
        W[w_0_k0, start+6] += 1

    # Last block: Column 14
    # Standard multiplication for A (2x2) * B_col (2x1) -> C_col (2x1)
    # C0 = A00*B0 + A01*B1
    # C1 = A10*B0 + A11*B1
    k_last = 14
    
    # Q1 = A00 * B0_k
    U[u_00, 49] += 1
    V[v_0_k0_last := 0*p + k_last, 49] += 1
    W[w_0_k0_last := k_last*n + 0, 49] += 1
    
    # Q2 = A01 * B1_k
    U[u_01, 50] += 1
    V[v_1_k0_last := 1*p + k_last, 50] += 1
    W[w_0_k0_last, 50] += 1
    
    # Q3 = A10 * B0_k
    U[u_10, 51] += 1
    V[v_0_k0_last, 51] += 1
    W[w_1_k0_last := k_last*n + 1, 51] += 1
    
    # Q4 = A11 * B1_k
    U[u_11, 52] += 1
    V[v_1_k0_last, 52] += 1
    W[w_1_k0_last, 52] += 1

    return {"U": U, "V": V, "W": W}
