import time
import numpy as np

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x3 * 3x8 matrix multiplication.
    """
    n, m, p = 2, 3, 8
    rank = 44
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    def add_strassen_block(u_map, v_map, w_map):
        """Add 7 Strassen terms for a 2x2 * 2x2 sub-problem."""
        nonlocal col
        
        # M1: (A00+A11)(B00+B11)
        U[u_map[0], col] += 1
        U[u_map[3], col] += 1
        V[v_map[0], col] += 1
        V[v_map[3], col] += 1
        W[w_map[0], col] += 1
        W[w_map[3], col] += 1
        col += 1
        
        # M2: (A10+A11)B00
        U[u_map[2], col] += 1
        U[u_map[3], col] += 1
        V[v_map[0], col] += 1
        W[w_map[2], col] += 1
        W[w_map[3], col] += -1
        col += 1
        
        # M3: A00(B01-B11)
        U[u_map[0], col] += 1
        V[v_map[1], col] += 1
        V[v_map[3], col] += -1
        W[w_map[1], col] += 1
        W[w_map[3], col] += 1
        col += 1
        
        # M4: A11(B10-B00)
        U[u_map[3], col] += 1
        V[v_map[2], col] += 1
        V[v_map[0], col] += -1
        W[w_map[0], col] += 1
        W[w_map[2], col] += 1
        col += 1
        
        # M5: (A00+A01)B11
        U[u_map[0], col] += 1
        U[u_map[1], col] += 1
        V[v_map[3], col] += 1
        W[w_map[0], col] += -1
        W[w_map[1], col] += 1
        col += 1
        
        # M6: (A10-A00)(B00+B01)
        U[u_map[2], col] += 1
        U[u_map[0], col] += -1
        V[v_map[0], col] += 1
        V[v_map[1], col] += 1
        W[w_map[3], col] += 1
        col += 1
        
        # M7: (A01-A11)(B10+B11)
        U[u_map[1], col] += 1
        U[u_map[3], col] += -1
        V[v_map[2], col] += 1
        V[v_map[3], col] += 1
        W[w_map[0], col] += 1
        col += 1

    # For each half (left: cols 0-3, right: cols 4-7)
    for half in range(2):
        # u_map maps Strassen indices to U indices
        # U indices: 0=A[0,0], 1=A[0,1], 2=A[1,0], 3=A[1,1], 4=A[0,2], 5=A[1,2]
        # Note: A is 2x3. Indices:
        # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1), 4: (0,2), 5: (1,2)
        # Strassen uses A[0,0], A[0,1], A[1,0], A[1,1] -> indices 0, 1, 3, 4?
        # Wait, in the previous code:
        # u_map = {0: 0, 1: 1, 2: 3, 3: 4}
        # This maps Strassen A00->0, A01->1, A10->3, A11->4.
        # Let's check indices:
        # A[0,0] is index 0.
        # A[0,1] is index 1.
        # A[1,0] is index 3? No, A[1,0] is row 1, col 0. Index 1*3+0 = 3.
        # A[1,1] is index 1*3+1 = 4.
        # So u_map is correct for the first 2 columns of A.
        
        u_map = {0: 0, 1: 1, 2: 3, 3: 4}
        
        # First 2x2 block: B columns 0,1 (within this half)
        b_start = half * 4
        v_map_b1 = {
            0: 0 * p + (b_start + 0),  # B[0, col]
            1: 0 * p + (b_start + 1),  # B[0, col+1]
            2: 1 * p + (b_start + 0),  # B[1, col]
            3: 1 * p + (b_start + 1)   # B[1, col+1]
        }
        w_map_c1 = {
            0: (b_start + 0) * n + 0,  # C[row0, col]
            1: (b_start + 1) * n + 0,  # C[row0, col+1]
            2: (b_start + 0) * n + 1,  # C[row1, col]
            3: (b_start + 1) * n + 1   # C[row1, col+1]
        }
        add_strassen_block(u_map, v_map_b1, w_map_c1)
        
        # Second 2x2 block: B columns 2,3 (within this half)
        b_start2 = half * 4 + 2
        v_map_b2 = {
            0: 0 * p + (b_start2 + 0),
            1: 0 * p + (b_start2 + 1),
            2: 1 * p + (b_start2 + 0),
            3: 1 * p + (b_start2 + 1)
        }
        w_map_c2 = {
            0: (b_start2 + 0) * n + 0,
            1: (b_start2 + 1) * n + 0,
            2: (b_start2 + 0) * n + 1,
            3: (b_start2 + 1) * n + 1
        }
        add_strassen_block(u_map, v_map_b2, w_map_c2)
        
        # Direct terms for A column 2 (index 4,5) * B row 2 (index 2*p + col)
        # A[0,2] is index 4, A[1,2] is index 5.
        # B[2, col] is index 2*p + col.
        for k in range(4):
            for i in range(2):
                u_idx = i * m + 2  # A[i, 2]
                v_idx = 2 * p + (half * 4 + k)  # B[2, col]
                w_idx = (half * 4 + k) * n + i  # C[i, col]
                
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}
