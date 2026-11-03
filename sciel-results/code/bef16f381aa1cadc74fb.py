import numpy as np
import time

DIMENSIONS = [2, 4, 8]
_START_TIME = time.time()
_TIME_LIMIT = 140.0  # Safety margin for 150s limit

def construct():
    """
    Rank-minimal tensor decomposition for 2x4 * 4x8 matrix multiplication.
    Uses the known rank-56 Strassen decomposition.
    """
    n, m, p = DIMENSIONS
    rank = 56
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen coefficients for 2x2 blocks
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    # U coefficients (A side)
    # Local A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    s_coeffs = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ]
    
    # V coefficients (B side)
    # Local B indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    t_coeffs = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1], # M3
        [-1, 0, 1, 0], # M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    
    # W coefficients (C side)
    # Local C indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Note: W index convention is k*n + i.
    # Local C block is 2x2.
    # c00 is C[0,0], c01 is C[0,1], c10 is C[1,0], c11 is C[1,1]
    c_coeffs = [
        [1, 0, 0, 1], # M1 -> C00, C11
        [0, 0, 1, -1], # M2 -> C10, C11
        [0, 1, 0, 1], # M3 -> C01, C11
        [1, 0, 1, 0], # M4 -> C00, C10
        [-1, 1, 0, 0], # M5 -> C00, C01
        [0, 0, 0, 1], # M6 -> C11
        [1, 0, 0, 0]  # M7 -> C00
    ]
    
    # Iterate over the 8 blocks: 2 row-blocks of A/B (s=0,1) and 4 col-blocks of B/C (j=0,1,2,3)
    # Actually, the partition is:
    # A is 2x4. Split into A_top (2x2) and A_bot (2x2)? No, A is 2 rows.
    # The standard Strassen block decomposition for 2x4 x 4x8:
    # Partition A into 2 blocks of 2x2? No, A has 2 rows.
    # We partition the 4 columns of A into 2 blocks of 2 columns.
    # We partition the 4 rows of B into 2 blocks of 2 rows.
    # We partition the 8 columns of B into 4 blocks of 2 columns.
    # This creates 2 (row blocks of B) * 4 (col blocks of B) = 8 independent 2x2x2 problems.
    
    for s in range(2):  # Row block of B (and corresponding col block of A)
        for j in range(4):  # Col block of B (and corresponding col block of C)
            
            # Determine global indices for this block
            # A: 2x4. Block s uses columns 2*s and 2*s+1.
            # A_col_start = 2*s
            # A indices:
            # a00: A[0, 2*s]
            # a01: A[0, 2*s+1]
            # a10: A[1, 2*s]
            # a11: A[1, 2*s+1]
            
            # B: 4x8. Block s (rows 2*s, 2*s+1) and j (cols 2*j, 2*j+1).
            # B_row_start = 2*s
            # B_col_start = 2*j
            # B indices:
            # b00: B[2*s, 2*j]
            # b01: B[2*s, 2*j+1]
            # b10: B[2*s+1, 2*j]
            # b11: B[2*s+1, 2*j+1]
            
            # C: 2x8. Block j (cols 2*j, 2*j+1).
            # C_col_start = 2*j
            # C indices:
            # c00: C[0, 2*j]
            # c01: C[0, 2*j+1]
            # c10: C[1, 2*j]
            # c11: C[1, 2*j+1]
            
            # Rank offset for this block
            r_start = (s * 4 + j) * 7
            
            for t in range(7):
                r = r_start + t
                s_c = s_coeffs[t]
                t_c = t_coeffs[t]
                c_c = c_coeffs[t]
                
                # Fill U
                # U index: i*m + j_global
                # i=0, j_global = 2*s
                U[0*4 + (2*s), r] += s_c[0]
                # i=0, j_global = 2*s+1
                U[0*4 + (2*s+1), r] += s_c[1]
                # i=1, j_global = 2*s
                U[1*4 + (2*s), r] += s_c[2]
                # i=1, j_global = 2*s+1
                U[1*4 + (2*s+1), r] += s_c[3]
                
                # Fill V
                # V index: j_global*p + k_global
                # j_global = 2*s, k_global = 2*j
                V[(2*s)*8 + (2*j), r] += t_c[0]
                # j_global = 2*s, k_global = 2*j+1
                V[(2*s)*8 + (2*j+1), r] += t_c[1]
                # j_global = 2*s+1, k_global = 2*j
                V[(2*s+1)*8 + (2*j), r] += t_c[2]
                # j_global = 2*s+1, k_global = 2*j+1
                V[(2*s+1)*8 + (2*j+1), r] += t_c[3]
                
                # Fill W
                # W index: k_global*n + i_global
                # k_global = 2*j, i_global = 0
                W[(2*j)*2 + 0, r] += c_c[0]
                # k_global = 2*j+1, i_global = 0
                W[(2*j+1)*2 + 0, r] += c_c[1]
                # k_global = 2*j, i_global = 1
                W[(2*j)*2 + 1, r] += c_c[2]
                # k_global = 2*j+1, i_global = 1
                W[(2*j+1)*2 + 1, r] += c_c[3]
    
    # Convert to lists for the return interface
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
