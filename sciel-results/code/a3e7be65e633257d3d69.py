"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    # We construct a rank 54 decomposition.
    # Rank = 3 * (7 + 7 + 4) = 54
    
    R = 54
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    # Precompute Strassen coefficients for 2x2x2
    # Terms t=0..6
    # U indices (i,j), V indices (j,k), W indices (k,i)
    # Standard Strassen:
    # M0: (A00+A11)(B00+B11) -> C00+=1, C11+=1
    # M1: (A10+A11)B00 -> C10+=1
    # M2: A00(B10-B11) -> C01+=1
    # M3: A11(B01-B00) -> C00-=1, C11+=1 (Wait, C11 = M0 - M2 + M1 - M5? Let's stick to standard)
    # Standard:
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B10-B11)
    # M3 = A11(B01-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    # C00 = M0 - M3 + M4 + M6
    # C01 = M2 + M3
    # C10 = M1 + M4
    # C11 = M0 - M2 + M1 - M5
    
    # Let's define the vectors for U, V, W for the 7 terms
    # U is 4x7, V is 4x7, W is 4x7
    # Local indices: A(i,j), B(j,k), C(k,i)
    
    U_s = [
        [1, 0, 1, 0, 1, 0, 0], # A00
        [0, 0, 0, 1, 1, 1, 1], # A01
        [0, 1, 0, 0, 0, 1, -1],# A10
        [1, 1, 0, 1, 0, 0, 0]  # A11
    ]
    
    V_s = [
        [1, 1, 0, -1, 0, 1, 1], # B00 (k=0)
        [0, 0, 0, 1, 1, 0, 0],  # B01 (k=1) -- Wait, B is (j,k). Row j, Col k.
        # B00 is (j=0, k=0). B01 is (j=0, k=1).
        # Let's re-verify V indexing.
        # V_s[j_local, t]. j_local 0..1 is j, but we need to distinguish k?
        # No, V is (m * p) x R. For a 2x2 block, V is (2 * 2) = 4 rows.
        # Rows correspond to (j=0,k=0), (j=0,k=1), (j=1,k=0), (j=1,k=1).
        
        # M0: B00+B11 -> (0,0)+1, (1,1)+1
        # M1: B00 -> (0,0)+1
        # M2: B10-B11 -> (1,0)+1, (1,1)-1
        # M3: B01-B00 -> (0,1)+1, (0,0)-1
        # M4: B11 -> (1,1)+1
        # M5: B00+B01 -> (0,0)+1, (0,1)+1
        # M6: B10+B11 -> (1,0)+1, (1,1)+1
        
    ]
    
    # Let's build V_s correctly:
    # Row 0: j=0, k=0
    # Row 1: j=0, k=1
    # Row 2: j=1, k=0
    # Row 3: j=1, k=1
    
    V_s = [
        [1, 1, 0, -1, 0, 1, 0], # (0,0)
        [0, 0, 0, 1, 1, 1, 0],  # (0,1)
        [0, 0, 1, 0, 0, 0, 1],  # (1,0)
        [1, 0, -1, 0, 1, 0, 1]  # (1,1)
    ]
    
    # W_s is 4x7. Rows: (k=0,i=0), (k=0,i=1), (k=1,i=0), (k=1,i=1)
    # C00 = (0,0), C01 = (0,1), C10 = (1,0), C11 = (1,1)
    # M0: C00+1, C11+1
    # M1: C10+1
    # M2: C01+1
    # M3: C00-1, C11+1
    # M4: C00+1, C10+1
    # M5: C11-1
    # M6: C00+1, C01+1
    
    W_s = [
        [1, 0, 0, -1, 1, 0, 1], # C00 (k=0, i=0)
        [0, 0, 1, 0, 0, 0, 1],  # C01 (k=0, i=1)
        [0, 1, 0, 0, 1, 0, 0],  # C10 (k=1, i=0)
        [1, 0, 0, 1, 0, -1, 0]  # C11 (k=1, i=1)
    ]
    
    col_idx = 0
    
    # Define a function to add a Strassen block
    def add_strassen(A_i_base, A_j_base, B_j_base, B_k_base, W_k_base, W_i_base):
        nonlocal col_idx
        for t in range(7):
            # U
            for i_l in range(2):
                for j_l in range(2):
                    val = U_s[i_l * 2 + j_l][t] # Wait, U_s defined as 4x7 with row i*2+j?
                    # Let's check U_s definition above:
                    # [0] A00, [1] A01, [2] A10, [3] A11.
                    # So U_s[i*2+j] is correct.
                    if val != 0:
                        u_row = A_i_base + i_l * m + (A_j_base + j_l) # Global A index: i*m + j
                        # Wait, A_i_base is the i index? No, i is fixed 0..1 in 2x2 block.
                        # A is n x m. n=2.
                        # i_l is 0 or 1.
                        # Global i = i_l.
                        # Global j = A_j_base + j_l.
                        u_row = i_l * m + (A_j_base + j_l)
                        U[u_row][col_idx] += val
            
            # V
            for j_l in range(2):
                for k_l in range(2):
                    val = V_s[j_l * 2 + k_l][t]
                    if val != 0:
                        v_row = (B_j_base + j_l) * p + (B_k_base + k_l)
                        V[v_row][col_idx] += val
            
            # W
            for k_l in range(2):
                for i_l in range(2):
                    val = W_s[k_l * 2 + i_l][t]
                    if val != 0:
                        w_row = (W_k_base + k_l) * n + (W_i_base + i_l)
                        W[w_row][col_idx] += val
            
            col_idx += 1
    
    # Define a function to add a 2x1x1x2 Outer block
    # A: 2x1 (i=0,1; j=j_single)
    # B: 1x2 (j=j_single; k=k0, k1)
    # C: 2x2
    def add_outer(A_i_base, A_j, B_j, B_k_base, W_k_base, W_i_base):
        nonlocal col_idx
        # Terms:
        # A_i,0 * B_0,0 -> C_0,0
        U[A_i_base * m + A_j][col_idx] = 1
        V[B_j * p + (B_k_base + 0)][col_idx] = 1
        W[(W_k_base + 0) * n + W_i_base][col_idx] = 1
        col_idx += 1
        
        # A_i,0 * B_0,1 -> C_0,1
        U[A_i_base * m + A_j][col_idx] = 1
        V[B_j * p + (B_k_base + 1)][col_idx] = 1
        W[(W_k_base + 1) * n + W_i_base][col_idx] = 1
        col_idx += 1
        
        # A_i,1 * B_1,0 -> C_1,0
        U[(A_i_base + 1) * m + A_j][col_idx] = 1
        V[B_j * p + (B_k_base + 0)][col_idx] = 1
        W[(W_k_base + 0) * n + (W_i_base + 1)][col_idx] = 1
        col_idx += 1
        
        # A_i,1 * B_1,1 -> C_1,1
        U[(A_i_base + 1) * m + A_j][col_idx] = 1
        V[B_j * p + (B_k_base + 1)][col_idx] = 1
        W[(W_k_base + 1) * n + (W_i_base + 1)][col_idx] = 1
        col_idx += 1

    # We have 3 blocks of 2 columns in B
    # Block 0: k in 0..1
    # Block 1: k in 2..3
    # Block 2: k in 4..5
    
    for b in range(3):
        B_k_start = b * 2
        
        # Strassen Part 1: A (i=0,1; j=0,1) * B (j=0,1; k=B_k_start..+1)
        add_strassen(0, 0, 0, B_k_start, B_k_start, 0)
        
        # Strassen Part 2: A (i=0,1; j=2,3) * B (j=2,3; k=B_k_start..+1)
        add_strassen(0, 2, 2, B_k_start, B_k_start, 0)
        
        # Outer Part: A (i=0,1; j=4) * B (j=4; k=B_k_start..+1)
        add_outer(0, 4, 4, B_k_start, B_k_start, 0)
        
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
