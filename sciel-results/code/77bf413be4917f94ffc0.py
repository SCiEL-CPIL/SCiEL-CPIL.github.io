import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Total schoolbook rank
    rank_schoolbook = n * m * p
    
    # We will use a hybrid approach:
    # 1. Identify disjoint 2x2x2 blocks and apply Strassen (rank 7 instead of 8).
    # 2. Apply schoolbook to the rest.
    
    # Define the blocks
    # i indices: 0, 1 (size 2)
    # j indices: 0,1 and 2,3 (size 2 each)
    # k indices: 0,1 and 2,3 and 4,5 (size 2 each)
    
    i_block = [0, 1]
    j_blocks = [[0, 1], [2, 3]]
    k_blocks = [[0, 1], [2, 3], [4, 5]]
    
    # Pre-allocate matrices with max possible rank (schoolbook)
    # We will fill them and then trim if we find a better way, 
    # but here we know the rank will be:
    # 6 blocks * 7 + (90 - 6*8) * 1 = 42 + (90 - 48) = 42 + 42 = 84.
    # Wait, 90 - 48 = 42 elements left.
    # Total rank = 42 + 42 = 84.
    
    rank = 84
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term to U, V, W
    def add_term(u_idx, v_idx, w_idx, coeff_u, coeff_v, coeff_w):
        nonlocal col
        U[u_idx][col] += coeff_u
        V[v_idx][col] += coeff_v
        W[w_idx][col] += coeff_w
        col += 1

    # Strassen coefficients
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    strassen_Us = [
        [1, 1, 0, 0, 1, 1, 0], # A00
        [0, 0, 1, 0, 0, 0, 1], # A01
        [0, 1, 0, 0, 0, 1, 0], # A10
        [1, 1, 0, 1, 0, 0, -1] # A11
    ]
    # Note: Indices in strassen_Us correspond to A00, A01, A10, A11
    # M1 uses A00, A11
    # M2 uses A10, A11
    # M3 uses A00
    # M4 uses A11
    # M5 uses A00, A01
    # M6 uses A10, A00
    # M7 uses A01, A11
    
    # Let's re-verify Strassen U coefficients for each M
    # M1: A00 + A11 -> [1, 0, 0, 1]
    # M2: A10 + A11 -> [0, 0, 1, 1]
    # M3: A00 -> [1, 0, 0, 0]
    # M4: A11 -> [0, 0, 0, 1]
    # M5: A00 + A01 -> [1, 1, 0, 0]
    # M6: A10 - A00 -> [-1, 0, 1, 0]
    # M7: A01 - A11 -> [0, 1, 0, -1]
    
    U_strassen = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ]
    
    # V coefficients
    # M1: B00 + B11 -> [1, 0, 0, 1]
    # M2: B00 -> [1, 0, 0, 0]
    # M3: B01 - B11 -> [0, 1, 0, -1]
    # M4: B10 - B00 -> [-1, 0, 1, 0]
    # M5: B11 -> [0, 0, 0, 1]
    # M6: B00 + B01 -> [1, 1, 0, 0]
    # M7: B10 + B11 -> [0, 0, 1, 1]
    
    V_strassen = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1], # M3
        [-1, 0, 1, 0], # M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    
    # W coefficients (for C00, C01, C10, C11)
    # C00 = M1 + M4 - M5 + M7 -> [1, 0, 0, 1, -1, 0, 1]
    # C01 = M3 + M5 -> [0, 0, 1, 0, 1, 0, 0]
    # C10 = M2 + M4 -> [0, 1, 0, 1, 0, 0, 0]
    # C11 = M1 - M2 + M3 + M6 -> [1, -1, 1, 0, 0, 1, 0]
    
    W_strassen = [
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 0], # C01
        [0, 1, 0, 1, 0, 0, 0], # C10
        [1, -1, 1, 0, 0, 1, 0] # C11
    ]
    
    # Map local indices to global indices
    # Local A: 00, 01, 10, 11
    # Local B: 00, 01, 10, 11
    # Local C: 00, 01, 10, 11
    
    covered = set()
    
    for j_idx in j_blocks:
        for k_idx in k_blocks:
            # i is always [0, 1]
            i_idx = i_block
            # Check if these indices are already covered
            # Since we are using disjoint blocks, they shouldn't be.
            
            # Get global indices
            # A indices: i * m + j
            # B indices: j * p + k
            # C indices: k * n + i
            
            # Local to Global mapping
            # A_local[0] -> A[i_idx[0], j_idx[0]]
            # A_local[1] -> A[i_idx[0], j_idx[1]]
            # A_local[2] -> A[i_idx[1], j_idx[0]]
            # A_local[3] -> A[i_idx[1], j_idx[1]]
            
            a_g_idx = [
                i_idx[0] * m + j_idx[0],
                i_idx[0] * m + j_idx[1],
                i_idx[1] * m + j_idx[0],
                i_idx[1] * m + j_idx[1]
            ]
            
            b_g_idx = [
                j_idx[0] * p + k_idx[0],
                j_idx[0] * p + k_idx[1],
                j_idx[1] * p + k_idx[0],
                j_idx[1] * p + k_idx[1]
            ]
            
            c_g_idx = [
                k_idx[0] * n + i_idx[0],
                k_idx[0] * n + i_idx[1],
                k_idx[1] * n + i_idx[0],
                k_idx[1] * n + i_idx[1]
            ]
            
            # Mark as covered
            for i in i_idx:
                for j in j_idx:
                    for k in k_idx:
                        covered.add((i, j, k))
                        
            # Add Strassen terms
            for m_id in range(7):
                # U
                for local_a, coeff in enumerate(U_strassen[m_id]):
                    if coeff != 0:
                        U[a_g_idx[local_a]][col] += coeff
                # V
                for local_b, coeff in enumerate(V_strassen[m_id]):
                    if coeff != 0:
                        V[b_g_idx[local_b]][col] += coeff
                # W
                for local_c, coeff in enumerate(W_strassen[local_c]): # Wait, W_strassen is indexed by C output
                    pass # Need to loop over C outputs
                
                # Correct W filling
                for c_out in range(4):
                    coeff = W_strassen[c_out][m_id]
                    if coeff != 0:
                        W[c_g_idx[c_out]][col] += coeff
                
                col += 1

    # Add schoolbook for the rest
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
