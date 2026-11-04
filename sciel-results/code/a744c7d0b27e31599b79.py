import numpy as np

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    # Rank 50
    rank = 50
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        nonlocal col
        if col >= rank:
            return
        U[u_idx, col] += u_val
        V[v_idx, col] += v_val
        W[w_idx, col] += w_val
        col += 1

    # Helper to add Strassen block for A[i1:i2, j1:j2] * B[j1:j2, k1:k2] -> C[k1:k2, i1:i2]
    def add_strassen_block(i1, j1, k1):
        # i1, j1, k1 are start indices. Size is 2x2x2.
        # Local indices 0,1 map to i1, i1+1 etc.
        
        # M1 = (A11 + A22) * (B11 + B22)
        # A11: (i1, j1), A22: (i1+1, j1+1)
        # B11: (j1, k1), B22: (j1+1, k1+1)
        # C11: (k1, i1), C22: (k1+1, i1+1)
        
        # We need to add 7 terms.
        # Each term is a product of a linear combo of A and a linear combo of B.
        # This results in multiple entries in U and V for a single column in W?
        # No, the decomposition is a sum of rank-1 tensors.
        # Each rank-1 tensor is $u \otimes v \otimes w$.
        # Strassen expresses the tensor as sum of 7 such tensors.
        # The vectors u, v, w have sparse entries.
        
        # Term 1: u1 = e_A11 + e_A22, v1 = e_B11 + e_B22, w1 = e_C11 + e_C22 - e_C? No.
        # The output C is a linear combination of the M's.
        # So W for each M has entries corresponding to the C indices it affects.
        
        # M1 contributes to C11, C22.
        # w1 has 1 at C11 and 1 at C22?
        # C11 = M1 + ...
        # C22 = M1 - ...
        # So w1 = e_C11 + e_C22?
        # Yes, because M1 is added to C11 and C22.
        
        # Let's define the 7 vectors u, v, w for Strassen.
        
        # Indices in local 2x2 space:
        # A: (0,0), (0,1), (1,0), (1,1)
        # B: (0,0), (0,1), (1,0), (1,1)
        # C: (0,0), (0,1), (1,0), (1,1)
        
        # M1: u = (1,0,0,1), v = (1,0,0,1), w = (1,0,0,1)
        # M2: u = (0,0,1,1), v = (1,0,0,0), w = (0,0,1,0)
        # M3: u = (1,1,0,0), v = (0,1,0,0), w = (0,1,0,0)
        # M4: u = (0,0,0,1), v = (0,0,1,0), w = (1,0,0,0)
        # M5: u = (1,1,0,0), v = (0,0,0,1), w = (0,1,0,0)
        # M6: u = (0,0,1,0), v = (1,1,0,0), w = (0,0,0,1)
        # M7: u = (1,0,0,0), v = (0,0,1,1), w = (1,0,0,0)
        
        # Wait, let's verify the C combinations.
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # So w vectors:
        # w1: C11=1, C22=1
        # w2: C21=1
        # w3: C12=1
        # w4: C11=1, C21=1
        # w5: C12=-1, C22=1  (Wait, C12 = M3+M5, C22 = M1-M2+M3+M6. M5 is in C12? No.
        # Let's re-check Strassen formulas.
        # M5 = (A11+A12)B22.
        # C12 = M3 + M5.
        # C22 = M1 - M2 + M3 + M6.
        # So M5 is in C12.
        # w5: C12=1.
        
        # Let's re-list w:
        # w1: (1, 0, 0, 1)  # C11, C22
        # w2: (0, 0, 1, 0)  # C21
        # w3: (0, 1, 0, 0)  # C12
        # w4: (1, 0, 1, 0)  # C11, C21
        # w5: (0, 1, 0, 0)  # C12
        # w6: (0, 0, 0, 1)  # C22
        # w7: (1, 0, 0, 0)  # C11
        
        # Wait, C11 = M1 + M4 - M5 + M7.
        # So w1 has 1 at C11. w4 has 1 at C11. w5 has -1 at C11? No, w5 is for M5.
        # M5 is subtracted in C11.
        # So w5 should have -1 at C11?
        # But M5 is added in C12.
        # So w5 has 1 at C12 and -1 at C11?
        # Yes.
        
        # Let's refine w vectors:
        # w1: (1, 0, 0, 1)
        # w2: (0, 0, 1, 0)
        # w3: (0, 1, 0, 0)
        # w4: (1, 0, 1, 0)
        # w5: (-1, 1, 0, 0)
        # w6: (0, 0, 0, 1)
        # w7: (1, 0, 0, 0)
        
        # u vectors:
        # u1: (1, 0, 0, 1) # A11, A22
        # u2: (0, 0, 1, 1) # A21, A22
        # u3: (1, 1, 0, 0) # A11, A12
        # u4: (0, 0, 0, 1) # A22
        # u5: (1, 1, 0, 0) # A11, A12
        # u6: (0, 0, 1, 0) # A21
        # u7: (1, 0, 0, 0) # A11
        
        # v vectors:
        # v1: (1, 0, 0, 1) # B11, B22
        # v2: (1, 0, 0, 0) # B11
        # v3: (0, 1, 0, 0) # B12
        # v4: (0, 0, 1, 0) # B21
        # v5: (0, 0, 0, 1) # B22
        # v6: (1, 1, 0, 0) # B11, B12
        # v7: (0, 0, 1, 1) # B21, B22
        
        # Now map local indices to global.
        # A local (r, c) -> global (i1+r, j1+c) -> index (i1+r)*3 + (j1+c)
        # B local (r, c) -> global (j1+r, k1+c) -> index (j1+r)*9 + (k1+c)
        # C local (r, c) -> global (k1+r, i1+c) -> index (k1+r)*2 + (i1+c)
        
        for t in range(7):
            # Get vectors
            u_vec = [u1, u2, u3, u4, u5, u6, u7][t]
            v_vec = [v1, v2, v3, v4, v5, v6, v7][t]
            w_vec = [w1, w2, w3, w4, w5, w6, w7][t]
            
            # This is not a single rank-1 term. It is a sum of rank-1 terms?
            # No, each M is a single multiplication of two linear forms.
            # So for each M, we have one column in U, V, W.
            # But U and V entries are sums of basis vectors.
            # So for a given column index `col`, U has non-zero entries at the indices specified by u_vec.
            
            # We add one column to U, V, W for each Strassen term.
            
            # U entries
            for r in range(2):
                for c in range(2):
                    if u_vec[r*2+c] != 0:
                        g_idx = (i1+r)*3 + (j1+c)
                        U[g_idx, col] += u_vec[r*2+c]
                        
            # V entries
            for r in range(2):
                for c in range(2):
                    if v_vec[r*2+c] != 0:
                        g_idx = (j1+r)*9 + (k1+c)
                        V[g_idx, col] += v_vec[r*2+c]
                        
            # W entries
            for r in range(2):
                for c in range(2):
                    if w_vec[r*2+c] != 0:
                        g_idx = (k1+r)*2 + (i1+c)
                        W[g_idx, col] += w_vec[r*2+c]
                        
            col += 1

    # Define the vectors
    u1 = [1, 0, 0, 1]
    u2 = [0, 0, 1, 1]
    u3 = [1, 1, 0, 0]
    u4 = [0, 0, 0, 1]
    u5 = [1, 1, 0, 0]
    u6 = [0, 0, 1, 0]
    u7 = [1, 0, 0, 0]
    
    v1 = [1, 0, 0, 1]
    v2 = [1, 0, 0, 0]
    v3 = [0, 1, 0, 0]
    v4 = [0, 0, 1, 0]
    v5 = [0, 0, 0, 1]
    v6 = [1, 1, 0, 0]
    v7 = [0, 0, 1, 1]
    
    w1 = [1, 0, 0, 1]
    w2 = [0, 0, 1, 0]
    w3 = [0, 1, 0, 0]
    w4 = [1, 0, 1, 0]
    w5 = [-1, 1, 0, 0]
    w6 = [0, 0, 0, 1]
    w7 = [1, 0, 0, 0]

    # Part 1: Strassen (0,0,0)
    add_strassen_block(0, 0, 0)
    
    # Part 2: Schoolbook (0,2,0) size 2x1x2
    # i in 0..1, j=2, k in 0..1
    for i in range(2):
        for k in range(2):
            U[i*3 + 2, col] = 1
            V[2*9 + k, col] = 1
            W[k*2 + i, col] = 1
            col += 1
            
    # Part 3: Strassen blocks for k=2,4,6
    add_strassen_block(0, 0, 2)
    add_strassen_block(0, 0, 4)
    add_strassen_block(0, 0, 6)
    
    # Part 3b: Schoolbook for k=8 (size 2x2x1)
    # i in 0..1, j in 0..1, k=8
    for i in range(2):
        for j in range(2):
            U[i*3 + j, col] = 1
            V[j*9 + 8, col] = 1
            W[8*2 + i, col] = 1
            col += 1
            
    # Part 4: Schoolbook for j=2, k in 2..8 (size 2x1x7)
    for i in range(2):
        for k in range(2, 9):
            U[i*3 + 2, col] = 1
            V[2*9 + k, col] = 1
            W[k*2 + i, col] = 1
            col += 1
            
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
