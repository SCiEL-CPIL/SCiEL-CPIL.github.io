import numpy as np

def construct():
    """
    Decompose 2x2 * 2x10 multiplication using block-diagonal Strassen decomposition.
    
    The multiplication is split into 5 independent 2x2 * 2x2 blocks.
    Each block uses the rank-7 Strassen algorithm.
    Total rank = 5 * 7 = 35.
    
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    U: shape (4, 35)
    V: shape (20, 35)
    W: shape (20, 35)
    """
    n, m, p = 2, 2, 10
    num_blocks = p // 2  # 5 blocks
    block_rank = 7
    total_rank = num_blocks * block_rank  # 35
    
    # Initialize matrices with zeros
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)
    
    # Define Strassen terms for 2x2 matrices
    # A = [[a, b], [c, d]] (indices 0,1,2,3 in U: a=0, b=1, c=2, d=3)
    # B = [[e, f], [g, h]] (indices 0,1,2,3 in V block: e=0, f=1, g=2, h=3)
    # C = A*B
    #
    # Strassen 7 products:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    #
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    # U factors (A indices):
    # 0: a+d -> [1, 0, 0, 1]
    # 1: c+d -> [0, 0, 1, 1]
    # 2: a   -> [1, 0, 0, 0]
    # 3: d   -> [0, 0, 0, 1]
    # 4: a+b -> [1, 1, 0, 0]
    # 5: c-a -> [-1, 0, 1, 0]
    # 6: b-d -> [0, 1, 0, -1]
    
    # V factors (B indices within block):
    # 0: e+h -> [1, 0, 0, 1]
    # 1: e   -> [1, 0, 0, 0]
    # 2: f-h -> [0, 1, 0, -1]
    # 3: g-e -> [-1, 0, 1, 0]
    # 4: h   -> [0, 0, 0, 1]
    # 5: e+f -> [1, 1, 0, 0]
    # 6: g+h -> [0, 0, 1, 1]
    
    # W factors (C indices: k*n+i)
    # C indices:
    # C[0,0] -> index 0*2+0 = 0
    # C[0,1] -> index 1*2+0 = 2
    # C[1,0] -> index 0*2+1 = 1
    # C[1,1] -> index 1*2+1 = 3
    
    # Term 0: M1 = (a+d)(e+h)
    # W: C[0,0]+=1, C[1,1]+=1 -> indices 0, 3
    # Term 1: M2 = (c+d)e
    # W: C[1,0]+=1, C[1,1]-=1 -> indices 1, 3
    # Term 2: M3 = a(f-h)
    # W: C[0,1]+=1, C[1,1]+=1 -> indices 2, 3
    # Term 3: M4 = d(g-e)
    # W: C[0,0]+=1, C[1,0]+=1 -> indices 0, 1
    # Term 4: M5 = (a+b)h
    # W: C[0,0]-=1, C[0,1]+=1 -> indices 0, 2
    # Term 5: M6 = (c-a)(e+f)
    # W: C[1,1]+=1 -> index 3
    # Term 6: M7 = (b-d)(g+h)
    # W: C[0,0]+=1 -> index 0
    
    u_terms = [
        [1, 0, 0, 1],  # M1
        [0, 0, 1, 1],  # M2
        [1, 0, 0, 0],  # M3
        [0, 0, 0, 1],  # M4
        [1, 1, 0, 0],  # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ]
    
    v_terms = [
        [1, 0, 0, 1],  # M1
        [1, 0, 0, 0],  # M2
        [0, 1, 0, -1], # M3
        [-1, 0, 1, 0], # M4
        [0, 0, 0, 1],  # M5
        [1, 1, 0, 0],  # M6
        [0, 0, 1, 1]   # M7
    ]
    
    # W terms are sparse vectors of length 4 (local C indices 0..3)
    # We will map these to global W indices later.
    # Local C indices: 0:(0,0), 1:(1,0), 2:(0,1), 3:(1,1)
    # Note: Global W index is k*n + i.
    # Local C[i,k] maps to global index k*n + i.
    # Local index 0 is C[0,0] -> global 0*2+0 = 0
    # Local index 1 is C[1,0] -> global 0*2+1 = 1
    # Local index 2 is C[0,1] -> global 1*2+0 = 2
    # Local index 3 is C[1,1] -> global 1*2+1 = 3
    
    w_terms = [
        {0: 1, 3: 1},   # M1
        {1: 1, 3: -1},  # M2
        {2: 1, 3: 1},   # M3
        {0: 1, 1: 1},   # M4
        {0: -1, 2: 1},  # M5
        {3: 1},         # M6
        {0: 1}          # M7
    ]
    
    # Fill the matrices
    for block_idx in range(num_blocks):
        base_col = block_idx * block_rank
        # B columns for this block: 2*block_idx and 2*block_idx + 1
        b_col_offset = 2 * block_idx
        # C columns for this block: 2*block_idx and 2*block_idx + 1
        c_col_offset = 2 * block_idx
        
        for t in range(block_rank):
            col = base_col + t
            
            # Fill U
            U[0:4, col] = u_terms[t]
            
            # Fill V
            # V indices: j*p + k
            # j is 0 or 1. k is 0 or 1 (local).
            # Local k=0 -> global k = b_col_offset
            # Local k=1 -> global k = b_col_offset + 1
            v_vals = v_terms[t]
            # v_vals[0] is coeff for e (j=0, k_local=0)
            # v_vals[1] is coeff for f (j=0, k_local=1)
            # v_vals[2] is coeff for g (j=1, k_local=0)
            # v_vals[3] is coeff for h (j=1, k_local=1)
            
            V[0 * p + (b_col_offset + 0), col] = v_vals[0]
            V[0 * p + (b_col_offset + 1), col] = v_vals[1]
            V[1 * p + (b_col_offset + 0), col] = v_vals[2]
            V[1 * p + (b_col_offset + 1), col] = v_vals[3]
            
            # Fill W
            # W indices: k*p + i  (Wait, task says W[k*n+i, r])
            # Task: "W[k*n+i,r]"
            # My previous analysis: W index is k*n + i.
            # Let's re-verify the task description.
            # "W[k*n+i,r]"
            # Yes.
            
            for local_c_idx, val in w_terms[t].items():
                # local_c_idx corresponds to C[i, k_local]
                # We need to map local_c_idx to i, k_local
                # 0: i=0, k_local=0
                # 1: i=1, k_local=0
                # 2: i=0, k_local=1
                # 3: i=1, k_local=1
                
                if local_c_idx == 0:
                    i = 0; k_local = 0
                elif local_c_idx == 1:
                    i = 1; k_local = 0
                elif local_c_idx == 2:
                    i = 0; k_local = 1
                else: # 3
                    i = 1; k_local = 1
                
                global_k = c_col_offset + k_local
                global_w_idx = global_k * n + i
                W[global_w_idx, col] = val

    # Convert to lists for return
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
