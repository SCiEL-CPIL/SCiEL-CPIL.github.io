import time

DIMENSIONS = [2, 2, 10]

def construct():
    n, m, p = DIMENSIONS
    # Rank 35
    rank = 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Patterns for 2x2x2 Strassen
    # Each pattern is (U_vec, V_indices, V_vals, W_indices, W_vals)
    # We define them relative to the local block
    # Local A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Local B indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) -> mapped to global
    # Local C indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) -> mapped to global
    
    # Pattern definitions
    # P1: U=[1,0,0,1], V=[1,0,0,1] (B00+B11), W=[1,0,0,1] (C00+C11)
    # P2: U=[0,0,1,1], V=[1,0,0,0] (B00), W=[0,0,1,-1] (C10-C11)
    # P3: U=[1,0,0,0], V=[0,1,0,-1] (B01-B11), W=[0,1,0,1] (C01+C11)
    # P4: U=[0,0,0,1], V=[0,0,1,-1] (B10-B00), W=[1,0,1,0] (C00+C10)
    # P5: U=[1,1,0,0], V=[0,0,0,1] (B11), W=[-1,1,0,0] (-C00+C01)
    # P6: U=[-1,0,1,0], V=[1,1,0,0] (B00+B01), W=[0,0,0,1] (C11)
    # P7: U=[0,1,0,-1], V=[0,0,1,1] (B10+B11), W=[1,0,0,0] (C00)
    
    patterns = [
        ([1, 0, 0, 1], [(0, 1), (3, 1)], [(0, 1), (3, 1)]), # P1
        ([0, 0, 1, 1], [(0, 1)], [(2, 1), (3, -1)]),        # P2
        ([1, 0, 0, 0], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),# P3
        ([0, 0, 0, 1], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),# P4
        ([1, 1, 0, 0], [(3, 1)], [(0, -1), (1, 1)]),        # P5
        ([-1, 0, 1, 0], [(0, 1), (1, 1)], [(3, 1)]),        # P6
        ([0, 1, 0, -1], [(2, 1), (3, 1)], [(0, 1)])         # P7
    ]
    
    col = 0
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # Global indices for B
        # B00 -> k0
        # B01 -> k1
        # B10 -> 10 + k0
        # B11 -> 10 + k1
        b_idx = {
            0: k0,
            1: k1,
            2: 10 + k0,
            3: 10 + k1
        }
        
        # Global indices for C (W)
        # C00 -> 2*k0
        # C01 -> 2*k1
        # C10 -> 2*k0 + 1
        # C11 -> 2*k1 + 1
        c_idx = {
            0: 2 * k0,
            1: 2 * k1,
            2: 2 * k0 + 1,
            3: 2 * k1 + 1
        }
        
        for u_vec, v_pairs, w_pairs in patterns:
            # U
            for i in range(4):
                U[i][col] = u_vec[i]
            
            # V
            for local_idx, val in v_pairs:
                V[b_idx[local_idx]][col] = val
                
            # W
            for local_idx, val in w_pairs:
                W[c_idx[local_idx]][col] = val
                
            col += 1
            
    return {"U": U, "V": V, "W": W}
