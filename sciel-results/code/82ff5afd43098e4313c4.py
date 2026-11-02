"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Data-driven approach: Explicit list of (U_idx, U_val, V_idx, V_val, W_idx, W_val)
    # Indices:
    # U: i*m + j (i=0..1, j=0..1) -> 0..3
    # V: j*p + k (j=0..1, k=0..7) -> 0..15
    # W: k*n + i (k=0..7, i=0..1) -> 0..15
    
    # Strassen template for 2x2 block
    # U indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # V indices for block b (k in 0,1): 
    #   j=0, k=2b -> 0*p + 2b = 2b
    #   j=0, k=2b+1 -> 2b+1
    #   j=1, k=2b -> 1*p + 2b = 8+2b
    #   j=1, k=2b+1 -> 9+2b
    # W indices for block b (k in 0,1):
    #   k=2b, i=0 -> 2b*n + 0 = 4b
    #   k=2b, i=1 -> 4b+1
    #   k=2b+1, i=0 -> (2b+1)n + 0 = 4b+2
    #   k=2b+1, i=1 -> 4b+3
    
    # Strassen steps (u_indices, u_vals, v_indices, v_vals, w_indices, w_vals)
    # Note: v_indices and w_indices are relative to the block (0..3)
    # We will map them to global indices.
    
    steps = [
        # P1: (A00+A11)(B00+B11) -> C00+=, C11+=
        # U: 0, 3 (1, 1)
        # V: 0, 3 (1, 1) [B00, B11]
        # W: 0, 3 (1, 1) [C00, C11]
        ([0, 3], [1, 1], [0, 3], [1, 1], [0, 3], [1, 1]),
        # P2: (A10+A11)B00 -> C10+=, C11-=
        # U: 2, 3 (1, 1)
        # V: 0 (1) [B00]
        # W: 1, 3 (1, -1) [C10, C11]
        ([2, 3], [1, 1], [0], [1], [1, 3], [1, -1]),
        # P3: A00(B01-B11) -> C01+=, C11+=
        # U: 0 (1)
        # V: 1, 3 (1, -1) [B01, B11]
        # W: 2, 3 (1, 1) [C01, C11]
        ([0], [1], [1, 3], [1, -1], [2, 3], [1, 1]),
        # P4: A11(B10-B00) -> C00+=, C10+=
        # U: 3 (1)
        # V: 2, 0 (1, -1) [B10, B00]
        # W: 0, 1 (1, 1) [C00, C10]
        ([3], [1], [2, 0], [1, -1], [0, 1], [1, 1]),
        # P5: (A00+A01)B11 -> C00-=, C01+=
        # U: 0, 1 (1, 1)
        # V: 3 (1) [B11]
        # W: 0, 2 (-1, 1) [C00, C01]
        ([0, 1], [1, 1], [3], [1], [0, 2], [-1, 1]),
        # P6: (A10-A00)(B00+B01) -> C11+=
        # U: 2, 0 (1, -1)
        # V: 0, 1 (1, 1) [B00, B01]
        # W: 3 (1) [C11]
        ([2, 0], [1, -1], [0, 1], [1, 1], [3], [1]),
        # P7: (A01-A11)(B10+B11) -> C00+=
        # U: 1, 3 (1, -1)
        # V: 2, 3 (1, 1) [B10, B11]
        # W: 0 (1) [C00]
        ([1, 3], [1, -1], [2, 3], [1, 1], [0], [1]),
    ]
    
    for block in range(4):
        for r in range(7):
            col = block * 7 + r
            u_idx_rel, u_vals, v_idx_rel, v_vals, w_idx_rel, w_vals = steps[r]
            
            # Map relative indices to global
            # U is global (shared)
            for idx_rel, val in zip(u_idx_rel, u_vals):
                U[idx_rel][col] += val
            
            # V is block-specific
            # Relative V indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            # Global V index: j*p + k
            # 0 -> j=0, k=0 -> 0*p + 2*block = 2*block
            # 1 -> j=0, k=1 -> 0*p + (2*block+1) = 2*block+1
            # 2 -> j=1, k=0 -> 1*p + 2*block = 8+2*block
            # 3 -> j=1, k=1 -> 1*p + (2*block+1) = 9+2*block
            v_map = [2*block, 2*block+1, 8+2*block, 9+2*block]
            for idx_rel, val in zip(v_idx_rel, v_vals):
                V[v_map[idx_rel]][col] += val
            
            # W is block-specific
            # Relative W indices: 0:(0,0), 1:(1,0), 2:(0,1), 3:(1,1)
            # Global W index: k*n + i
            # 0 -> k=0, i=0 -> 2*block*2 + 0 = 4*block
            # 1 -> k=0, i=1 -> 4*block + 1
            # 2 -> k=1, i=0 -> (2*block+1)*2 + 0 = 4*block+2
            # 3 -> k=1, i=1 -> 4*block+3
            w_map = [4*block, 4*block+1, 4*block+2, 4*block+3]
            for idx_rel, val in zip(w_idx_rel, w_vals):
                W[w_map[idx_rel]][col] += val
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
