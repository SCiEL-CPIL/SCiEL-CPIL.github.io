"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n, m, p = DIMENSIONS
    # Try rank 20 first, fall back to rank 21 if verification fails
    rank = 20
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # For 2x2x6, use a rank-20 decomposition
    # Based on exploiting shared structure across the three 2x2 blocks
    # We use 7 products per block but merge one component across blocks
    
    # Strassen coefficients for A (U matrix) - same as before
    U_coeffs = [
        {0: 1, 3: 1},       # M1: A00 + A11
        {2: 1, 3: 1},       # M2: A10 + A11
        {0: 1},             # M3: A00
        {3: 1},             # M4: A11
        {0: 1, 1: 1},       # M5: A00 + A01
        {2: 1, 0: -1},      # M6: A10 - A00
        {1: 1, 3: -1}       # M7: A01 - A11
    ]
    
    # V coefficients - for rank 20, we modify the M7 terms
    # to create a shared component
    V_coeffs = [
        [(0, 0, 1), (1, 1, 1)],      # M1: B00 + B11
        [(0, 0, 1)],                 # M2: B00
        [(0, 1, 1), (1, 1, -1)],     # M3: B01 - B11
        [(1, 0, 1), (0, 0, -1)],     # M4: B10 - B00
        [(1, 1, 1)],                 # M5: B11
        [(0, 0, 1), (0, 1, 1)],      # M6: B00 + B01
        [(1, 0, 1), (1, 1, 1)]       # M7: B10 + B11
    ]
    
    W_map = [
        [(0, 0, 1), (1, 1, 1)],   # k=0 (M1) -> C00, C11
        [(1, 0, 1), (1, 1, -1)],  # k=1 (M2) -> C10, C11
        [(0, 1, 1), (1, 1, 1)],   # k=2 (M3) -> C01, C11
        [(0, 0, 1), (1, 0, 1)],   # k=3 (M4) -> C00, C10
        [(0, 0, -1), (0, 1, 1)],  # k=4 (M5) -> C00, C01
        [(1, 1, 1)],              # k=5 (M6) -> C11
        [(0, 0, 1)]               # k=6 (M7) -> C00
    ]
    
    # For rank 20: use 7 products for blocks 0 and 1, and 6 products for block 2
    # by merging M7 of block 2 with a linear combination
    # Actually, let's try: 7+7+6 = 20
    
    # Block 0: products 0-6
    # Block 1: products 7-13
    # Block 2: products 14-19 (only 6 products, modified)
    
    for r in range(3):
        num_products = 7 if r < 2 else 6
        base_col = r * 7 if r < 2 else 14
        
        for k in range(num_products):
            col = base_col + k
            
            # U
            for idx, val in U_coeffs[k].items():
                U[idx][col] = val
                
            # V
            for j_local, k_local, val in V_coeffs[k]:
                global_k = 2 * r + k_local
                v_idx = j_local * p + global_k
                V[v_idx][col] = val
                
            # W
            for i, j_local, val in W_map[k]:
                global_k = 2 * r + j_local
                w_idx = global_k * n + i
                W[w_idx][col] = val
    
    # Verify the decomposition
    try:
        import numpy as np
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Build tensor T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
        # where i indexes A (2x2=4), j indexes B (2x6=12), k indexes C (6x2=12)
        # C = A*B, so C[a,c] = sum_b A[a,b]*B[b,c]
        # Tensor: T[a,c,b] = delta[a,b] (for the multiplication tensor)
        
        # Check: for each (a, b, c), sum_r U[a*2+b, r] * V[b*6+c, r] * W[c*2+a, r] should be 1 if a==b else 0
        # Wait, the convention is W[k*n+i, r] where k is the column index of C and i is the row
        # C has shape (n, p) = (2, 6), so C[i, j] where i in 0..1, j in 0..5
        # W index = j * n + i = j*2 + i
        
        max_err = 0.0
        for a in range(n):
            for b in range(m):
                for c in range(p):
                    expected = 1.0 if a == b else 0.0
                    actual = sum(U_np[a*m + b, r] * V_np[b*p + c, r] * W_np[c*n + a, r] for r in range(rank))
                    max_err = max(max_err, abs(actual - expected))
        
        if max_err > 1e-9:
            # Rank 20 didn't work, fall back to rank 21
            rank = 21
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            
            for r in range(3):
                for k in range(7):
                    col = r * 7 + k
                    
                    for idx, val in U_coeffs[k].items():
                        U[idx][col] = val
                        
                    for j_local, k_local, val in V_coeffs[k]:
                        global_k = 2 * r + k_local
                        v_idx = j_local * p + global_k
                        V[v_idx][col] = val
                        
                    for i, j_local, val in W_map[k]:
                        global_k = 2 * r + j_local
                        w_idx = global_k * n + i
                        W[w_idx][col] = val
    except Exception:
        # If numpy fails, just return rank 21
        rank = 21
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for r in range(3):
            for k in range(7):
                col = r * 7 + k
                for idx, val in U_coeffs[k].items():
                    U[idx][col] = val
                for j_local, k_local, val in V_coeffs[k]:
                    global_k = 2 * r + k_local
                    v_idx = j_local * p + global_k
                    V[v_idx][col] = val
                for i, j_local, val in W_map[k]:
                    global_k = 2 * r + j_local
                    w_idx = global_k * n + i
                    W[w_idx][col] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
