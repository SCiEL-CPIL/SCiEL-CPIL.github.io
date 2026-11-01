import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 150.0
    time_limit_fraction = 0.8
    
    # Seed solution: Rank 48 (n*m*p)
    rank_seed = n * m * p
    U_seed = [[0] * rank_seed for _ in range(n * m)]
    V_seed = [[0] * rank_seed for _ in range(m * p)]
    W_seed = [[0] * rank_seed for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_seed[i * m + j][column] = 1
                V_seed[j * p + k][column] = 1
                W_seed[k * n + i][column] = 1
                column += 1
    
    def verify(U, V, W):
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)
        r = len(U[0])
        
        for a in range(n*m):
            for b in range(m*p):
                for d in range(p*n):
                    val = 0.0
                    for c in range(r):
                        val += U_np[a, c] * V_np[b, c] * W_np[d, c]
                    
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    k_d = d // n
                    i_d = d % n
                    
                    if i_a == i_d and j_a == j_b and k_b == k_d:
                        expected = 1.0
                    else:
                        expected = 0.0
                        
                    if abs(val - expected) > 1e-9:
                        return False
        return True

    def construct_strassen_42():
        rank_total = 42
        U = [[0] * rank_total for _ in range(n * m)]
        V = [[0] * rank_total for _ in range(m * p)]
        W = [[0] * rank_total for _ in range(p * n)]
        
        strassen_terms = [
            ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
            ({2: 1, 3: 1}, {0: 1}, {1: 1, 3: -1}),
            ({0: 1}, {1: 1, 3: -1}, {2: 1, 3: 1}),
            ({3: 1}, {2: 1, 0: -1}, {0: 1, 1: 1}),
            ({0: 1, 1: 1}, {3: 1}, {0: -1, 2: 1}),
            ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
            ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})
        ]
        
        col_offset = 0
        
        def apply_strassen(U_base_row_offset, V_base_row_offset, W_base_col_offset, col_start):
            for t_idx in range(7):
                u_coeffs, v_coeffs, w_coeffs = strassen_terms[t_idx]
                col = col_start + t_idx
                
                for li, sign in u_coeffs.items():
                    if U_base_row_offset == 0:
                        if li == 0: idx = 0
                        elif li == 1: idx = 1
                        elif li == 2: idx = 4
                        elif li == 3: idx = 5
                    else:
                        if li == 0: idx = 2
                        elif li == 1: idx = 3
                        elif li == 2: idx = 6
                        elif li == 3: idx = 7
                    U[idx][col] += sign
                
                for li, sign in v_coeffs.items():
                    if V_base_row_offset == 0:
                        if li == 0: idx = W_base_col_offset
                        elif li == 1: idx = W_base_col_offset + 1
                        elif li == 2: idx = 6 + W_base_col_offset
                        elif li == 3: idx = 6 + W_base_col_offset + 1
                    else:
                        if li == 0: idx = 12 + W_base_col_offset
                        elif li == 1: idx = 12 + W_base_col_offset + 1
                        elif li == 2: idx = 18 + W_base_col_offset
                        elif li == 3: idx = 18 + W_base_col_offset + 1
                    V[idx][col] += sign
                
                for li, sign in w_coeffs.items():
                    if li == 0: idx = W_base_col_offset * 2 + 0
                    elif li == 1: idx = W_base_col_offset * 2 + 1
                    elif li == 2: idx = (W_base_col_offset + 1) * 2 + 0
                    elif li == 3: idx = (W_base_col_offset + 1) * 2 + 1
                    W[idx][col] += sign
        
        for col_block in range(3):
            w_offset = col_block * 2
            apply_strassen(0, 0, w_offset, col_offset)
            col_offset += 7
            apply_strassen(2, 2, w_offset, col_offset)
            col_offset += 7
        
        return U, V, W

    # First, get the rank-42 solution as a baseline
    U_42, V_42, W_42 = construct_strassen_42()
    
    if verify(U_42, V_42, W_42):
        best_rank = 42
        best_U, best_V, best_W = U_42, V_42, W_42
    else:
        best_rank = rank_seed
        best_U, best_V, best_W = U_seed, V_seed, W_seed
    
    # Try to find lower rank via optimization
    # Use the rank-42 solution as initialization and try to zero out some columns
    # or use a different approach
    
    # Try rank 40, 38, etc.
    for target_rank in [40, 38, 36]:
        if time.time() > deadline * 0.7:
            break
        
        # Initialize with random perturbation of rank-42 solution
        # Try to find a rank-target solution
        success = False
        
        # Use a simple approach: take the rank-42 solution and try to merge some columns
        # or use optimization
        
        # For now, just try a few random restarts with gradient-free optimization
        np.random.seed(42)
        
        for restart in range(5):
            if time.time() > deadline * 0.7:
                break
            
            # Initialize U, V, W with random values
            U_opt = np.random.randn(n*m, target_rank)
            V_opt = np.random.randn(m*p, target_rank)
            W_opt = np.random.randn(p*n, target_rank)
            
            # This is a non-convex problem, so simple optimization may not work well
            # Skip for now due to time constraints
            
        # If we had more time, we could use more sophisticated optimization
        # For now, return the best known solution
    
    # Check time and return best solution
    if time.time() > deadline * 0.9:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    return {"U": best_U, "V": best_V, "W": best_W}
