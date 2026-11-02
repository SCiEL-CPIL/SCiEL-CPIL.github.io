import numpy as np
import time

DIMENSIONS = [2, 2, 6]

def construct():
    n, m, p = DIMENSIONS
    target_rank = 20
    
    # Fallback: Rank 21 Strassen
    def get_rank_21():
        rank = 21
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        strassen = [
            ({0: 1, 3: 1}, {0: 1, 3: 1}, {(0, 0): 1, (1, 1): 1}),
            ({2: 1, 3: 1}, {0: 1}, {(1, 0): 1, (1, 1): -1}),
            ({0: 1}, {1: 1, 3: -1}, {(0, 1): 1, (1, 1): 1}),
            ({3: 1}, {2: 1, 0: -1}, {(0, 0): 1, (1, 0): 1}),
            ({0: 1, 1: 1}, {3: 1}, {(0, 0): -1, (0, 1): 1}),
            ({2: 1, 0: -1}, {0: 1, 1: 1}, {(1, 1): 1}),
            ({1: 1, 3: -1}, {2: 1, 3: 1}, {(0, 0): 1})
        ]
        
        r = 0
        for b in range(3):
            for t in range(7):
                u_coeffs, v_coeffs, w_coeffs = strassen[t]
                for idx, val in u_coeffs.items():
                    U[idx][r] = val
                for l, val in v_coeffs.items():
                    if l == 0: v_idx = 0 * p + 2 * b
                    elif l == 1: v_idx = 0 * p + 2 * b + 1
                    elif l == 2: v_idx = 1 * p + 2 * b
                    elif l == 3: v_idx = 1 * p + 2 * b + 1
                    V[v_idx][r] = val
                for (i, k_local), val in w_coeffs.items():
                    k_global = 2 * b + k_local
                    w_idx = k_global * n + i
                    W[w_idx][r] = val
                r += 1
        return {"U": U, "V": V, "W": W}

    # Attempt rank-20 search with time limit
    start_time = time.time()
    time_limit = 120.0  # seconds
    
    def verify_decomposition(U, V, W, rank):
        """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
        # Check dimensions
        if len(U) != n * m or len(U[0]) != rank:
            return False
        if len(V) != m * p or len(V[0]) != rank:
            return False
        if len(W) != p * n or len(W[0]) != rank:
            return False
        
        # Build the multiplication tensor from U, V, W
        # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
        # where i indexes (row, col) of A, j indexes (row, col) of B, k indexes (row, col) of C
        # A is n x m, B is m x p, C is n x p
        # i = a_row * m + a_col
        # j = b_row * p + b_col  
        # k = c_row * n + c_col
        
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # For each output element C[i,j], the coefficient should be:
        # sum over a_col,b_row where a_col==b_row of 1
        # T[i,j,k] should equal 1 if i_row==k_row and i_col==j_row, else 0
        
        for a_row in range(n):
            for a_col in range(m):
                i = a_row * m + a_col
                for b_row in range(m):
                    for b_col in range(p):
                        j = b_row * p + b_col
                        for c_row in range(n):
                            for c_col in range(p):
                                k = c_row * n + c_col
                                # Expected value: 1 if a_row==c_row and a_col==b_row, else 0
                                expected = 1.0 if (a_row == c_row and a_col == b_row) else 0.0
                                # Computed value: sum_r U[i,r] * V[j,r] * W[k,r]
                                computed = 0.0
                                for r in range(rank):
                                    computed += U_arr[i, r] * V_arr[j, r] * W_arr[k, r]
                                if abs(computed - expected) > 1e-6:
                                    return False
        return True

    def try_rank_20_search():
        """Attempt to find a rank-20 decomposition using random search."""
        deadline = start_time + time_limit - 10.0  # 10 second safety margin
        
        # Strategy: Start from rank-21 and try to merge/simplify
        # Or use random initialization with gradient-free optimization
        
        # Let's try a structured approach: 
        # For 2x2 x 2x6, we can think of it as 3 blocks of 2x2 x 2x2
        # Each block has rank 7, total 21. 
        # To get rank 20, we need to share one factor across blocks.
        
        # Alternative: Use the fact that the tensor has symmetries
        # Let's try random search with refinement
        
        best_solution = None
        
        # Try several random initializations
        np.random.seed(42)
        
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Random initialization
            U = np.random.randn(n * m, target_rank) * 0.5
            V = np.random.randn(m * p, target_rank) * 0.5
            W = np.random.randn(p * n, target_rank) * 0.5
            
            # Simple iterative refinement: try to reduce error
            # Use a simple coordinate descent approach
            for it in range(100):
                if time.time() > deadline:
                    break
                    
                # Compute current error
                error = 0.0
                for a_row in range(n):
                    for a_col in range(m):
                        i = a_row * m + a_col
                        for b_row in range(m):
                            for b_col in range(p):
                                j = b_row * p + b_col
                                for c_row in range(n):
                                    for c_col in range(p):
                                        k = c_row * n + c_col
                                        expected = 1.0 if (a_row == c_row and a_col == b_row) else 0.0
                                        computed = np.sum(U[i, :] * V[j, :] * W[k, :])
                                        error += (computed - expected) ** 2
                
                if error < 1e-10:
                    # Found a valid solution!
                    # Convert to list format
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = W.tolist()
                    # Round to nearest half-integer if close
                    U_rounded = [[round(x * 2) / 2 for x in row] for row in U_list]
                    V_rounded = [[round(x * 2) / 2 for x in row] for row in V_list]
                    W_rounded = [[round(x * 2) / 2 for x in row] for row in W_list]
                    
                    # Verify the rounded version
                    if verify_decomposition(U_rounded, V_rounded, W_rounded, target_rank):
                        return {"U": U_rounded, "V": V_rounded, "W": W_rounded}
                    else:
                        # Use unrounded if it works
                        if verify_decomposition(U_list, V_list, W_list, target_rank):
                            return {"U": U_list, "V": V_list, "W": W_list}
                
                # Coordinate descent: update one variable at a time
                # For simplicity, just do random perturbations
                var_type = np.random.randint(3)  # 0=U, 1=V, 2=W
                if var_type == 0:
                    idx = np.random.randint(n * m)
                    U[idx, :] += np.random.randn(target_rank) * 0.1
                elif var_type == 1:
                    idx = np.random.randint(m * p)
                    V[idx, :] += np.random.randn(target_rank) * 0.1
                else:
                    idx = np.random.randint(p * n)
                    W[idx, :] += np.random.randn(target_rank) * 0.1
        
        return None

    # Try rank-20 search
    result = try_rank_20_search()
    if result is not None:
        return result
    
    # Fallback to rank-21
    return get_rank_21()
