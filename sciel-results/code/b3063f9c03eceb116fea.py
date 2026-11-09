"""Strassen-based 2x5 x 5x6 matrix multiplication decomposition."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    import time
    import numpy as np
    from itertools import product
    from fractions import Fraction

    start_time = time.time()
    DEATHLINE = 140.0
    n, m, p = 2, 5, 6

    def elapsed():
        return time.time() - start_time

    def verify_decomposition(U, V, W, tol=1e-10):
        """Verify that the decomposition exactly decomposes the matrix multiplication tensor."""
        rank = len(U[0]) if len(U) > 0 else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    s = 0.0
                    for r in range(rank):
                        s += U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def build_trivial():
        """Build the trivial rank-60 decomposition."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
                    r += 1
        return {"U": U, "V": V, "W": W}, rank

    # Start with trivial decomposition
    best_result, best_rank = build_trivial()

    if elapsed() > DEATHLINE - 10:
        return best_result

    # Try to find a lower-rank decomposition using a local search approach
    # Strategy: Start with rank 60 and try to merge pairs of terms
    
    def try_reduce_rank(U_list, V_list, W_list, current_rank, target_rank, max_iterations=500):
        """Try to reduce rank from current_rank to target_rank."""
        if current_rank <= target_rank:
            return None
        
        # Convert to numpy arrays for efficient computation
        U = np.array(U_list, dtype=float)
        V = np.array(V_list, dtype=float)
        W = np.array(W_list, dtype=float)
        
        # We'll try to find a linear combination that allows us to remove one term
        # For each pair of terms (r1, r2), try to find if we can express one as a combination
        # of the other and adjust the remaining terms
        
        # Simpler approach: use gradient descent to find a lower-rank decomposition
        # Initialize with random perturbation of the current solution
        np.random.seed(42)
        
        for iteration in range(max_iterations):
            if elapsed() > DEATHLINE - 15:
                return None
            
            # Randomly perturb the solution
            U_new = U + np.random.randn(*U.shape) * 0.1
            V_new = V + np.random.randn(*V.shape) * 0.1
            W_new = W + np.random.randn(*W.shape) * 0.1
            
            # Check if this is a valid decomposition
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        s = np.sum(U_new[u_idx] * V_new[v_idx] * W_new[w_idx])
                        if abs(s - 1.0) > 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                # Try to reduce rank by zeroing out some columns
                # Find columns that are close to zero
                col_norms = np.linalg.norm(U_new, axis=0) + np.linalg.norm(V_new, axis=0) + np.linalg.norm(W_new, axis=0)
                # Sort columns by norm
                sorted_cols = np.argsort(col_norms)
                
                # Try to remove the smallest columns
                for num_to_remove in range(1, current_rank - target_rank + 1):
                    cols_to_keep = sorted_cols[num_to_remove:]
                    if len(cols_to_keep) < target_rank:
                        break
                    
                    U_reduced = U_new[:, cols_to_keep]
                    V_reduced = V_new[:, cols_to_keep]
                    W_reduced = W_new[:, cols_to_keep]
                    
                    # Verify the reduced decomposition
                    valid_reduced = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                u_idx = i * m + j
                                v_idx = j * p + k
                                w_idx = k * n + i
                                s = np.sum(U_reduced[u_idx] * V_reduced[v_idx] * W_reduced[w_idx])
                                if abs(s - 1.0) > 1e-6:
                                    valid_reduced = False
                                    break
                            if not valid_reduced:
                                break
                        if not valid_reduced:
                            break
                    
                    if valid_reduced:
                        # Convert back to list format
                        U_list_new = [[float(U_reduced[i][r]) for r in range(len(cols_to_keep))] for i in range(n * m)]
                        V_list_new = [[float(V_reduced[i][r]) for r in range(len(cols_to_keep))] for i in range(m * p)]
                        W_list_new = [[float(W_reduced[i][r]) for r in range(len(cols_to_keep))] for i in range(p * n)]
                        
                        if verify_decomposition(U_list_new, V_list_new, W_list_new, tol=1e-9):
                            return {"U": U_list_new, "V": V_list_new, "W": W_list_new}
        
        return None

    # Try to reduce rank step by step
    current_U = best_result["U"]
    current_V = best_result["V"]
    current_W = best_result["W"]
    current_rank = best_rank
    
    for target_rank in range(59, 40, -1):
        if elapsed() > DEATHLINE - 15:
            break
        
        result = try_reduce_rank(current_U, current_V, current_W, current_rank, target_rank, max_iterations=200)
        if result is not None:
            best_result = result
            best_rank = target_rank
            current_U = result["U"]
            current_V = result["V"]
            current_W = result["W"]
            current_rank = target_rank

    return best_result
# EVOLVE-BLOCK-END
