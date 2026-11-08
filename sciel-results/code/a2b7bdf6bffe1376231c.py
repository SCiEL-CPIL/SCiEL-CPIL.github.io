import time
import sys
import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    def make_rank28():
        rank = 28
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        r = 0
        # Part 1: A2 B2 (j=2, k=0..4) - 10 terms
        for i in range(n):
            for k in range(p):
                U[i * m + 2][r] = 1
                V[2 * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1
        # Part 2: A1 B13 (j=0,1, k=4) - 4 terms
        for i in range(n):
            for j in range(2):
                U[i * m + j][r] = 1
                V[j * p + 4][r] = 1
                W[4 * n + i][r] = 1
                r += 1
        # Part 3: A1 B11 (j=0,1, k=0,1) using Strassen - 7 terms
        U[0 * m + 0][r] = 1; U[1 * m + 1][r] = 1; V[0 * p + 0][r] = 1; V[1 * p + 1][r] = 1; W[0 * n + 0][r] = 1; W[1 * n + 1][r] = 1; r += 1
        U[1 * m + 0][r] = 1; U[1 * m + 1][r] = 1; V[0 * p + 0][r] = 1; W[0 * n + 1][r] = 1; W[1 * n + 1][r] = -1; r += 1
        U[0 * m + 0][r] = 1; V[0 * p + 1][r] = 1; V[1 * p + 1][r] = -1; W[1 * n + 0][r] = 1; W[1 * n + 1][r] = 1; r += 1
        U[1 * m + 1][r] = 1; V[1 * p + 0][r] = 1; V[0 * p + 0][r] = -1; W[0 * n + 0][r] = 1; W[0 * n + 1][r] = 1; r += 1
        U[0 * m + 0][r] = 1; U[0 * m + 1][r] = 1; V[1 * p + 1][r] = 1; W[0 * n + 0][r] = -1; W[1 * n + 0][r] = 1; r += 1
        U[1 * m + 0][r] = 1; U[0 * m + 0][r] = -1; V[0 * p + 0][r] = 1; V[0 * p + 1][r] = 1; W[1 * n + 1][r] = 1; r += 1
        U[0 * m + 1][r] = 1; U[1 * m + 1][r] = -1; V[1 * p + 0][r] = 1; V[1 * p + 1][r] = 1; W[0 * n + 0][r] = 1; r += 1
        # Part 4: A1 B12 (j=0,1, k=2,3) using Strassen - 7 terms
        U[0 * m + 0][r] = 1; U[1 * m + 1][r] = 1; V[0 * p + 2][r] = 1; V[1 * p + 3][r] = 1; W[2 * n + 0][r] = 1; W[3 * n + 1][r] = 1; r += 1
        U[1 * m + 0][r] = 1; U[1 * m + 1][r] = 1; V[0 * p + 2][r] = 1; W[2 * n + 1][r] = 1; W[3 * n + 1][r] = -1; r += 1
        U[0 * m + 0][r] = 1; V[0 * p + 3][r] = 1; V[1 * p + 3][r] = -1; W[3 * n + 0][r] = 1; W[3 * n + 1][r] = 1; r += 1
        U[1 * m + 1][r] = 1; V[1 * p + 2][r] = 1; V[0 * p + 2][r] = -1; W[2 * n + 0][r] = 1; W[2 * n + 1][r] = 1; r += 1
        U[0 * m + 0][r] = 1; U[0 * m + 1][r] = 1; V[1 * p + 3][r] = 1; W[2 * n + 0][r] = -1; W[3 * n + 0][r] = 1; r += 1
        U[1 * m + 0][r] = 1; U[0 * m + 0][r] = -1; V[0 * p + 2][r] = 1; V[0 * p + 3][r] = 1; W[3 * n + 1][r] = 1; r += 1
        U[0 * m + 1][r] = 1; U[1 * m + 1][r] = -1; V[1 * p + 2][r] = 1; V[1 * p + 3][r] = 1; W[2 * n + 0][r] = 1; r += 1
        
        return {"U": U, "V": V, "W": W}
    
    def validate(U, V, W, rank):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for rr in range(rank):
                        s += U[i*m+j, rr] * V[j*p+k, rr] * W[k*n+i, rr]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def tensor_norm(U, V, W, rank):
        """Compute the Frobenius norm of the tensor represented by U, V, W."""
        total = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for rr in range(rank):
                        s += U[i*m+j, rr] * V[j*p+k, rr] * W[k*n+i, rr]
                    total += s * s
        return np.sqrt(total)
    
    def greedy_rank_reduction(base_U, base_V, base_W, base_rank, max_iters=200):
        """
        Structured greedy rank reduction:
        1. Start with base_rank terms
        2. Iteratively try to remove one term and compensate by adjusting others
        3. Use the residual norm as a guide
        """
        U = np.array(base_U, dtype=float)
        V = np.array(base_V, dtype=float)
        W = np.array(base_W, dtype=float)
        
        best_rank = base_rank
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for target_rank in range(base_rank - 1, 20, -1):
            if time.time() > deadline - 5:
                break
            
            # Try to reduce from current rank to target_rank
            current_U, current_V, current_W = best_U, best_V, best_W
            current_rank = best_rank
            
            for attempt in range(max_iters):
                if time.time() > deadline - 5:
                    break
                
                # Randomly select a term to remove
                remove_idx = np.random.randint(0, current_rank)
                
                U_rem = np.delete(current_U, remove_idx, axis=1)
                V_rem = np.delete(current_V, remove_idx, axis=1)
                W_rem = np.delete(current_W, remove_idx, axis=1)
                rem_rank = current_rank - 1
                
                if rem_rank >= target_rank:
                    continue
                
                # Compute the error tensor from removing this term
                error = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            error[i, j, k] = -current_U[i*m+j, remove_idx] * current_V[j*p+k, remove_idx] * current_W[k*n+i, remove_idx]
                
                # Try to find a rank-1 approximation to the error
                # Use SVD on the flattened error tensor
                err_flat = error.reshape(n*m*p)
                
                # Try multiple random rank-1 approximations
                found = False
                for trial in range(20):
                    # Random direction in U space
                    u_dir = np.random.randn(n*m)
                    u_dir /= np.linalg.norm(u_dir) + 1e-15
                    
                    # Project error onto this direction
                    # error[i,j,k] ≈ u_dir[i*m+j] * v_dir[j*p+k] * w_dir[k*n+i]
                    # For fixed u_dir, solve for v_dir and w_dir
                    v_dir = np.zeros(m*p)
                    w_dir = np.zeros(p*n)
                    
                    for j in range(m):
                        for k in range(p):
                            # Sum over i
                            v_val = 0.0
                            for i in range(n):
                                v_val += error[i, j, k] * u_dir[i*m+j]
                            v_dir[j*p+k] = v_val
                    
                    for k in range(p):
                        for i in range(n):
                            w_val = 0.0
                            for j in range(m):
                                w_val += error[i, j, k] * u_dir[i*m+j] * v_dir[j*p+k]
                            w_dir[k*n+i] = w_val
                    
                    # Check if this rank-1 term helps
                    U_try = np.column_stack([U_rem, u_dir])
                    V_try = np.column_stack([V_rem, v_dir])
                    W_try = np.column_stack([W_rem, w_dir])
                    
                    # Validate
                    valid = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = 0.0
                                for rr in range(rem_rank + 1):
                                    s += U_try[i*m+j, rr] * V_try[j*p+k, rr] * W_try[k*n+i, rr]
                                if abs(s - 1.0) > 1e-6:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    
                    if valid:
                        best_U, best_V, best_W = U_try, V_try, W_try
                        best_rank = rem_rank + 1
                        found = True
                        break
                
                if found and best_rank <= target_rank:
                    break
                
                # If we couldn't reduce, try perturbing the remaining terms
                if not found:
                    U_try = U_rem.copy()
                    V_try = V_rem.copy()
                    W_try = W_rem.copy()
                    
                    # Small perturbations
                    for _ in range(5):
                        idx_u = np.random.randint(0, n*m)
                        idx_v = np.random.randint(0, m*p)
                        idx_w = np.random.randint(0, p*n)
                        col_u = np.random.randint(0, rem_rank)
                        col_v = np.random.randint(0, rem_rank)
                        col_w = np.random.randint(0, rem_rank)
                        
                        U_try[idx_u, col_u] += np.random.choice([-1, -0.5, 0.5, 1])
                        V_try[idx_v, col_v] += np.random.choice([-1, -0.5, 0.5, 1])
                        W_try[idx_w, col_w] += np.random.choice([-1, -0.5, 0.5, 1])
                    
                    valid = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = 0.0
                                for rr in range(rem_rank):
                                    s += U_try[i*m+j, rr] * V_try[j*p+k, rr] * W_try[k*n+i, rr]
                                if abs(s - 1.0) > 1e-6:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    
                    if valid:
                        best_U, best_V, best_W = U_try, V_try, W_try
                        best_rank = rem_rank
                        break
        
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist(), "rank": best_rank}
    
    def local_search_polish(U, V, W, rank, max_iters=500):
        """
        Local search to find a lower rank decomposition by perturbing entries.
        """
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        best_rank = rank
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for target_rank in range(rank - 1, 20, -1):
            if time.time() > deadline - 5:
                break
            
            for attempt in range(max_iters):
                if time.time() > deadline - 5:
                    break
                
                # Try removing a random term and compensating
                remove_idx = np.random.randint(0, best_rank)
                
                U_rem = np.delete(best_U, remove_idx, axis=1)
                V_rem = np.delete(best_V, remove_idx, axis=1)
                W_rem = np.delete(best_W, remove_idx, axis=1)
                rem_rank = best_rank - 1
                
                if rem_rank >= target_rank:
                    continue
                
                # Perturb the remaining terms
                U_try = U_rem.copy()
                V_try = V_rem.copy()
                W_try = W_rem.copy()
                
                for _ in range(8):
                    idx_u = np.random.randint(0, n*m)
                    idx_v = np.random.randint(0, m*p)
                    idx_w = np.random.randint(0, p*n)
                    col_u = np.random.randint(0, rem_rank)
                    col_v = np.random.randint(0, rem_rank)
                    col_w = np.random.randint(0, rem_rank)
                    
                    U_try[idx_u, col_u] += np.random.choice([-1, -0.5, -0.25, 0.25, 0.5, 1])
                    V_try[idx_v, col_v] += np.random.choice([-1, -0.5, -0.25, 0.25, 0.5, 1])
                    W_try[idx_w, col_w] += np.random.choice([-1, -0.5, -0.25, 0.25, 0.5, 1])
                
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for rr in range(rem_rank):
                                s += U_try[i*m+j, rr] * V_try[j*p+k, rr] * W_try[k*n+i, rr]
                            if abs(s - 1.0) > 1e-6:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    best_U, best_V, best_W = U_try, V_try, W_try
                    best_rank = rem_rank
                    break
        
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist(), "rank": best_rank}
    
    best = make_rank28()
    best_rank = 28
    
    # Try greedy rank reduction
    if time.time() < deadline - 20:
        result = greedy_rank_reduction(best["U"], best["V"], best["W"], best_rank, max_iters=150)
        if result["rank"] < best_rank:
            best = {"U": result["U"], "V": result["V"], "W": result["W"]}
            best_rank = result["rank"]
    
    # Try local search polish
    if time.time() < deadline - 10:
        result = local_search_polish(best["U"], best["V"], best["W"], best_rank, max_iters=400)
        if result["rank"] < best_rank:
            best = {"U": result["U"], "V": result["V"], "W": result["W"]}
            best_rank = result["rank"]
    
    return best
# EVOLVE-BLOCK-END
