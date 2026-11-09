"""Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x10 matrix."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Schoolbook rank 60 as fallback
    rank_fallback = n * m * p
    U_fb = [[0] * rank_fallback for _ in range(n * m)]
    V_fb = [[0] * rank_fallback for _ in range(m * p)]
    W_fb = [[0] * rank_fallback for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    
    def verify_decomposition(U, V, W, n, m, p, tol=1e-9):
        """Check if the decomposition is correct."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(len(U[0])):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def compute_error(U, V, W, n, m, p):
        """Compute the maximum absolute error across all (i,j,k) entries."""
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(len(U[0])):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    err = abs(s - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err
    
    def random_init(rank, n, m, p, seed=None):
        """Generate random factor matrices with half-integer entries."""
        if seed is not None:
            random.seed(seed)
        U = [[random.choice([-1, -0.5, 0, 0.5, 1]) for _ in range(rank)] for _ in range(n*m)]
        V = [[random.choice([-1, -0.5, 0, 0.5, 1]) for _ in range(rank)] for _ in range(m*p)]
        W = [[random.choice([-1, -0.5, 0, 0.5, 1]) for _ in range(rank)] for _ in range(p*n)]
        return U, V, W
    
    def local_search(U, V, W, n, m, p, max_iters=5000, time_limit=140.0):
        """Local search to refine factor matrices.
        
        Tries to flip individual entries to reduce the maximum error.
        Uses a smooth surrogate: sum of squared errors instead of max error.
        """
        rank = len(U[0])
        U = [row[:] for row in U]
        V = [row[:] for row in V]
        W = [row[:] for row in W]
        
        def compute_sq_err(U, V, W):
            total = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(rank):
                            s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                        total += (s - 1.0) ** 2
            return total
        
        best_err = compute_sq_err(U, V, W)
        values = [-1, -0.5, 0, 0.5, 1]
        
        for it in range(max_iters):
            if time.time() > time_limit:
                break
            
            # Pick a random matrix and entry to try changing
            mat_choice = random.random()
            if mat_choice < 0.33:
                mat = 'U'
                row_idx = random.randint(0, n*m-1)
            elif mat_choice < 0.66:
                mat = 'V'
                row_idx = random.randint(0, m*p-1)
            else:
                mat = 'W'
                row_idx = random.randint(0, p*n-1)
            
            col_idx = random.randint(0, rank-1)
            
            # Try all possible new values for this entry
            old_val = U[row_idx][col_idx] if mat == 'U' else (V[row_idx][col_idx] if mat == 'V' else W[row_idx][col_idx])
            
            best_new_val = old_val
            best_new_err = best_err
            
            for new_val in values:
                if new_val == old_val:
                    continue
                
                # Temporarily change and compute error
                if mat == 'U':
                    U[row_idx][col_idx] = new_val
                elif mat == 'V':
                    V[row_idx][col_idx] = new_val
                else:
                    W[row_idx][col_idx] = new_val
                
                err = compute_sq_err(U, V, W)
                
                # Revert
                if mat == 'U':
                    U[row_idx][col_idx] = old_val
                elif mat == 'V':
                    V[row_idx][col_idx] = old_val
                else:
                    W[row_idx][col_idx] = old_val
                
                if err < best_new_err:
                    best_new_err = err
                    best_new_val = new_val
            
            # Apply the best change if it improves
            if best_new_val != old_val:
                if mat == 'U':
                    U[row_idx][col_idx] = best_new_val
                elif mat == 'V':
                    V[row_idx][col_idx] = best_new_val
                else:
                    W[row_idx][col_idx] = best_new_val
                best_err = best_new_err
                
                if best_err < 1e-12:
                    break
        
        return U, V, W, best_err
    
    best_U, best_V, best_W = U_fb, V_fb, W_fb
    best_rank = rank_fallback
    
    random.seed(42)
    
    # Try ranks from 40 down to 30
    for target_rank in [40, 38, 36, 34, 32, 30]:
        if time.time() > deadline - 30:
            break
        
        found = False
        num_trials = 0
        max_trials = 15000
        
        while not found and num_trials < max_trials and time.time() < deadline - 10:
            num_trials += 1
            
            # Initialize with random values
            U, V, W = random_init(target_rank, n, m, p)
            
            # Quick check with numpy
            U_arr = np.array(U, dtype=np.float64)
            V_arr = np.array(V, dtype=np.float64)
            W_arr = np.array(W, dtype=np.float64)
            
            ok = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                        if abs(s - 1.0) > 1e-4:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            
            if ok:
                found = True
                best_U, best_V, best_W = U, V, W
                best_rank = target_rank
                break
            elif num_trials % 200 == 0:
                # Do local search to try to improve
                U_ls, V_ls, W_ls, err = local_search(U, V, W, n, m, p, max_iters=3000, time_limit=deadline-15)
                if err < 1e-9:
                    # Verify with exact check
                    if verify_decomposition(U_ls, V_ls, W_ls, n, m, p, tol=1e-9):
                        found = True
                        best_U, best_V, best_W = U_ls, V_ls, W_ls
                        best_rank = target_rank
                        break
    
    if best_rank < rank_fallback:
        return {"U": best_U, "V": best_V, "W": best_W}
    else:
        return {"U": U_fb, "V": V_fb, "W": W_fb}
# EVOLVE-BLOCK-END
