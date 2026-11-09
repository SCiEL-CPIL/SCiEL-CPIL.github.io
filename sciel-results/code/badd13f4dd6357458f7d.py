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
        """Generate factor matrices with half-integer entries using multiple strategies."""
        if seed is not None:
            random.seed(seed)
        
        strategy = random.randint(0, 3)
        values = [-1, -0.5, 0, 0.5, 1]
        
        if strategy == 0:
            # Pure random
            U = [[random.choice(values) for _ in range(rank)] for _ in range(n*m)]
            V = [[random.choice(values) for _ in range(rank)] for _ in range(m*p)]
            W = [[random.choice(values) for _ in range(rank)] for _ in range(p*n)]
        elif strategy == 1:
            # Sparse random (more zeros)
            sparse_vals = [-1, 0, 1]
            U = [[random.choice(sparse_vals) for _ in range(rank)] for _ in range(n*m)]
            V = [[random.choice(sparse_vals) for _ in range(rank)] for _ in range(m*p)]
            W = [[random.choice(sparse_vals) for _ in range(rank)] for _ in range(p*n)]
        elif strategy == 2:
            # Block structure: divide rank into blocks
            block_size = max(1, rank // 4)
            U = [[0] * rank for _ in range(n*m)]
            V = [[0] * rank for _ in range(m*p)]
            W = [[0] * rank for _ in range(p*n)]
            for block in range(rank // block_size + 1):
                start = block * block_size
                end = min(start + block_size, rank)
                for c in range(start, end):
                    for i in range(n):
                        for j in range(m):
                            U[i*m+j][c] = random.choice(values)
                    for j in range(m):
                        for k in range(p):
                            V[j*p+k][c] = random.choice(values)
                    for k in range(p):
                        for i in range(n):
                            W[k*n+i][c] = random.choice(values)
        else:
            # Kronecker-like structure
            U = [[0] * rank for _ in range(n*m)]
            V = [[0] * rank for _ in range(m*p)]
            W = [[0] * rank for _ in range(p*n)]
            
            # Create patterns based on indices
            for i in range(n):
                for j in range(m):
                    for c in range(rank):
                        if (i + j) % 3 == 0:
                            U[i*m+j][c] = random.choice([-1, 1])
                        elif (i + j) % 3 == 1:
                            U[i*m+j][c] = random.choice([-0.5, 0.5])
                        else:
                            U[i*m+j][c] = random.choice(values)
            
            for j in range(m):
                for k in range(p):
                    for c in range(rank):
                        if (j + k) % 3 == 0:
                            V[j*p+k][c] = random.choice([-1, 1])
                        elif (j + k) % 3 == 1:
                            V[j*p+k][c] = random.choice([-0.5, 0.5])
                        else:
                            V[j*p+k][c] = random.choice(values)
            
            for k in range(p):
                for i in range(n):
                    for c in range(rank):
                        if (k + i) % 3 == 0:
                            W[k*n+i][c] = random.choice([-1, 1])
                        elif (k + i) % 3 == 1:
                            W[k*n+i][c] = random.choice([-0.5, 0.5])
                        else:
                            W[k*n+i][c] = random.choice(values)
        
        return U, V, W
    
    def local_search(U, V, W, n, m, p, max_iters=5000, time_limit=140.0):
        """Local search with simulated annealing to refine factor matrices.
        
        Tries to flip individual entries to reduce the error.
        Uses simulated annealing to escape local minima.
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
        
        # Simulated annealing parameters
        start_temp = 1.0
        end_temp = 0.01
        cooling_rate = 0.999
        
        current_temp = start_temp
        no_improve_count = 0
        max_no_improve = 1000
        
        for it in range(max_iters):
            if time.time() > time_limit:
                break
            
            # Cool down
            current_temp = max(end_temp, current_temp * cooling_rate)
            
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
                
                # Simulated annealing acceptance
                if err < best_err:
                    best_new_err = err
                    best_new_val = new_val
                else:
                    # Accept worse solutions with probability exp(-delta/temp)
                    delta = err - best_err
                    if delta < 100 * current_temp:  # Avoid overflow
                        accept_prob = np.exp(-delta / current_temp)
                        if random.random() < accept_prob:
                            best_new_err = err
                            best_new_val = new_val
            
            # Apply the best change
            if best_new_val != old_val:
                if mat == 'U':
                    U[row_idx][col_idx] = best_new_val
                elif mat == 'V':
                    V[row_idx][col_idx] = best_new_val
                else:
                    W[row_idx][col_idx] = best_new_val
                
                if best_new_err < best_err:
                    best_err = best_new_err
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                
                if best_err < 1e-12:
                    break
            
            # If stuck for too long, restart with higher temperature
            if no_improve_count > max_no_improve:
                current_temp = min(start_temp, current_temp * 2)
                no_improve_count = 0
        
        return U, V, W, best_err
    
    best_U, best_V, best_W = U_fb, V_fb, W_fb
    best_rank = rank_fallback
    
    random.seed(42)
    
    # Try ranks from 50 down to 30 (start higher to find better solutions first)
    for target_rank in [50, 48, 46, 44, 42, 40, 38, 36, 34, 32, 30]:
        if time.time() > deadline - 25:
            break
        
        found = False
        num_trials = 0
        max_trials = 20000
        
        while not found and num_trials < max_trials and time.time() < deadline - 15:
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
            elif num_trials % 100 == 0:
                # Do local search to try to improve
                U_ls, V_ls, W_ls, err = local_search(U, V, W, n, m, p, max_iters=5000, time_limit=deadline-20)
                if err < 1e-8:
                    # Final polish with tighter tolerance
                    U_polish, V_polish, W_polish, err_polish = local_search(
                        U_ls, V_ls, W_ls, n, m, p, 
                        max_iters=10000, time_limit=deadline-18
                    )
                    if err_polish < 1e-10:
                        # Verify with exact check
                        if verify_decomposition(U_polish, V_polish, W_polish, n, m, p, tol=1e-9):
                            found = True
                            best_U, best_V, best_W = U_polish, V_polish, W_polish
                            best_rank = target_rank
                            break
                        elif verify_decomposition(U_ls, V_ls, W_ls, n, m, p, tol=1e-9):
                            found = True
                            best_U, best_V, best_W = U_ls, V_ls, W_ls
                            best_rank = target_rank
                            break
    
    if best_rank < rank_fallback:
        return {"U": best_U, "V": best_V, "W": best_W}
    else:
        return {"U": U_fb, "V": V_fb, "W": W_fb}
# EVOLVE-BLOCK-END
