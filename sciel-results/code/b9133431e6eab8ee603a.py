import time
import random
import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    # Pre-compute index arrays for vectorized evaluation
    # Each tensor index (i,j,k) maps to U[i*m+j], V[j*p+k], W[k*n+i]
    indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    num_indices = len(indices)

    u_rows = np.array([i * m + j for i, j, k in indices])
    v_rows = np.array([j * p + k for i, j, k in indices])
    w_rows = np.array([k * n + i for i, j, k in indices])

    # Schoolbook fallback (rank = n*m*p = 112)
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    # Vectorized evaluation
    def evaluate(U_arr, V_arr, W_arr):
        U_sel = U_arr[u_rows]
        V_sel = V_arr[v_rows]
        W_sel = W_arr[w_rows]
        return np.sum(U_sel * V_sel * W_sel, axis=1)

    def is_valid(T):
        return bool(np.all(T == 1))

    def get_error_indices(T):
        return np.where(T != 1)[0]

    # Strassen-based construction: 4x4 * 4x4 multiplication at rank 21 (best known for 4x4 is 21)
    # But we need 4x4 * 4x7. We can construct 4x4 * 4x4 at some rank R44,
    # then extend to 4x4 * 4x7 by adding 3 more columns (rank 3*4 = 12 more), total rank R44 + 12.
    # However, the best known rank for 4x4 matrix multiplication is 21 (Stothers).
    # For 4x4 * 4x7, we can use rank 21 + 3*4 = 33? No, that's not how it works.
    
    # Actually, let's think differently. We can use a rank R decomposition of 4x4*4x4,
    # and then for the extra 3 columns of B (making it 4x7), we need 3 additional "layers".
    # Each layer of 4x4*4x1 needs rank 4 (since it's just a linear map from 4-dim to 4-dim,
    # which can be done in rank 4). So total rank would be R44 + 3*4 = R44 + 12.
    
    # For 4x4*4x4, the best known rank is 21. So we'd get rank 33.
    # But we don't have an explicit rank-21 decomposition handy. Let's use rank 16 (schoolbook)
    # for the 4x4*4x4 part, giving total rank 16 + 12 = 28. That's the lower bound!
    # Wait, the lower bound is max(n*m, m*p, n*p) = max(16, 28, 28) = 28.
    # So rank 28 might be achievable.
    
    # Let's try to construct rank 28 directly.
    # For 4x4 * 4x7:
    # U: 16 x 28, V: 28 x 28, W: 28 x 16 (wait, W is p*n x rank = 28 x 28? No, W is (p*n, rank) = (7*4, 28) = (28, 28))
    
    # Actually, let me reconsider. The schoolbook for 4x4*4x7 has rank 112.
    # We want to do better. Let's try random search at various ranks, starting from 28 upward.
    
    # First, let's try to find a rank-28 solution via local search with better initialization.
    # Strategy: initialize with a "structured" pattern rather than pure random.
    
    def try_rank_random(R, max_restarts=5, max_iters=5000):
        """Try to find a rank-R solution via randomized local search."""
        for restart in range(max_restarts):
            if time.time() > deadline - 3:
                return None
            
            # Initialize with sparse random entries
            U_arr = np.zeros((n * m, R), dtype=float)
            V_arr = np.zeros((m * p, R), dtype=float)
            W_arr = np.zeros((p * n, R), dtype=float)
            
            # Seed with some 1s and -1s
            for r in range(R):
                for _ in range(3):
                    U_arr[random.randint(0, n*m-1), r] = random.choice([1, -1])
                for _ in range(3):
                    V_arr[random.randint(0, m*p-1), r] = random.choice([1, -1])
                for _ in range(3):
                    W_arr[random.randint(0, p*n-1), r] = random.choice([1, -1])
            
            best_error = num_indices + 1
            stuck = 0
            
            for it in range(max_iters):
                if time.time() > deadline - 2:
                    return None
                
                T = evaluate(U_arr, V_arr, W_arr)
                err_idx = get_error_indices(T)
                error_count = len(err_idx)
                
                if error_count == 0:
                    return {
                        "U": U_arr.astype(int).tolist(),
                        "V": V_arr.astype(int).tolist(),
                        "W": W_arr.astype(int).tolist()
                    }
                
                if error_count < best_error:
                    best_error = error_count
                    stuck = 0
                else:
                    stuck += 1
                
                if stuck > 100:
                    # Perturb
                    for _ in range(random.randint(3, 10)):
                        var = random.choice(['U', 'V', 'W'])
                        if var == 'U':
                            U_arr[random.randint(0, n*m-1), random.randint(0, R-1)] = random.choice([0, 1, -1, 2, -2])
                        elif var == 'V':
                            V_arr[random.randint(0, m*p-1), random.randint(0, R-1)] = random.choice([0, 1, -1, 2, -2])
                        else:
                            W_arr[random.randint(0, p*n-1), random.randint(0, R-1)] = random.choice([0, 1, -1, 2, -2])
                    stuck = 0
                    continue
                
                # Pick a random error and try to fix it
                if len(err_idx) == 0:
                    break
                    
                t = random.choice(err_idx.tolist())
                i, j, k = indices[t]
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                current_sum = int(round(T[t]))
                
                # Try to adjust one entry
                r = random.randint(0, R - 1)
                u_r = int(round(U_arr[u_idx, r]))
                v_r = int(round(V_arr[v_idx, r]))
                w_r = int(round(W_arr[w_idx, r]))
                contrib = u_r * v_r * w_r
                
                target = 1 - current_sum + contrib
                
                # Try adjusting U
                if v_r != 0 and w_r != 0:
                    denom = v_r * w_r
                    if target % denom == 0:
                        new_u = target // denom
                        if -50 <= new_u <= 50:
                            U_arr[u_idx, r] = new_u
                            continue
                
                # Try adjusting V
                if u_r != 0 and w_r != 0:
                    denom = u_r * w_r
                    if target % denom == 0:
                        new_v = target // denom
                        if -50 <= new_v <= 50:
                            V_arr[v_idx, r] = new_v
                            continue
                
                # Try adjusting W
                if u_r != 0 and v_r != 0:
                    denom = u_r * v_r
                    if target % denom == 0:
                        new_w = target // denom
                        if -50 <= new_w <= 50:
                            W_arr[w_idx, r] = new_w
                            continue
                
                # If none worked, do a small random perturbation
                var = random.choice(['U', 'V', 'W'])
                if var == 'U':
                    U_arr[random.randint(0, n*m-1), random.randint(0, R-1)] = random.choice([0, 1, -1, 2, -2])
                elif var == 'V':
                    V_arr[random.randint(0, m*p-1), random.randint(0, R-1)] = random.choice([0, 1, -1, 2, -2])
                else:
                    W_arr[random.randint(0, p*n-1), random.randint(0, R-1)] = random.choice([0, 1, -1, 2, -2])
        
        return None

    # Try ranks from 28 upward
    best_solution = None
    best_rank = rank_school

    for R in range(28, min(112, 60)):
        if time.time() > deadline - 10:
            break
            
        if R >= best_rank:
            continue
        
        # More restarts for lower ranks
        restarts = 10 if R <= 35 else 5
        iters = 8000 if R <= 35 else 5000
        
        sol = try_rank_random(R, max_restarts=restarts, max_iters=iters)
        if sol is not None:
            best_solution = sol
            best_rank = R
            break
    
    if best_solution is not None:
        return best_solution

    # Fallback: use a simple block construction
    # For 4x4 * 4x7, we can decompose B into 7 columns.
    # Use schoolbook for each column: rank 4*4*7 = 112
    # Or, use a smarter approach: treat it as 7 separate 4x4 * 4x1 multiplications.
    # Each 4x4 * 4x1 can be done in rank 4 (just 4 multiplications).
    # Total rank: 7 * 4 = 28. But this doesn't work because the A matrix is shared.
    
    # Actually, let's just return the schoolbook solution if we couldn't find better.
    return {"U": U_school, "V": V_school, "W": W_school}
