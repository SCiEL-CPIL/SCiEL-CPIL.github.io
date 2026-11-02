import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    target = np.zeros((nm, mp, pn), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1.0
    
    # We search for integer matrices U_int, V_int, W_int
    # such that sum_k U_int[ak] V_int[bk] W_int[ck] = 8 * target[abc]
    # This corresponds to U = U_int/2, V = V_int/2, W = W_int/2
    target_int = (target * 8).astype(np.int64)
    
    def compute_error_int(U, V, W):
        # U, V, W are int64 arrays
        T_approx = np.einsum('ak,bk,ck->abc', U, V, W)
        diff = T_approx - target_int
        return np.sum(diff**2)
    
    def check_solution(U_int, V_int, W_int):
        err = compute_error_int(U_int, V_int, W_int)
        if err == 0:
            # Convert to float half-integers
            U = U_int.astype(np.float64) / 2.0
            V = V_int.astype(np.float64) / 2.0
            W = W_int.astype(np.float64) / 2.0
            return U, V, W
        return None

    def try_rank_int(r, max_restarts=50, max_steps_per_restart=5000):
        """
        Try to find an integer decomposition of rank r.
        Uses hill climbing on the integer lattice.
        """
        if time.time() > deadline - 10:
            return None
            
        best_err = float('inf')
        best_U, best_V, best_W = None, None, None
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            # Initialize with random integers in [-4, 4]
            U = np.random.randint(-4, 5, (nm, r), dtype=np.int64)
            V = np.random.randint(-4, 5, (mp, r), dtype=np.int64)
            W = np.random.randint(-4, 5, (pn, r), dtype=np.int64)
            
            # Occasionally try a "schoolbook-like" initialization
            if restart % 5 == 0:
                U = np.zeros((nm, r), dtype=np.int64)
                V = np.zeros((mp, r), dtype=np.int64)
                W = np.zeros((pn, r), dtype=np.int64)
                # Distribute 24 terms into r columns
                terms = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
                for idx, (i, j, k) in enumerate(terms):
                    col = idx % r
                    U[i*m+j, col] += 2 # 1.0 / 0.5 = 2
                    V[j*p+k, col] += 2
                    W[k*n+i, col] += 2
                # Add some noise
                U += np.random.randint(-1, 2, (nm, r), dtype=np.int64)
                V += np.random.randint(-1, 2, (mp, r), dtype=np.int64)
                W += np.random.randint(-1, 2, (pn, r), dtype=np.int64)
            
            curr_err = compute_error_int(U, V, W)
            
            if curr_err < best_err:
                best_err = curr_err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if curr_err == 0:
                    return check_solution(U, V, W)
            
            # Hill Climbing
            steps = 0
            while steps < max_steps_per_restart:
                if time.time() > deadline - 5:
                    break
                    
                # Pick a random matrix
                mat_idx = np.random.choice(3)
                if mat_idx == 0:
                    mat = U
                    shape = (nm, r)
                elif mat_idx == 1:
                    mat = V
                    shape = (mp, r)
                else:
                    mat = W
                    shape = (pn, r)
                    
                # Pick random index
                idx1 = np.random.randint(shape[0])
                idx2 = np.random.randint(shape[1])
                
                # Pick delta
                delta = np.random.choice([-2, -1, 1, 2])
                
                # Apply change
                old_val = mat[idx1, idx2]
                mat[idx1, idx2] = old_val + delta
                
                new_err = compute_error_int(U, V, W)
                
                if new_err < curr_err:
                    curr_err = new_err
                    if curr_err < best_err:
                        best_err = curr_err
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        if curr_err == 0:
                            return check_solution(U, V, W)
                else:
                    # Revert
                    mat[idx1, idx2] = old_val
                    # If stuck, maybe do a larger jump?
                    # For now, just revert.
                    
                steps += 1
        
        # If we found a very good solution but not 0, check if it's close enough
        # and try to polish
        if best_U is not None and best_err < 100:
            # Try a few more steps on the best solution found
            U = best_U.copy()
            V = best_V.copy()
            W = best_W.copy()
            curr_err = best_err
            for _ in range(1000):
                if time.time() > deadline - 2:
                    break
                mat_idx = np.random.choice(3)
                if mat_idx == 0: mat = U
                elif mat_idx == 1: mat = V
                else: mat = W
                
                idx1 = np.random.randint(mat.shape[0])
                idx2 = np.random.randint(mat.shape[1])
                delta = np.random.choice([-1, 1])
                
                old_val = mat[idx1, idx2]
                mat[idx1, idx2] = old_val + delta
                new_err = compute_error_int(U, V, W)
                
                if new_err < curr_err:
                    curr_err = new_err
                    if curr_err == 0:
                        return check_solution(U, V, W)
                else:
                    mat[idx1, idx2] = old_val
            
        return None

    # Try ranks from low to high
    # Start from 16, go up to 23
    for rank in range(16, 24):
        if time.time() > deadline - 15:
            break
        
        # Adjust restarts and steps based on rank
        # Lower ranks are harder, need more restarts? Or are they just impossible?
        # Let's give each rank a reasonable budget.
        res = try_rank_int(rank, max_restarts=20, max_steps_per_restart=2000)
        if res is not None:
            U, V, W = res
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to rank 24 (Schoolbook)
    rank = n * m * p
    U = np.zeros((nm, rank), dtype=np.float64)
    V = np.zeros((mp, rank), dtype=np.float64)
    W = np.zeros((pn, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
