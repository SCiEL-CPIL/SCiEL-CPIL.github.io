import time
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 64
    
    # Schoolbook decomposition
    U_school = np.zeros((n*m, rank_schoolbook))
    V_school = np.zeros((m*p, rank_schoolbook))
    W_school = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
                
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_school, V_school, W_school

    def validate(U_try, V_try, W_try, rank):
        """Check if the decomposition is valid (exact integer)."""
        # Reconstruct tensor T[i, j, k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        # Using einsum for speed
        try:
            # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
            # We need to reshape to (n, m, rank), (m, p, rank), (p, n, rank)
            U_res = U_try.reshape(n, m, rank)
            V_res = V_try.reshape(m, p, rank)
            W_res = W_try.reshape(p, n, rank)
            
            # T[i, j, k] = sum_r U_res[i, j, r] * V_res[j, k, r] * W_res[k, i, r]
            # This is a bit tricky with einsum directly because indices are permuted.
            # Let's just do it with loops or optimized numpy if possible.
            # Given small dimensions, a loop is fast enough for validation.
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        total = 0.0
                        # Sum over r
                        # U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]
                        total = np.dot(np.dot(U_res[i, j, :], V_res[j, k, :]), W_res[k, i, :])
                        if total != 1.0:
                            return False
            return True
        except Exception:
            return False

    # Search for lower rank
    # We try ranks from 63 down to 50
    for target_rank in range(63, 49, -1):
        if time.time() > deadline - 5:
            break
        
        # Try multiple random initializations
        for attempt in range(100):
            if time.time() > deadline - 3:
                break
            
            np.random.seed(attempt * 12345 + target_rank)
            
            # Initialize with sparse random integers {-1, 0, 1}
            # Bias towards sparsity
            U_try = np.random.choice([-1, 0, 1], size=(n*m, target_rank))
            V_try = np.random.choice([-1, 0, 1], size=(m*p, target_rank))
            W_try = np.random.choice([-1, 0, 1], size=(p*n, target_rank))
            
            # Local Search: Hill Climbing
            # Objective: Minimize error. Since we need exact 1, we can look for 0 error.
            # Let's compute the error vector first to see if we are close.
            
            # Fast error check
            def get_error(U_t, V_t, W_t, r):
                U_r = U_t.reshape(n, m, r)
                V_r = V_t.reshape(m, p, r)
                W_r = W_t.reshape(p, n, r)
                err = 0.0
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(np.dot(U_r[i, j, :], V_r[j, k, :]), W_r[k, i, :])
                            diff = abs(val - 1.0)
                            err += diff
                            if diff > max_err:
                                max_err = diff
                return err, max_err

            current_err, _ = get_error(U_try, V_try, W_try, target_rank)
            
            # If we start with 0 error, we are done (unlikely)
            # Run hill climbing
            no_improve_count = 0
            for step in range(500): # Limited steps per attempt
                if time.time() > deadline - 2:
                    break
                
                # Pick a random matrix and entry
                mat_idx = np.random.randint(3)
                if mat_idx == 0:
                    mat = U_try
                    dim1, dim2 = n*m, target_rank
                elif mat_idx == 1:
                    mat = V_try
                    dim1, dim2 = m*p, target_rank
                else:
                    mat = W_try
                    dim1, dim2 = p*n, target_rank
                
                idx1 = np.random.randint(dim1)
                idx2 = np.random.randint(dim2)
                
                old_val = mat[idx1, idx2]
                
                # Try perturbing
                # Candidates: 0, 1, -1
                # Don't keep the same value
                candidates = [0, 1, -1]
                if old_val == 0:
                    candidates = [1, -1]
                elif old_val == 1:
                    candidates = [0, -1]
                else:
                    candidates = [0, 1]
                
                new_val = np.random.choice(candidates)
                mat[idx1, idx2] = new_val
                
                new_err, _ = get_error(U_try, V_try, W_try, target_rank)
                
                if new_err < current_err:
                    current_err = new_err
                    no_improve_count = 0
                    if current_err == 0:
                        break
                else:
                    # Revert
                    mat[idx1, idx2] = old_val
                    no_improve_count += 1
                    if no_improve_count > 50:
                        break
            
            if current_err == 0:
                best_U, best_V, best_W = U_try, V_try, W_try
                best_rank = target_rank
                break
                
        if best_rank < 64:
            break

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
