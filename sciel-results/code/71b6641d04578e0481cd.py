import numpy as np
import time
import random

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 115  # Safety margin
    
    N_u = n * m
    N_v = m * p
    N_w = p * n
    N_eq = n * m * p
    
    def compute_tensor(U, V, W, r):
        """Compute the full tensor T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]"""
        T = np.zeros((n, m, p))
        # Vectorized computation for speed
        # U: (n*m, r) -> (n, m, r)
        # V: (m*p, r) -> (m, p, r)
        # W: (p*n, r) -> (p, n, r)
        
        # Reshape
        U_r = U.reshape(n, m, r)
        V_r = V.reshape(m, p, r)
        W_r = W.reshape(p, n, r)
        
        # We want T[i, j, k] = sum_c U[i, j, c] * V[j, k, c] * W[k, i, c]
        # Note indices:
        # U[i, j, c]
        # V[j, k, c]
        # W[k, i, c] -> W is indexed by k*n+i, so W_r[k, i, c]
        
        # Iterate over c to accumulate
        for c in range(r):
            u_c = U_r[:, :, c] # (n, m)
            v_c = V_r[:, :, c] # (m, p)
            w_c = W_r[:, :, c] # (p, n)
            
            # T[i, j, k] += u_c[i, j] * v_c[j, k] * w_c[k, i]
            # This is a tensor contraction.
            # prod[i, j, k] = u_c[i, j] * v_c[j, k] * w_c[k, i]
            # u_c[:, :, None] * v_c[None, :, :] * w_c.T[:, None, :] ? No.
            # w_c is (p, n). We need w_c[k, i].
            # So w_c.T is (n, p).
            # We need to align i, j, k.
            # u_c[i, j]
            # v_c[j, k]
            # w_c[k, i]
            
            # Let's construct the product term by term or use broadcasting carefully.
            # Term = u_c[i,j] * v_c[j,k] * w_c[k,i]
            
            # Step 1: u_c[i,j] * v_c[j,k] -> (n, m, p)
            # u_c[:, :, None] * v_c[None, :, :] -> (n, m, p)
            uv = u_c[:, :, None] * v_c[None, :, :]
            
            # Step 2: Multiply by w_c[k,i]
            # w_c is (p, n). We need to index by k and i.
            # uv is (n, m, p). We need to multiply uv[i, j, k] by w_c[k, i].
            # w_c.T is (n, p).
            # We need w_c[k, i] for each i, k.
            # Let's reshape w_c to (1, 1, p, n) ? No.
            # w_c.T[i, k] is what we want for the (i, k) slice of uv.
            # So we can do:
            # w_term = w_c.T[:, None, :] # (n, 1, p)
            # T += uv * w_term
            
            w_term = w_c.T[:, None, :] # (n, 1, p)
            T += uv * w_term
            
        return T

    def is_valid(U, V, W, r, atol=1e-6):
        T = compute_tensor(U, V, W, r)
        return np.allclose(T, 1.0, atol=atol)

    def snap_and_validate(U, V, W, r, snap_func):
        U_s = snap_func(U)
        V_s = snap_func(V)
        W_s = snap_func(W)
        if is_valid(U_s, V_s, W_s, r):
            return U_s, V_s, W_s
        return None

    # Initialize with Schoolbook
    rank_schoolbook = N_u * N_v // N_w # No, rank is n*m*p
    rank_schoolbook = n * m * p
    
    U_school = np.zeros((N_u, rank_schoolbook))
    V_school = np.zeros((N_v, rank_schoolbook))
    W_school = np.zeros((N_w, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()
    best_rank = rank_schoolbook
    
    # Try to reduce rank using local search from random starts
    # We will try ranks from best_rank - 1 down to a lower bound
    # But random search for low rank is hard.
    # Let's try to improve the schoolbook rank by removing terms and adjusting others.
    
    def local_search(U, V, W, r, max_iter=10000):
        """
        Local search to minimize error. 
        Returns (U, V, W) if valid, else None.
        """
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        # Calculate initial error
        T = compute_tensor(U, V, W, r)
        error = np.sum((T - 1.0)**2)
        
        # If already valid
        if error < 1e-6:
            return U, V, W
        
        best_error = error
        best_U = U.copy()
        best_V = V.copy()
        best_W = W.copy()
        
        # Define possible moves: change one element in U, V, or W by +/- 1
        # To speed up, we can only consider changing elements that are non-zero or near zero?
        # For now, random selection of indices.
        
        for it in range(max_iter):
            if time.time() - start_time > time_limit:
                return None
                
            # Pick a random matrix and index
            mat_choice = random.choice(['U', 'V', 'W'])
            if mat_choice == 'U':
                idx = (random.randint(0, N_u - 1), random.randint(0, r - 1))
                delta = random.choice([-1, 1])
                old_val = U[idx]
                U[idx] += delta
                new_T = compute_tensor(U, V, W, r)
                new_error = np.sum((new_T - 1.0)**2)
                if new_error < best_error:
                    best_error = new_error
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                    if new_error < 1e-6:
                        return best_U, best_V, best_W
                else:
                    U[idx] = old_val
                    
            elif mat_choice == 'V':
                idx = (random.randint(0, N_v - 1), random.randint(0, r - 1))
                delta = random.choice([-1, 1])
                old_val = V[idx]
                V[idx] += delta
                new_T = compute_tensor(U, V, W, r)
                new_error = np.sum((new_T - 1.0)**2)
                if new_error < best_error:
                    best_error = new_error
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                    if new_error < 1e-6:
                        return best_U, best_V, best_W
                else:
                    V[idx] = old_val
                    
            else: # W
                idx = (random.randint(0, N_w - 1), random.randint(0, r - 1))
                delta = random.choice([-1, 1])
                old_val = W[idx]
                W[idx] += delta
                new_T = compute_tensor(U, V, W, r)
                new_error = np.sum((new_T - 1.0)**2)
                if new_error < best_error:
                    best_error = new_error
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                    if new_error < 1e-6:
                        return best_U, best_V, best_W
                else:
                    W[idx] = old_val
                    
        # Final snap attempt
        if best_error < 1e-3:
            res = snap_and_validate(best_U, best_V, best_W, r, np.round)
            if res:
                return res
            res = snap_and_validate(best_U, best_V, best_W, r, lambda x: np.round(x*2)/2)
            if res:
                return res
                
        return None

    # Strategy: Try to reduce rank from schoolbook by 1, 2, etc.
    # Start with schoolbook, remove one term, and try to fix the error with local search.
    
    for r_new in range(rank_schoolbook - 1, max(0, rank_schoolbook - 20), -1):
        if time.time() - start_time > time_limit:
            break
            
        # Try multiple random removals
        for _ in range(10):
            if time.time() - start_time > time_limit:
                break
                
            # Remove a random term
            c_remove = random.randint(0, rank_schoolbook - 1)
            keep_idx = [i for i in range(rank_schoolbook) if i != c_remove]
            
            U_init = U_school[:, keep_idx]
            V_init = V_school[:, keep_idx]
            W_init = W_school[:, keep_idx]
            
            # Add some random noise to help escape local minima?
            # Or just run local search.
            result = local_search(U_init, V_init, W_init, r_new, max_iter=5000)
            if result:
                best_U, best_V, best_W = result
                best_rank = r_new
                break # Move to next lower rank
        
        if best_rank == r_new:
            continue # Found a solution for this rank, try to go lower
        else:
            break # Couldn't find solution for this rank, try next? Or stop?
            # Actually, if we fail for r_new, we might succeed for r_new+1?
            # But we are going downwards. If we fail for r_new, we should probably stop or try more attempts.
            # Let's try a few more attempts for the same rank.
            pass

    # If no reduction found, return schoolbook
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
