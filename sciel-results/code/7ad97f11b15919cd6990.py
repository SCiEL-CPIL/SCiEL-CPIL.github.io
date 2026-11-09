import numpy as np
import time
import random

def construct():
    n, m, p = 3, 3, 3
    start_time = time.time()
    time_limit = 110.0
    
    # Fallback Rank 27
    U_fb = [[0]*27 for _ in range(9)]
    V_fb = [[0]*27 for _ in range(9)]
    W_fb = [[0]*27 for _ in range(9)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j][col] = 1
                V_fb[j*p+k][col] = 1
                W_fb[k*n+i][col] = 1
                col += 1
    
    # Target ranks to try, starting from low
    # We know rank 20 is possible. Let's try 20, 21, 22...
    # But finding rank 20 is hard. Maybe 21-26 is easier?
    # Actually, if we find ANY rank < 27, we improve.
    
    def get_error(U, V, W):
        # U, V, W are numpy arrays of shape (9, R)
        # Reshape to (3,3,R)
        U_t = U.reshape(3, 3, -1)
        V_t = V.reshape(3, 3, -1)
        W_t = W.reshape(3, 3, -1)
        
        # Compute T_ijk = sum_r U_ijk_r V_jkr_r W_kir_r
        # Note indices:
        # U[i, j, r]
        # V[j, k, r]
        # W[k, i, r] -> W_t is (k, i, r) in memory if reshaped from (k*n+i, r)?
        # W is (9, R). Index k*n+i.
        # So W_t[k, i, r] is correct if we reshape (9, R) to (3, 3, R).
        
        T_approx = np.einsum('ijr,jkr,ki r->ijk', U_t, V_t, W_t)
        # Wait, W_t is (3,3,R). The indices are k, i.
        # So W_t[k, i, r].
        # The einsum string:
        # U: i, j, r
        # V: j, k, r
        # W: k, i, r
        # Output: i, j, k
        
        # Let's verify the einsum string.
        # 'ijr,jkr,ki r->ijk'
        # U_t[i,j,r] * V_t[j,k,r] * W_t[k,i,r]
        # This sums over r.
        # Result T[i,j,k].
        # We want T[i,j,k] = 1 for all i,j,k.
        
        error = np.sum((T_approx - 1.0)**2)
        return error

    def local_search(U, V, W, max_iter=1000):
        U = U.copy()
        V = V.copy()
        W = W.copy()
        current_error = get_error(U, V, W)
        
        if current_error < 1e-9:
            return U, V, W, 0.0
            
        for _ in range(max_iter):
            if time.time() - start_time > time_limit:
                break
                
            # Pick a random matrix and a random entry to flip
            mat_choice = random.choice([U, V, W])
            idx1 = random.randint(0, 8)
            idx2 = random.randint(0, mat_choice.shape[1]-1)
            
            old_val = mat_choice[idx1, idx2]
            
            # Try a new value from {-2, -1, 0, 1, 2}
            new_val = random.choice([-2, -1, 0, 1, 2])
            if new_val == old_val:
                continue
                
            mat_choice[idx1, idx2] = new_val
            new_error = get_error(U, V, W)
            
            if new_error < current_error:
                current_error = new_error
                if current_error < 1e-9:
                    return U, V, W, 0.0
            else:
                # Revert
                mat_choice[idx1, idx2] = old_val
                
        return U, V, W, current_error

    # Search for solutions with rank 20 to 26
    # Start with 20, go up to 26
    for rank in range(20, 27):
        if time.time() - start_time > time_limit:
            break
            
        # Try a few random initializations
        for attempt in range(50):
            if time.time() - start_time > time_limit:
                break
                
            # Initialize with random integers
            # Use a sparse initialization? Or dense?
            # Dense random is better for local search to move around.
            U = np.random.choice([-2, -1, 0, 1, 2], size=(9, rank))
            V = np.random.choice([-2, -1, 0, 1, 2], size=(9, rank))
            W = np.random.choice([-2, -1, 0, 1, 2], size=(9, rank))
            
            U, V, W, err = local_search(U, V, W, max_iter=500)
            
            if err < 1e-9:
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }

    return {
        "U": U_fb,
        "V": V_fb,
        "W": W_fb
    }
