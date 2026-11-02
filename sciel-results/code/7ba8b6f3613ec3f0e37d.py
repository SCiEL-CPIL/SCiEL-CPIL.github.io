import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 3, 3, 4
    
    # Target tensor
    T_target = np.zeros((n, m, p), dtype=np.int64)
    active_triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i, j, k] = 1
                active_triples.append((i, j, k))
    
    # Schoolbook fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=np.int64)
        V = np.zeros((m * p, rank), dtype=np.int64)
        W = np.zeros((p * n, rank), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W, r):
        """
        Verify decomposition using einsum.
        """
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        # Reshape to (n, m, r), (m, p, r), (p, n, r)
        U_r = U.reshape(n, m, r)
        V_r = V.reshape(m, p, r)
        W_r = W.reshape(p, n, r)
        
        # Compute tensor T[i, j, k] = sum_alpha U[i, j, alpha] * V[j, k, alpha] * W[k, i, alpha]
        # Note: W index is k*n + i, so W_r[k, i, alpha]
        # einsum: 'ija, jka, kia -> ijk'
        T_calc = np.einsum('ija,jka,kia->ijk', U_r, V_r, W_r, optimize=True)
        
        # Check against T_target
        # T_target is (n, m, p) which is (i, j, k)
        # T_calc is (i, j, k)
        return np.all(T_calc == T_target)
    
    def error_metric(U, V, W, r):
        """
        Compute squared error.
        """
        U_r = U.reshape(n, m, r)
        V_r = V.reshape(m, p, r)
        W_r = W.reshape(p, n, r)
        
        T_calc = np.einsum('ija,jka,kia->ijk', U_r, V_r, W_r, optimize=True)
        
        # Error on active triples (should be 1)
        err_active = np.sum((T_calc - 1)**2)
        
        # Error on inactive triples (should be 0)
        # T_target is 0 for inactive
        # So err_inactive = sum(T_calc^2) for inactive
        # We can just compute sum(T_calc^2) over all, and subtract active part?
        # No, active part is (1-1)^2 = 0.
        # So total error = sum((T_calc - T_target)^2)
        err = np.sum((T_calc - T_target)**2)
        return err
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Try ranks 18, 24
    ranks_to_try = [18, 24]
    
    for r in ranks_to_try:
        if time.time() > deadline - 10:
            break
            
        # Local search
        U = np.zeros((n * m, r), dtype=np.int64)
        V = np.zeros((m * p, r), dtype=np.int64)
        W = np.zeros((p * n, r), dtype=np.int64)
        
        # Initialize with random sparse values
        for _ in range(100):
            if time.time() > deadline - 5:
                break
                
            # Random initialization
            U[:] = 0
            V[:] = 0
            W[:] = 0
            
            # Each column of U, V, W has some non-zeros
            for alpha in range(r):
                # U column
                nz_u = np.random.choice(n * m, size=max(1, int(r * 0.2)), replace=False)
                U[nz_u, alpha] = np.random.choice([-1, 1], size=len(nz_u))
                
                # V column
                nz_v = np.random.choice(m * p, size=max(1, int(r * 0.2)), replace=False)
                V[nz_v, alpha] = np.random.choice([-1, 1], size=len(nz_v))
                
                # W column
                nz_w = np.random.choice(p * n, size=max(1, int(r * 0.2)), replace=False)
                W[nz_w, alpha] = np.random.choice([-1, 1], size=len(nz_w))
            
            current_err = error_metric(U, V, W, r)
            if current_err == 0:
                best_result = {"U": U, "V": V, "W": W}
                best_rank = r
                break
            
            # Local search
            for step in range(500):
                if time.time() > deadline - 5:
                    break
                    
                # Mutate one entry
                mat = np.random.choice(['U', 'V', 'W'])
                if mat == 'U':
                    idx = np.random.randint(0, n * m)
                    col = np.random.randint(0, r)
                    old_val = U[idx, col]
                    U[idx, col] = np.random.choice([-2, -1, 0, 1, 2])
                elif mat == 'V':
                    idx = np.random.randint(0, m * p)
                    col = np.random.randint(0, r)
                    old_val = V[idx, col]
                    V[idx, col] = np.random.choice([-2, -1, 0, 1, 2])
                else:
                    idx = np.random.randint(0, p * n)
                    col = np.random.randint(0, r)
                    old_val = W[idx, col]
                    W[idx, col] = np.random.choice([-2, -1, 0, 1, 2])
                
                new_err = error_metric(U, V, W, r)
                if new_err < current_err:
                    current_err = new_err
                    if current_err == 0:
                        break
                else:
                    # Revert
                    if mat == 'U':
                        U[idx, col] = old_val
                    elif mat == 'V':
                        V[idx, col] = old_val
                    else:
                        W[idx, col] = old_val
                        
            if current_err == 0:
                best_result = {"U": U, "V": V, "W": W}
                best_rank = r
                break
                
    return best_result
