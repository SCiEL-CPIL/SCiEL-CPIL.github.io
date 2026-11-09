import numpy as np
import random
import time

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 115
    
    # We search for integer matrices U_int, V_int, W_int
    # such that sum_r U_int[i*m+j, r] * V_int[j*p+k, r] * W_int[k*n+i, r] == 8
    # Then U = U_int / 2, etc.
    # Entries are in {-2, -1, 0, 1, 2}
    
    target_val = 8
    values = np.array([-2, -1, 0, 1, 2], dtype=np.int64)
    
    def compute_error(U_int, V_int, W_int):
        """
        U_int: (nm, r)
        V_int: (mp, r)
        W_int: (pn, r)
        Returns sum of squared differences from target_val.
        """
        # Reshape to 3D
        # U: (n, m, r)
        # V: (m, p, r)
        # W: (p, n, r)
        U_3d = U_int.reshape(n, m, -1)
        V_3d = V_int.reshape(m, p, -1)
        W_3d = W_int.reshape(p, n, -1)
        
        # T[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
        # Note: W index is (k, i, r)
        T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        
        # Error
        err = np.sum((T - target_val) ** 2)
        return err
    
    def local_search(U_int, V_int, W_int, max_steps=5000):
        """
        Perform local search to minimize error.
        """
        current_err = compute_error(U_int, V_int, W_int)
        if current_err == 0:
            return True
        
        r = U_int.shape[1]
        # Pre-allocate views or just use indices
        # We will modify in place
        
        for step in range(max_steps):
            if time.time() > deadline - 5:
                break
            
            # Pick a random matrix and position
            mat_idx = random.randint(0, 2)
            if mat_idx == 0:
                mat = U_int
                row = random.randint(0, nm - 1)
                col = random.randint(0, r - 1)
            elif mat_idx == 1:
                mat = V_int
                row = random.randint(0, mp - 1)
                col = random.randint(0, r - 1)
            else:
                mat = W_int
                row = random.randint(0, pn - 1)
                col = random.randint(0, r - 1)
            
            old_val = mat[row, col]
            # Pick a new value different from old
            new_val = random.choice(values.tolist())
            if new_val == old_val:
                continue
                
            mat[row, col] = new_val
            new_err = compute_error(U_int, V_int, W_int)
            
            if new_err < current_err:
                current_err = new_err
                if current_err == 0:
                    return True
            else:
                mat[row, col] = old_val
                
        return current_err == 0

    def init_random(r):
        U_int = np.random.choice(values, size=(nm, r))
        V_int = np.random.choice(values, size=(mp, r))
        W_int = np.random.choice(values, size=(pn, r))
        return U_int, V_int, W_int

    def init_schoolbook_int(r):
        """
        Initialize with schoolbook pattern scaled to integers.
        Schoolbook: U[i*m+j, c]=1, V[j*p+k, c]=1, W[k*n+i, c]=1 for c corresponding to (i,j,k).
        Here we need sum = 8.
        If we use 1s, sum is 1. We need 8.
        So we can use 2s? 2*2*2 = 8.
        So U=2, V=2, W=2 for the active term.
        """
        U_int = np.zeros((nm, r), dtype=np.int64)
        V_int = np.zeros((mp, r), dtype=np.int64)
        W_int = np.zeros((pn, r), dtype=np.int64)
        
        # If r >= 60, we can use schoolbook directly with 2s
        if r >= 60:
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_int[i*m+j, col] = 2
                        V_int[j*p+k, col] = 2
                        W_int[k*n+i, col] = 2
                        col += 1
            return U_int, V_int, W_int
        else:
            # Random init is better for lower ranks
            return init_random(r)

    # Search for minimal rank
    # Lower bound is not trivial, but let's try from 20 upwards
    # We will try a few random restarts for each rank
    
    best_result = None
    
    # Try ranks from low to high
    # 60 is the schoolbook rank.
    # Let's try 30 to 60.
    
    # To save time, we can start with a few low rank attempts, then go higher.
    ranks_to_try = list(range(20, 61))
    
    # Shuffle a bit to avoid getting stuck if 20-30 are hard
    # But lower is better, so we want the first one found.
    
    for r in ranks_to_try:
        if time.time() > deadline - 10:
            break
            
        # Number of attempts decreases as rank increases?
        # Actually, lower ranks are harder. So more attempts for low ranks.
        attempts = 100 if r < 40 else 50
        
        for attempt in range(attempts):
            if time.time() > deadline - 10:
                break
                
            if attempt == 0:
                U_int, V_int, W_int = init_schoolbook_int(r)
            else:
                U_int, V_int, W_int = init_random(r)
                
            # Run local search
            # Limit steps per attempt to ensure we try multiple attempts
            max_steps_per_attempt = 2000
            if local_search(U_int, V_int, W_int, max_steps_per_attempt):
                # Found a solution
                # Convert to half-integers
                U = (U_int / 2.0).tolist()
                V = (V_int / 2.0).tolist()
                W = (W_int / 2.0).tolist()
                return {"U": U, "V": V, "W": W}

    # Fallback to schoolbook (rank 60)
    r = 60
    U_int, V_int, W_int = init_schoolbook_int(r)
    U = (U_int / 2.0).tolist()
    V = (V_int / 2.0).tolist()
    W = (W_int / 2.0).tolist()
    return {"U": U, "V": V, "W": W}
