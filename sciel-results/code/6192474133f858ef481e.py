<<<<<<< SEARCH
import numpy as np
import time
import random

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 12
    mp = m * p  # 20
    pn = p * n  # 15
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank
    
    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try to find lower rank via random search
    # We'll try ranks from 55 down to 50
    np.random.seed(42)
    random.seed(42)
    
    def verify(U, V, W, tol=1e-9):
        """Check if T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] for all i,j,k"""
        n, m, p = DIMENSIONS
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(U.shape[1]):
                        s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def try_rank(r):
        """Try to find a rank-r decomposition"""
        start_time = time.time()
        deadline = start_time + 25  # 25 seconds per rank attempt
        
        for attempt in range(30):
            if time.time() > deadline:
                break
            
            # Random initialization
            U = np.random.randn(nm, r)
            V = np.random.randn(mp, r)
            W = np.random.randn(pn, r)
            
            # Normalize columns to help convergence
            for c in range(r):
                norm = np.sqrt(np.sum(U[:, c]**2) + np.sum(V[:, c]**2) + np.sum(W[:, c]**2))
                if norm > 1e-10:
                    U[:, c] /= norm
                    V[:, c] /= norm
                    W[:, c] /= norm
            
            # Local search: randomly perturb and accept if error decreases
            def compute_error(U, V, W):
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for c in range(r):
                                s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                            err += (s - 1.0) ** 2
                return err
            
            current_err = compute_error(U, V, W)
            
            # Simulated annealing style
            temperature = 1.0
            for step in range(500):
                if time.time() > deadline:
                    break
                
                # Pick a random element to perturb
                mat = random.choice(['U', 'V', 'W'])
                row = random.randrange(mat[0] if False else (nm if mat=='U' else mp if mat=='V' else pn))
                col = random.randrange(r)
                
                delta = random.gauss(0, 0.1 * temperature)
                
                old_val = 0
                if mat == 'U':
                    old_val = U[row, col]
                    U[row, col] += delta
                elif mat == 'V':
                    old_val = V[row, col]
                    V[row, col] += delta
                else:
                    old_val = W[row, col]
                    W[row, col] += delta
                
                new_err = compute_error(U, V, W)
                
                if new_err < current_err or random.random() < np.exp(-(new_err - current_err) / max(temperature, 1e-6)):
                    current_err = new_err
                    temperature = max(temperature * 0.995, 0.01)
                else:
                    # Revert
                    if mat == 'U':
                        U[row, col] = old_val
                    elif mat == 'V':
                        V[row, col] = old_val
                    else:
                        W[row, col] = old_val
            
            # Check if we got an exact solution
            if current_err < 1e-6:
                # Try to clean up: snap to rational-like values
                if verify(U, V, W, tol=1e-8):
                    return U, V, W, r
                # Try with tighter tolerance
                if verify(U, V, W, tol=1e-10):
                    return U, V, W, r
        
        return None
    
    # Try ranks from 55 down to 50
    for r in range(55, 49, -1):
        result = try_rank(r)
        if result is not None:
            U, V, W, rk = result
            best_rank = rk
            best_U, best_V, best_W = U, V, W
            break
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
=======
import numpy as np
import time
import random

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 12
    mp = m * p  # 20
    pn = p * n  # 15
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank
    
    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try to find lower rank via random search
    # We'll try ranks from 55 down to 50
    np.random.seed(42)
    random.seed(42)
    
    def verify(U, V, W, tol=1e-9):
        """Check if T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] for all i,j,k"""
        n, m, p = DIMENSIONS
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(U.shape[1]):
                        s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def try_rank(r, time_budget):
        """Try to find a rank-r decomposition"""
        start_time = time.time()
        deadline = start_time + time_budget
        
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Random initialization
            U = np.random.randn(nm, r)
            V = np.random.randn(mp, r)
            W = np.random.randn(pn, r)
            
            # Normalize columns to help convergence
            for c in range(r):
                norm = np.sqrt(np.sum(U[:, c]**2) + np.sum(V[:, c]**2) + np.sum(W[:, c]**2))
                if norm > 1e-10:
                    U[:, c] /= norm
                    V[:, c] /= norm
                    W[:, c] /= norm
            
            # Local search: randomly perturb and accept if error decreases
            def compute_error(U, V, W):
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for c in range(r):
                                s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                            err += (s - 1.0) ** 2
                return err
            
            current_err = compute_error(U, V, W)
            
            # Simulated annealing style
            temperature = 1.0
            for step in range(800):
                if time.time() > deadline:
                    break
                
                # Pick a random element to perturb
                mat = random.choice(['U', 'V', 'W'])
                row = random.randrange(nm if mat=='U' else mp if mat=='V' else pn)
                col = random.randrange(r)
                
                delta = random.gauss(0, 0.1 * temperature)
                
                old_val = 0
                if mat == 'U':
                    old_val = U[row, col]
                    U[row, col] += delta
                elif mat == 'V':
                    old_val = V[row, col]
                    V[row, col] += delta
                else:
                    old_val = W[row, col]
                    W[row, col] += delta
                
                new_err = compute_error(U, V, W)
                
                if new_err < current_err or random.random() < np.exp(-(new_err - current_err) / max(temperature, 1e-6)):
                    current_err = new_err
                    temperature = max(temperature * 0.995, 0.01)
                else:
                    # Revert
                    if mat == 'U':
                        U[row, col] = old_val
                    elif mat == 'V':
                        V[row, col] = old_val
                    else:
                        W[row, col] = old_val
            
            # Check if we got an exact solution
            if current_err < 1e-6:
                # Try to clean up: snap to rational-like values
                if verify(U, V, W, tol=1e-8):
                    return U, V, W, r
                # Try with tighter tolerance
                if verify(U, V, W, tol=1e-10):
                    return U, V, W, r
        
        return None
    
    # Global deadline guard: 150s wall-clock limit, keep 15s safety margin
    global_start = time.time()
    global_deadline = global_start + 135  # 15s safety margin
    
    # Try ranks from 55 down to 50
    for r in range(55, 49, -1):
        if time.time() > global_deadline:
            break
        time_budget = min(25, max(5, global_deadline - time.time()))
        if time_budget < 5:
            break
        result = try_rank(r, time_budget)
        if result is not None:
            U, V, W, rk = result
            best_rank = rk
            best_U, best_V, best_W = U, V, W
            break
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
>>>>>>> REPLACE
