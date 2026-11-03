import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    total_entries = n * m * p
    
    # Precompute index arrays for fast verification
    idx_u = np.array([i * m + j for i in range(n) for j in range(m) for k in range(p)])
    idx_v = np.array([j * p + k for i in range(n) for j in range(m) for k in range(p)])
    idx_w = np.array([k * n + i for i in range(n) for j in range(m) for k in range(p)])
    
    def verify(U, V, W, tol=1e-9):
        """Verify that the decomposition is exact"""
        U_sel = U[idx_u, :]
        V_sel = V[idx_v, :]
        W_sel = W[idx_w, :]
        res = np.sum(U_sel * V_sel * W_sel, axis=1)
        return np.allclose(res, np.ones(total_entries), atol=tol, rtol=0)
    
    def max_error(U, V, W):
        """Return the maximum absolute error"""
        U_sel = U[idx_u, :]
        V_sel = V[idx_v, :]
        W_sel = W[idx_w, :]
        res = np.sum(U_sel * V_sel * W_sel, axis=1)
        return np.max(np.abs(res - 1.0))
    
    def is_half_integer(mat, tol=1e-9):
        """Check if all entries are integers or half-integers"""
        doubled = mat * 2
        return np.allclose(doubled, np.round(doubled), atol=tol)
    
    def get_schoolbook():
        """Return the trivial rank-30 decomposition"""
        R = n * m * p
        U = np.zeros((nm, R))
        V = np.zeros((mp, R))
        W = np.zeros((pn, R))
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, r] = 1.0
                    V[j*p+k, r] = 1.0
                    W[k*n+i, r] = 1.0
                    r += 1
        return U, V, W
    
    def try_find_rank(R, time_limit=5.0, seed=0):
        """
        Attempt to find a rank R decomposition using alternating least squares
        with multiple initialization strategies.
        """
        if R >= n * m * p:
            return None
        
        rng = np.random.RandomState(seed)
        start_time = time.time()
        max_restarts = 20
        
        for restart in range(max_restarts):
            if time.time() - start_time > time_limit:
                break
            
            # Choose initialization strategy
            if restart % 4 == 0:
                # Random initialization
                U = rng.uniform(-1, 1, (nm, R))
                V = rng.uniform(-1, 1, (mp, R))
                W = rng.uniform(-1, 1, (pn, R))
            elif restart % 4 == 1:
                # Sparse random with ±1 entries
                U = np.zeros((nm, R))
                V = np.zeros((mp, R))
                W = np.zeros((pn, R))
                for r in range(R):
                    U[rng.randint(0, nm), r] = rng.choice([-1.0, 1.0])
                    V[rng.randint(0, mp), r] = rng.choice([-1.0, 1.0])
                    W[rng.randint(0, pn), r] = rng.choice([-1.0, 1.0])
            elif restart % 4 == 2:
                # Structured initialization based on Strassen-like patterns
                U = np.zeros((nm, R))
                V = np.zeros((mp, R))
                W = np.zeros((pn, R))
                
                # Try to create a structured pattern
                for r in range(R):
                    # Each rank term touches a subset of indices
                    i = r % n
                    j = (r // n) % m
                    k = (r // (n*m)) % p
                    
                    # Add some structure
                    for di in range(n):
                        for dj in range(m):
                            for dk in range(p):
                                if (di + dj + dk) % 3 == r % 3:
                                    U[(i+di)%n * m + (j+dj)%m, r] += rng.choice([-1.0, 1.0])
                                    V[(j+dj)%m * p + (k+dk)%p, r] += rng.choice([-1.0, 1.0])
                                    W[(k+dk)%p * n + (i+di)%n, r] += rng.choice([-1.0, 1.0])
            else:
                # Block-structured initialization
                U = np.zeros((nm, R))
                V = np.zeros((mp, R))
                W = np.zeros((pn, R))
                
                # Divide ranks into blocks
                block_size = max(1, R // 4)
                for block in range(4):
                    r_start = block * block_size
                    r_end = min(R, r_start + block_size)
                    
                    for r in range(r_start, r_end):
                        # Each block focuses on a subset of the tensor
                        i_range = [i for i in range(n) if i % 2 == block % 2]
                        j_range = [j for j in range(m) if j % 2 == block % 2] or [j for j in range(m)]
                        k_range = [k for k in range(p) if k % 2 == block % 2] or [k for k in range(p)]
                        
                        for i in i_range:
                            for j in j_range:
                                for k in k_range:
                                    if rng.random() < 0.5:
                                        U[i*m+j, r] = rng.choice([-1.0, 1.0])
                                        V[j*p+k, r] = rng.choice([-1.0, 1.0])
                                        W[k*n+i, r] = rng.choice([-1.0, 1.0])
            
            # Alternating Least Squares
            best_err = float('inf')
            for it in range(100):
                if time.time() - start_time > time_limit * 0.7:
                    break
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        V_sub = V[j*p:(j+1)*p, :]
                        W_rows = W[np.arange(p)*n + i, :]
                        A = V_sub * W_rows
                        b = np.ones(p)
                        try:
                            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
                            U[i*m+j, :] = sol
                        except:
                            pass
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        U_rows = U[np.arange(n)*m + j, :]
                        W_rows = W[k*n:(k+1)*n, :]
                        A = U_rows * W_rows
                        b = np.ones(n)
                        try:
                            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
                            V[j*p+k, :] = sol
                        except:
                            pass
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        U_rows = U[i*m:(i+1)*m, :]
                        V_rows = V[np.arange(m)*p + k, :]
                        A = U_rows * V_rows
                        b = np.ones(m)
                        try:
                            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
                            W[k*n+i, :] = sol
                        except:
                            pass
                
                # Check error periodically
                if it % 5 == 0 or it == 99:
                    err = max_error(U, V, W)
                    if err < best_err:
                        best_err = err
                        if err < 1e-12:
                            break
            
            if best_err > 1e-6:
                continue
            
            # Try rounding to half-integers
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            err_r = max_error(U_r, V_r, W_r)
            if err_r < 1e-9:
                return U_r, V_r, W_r
            
            # Local search on half-integers
            if err_r < 1.0:
                current_U, current_V, current_W = U_r.copy(), V_r.copy(), W_r.copy()
                current_err = err_r
                
                for local_iter in range(50):
                    if time.time() - start_time > time_limit:
                        break
                    
                    found_improvement = False
                    
                    # Try flipping entries in U
                    for r_idx in range(nm):
                        for c_idx in range(R):
                            orig = current_U[r_idx, c_idx]
                            for delta in [-0.5, 0.5]:
                                new_val = orig + delta
                                if abs(new_val * 2 - round(new_val * 2)) > 1e-9:
                                    continue
                                
                                current_U[r_idx, c_idx] = new_val
                                new_err = max_error(current_U, current_V, current_W)
                                if new_err < current_err - 1e-12:
                                    current_err = new_err
                                    found_improvement = True
                                    if new_err < 1e-9:
                                        return current_U, current_V, current_W
                                else:
                                    current_U[r_idx, c_idx] = orig
                    
                    # Try flipping entries in V
                    for r_idx in range(mp):
                        for c_idx in range(R):
                            orig = current_V[r_idx, c_idx]
                            for delta in [-0.5, 0.5]:
                                new_val = orig + delta
                                if abs(new_val * 2 - round(new_val * 2)) > 1e-9:
                                    continue
                                
                                current_V[r_idx, c_idx] = new_val
                                new_err = max_error(current_U, current_V, current_W)
                                if new_err < current_err - 1e-12:
                                    current_err = new_err
                                    found_improvement = True
                                    if new_err < 1e-9:
                                        return current_U, current_V, current_W
                                else:
                                    current_V[r_idx, c_idx] = orig
                    
                    # Try flipping entries in W
                    for r_idx in range(pn):
                        for c_idx in range(R):
                            orig = current_W[r_idx, c_idx]
                            for delta in [-0.5, 0.5]:
                                new_val = orig + delta
                                if abs(new_val * 2 - round(new_val * 2)) > 1e-9:
                                    continue
                                
                                current_W[r_idx, c_idx] = new_val
                                new_err = max_error(current_U, current_V, current_W)
                                if new_err < current_err - 1e-12:
                                    current_err = new_err
                                    found_improvement = True
                                    if new_err < 1e-9:
                                        return current_U, current_V, current_W
                                else:
                                    current_W[r_idx, c_idx] = orig
                    
                    if not found_improvement:
                        break
                
                if max_error(current_U, current_V, current_W) < 1e-9:
                    return current_U, current_V, current_W
        
        return None
    
    # Main Logic
    U_sb, V_sb, W_sb = get_schoolbook()
    best_rank = n * m * p
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    start_time = time.time()
    timeout = 130.0  # Safety margin before 150s kill
    
    # Try ranks from 29 down to 25
    for R in range(29, 24, -1):
        if time.time() - start_time > timeout:
            break
        
        found = False
        num_seeds = 40
        time_per_seed = min(2.5, (timeout - (time.time() - start_time)) / num_seeds)
        
        for seed in range(num_seeds):
            if time.time() - start_time > timeout:
                break
            res = try_find_rank(R, time_limit=time_per_seed, seed=seed)
            if res is not None:
                U, V, W = res
                best_rank = R
                best_U, best_V, best_W = U, V, W
                found = True
                break
        
        if found:
            continue
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
