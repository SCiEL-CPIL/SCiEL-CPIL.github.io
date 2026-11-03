import time
import numpy as np
from scipy.optimize import minimize, LinearConstraint

DIMENSIONS = [3, 3, 8]
DEADLINE_S = 110  # safety margin from 120s limit

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    def time_left():
        return DEADLINE_S - (time.time() - start_time)
    
    def schoolbook():
        rank = n*m*p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def validate(U, V, W, tol=1e-10):
        """Check that sum_c U[:,c] V[:,c] W[:,c] == multiplication tensor."""
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        rank = U.shape[1]
        if V.shape[1] != rank or W.shape[1] != rank:
            return False
        # Compute tensor: T[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
        # Build expected tensor
        T_expected = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_expected[i,j,k] = 1.0
        
        # Build T from factors
        T_computed = np.zeros((n, m, p))
        for c in range(rank):
            u_c = U[:, c]
            v_c = V[:, c]
            w_c = W[:, c]
            # Reshape to (n,m), (m,p), (p,n)
            u_mat = u_c.reshape(n, m)
            v_mat = v_c.reshape(m, p)
            w_mat = w_c.reshape(p, n)
            # Outer contribution: T[i,j,k] += u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
            for i in range(n):
                for j in range(m):
                    if abs(u_mat[i,j]) < 1e-15:
                        continue
                    for k in range(p):
                        T_computed[i,j,k] += u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
        
        return np.max(np.abs(T_computed - T_expected)) < tol
    
    # Start with schoolbook as fallback
    best_result = schoolbook()
    best_rank = n*m*p
    
    # Try to find lower rank via random search + optimization
    # The tensor for matmul (n,m,p) has structure: T[i,j,k] = 1 for all (i,j,k)
    # A rank-r decomposition gives matrices U (nm x r), V (mp x r), W (pn x r)
    # such that for all (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Strategy: random initialization + gradient-free local search
    # Since n=3, m=3, p=8, the tensor is 3x3x8 = 72 entries, all 1.
    
    target_ranks = [60, 58, 56, 54, 52, 50, 48, 46, 44, 42, 40, 38, 36, 34, 32, 30, 28, 26, 24]
    
    rng = np.random.default_rng(42)
    
    for target_rank in target_ranks:
        if time_left() < 20:
            break
        if target_rank >= best_rank:
            continue
            
        # Try multiple random restarts
        max_restarts = 30
        for attempt in range(max_restarts):
            if time_left() < 15:
                    break
                
            # Random initialization
            U = rng.standard_normal((nm, target_rank))
            V = rng.standard_normal((mp, target_rank))
            W = rng.standard_normal((pn, target_rank))
            
            # Scale to make the sum of products closer to 1
            # For each (i,j,k), we need sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1
            # A rough scale: each product should be ~1/target_rank
            scale = (1.0 / target_rank) ** (1.0/3.0)
            U *= scale
            V *= scale
            W *= scale
            
            # Local search: perturb entries to reduce error
            def compute_error(U, V, W):
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for c in range(target_rank):
                                s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            err += (s - 1.0)**2
                return err
            
            current_err = compute_error(U, V, W)
            if current_err < 1e-8:
                if validate(U.tolist(), V.tolist(), W.tolist()):
                    best_rank = target_rank
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    return best_result
                break
            
            # Simple coordinate descent / local search
            lr = 0.01
            for step in range(200):
                if time_left() < 12:
                    break
                if current_err < 1e-10:
                    break
                    
                # Randomly pick a variable to update
                idx = rng.integers(0, 3)
                if idx == 0:
                    r, c = rng.integers(0, nm), rng.integers(0, target_rank)
                    old = U[r, c]
                    U[r, c] = old + lr * rng.standard_normal()
                    new_err = compute_error(U, V, W)
                    if new_err < current_err:
                        current_err = new_err
                    else:
                        U[r, c] = old
                elif idx == 1:
                    r, c = rng.integers(0, mp), rng.integers(0, target_rank)
                    old = V[r, c]
                    V[r, c] = old + lr * rng.standard_normal()
                    new_err = compute_error(U, V, W)
                    if new_err < current_err:
                        current_err = new_err
                    else:
                        V[r, c] = old
                else:
                    r, c = rng.integers(0, pn), rng.integers(0, target_rank)
                    old = W[r, c]
                    W[r, c] = old + lr * rng.standard_normal()
                    new_err = compute_error(U, V, W)
                    if new_err < current_err:
                        current_err = new_err
                    else:
                        W[r, c] = old
            
            if current_err < 1e-8:
                if validate(U.tolist(), V.tolist(), W.tolist()):
                    best_rank = target_rank
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    return best_result
    
    return best_result
