import numpy as np
import time
from scipy.optimize import nnls

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import nnls

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Precompute indices for vectorized operations
    # We need to map (i, j, k) to indices in U, V, W
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    # Total number of tensor entries
    num_entries = n * m * p
    
    # Index arrays for U, V, W corresponding to each (i, j, k) entry
    # Shape: (num_entries,)
    idx_u = np.array([i * m + j for i in range(n) for j in range(m) for k in range(p)])
    idx_v = np.array([j * p + k for i in range(n) for j in range(m) for k in range(p)])
    idx_w = np.array([k * n + i for i in range(n) for j in range(m) for k in range(p)])
    
    def verify_solution(U, V, W, tol=1e-9):
        """
        Verify if U, V, W form a valid exact decomposition.
        U: (nm, R), V: (mp, R), W: (pn, R)
        """
        R = U.shape[1]
        # Vectorized check:
        # res[t] = sum_r U[idx_u[t], r] * V[idx_v[t], r] * W[idx_w[t], r]
        U_sel = U[idx_u, :] # (num_entries, R)
        V_sel = V[idx_v, :] # (num_entries, R)
        W_sel = W[idx_w, :] # (num_entries, R)
        
        res = np.sum(U_sel * V_sel * W_sel, axis=1)
        
        return np.allclose(res, np.ones(num_entries), atol=tol, rtol=0)

    def get_schoolbook():
        """
        Return the standard schoolbook decomposition (Rank = n*m*p).
        Entries are 0 or 1.
        """
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

    def try_rank_nnls(R, time_limit=2.0, seed=0):
        """
        Attempt to find a rank R decomposition using alternating NNLS.
        Returns (U, V, W) if found and valid, else None.
        """
        if R >= n * m * p:
            return None # Schoolbook is better handled separately
        
        rng = np.random.RandomState(seed)
        
        # Initialize with small random values
        U = rng.uniform(0.1, 1.0, (nm, R))
        V = rng.uniform(0.1, 1.0, (mp, R))
        W = rng.uniform(0.1, 1.0, (pn, R))
        
        best_err = float('inf')
        best_solution = None
        
        start_time = time.time()
        max_iters = 5000
        
        for it in range(max_iters):
            if time.time() - start_time > time_limit:
                break
                
            # Update U
            # For each row i*m+j of U, we solve A * u_row = b
            # A is (p, R), b is (p,)
            # A[k, c] = V[j*p+k, c] * W[k*n+i, c]
            # b[k] = 1
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, R))
                    for k in range(p):
                        A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    b = np.ones(p)
                    try:
                        u_sol, _ = nnls(A, b)
                        U[i*m+j, :] = u_sol
                    except:
                        pass
            
            # Update V
            # For each row j*p+k of V, solve A * v_row = b
            # A is (n, R), b is (n,)
            # A[i, c] = U[i*m+j, c] * W[k*n+i, c]
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, R))
                    for i in range(n):
                        A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    b = np.ones(n)
                    try:
                        v_sol, _ = nnls(A, b)
                        V[j*p+k, :] = v_sol
                    except:
                        pass
                        
            # Update W
            # For each row k*n+i of W, solve A * w_row = b
            # A is (m, R), b is (m,)
            # A[j, c] = U[i*m+j, c] * V[j*p+k, c]
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, R))
                    for j in range(m):
                        A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    b = np.ones(m)
                    try:
                        w_sol, _ = nnls(A, b)
                        W[k*n+i, :] = w_sol
                    except:
                        pass

            # Check error
            U_sel = U[idx_u, :]
            V_sel = V[idx_v, :]
            W_sel = W[idx_w, :]
            res = np.sum(U_sel * V_sel * W_sel, axis=1)
            err = np.sum((res - 1.0)**2)
            
            if err < best_err:
                best_err = err
                best_solution = (U.copy(), V.copy(), W.copy())
                
            if err < 1e-12:
                # Found a float solution. Try to round to half-integers.
                # The problem requires integer/half-integer entries.
                # We try rounding to nearest 0.5
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                if verify_solution(U_r, V_r, W_r):
                    return U_r, V_r, W_r
                
                # Try integer rounding
                U_i = np.round(U)
                V_i = np.round(V)
                W_i = np.round(W)
                if verify_solution(U_i, V_i, W_i):
                    return U_i, V_i, W_i

        # If we exited loop without finding exact, check best solution
        if best_solution is not None and best_err < 1e-10:
            U, V, W = best_solution
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            if verify_solution(U_r, V_r, W_r):
                return U_r, V_r, W_r
                
        return None

    # Main Logic
    # 1. Get the guaranteed schoolbook solution (Rank 30)
    U_sb, V_sb, W_sb = get_schoolbook()
    best_rank = n * m * p
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # 2. Try to find lower ranks
    # Theoretical lower bound is max(nm, mp, pn) = 15
    # We try a few ranks below 30.
    # Given time constraints, we prioritize ranks close to 30 as they are more likely to exist if any do.
    
    start_time = time.time()
    timeout = 110.0 # Leave 10s for safety
    
    # Try ranks 29, 28, 27... down to lower_bound
    for R in range(best_rank - 1, 14, -1):
        if time.time() - start_time > timeout:
            break
            
        # Use different seeds for diversity
        for seed in [42, 123, 999]:
            if time.time() - start_time > timeout:
                break
            res = try_rank_nnls(R, time_limit=1.5, seed=seed)
            if res is not None:
                U, V, W = res
                best_rank = R
                best_U, best_V, best_W = U, V, W
                break # Found a solution for this rank, move to lower rank
        
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
