"""Tensor decomposition search for 5x5x5 matrix multiplication."""
import time
import numpy as np
import itertools

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 5x5 matrix multiplication.
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    """
    n, m, p = 5, 5, 5
    start_time = time.time()
    deadline = start_time + 125.0  # safety margin before 150s kill

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def round_to_half(x):
        """Round to nearest half-integer."""
        return round(x * 2) / 2.0

    def verify_fast(U, V, W, rank):
        """Verify exact decomposition using numpy. Returns max error."""
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err

    def try_rank(rank, max_restarts=8, max_iters=120):
        """Try to find exact rank decomposition via alternating minimization."""
        if time.time() > deadline - 15:
            return None
        
        for restart in range(max_restarts):
            if time.time() > deadline - 8:
                return None
            
            np.random.seed(rank * 7919 + restart * 104729 + 17)
            
            # Random initialization with small values
            U = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(n*m, rank)).astype(float)
            V = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(m*p, rank)).astype(float)
            W = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(p*n, rank)).astype(float)
            
            best_err = float('inf')
            best_state = (U.copy(), V.copy(), W.copy())
            
            for it in range(max_iters):
                if time.time() > deadline - 3:
                    return None
                
                # Update U: for each (i,j), solve U[i*m+j,:] @ diag(V[j*p+k,:]*W[k*n+i,:]) = ones(p)
                for i in range(n):
                    for j in range(m):
                        # Build system: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k
                        # This is: U[i*m+j,:] @ M = ones(p) where M[k,c] = V[j*p+k,c]*W[k*n+i,c]
                        M = V[j*p:j*p+p, :] * W[0:p*n, :][:p, :]  # shape (p, rank)
                        # Actually: M[k,c] = V[j*p+k, c] * W[k*n+i, c]
                        M = np.zeros((p, rank))
                        for k in range(p):
                            M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        # Solve: U[i*m+j,:] @ M.T = ones(p)
                        # i.e., M @ U[i*m+j,:].T = ones(p)
                        try:
                            U[i*m+j, :] = np.linalg.lstsq(M.T, np.ones(p), rcond=None)[0]
                        except:
                            pass
                
                # Update V: for each (j,k), solve V[j*p+k,:] @ diag(U[i*m+j,:]*W[k*n+i,:]) = ones(n)
                for j in range(m):
                    for k in range(p):
                        M = np.zeros((n, rank))
                        for i in range(n):
                            M[i, :] = U[i*m+j, :] * W[k*n+i, :]
                        try:
                            V[j*p+k, :] = np.linalg.lstsq(M.T, np.ones(n), rcond=None)[0]
                        except:
                            pass
                
                # Update W: for each (k,i), solve W[k*n+i,:] @ diag(U[i*m+j,:]*V[j*p+k,:]) = ones(m)
                for k in range(p):
                    for i in range(n):
                        M = np.zeros((m, rank))
                        for j in range(m):
                            M[j, :] = U[i*m+j, :] * V[j*p+k, :]
                        try:
                            W[k*n+i, :] = np.linalg.lstsq(M.T, np.ones(m), rcond=None)[0]
                        except:
                            pass
                
                # Check convergence every 2 iterations
                if it % 2 == 1 or it == max_iters - 1:
                    max_err = verify_fast(U, V, W, rank)
                    
                    if max_err < best_err:
                        best_err = max_err
                        best_state = (U.copy(), V.copy(), W.copy())
                    
                    if max_err < 1e-10:
                        # Try rounding
                        U_r = np.array([[round_to_half(x) for x in U[i]] for i in range(n*m)])
                        V_r = np.array([[round_to_half(x) for x in V[j]] for j in range(m*p)])
                        W_r = np.array([[round_to_half(x) for x in W[k]] for k in range(p*n)])
                        err_r = verify_fast(U_r, V_r, W_r, rank)
                        if err_r < 1e-10:
                            return {
                                "U": U_r.tolist(),
                                "V": V_r.tolist(),
                                "W": W_r.tolist()
                            }
                    
                    # If close, try local search with rounding
                    if max_err < 1e-5:
                        for attempt in range(30):
                            if time.time() > deadline - 2:
                                break
                            U_test = U.copy()
                            V_test = V.copy()
                            W_test = W.copy()
                            
                            # Randomly perturb a few entries
                            for _ in range(3):
                                mat = np.random.choice([0, 1, 2])
                                if mat == 0:
                                    row = np.random.randint(0, U.shape[0])
                                    col = np.random.randint(0, rank)
                                    U_test[row, col] += np.random.choice([-1, 1]) * 0.5
                                elif mat == 1:
                                    row = np.random.randint(0, V.shape[0])
                                    col = np.random.randint(0, rank)
                                    V_test[row, col] += np.random.choice([-1, 1]) * 0.5
                                else:
                                    row = np.random.randint(0, W.shape[0])
                                    col = np.random.randint(0, rank)
                                    W_test[row, col] += np.random.choice([-1, 1]) * 0.5
                            
                            # Round and verify
                            U_r = np.array([[round_to_half(x) for x in U_test[i]] for i in range(n*m)])
                            V_r = np.array([[round_to_half(x) for x in V_test[j]] for j in range(m*p)])
                            W_r = np.array([[round_to_half(x) for x in W_test[k]] for k in range(p*n)])
                            err_r = verify_fast(U_r, V_r, W_r, rank)
                            if err_r < 1e-10:
                                return {
                                    "U": U_r.tolist(),
                                    "V": V_r.tolist(),
                                    "W": W_r.tolist()
                                }
                    
                    if max_err < 1e-10:
                        break
            
            # Try best state
            if best_err < 1e-8:
                U, V, W = best_state
                U_r = np.array([[round_to_half(x) for x in U[i]] for i in range(n*m)])
                V_r = np.array([[round_to_half(x) for x in V[j]] for j in range(m*p)])
                W_r = np.array([[round_to_half(x) for x in W[k]] for k in range(p*n)])
                err_r = verify_fast(U_r, V_r, W_r, rank)
                if err_r < 1e-10:
                    return {
                        "U": U_r.tolist(),
                        "V": V_r.tolist(),
                        "W": W_r.tolist()
                    }
        
        return None

    # Try ranks in strategic order
    rank_order = [49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 60, 62, 65, 70, 80, 100, 125]
    
    for rank in rank_order:
        if time.time() > deadline - 20:
            break
        if rank >= 125:
            return schoolbook()
        
        result = try_rank(rank, max_restarts=6, max_iters=100)
        if result is not None:
            return result
    
    return schoolbook()
