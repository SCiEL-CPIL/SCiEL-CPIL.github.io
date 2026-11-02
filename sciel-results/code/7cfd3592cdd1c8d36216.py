import numpy as np
import time

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    
    # Helper to get indices
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    triplets = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    def get_residuals(U, V, W):
        res = {}
        for (i, j, k) in triplets:
            val = np.sum(U[u_idx(i, j)] * V[v_idx(j, k)] * W[w_idx(k, i)])
            res[(i, j, k)] = val - 1.0
        return res

    def total_error(U, V, W):
        res = get_residuals(U, V, W)
        return np.sum([r*r for r in res.values()])

    def schoolbook(rank=None):
        if rank is None:
            rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if col >= rank:
                        break
                    U[u_idx(i, j), col] = 1.0
                    V[v_idx(j, k), col] = 1.0
                    W[w_idx(k, i), col] = 1.0
                    col += 1
                if col >= rank:
                    break
            if col >= rank:
                break
        return U, V, W

    def alternating_linear_solve(U, V, W, max_iters=200, tol=1e-10):
        """
        Alternating linear solves: fix two factors, solve for the third.
        """
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        for iteration in range(max_iters):
            prev_err = total_error(U, V, W)
            
            # Solve for U: fix V, W
            for i in range(n):
                for j in range(m):
                    ui = u_idx(i, j)
                    # Equations: for each k, sum_r U[ui,r] * V[v_idx(j,k),r] * W[w_idx(k,i),r] = 1
                    # Build A matrix: A[k,r] = V[v_idx(j,k),r] * W[w_idx(k,i),r]
                    A = np.zeros((p, W.shape[1]))
                    b = np.ones(p)
                    for k in range(p):
                        vi = v_idx(j, k)
                        wi = w_idx(k, i)
                        A[k, :] = V[vi] * W[wi]
                    # Solve A @ u_row = b (least squares)
                    u_row, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    U[ui, :] = u_row
            
            # Solve for V: fix U, W
            for j in range(m):
                for k in range(p):
                    vi = v_idx(j, k)
                    # Equations: for each i, sum_r V[vi,r] * U[u_idx(i,j),r] * W[w_idx(k,i),r] = 1
                    A = np.zeros((n, U.shape[1]))
                    b = np.ones(n)
                    for i in range(n):
                        ui = u_idx(i, j)
                        wi = w_idx(k, i)
                        A[i, :] = U[ui] * W[wi]
                    v_row, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    V[vi, :] = v_row
            
            # Solve for W: fix U, V
            for k in range(p):
                for i in range(n):
                    wi = w_idx(k, i)
                    # Equations: for each j, sum_r W[wi,r] * U[u_idx(i,j),r] * V[v_idx(j,k),r] = 1
                    A = np.zeros((m, V.shape[1]))
                    b = np.ones(m)
                    for j in range(m):
                        ui = u_idx(i, j)
                        vi = v_idx(j, k)
                        A[j, :] = U[ui] * V[vi]
                    w_row, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    W[wi, :] = w_row
            
            curr_err = total_error(U, V, W)
            if curr_err < tol:
                break
            if abs(prev_err - curr_err) < 1e-12:
                break
        
        return U, V, W

    def randomized_algebraic_search(max_restarts=100):
        best_rank = n * m * p
        best_U, best_V, best_W = schoolbook()
        best_err = float('inf')
        
        start_time = time.time()
        deadline = 130  # 20s margin before 150s limit
        
        # Try ranks from 40 to 48
        for r in range(40, 49):
            if time.time() > deadline:
                break
            
            for restart in range(max_restarts):
                if time.time() > deadline:
                    break
                
                # Initialize with random factors
                U_init = np.random.randn(n*m, r)
                V_init = np.random.randn(m*p, r)
                W_init = np.random.randn(p*n, r)
                
                # Run alternating linear solve
                U_opt, V_opt, W_opt = alternating_linear_solve(U_init, V_init, W_init, max_iters=100)
                
                err = total_error(U_opt, V_opt, W_opt)
                
                if err < 1e-8:
                    # Success!
                    return U_opt, V_opt, W_opt, r
                
                # Keep the best found so far (lowest error)
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U_opt, V_opt, W_opt
                    best_rank = r
        
        return best_U, best_V, best_W, best_rank

    U, V, W, rank = randomized_algebraic_search()
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
