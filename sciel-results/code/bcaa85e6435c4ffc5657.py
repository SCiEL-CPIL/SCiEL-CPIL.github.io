import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 2, 9]

def construct():
    n, m, p = DIMENSIONS
    t_start = time.time()
    deadline = t_start + 130.0  # Safety margin before 150s limit
    
    # Target tensor: T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We want to find rank R such that T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Note: The problem statement says U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # And the decomposition is sum_r U[:,r] V[:,r] W[:,r] = T
    # Specifically, T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Indices for the tensor T
    # i: 0..n-1
    # j: 0..m-1
    # k: 0..p-1
    # Index in U: i*m + j
    # Index in V: j*p + k
    # Index in W: k*n + i
    
    def residuals(params, R):
        # params: flattened array of length R * (n*m + m*p + p*n)
        # Split into U, V, W
        size_u = n * m
        size_v = m * p
        size_w = p * n
        total_size = R * (size_u + size_v + size_w)
        
        U_flat = params[:R * size_u].reshape(R, size_u)
        V_flat = params[R * size_u : R * (size_u + size_v)].reshape(R, size_v)
        W_flat = params[R * (size_u + size_v) :].reshape(R, size_w)
        
        # U is (n*m, R), so U_flat is (R, n*m), we need to transpose or index carefully
        # Let's keep U as (R, n*m) for now, so U[r, i*m+j]
        # V as (R, m*p), so V[r, j*p+k]
        # W as (R, p*n), so W[r, k*n+i]
        
        res = np.zeros(n * m * p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    ui = i * m + j
                    vj = j * p + k
                    wk = k * n + i
                    val = np.sum(U_flat[:, ui] * V_flat[:, vj] * W_flat[:, wk])
                    res[idx] = val - 1.0
                    idx += 1
        return res

    best_rank = n * m * p
    best_params = None
    
    # Try ranks from lower bound up
    # Lower bound: max(n*m, m*p, p*n) is not quite right, but rank >= max(n*m, m*p, p*n) is a loose bound?
    # Actually, rank >= max(n*m, m*p, p*n) is not a standard bound.
    # Rank >= max(n, m, p) is trivial.
    # For 2x2x2, rank is 7.
    # For 2x2x9, let's start trying from 10 up to 36.
    
    # We need to be careful with time.
    # Let's try a few ranks.
    # If R is too small, it won't converge.
    
    # Heuristic: The schoolbook rank is 36.
    # Strassen-like blocks might give ~30.
    # Let's try 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36.
    
    # To save time, we can start with a rank we suspect is close, e.g., 20-30.
    # Or we can do a quick check for low ranks.
    
    # Let's try a range.
    # Note: least_squares can be slow if R is large.
    # Number of parameters: R * (4 + 18 + 18) = 40R.
    # Number of residuals: 36.
    # If R > 36, we are over-parameterized, which is fine but might be slower to converge or have flat valleys.
    # We want the minimal rank, so we should start low.
    
    # However, solving for rank 10 with 400 params and 36 residuals is an underdetermined system,
    # but least_squares finds the minimum norm solution (roughly).
    # Actually, if R is small, it might not fit exactly. We need residual < 1e-8.
    
    # Let's iterate R from 10 to 36.
    # But 10 to 36 is 27 attempts. Each attempt might take seconds.
    # 27 * 5s = 135s. Too risky.
    
    # Let's be smarter.
    # We know rank 36 works.
    # We can try to jump to 20, 15, 10.
    
    # Strategy:
    # 1. Try R=10. If fails, R=12, R=14...
    # 2. Use a time limit per attempt.
    
    # Actually, for small R, the optimization might get stuck in local minima or fail to converge to 0.
    # For R >= 36, it should converge quickly.
    
    # Let's try a binary search or step search?
    # Since we want minimal rank, linear search from low to high is correct.
    
    # Let's limit the number of attempts to fit in time.
    # Max attempts: 10.
    # Ranks to try: 10, 12, 14, 16, 18, 20, 22, 24, 26, 28.
    # If none work, fallback to 36.
    
    # But what if the rank is 30? We'd miss it.
    # Let's try: 10, 15, 20, 25, 30, 35, 36.
    
    ranks_to_try = [10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36]
    
    # We will iterate through these.
    # To speed up, we can use a good initial guess.
    # Random initialization is standard.
    
    # We can also scale the parameters.
    
    for R in ranks_to_try:
        if time.time() > deadline - 10:
            break
            
        size_u = n * m
        size_v = m * p
        size_w = p * n
        n_params = R * (size_u + size_v + size_w)
        
        # Random initialization
        np.random.seed(42 + R) # Reproducibility
        x0 = np.random.randn(n_params) * 0.1
        
        # We need to handle the case where R is small.
        # If R is too small, it won't converge.
        
        try:
            # Use least_squares
            # x_scale='jac' or 'jac-1' can help
            result = least_squares(
                residuals, 
                x0, 
                args=(R,), 
                method='lm', # Levenberg-Marquardt, good for this
                max_nfev=1000,
                x_scale='jac-1'
            )
            
            if result.success or np.min(result.cost) < 1e-12:
                # Check residual
                res = residuals(result.x, R)
                if np.max(np.abs(res)) < 1e-8:
                    best_rank = R
                    best_params = result.x
                    break
        except Exception as e:
            pass
    
    # If we didn't find a rank < 36, use 36 (schoolbook)
    if best_params is None:
        best_rank = n * m * p
        # Construct schoolbook
        U = np.zeros((n * m, best_rank))
        V = np.zeros((m * p, best_rank))
        W = np.zeros((p * n, best_rank))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Extract U, V, W from best_params
    size_u = n * m
    size_v = m * p
    size_w = p * n
    
    U_flat = best_params[:best_rank * size_u].reshape(best_rank, size_u)
    V_flat = best_params[best_rank * size_u : best_rank * (size_u + size_v)].reshape(best_rank, size_v)
    W_flat = best_params[best_rank * (size_u + size_v) :].reshape(best_rank, size_w)
    
    # The problem requires U to be (n*m, rank), V to be (m*p, rank), W to be (p*n, rank)
    # My U_flat is (rank, n*m). So I need to transpose.
    U = U_flat.T
    V = V_flat.T
    W = W_flat.T
    
    # Verify
    # T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # Let's do a quick check
    err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                err = max(err, abs(val - 1.0))
    
    if err > 1e-5:
        # Fallback to schoolbook if verification fails
        best_rank = n * m * p
        U = np.zeros((n * m, best_rank))
        V = np.zeros((m * p, best_rank))
        W = np.zeros((p * n, best_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
