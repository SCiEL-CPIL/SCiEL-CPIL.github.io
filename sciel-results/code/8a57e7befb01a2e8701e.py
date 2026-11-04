import time
import numpy as np

DIMENSIONS = [3, 4, 6]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that the decomposition is valid."""
    rank = len(U[0]) if U else 0
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Check the 1s
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
                if abs(val - 1.0) > tol:
                    return False
                    
    # Check the 0s
    support = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                support.add((u, v, w))
                
    for u in range(n*m):
        for v in range(m*p):
            for w in range(p*n):
                if (u, v, w) in support:
                    continue
                val = sum(U[u, c] * V[v, c] * W[w, c] for c in range(rank))
                if abs(val) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def _als_decompose(n, m, p, rank, seed, max_iter=100, tol=1e-8):
    """
    Attempt to find a decomposition of given rank using Alternating Least Squares.
    """
    rng = np.random.RandomState(seed)
    
    # Initialize V and W randomly
    V = rng.randn(m * p, rank) * 0.1
    W = rng.randn(p * n, rank) * 0.1
    
    # U will be computed
    U = np.zeros((n * m, rank))
    
    best_err = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        # 1. Solve for U
        # For each i, j, solve min_u sum_k (sum_r u_r V[j,k,r] W[k,i,r] - 1)^2
        # A[i,j] is p x R
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, rank))
                for k in range(p):
                    for r in range(rank):
                        A[k, r] = V[j*p + k, r] * W[k*n + i, r]
                b = np.ones(p)
                # Solve A u = b in least squares sense
                # u is (R, 1)
                try:
                    u, residuals, rank_s, sv = np.linalg.lstsq(A, b, rcond=None)
                    U[i*m + j, :] = u
                except:
                    U[i*m + j, :] = 0
        
        # 2. Solve for V
        # For each j, k, solve min_v sum_i (sum_r v_r U[i,j,r] W[k,i,r] - 1)^2
        # A[j,k] is n x R
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, rank))
                for i in range(n):
                    for r in range(rank):
                        A[i, r] = U[i*m + j, r] * W[k*n + i, r]
                b = np.ones(n)
                try:
                    v, residuals, rank_s, sv = np.linalg.lstsq(A, b, rcond=None)
                    V[j*p + k, :] = v
                except:
                    V[j*p + k, :] = 0
        
        # 3. Solve for W
        # For each k, i, solve min_w sum_j (sum_r w_r U[i,j,r] V[j,k,r] - 1)^2
        # A[k,i] is m x R
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, rank))
                for j in range(m):
                    for r in range(rank):
                        A[j, r] = U[i*m + j, r] * V[j*p + k, r]
                b = np.ones(m)
                try:
                    w, residuals, rank_s, sv = np.linalg.lstsq(A, b, rcond=None)
                    W[k*n + i, :] = w
                except:
                    W[k*n + i, :] = 0
        
        # Calculate error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m + j, :], V[j*p + k, :] * W[k*n + i, :])
                    err += (val - 1.0) ** 2
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < tol:
            break
            
    if best_err < 1e-6:
        return best_U, best_V, best_W, best_err
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = {"U": U, "V": V, "W": W}
    
    # Lower bound: max(n*p, m*p, n*m) = max(18, 24, 12) = 24
    min_rank = max(n*p, m*p, n*m)
    
    # Try ranks from min_rank upward
    # We limit the search to a reasonable range to stay within time limits
    max_rank_to_try = min(35, n * m * p)
    
    for rank in range(min_rank, max_rank_to_try + 1):
        if time.time() > deadline - 10:
            break
        
        # Try multiple seeds for each rank
        max_seeds = 10 if rank < 30 else 5
        for seed in range(max_seeds):
            if time.time() > deadline - 10:
                break
            
            result = _als_decompose(n, m, p, rank, seed, max_iter=50)
            if result is not None:
                U_opt, V_opt, W_opt, err = result
                # Round to nearest 0.5 if close
                U_round = np.round(U_opt * 2) / 2
                V_round = np.round(V_opt * 2) / 2
                W_round = np.round(W_opt * 2) / 2
                
                U_list = U_round.tolist()
                V_list = V_round.tolist()
                W_list = W_round.tolist()
                
                if _verify(U_list, V_list, W_list, n, m, p):
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = rank
                    return best_result
    
    return best_result
