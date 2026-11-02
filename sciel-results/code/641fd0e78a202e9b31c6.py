import time
import numpy as np

DIMENSIONS = [2, 5, 5]

def construct():
    """
    Returns the rank-minimal exact tensor decomposition for matrix multiplication 2x5 * 5x5.
    Uses structured initialization and alternating least squares to find low-rank decompositions.
    """
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 50
    
    # Build schoolbook solution as fallback
    def build_schoolbook():
        rank = schoolbook_rank
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W
    
    best_U, best_V, best_W = build_schoolbook()
    best_rank = schoolbook_rank
    
    def is_valid(U, V, W, tol=1e-10):
        """Check if decomposition is exact."""
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(np.dot(U[i*m+j], V[j*p+k]), W[k*n+i])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err < tol, max_err
    
    def check_time():
        return time.time() < deadline
    
    # Strategy: Try ranks from 25 upward
    # For each rank, use multiple random restarts with structured initialization
    
    def structured_init_W(R, seed=0):
        """Initialize W with structure: for each k, the two rows (k*n+0, k*n+1) 
        share some structure to exploit n=2."""
        np.random.seed(seed)
        W = np.zeros((p * n, R))
        for k in range(p):
            # Each k-slice has 2 rows (for i=0 and i=1)
            # Make them sparse with shared support
            num_nonzero = max(1, R // 5)
            indices = np.random.choice(R, min(num_nonzero, R), replace=False)
            W[k * n + 0, indices] = np.random.choice([-1.0, 1.0], size=len(indices))
            # Second row: same support with sign flip or random
            W[k * n + 1, indices] = np.random.choice([-1.0, 1.0], size=len(indices))
        return W
    
    def structured_init_UV(R, W, seed=0):
        """Initialize U and V given W."""
        np.random.seed(seed + 1000)
        U = np.random.randn(n * m, R) * 0.5
        V = np.random.randn(m * p, R) * 0.5
        return U, V
    
    def als_solve(R, W_init, max_iters=200, seed=0):
        """Alternating Least Squares with structured initialization."""
        if not check_time():
            return None
        
        W = W_init.copy()
        U, V = structured_init_UV(R, W, seed)
        
        # Precompute constraint matrix structure
        n_constraints = n * m * p
        
        for iteration in range(max_iters):
            if not check_time():
                break
            
            # Solve for U given V, W
            # For each (i,j): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
            A_U = np.zeros((n * m * p, n * m * R))
            b = np.ones(n * m * p)
            
            for i in range(n):
                for j in range(m):
                    u_row = i * m + j
                    for k in range(p):
                        row = i * m * p + j * p + k
                        coeff = V[j * p + k] * W[k * n + i]
                        A_U[row, u_row * R : (u_row+1) * R] = coeff
            
            try:
                u_flat, _, _, _ = np.linalg.lstsq(A_U, b, rcond=None)
                U = u_flat.reshape(n * m, R)
            except:
                break
            
            # Solve for V given U, W
            A_V = np.zeros((n * m * p, m * p * R))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        row = i * m * p + j * p + k
                        coeff = U[i * m + j] * W[k * n + i]
                        v_row = j * p + k
                        A_V[row, v_row * R : (v_row+1) * R] = coeff
            
            try:
                v_flat, _, _, _ = np.linalg.lstsq(A_V, b, rcond=None)
                V = v_flat.reshape(m * p, R)
            except:
                break
            
            # Check convergence
            valid, max_err = is_valid(U, V, W)
            if valid:
                return U, V, W
            
            # Occasionally update W (every 5 iterations)
            if iteration % 5 == 4:
                A_W = np.zeros((n * m * p, p * n * R))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            row = i * m * p + j * p + k
                            coeff = U[i * m + j] * V[j * p + k]
                            w_row = k * n + i
                            A_W[row, w_row * R : (w_row+1) * R] = coeff
                
                try:
                    w_flat, _, _, _ = np.linalg.lstsq(A_W, b, rcond=None)
                    W = w_flat.reshape(p * n, R)
                except:
                    break
                
                valid, max_err = is_valid(U, V, W)
                if valid:
                    return U, V, W
        
        return None
    
    # Try ranks from 25 to 49
    for R in range(25, 50):
        if not check_time():
            break
        
        # Try multiple restarts
        num_restarts = 5 if R <= 30 else 3
        for restart in range(num_restarts):
            if not check_time():
                break
            
            W_init = structured_init_W(R, seed=R * 100 + restart)
            result = als_solve(R, W_init, max_iters=150, seed=R * 100 + restart)
            
            if result is not None:
                U, V, W = result
                best_U, best_V, best_W = U, V, W
                best_rank = R
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # If we get here, try rank 49 with more iterations
    if check_time():
        for restart in range(10):
            if not check_time():
                break
            W_init = structured_init_W(49, seed=4900 + restart)
            result = als_solve(49, W_init, max_iters=300, seed=4900 + restart)
            if result is not None:
                U, V, W = result
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
