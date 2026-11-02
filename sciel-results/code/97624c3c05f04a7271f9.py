"""Find rank-minimal tensor decomposition for 2x3 * 3x4 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 3, 4]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying
    a 2x3 matrix by a 3x4 matrix.
    
    The output tensor W has indices W[k*n + i, r] where:
    - i is the row index of A (0..n-1)
    - k is the column index of B (0..p-1) 
    - r is the rank index
    """
    n, m, p = 2, 3, 4
    start_time = time.time()
    deadline = start_time + 115.0 # 5s safety margin
    
    # Precompute indices for the tensor equation
    # T[i, j, k] = 1
    # Indices:
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    # We want to solve for U, V, W
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    def check_decomposition(U, V, W):
        """Check if U, V, W form a valid exact decomposition."""
        rank = U.shape[1]
        # U: (n*m, rank)
        # V: (m*p, rank)
        # W: (p*n, rank)
        
        # Efficient check using einsum or loops
        # Since dimensions are small, loops are fine, but let's try to be slightly faster.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    # Sum over rank
                    # Vectorized dot product
                    total = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True

    def compute_error(U, V, W):
        """Compute the total error of the decomposition."""
        total_error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    s_val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                    total_error += (1.0 - s_val) ** 2
        return total_error

    def local_search(U, V, W, max_iter=100):
        """Perform local search by perturbing individual elements."""
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_error = compute_error(best_U, best_V, best_W)
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            
            # Try perturbing a random element in each matrix
            improved = False
            
            # Perturb U
            u_idx = np.random.randint(n * m)
            r_idx = np.random.randint(U.shape[1])
            old_val = best_U[u_idx, r_idx]
            for delta in [-0.5, 0.5]:
                best_U[u_idx, r_idx] = old_val + delta
                err = compute_error(best_U, best_V, best_W)
                if err < best_error - 1e-10:
                    best_error = err
                    improved = True
                    break
            best_U[u_idx, r_idx] = old_val
            
            # Perturb V
            v_idx = np.random.randint(m * p)
            r_idx = np.random.randint(V.shape[1])
            old_val = best_V[v_idx, r_idx]
            for delta in [-0.5, 0.5]:
                best_V[v_idx, r_idx] = old_val + delta
                err = compute_error(best_U, best_V, best_W)
                if err < best_error - 1e-10:
                    best_error = err
                    improved = True
                    break
            best_V[v_idx, r_idx] = old_val
            
            # Perturb W
            w_idx = np.random.randint(p * n)
            r_idx = np.random.randint(W.shape[1])
            old_val = best_W[w_idx, r_idx]
            for delta in [-0.5, 0.5]:
                best_W[w_idx, r_idx] = old_val + delta
                err = compute_error(best_U, best_V, best_W)
                if err < best_error - 1e-10:
                    best_error = err
                    improved = True
                    break
            best_W[w_idx, r_idx] = old_val
            
            if not improved:
                break
        
        return best_U, best_V, best_W, best_error

    def try_rank(r, max_attempts=100):
        """
        Try to find a rank-r decomposition using multiple strategies.
        """
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            # Strategy 1: Structured initialization based on tensor structure
            # Initialize with small random values
            if attempt % 3 == 0:
                # Random initialization
                U = np.random.uniform(-1, 1, size=(n*m, r))
                V = np.random.uniform(-1, 1, size=(m*p, r))
                W = np.random.uniform(-1, 1, size=(p*n, r))
            elif attempt % 3 == 1:
                # Sparse initialization: only a few non-zero entries per row
                U = np.zeros((n*m, r))
                V = np.zeros((m*p, r))
                W = np.zeros((p*n, r))
                for i in range(n*m):
                    idxs = np.random.choice(r, size=2, replace=False)
                    U[i, idxs] = np.random.choice([-1, 1], size=2)
                for j in range(m*p):
                    idxs = np.random.choice(r, size=2, replace=False)
                    V[j, idxs] = np.random.choice([-1, 1], size=2)
                for k in range(p*n):
                    idxs = np.random.choice(r, size=2, replace=False)
                    W[k, idxs] = np.random.choice([-1, 1], size=2)
            else:
                # Block structure initialization
                U = np.zeros((n*m, r))
                V = np.zeros((m*p, r))
                W = np.zeros((p*n, r))
                # Assign blocks of rows to different rank indices
                for i in range(n):
                    for j in range(m):
                        idx = (i * m + j) % r
                        U[i*m + j, idx] = 1.0
                for j in range(m):
                    for k in range(p):
                        idx = (j * p + k) % r
                        V[j*p + k, idx] = 1.0
                for k in range(p):
                    for i in range(n):
                        idx = (k * n + i) % r
                        W[k*n + i, idx] = 1.0
            
            # Gradient Descent
            lr = 0.005
            for step in range(30):
                if time.time() > deadline:
                    return None
                
                # Compute S for all i,j,k
                S = np.zeros((n, m, p))
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u_idx = i * m + j
                            v_idx = j * p + k
                            w_idx = k * n + i
                            
                            # S = dot(U[u], V[v] * W[w])
                            s_val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                            S[i, j, k] = s_val
                            
                            error = 1.0 - s_val
                            
                            # Gradients
                            grad_U[u_idx, :] += -2.0 * error * (V[v_idx, :] * W[w_idx, :])
                            grad_V[v_idx, :] += -2.0 * error * (U[u_idx, :] * W[w_idx, :])
                            grad_W[w_idx, :] += -2.0 * error * (U[u_idx, :] * V[v_idx, :])
                
                # Update
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Check if we're converging
                if np.max(np.abs(grad_U)) < 1e-6:
                    break
            
            # Round to nearest half-integer
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            # Check
            if check_decomposition(U_rounded, V_rounded, W_rounded):
                return U_rounded, V_rounded, W_rounded
            
            # If close, try local search
            err = compute_error(U_rounded, V_rounded, W_rounded)
            if err < 1.0:
                U_ls, V_ls, W_ls, err_ls = local_search(U_rounded, V_rounded, W_rounded, max_iter=50)
                if err_ls < 1e-9:
                    if check_decomposition(U_ls, V_ls, W_ls):
                        return U_ls, V_ls, W_ls
            
            # Also try pure random integer/half-integer search as a fallback
            if attempt % 5 == 4:
                U_int = np.random.randint(-4, 5, size=(n*m, r))
                V_int = np.random.randint(-4, 5, size=(m*p, r))
                W_int = np.random.randint(-4, 5, size=(p*n, r))
                
                U_rand = U_int / 2.0
                V_rand = V_int / 2.0
                W_rand = W_int / 2.0
                
                if check_decomposition(U_rand, V_rand, W_rand):
                    return U_rand, V_rand, W_rand
                
        return None

    # Construct schoolbook rank 24 as fallback
    rank_schoolbook = n * m * p
    U_school = np.zeros((n*m, rank_schoolbook), dtype=float)
    V_school = np.zeros((m*p, rank_schoolbook), dtype=float)
    W_school = np.zeros((p*n, rank_schoolbook), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1.0
                V_school[j*p + k, col] = 1.0
                W_school[k*n + i, col] = 1.0
                col += 1
                
    best_result = (U_school, V_school, W_school)
    best_rank = rank_schoolbook
    
    # Try to find lower ranks
    ranks_to_try = [20, 21, 22, 23]
    
    for r in ranks_to_try:
        if time.time() > deadline - 10.0:
            break
            
        # Allocate more attempts to lower ranks
        attempts = 80
        if r == 20:
            attempts = 120
        
        result = try_rank(r, attempts)
        if result is not None:
            U, V, W = result
            best_result = (U, V, W)
            best_rank = r
            if r == 20:
                break
        
    U, V, W = best_result
    
    # Convert to lists of lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
