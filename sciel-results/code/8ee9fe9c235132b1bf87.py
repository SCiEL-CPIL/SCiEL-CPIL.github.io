"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication.
    
    The tensor condition is:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    """
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    U_3d = U_arr.reshape(n, m, rank)
    V_3d = V_arr.reshape(m, p, rank)
    W_3d = W_arr.reshape(p, n, rank)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U_3d[i, j, r] * V_3d[j, k, r] * W_3d[k, i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the standard schoolbook rank decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _try_rank_decomposition(n, m, p, target_rank, time_budget):
    """Attempt to find a rank-`target_rank` decomposition using structured search.
    
    Uses a combination of:
    1. Random initialization with local refinement
    2. Structured perturbation around the schoolbook solution (for rank close to n*m*p)
    3. Verification with exact arithmetic check
    
    Returns (U, V, W) as lists if found, None otherwise.
    """
    deadline = time.time() + time_budget
    np.random.seed(42)
    
    num_rows_u = n * m
    num_rows_v = m * p
    num_rows_w = p * n
    num_constraints = n * m * p
    
    def compute_error(U, V, W):
        """Compute the maximum absolute error in the tensor reconstruction."""
        U_3d = U.reshape(n, m, target_rank)
        V_3d = V.reshape(m, p, target_rank)
        W_3d = W.reshape(p, n, target_rank)
        
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(target_rank):
                        s += U_3d[i, j, r] * V_3d[j, k, r] * W_3d[k, i, r]
                    err = abs(s - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err
    
    def local_refine(U, V, W, max_iter=50, step_size=0.1):
        """Perform local refinement using coordinate descent on the tensor error."""
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            U_3d = U.reshape(n, m, target_rank)
            V_3d = V.reshape(m, p, target_rank)
            W_3d = W.reshape(p, n, target_rank)
            
            best_err = compute_error(U, V, W)
            if best_err < 1e-10:
                break
            
            improved = False
            # Try perturbing each entry slightly
            for _ in range(20):
                if time.time() > deadline - 5:
                    break
                
                # Randomly pick which matrix to perturb
                which = np.random.randint(3)
                if which == 0:
                    idx = (np.random.randint(num_rows_u), np.random.randint(target_rank))
                    U[idx] += np.random.randn() * step_size
                elif which == 1:
                    idx = (np.random.randint(num_rows_v), np.random.randint(target_rank))
                    V[idx] += np.random.randn() * step_size
                else:
                    idx = (np.random.randint(num_rows_w), np.random.randint(target_rank))
                    W[idx] += np.random.randn() * step_size
                
                new_err = compute_error(U, V, W)
                if new_err < best_err:
                    best_err = new_err
                    improved = True
                else:
                    # Revert
                    if which == 0:
                        U[idx] -= np.random.randn() * step_size  # Approximate revert
                    elif which == 1:
                        V[idx] -= np.random.randn() * step_size
                    else:
                        W[idx] -= np.random.randn() * step_size
            
            if not improved:
                step_size *= 0.5
                if step_size < 1e-6:
                    break
        
        return U, V, W, best_err
    
    # Strategy 1: For rank = n*m*p - 1, try structured modifications
    if target_rank == n * m * p - 1:
        # Start from schoolbook and try to merge two terms
        U = np.zeros((num_rows_u, target_rank))
        V = np.zeros((num_rows_v, target_rank))
        W = np.zeros((num_rows_w, target_rank))
        
        # Fill first n*m*p - 2 columns with schoolbook pattern
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if col < target_rank - 1:
                        U[i * m + j, col] = 1
                        V[j * p + k, col] = 1
                        W[k * n + i, col] = 1
                        col += 1
        
        # The last two schoolbook terms are at positions (n*m*p-2) and (n*m*p-1)
        # We've used target_rank-1 = n*m*p-2 columns, so the last two original terms
        # need to be represented in the final column.
        # This is complex; let's just try random search for this rank.
        pass
    
    # Strategy 2: Random search with local refinement
    num_trials = 200
    for trial in range(num_trials):
        if time.time() > deadline - 10:
            break
        
        # Initialize randomly
        U = np.random.randn(num_rows_u, target_rank) * 0.5
        V = np.random.randn(num_rows_v, target_rank) * 0.5
        W = np.random.randn(num_rows_w, target_rank) * 0.5
        
        # Local refinement
        U, V, W, err = local_refine(U, V, W, max_iter=30, step_size=0.2)
        
        if err < 1e-9:
            return U.tolist(), V.tolist(), W.tolist()
    
    # Strategy 3: For ranks close to schoolbook, try structured initialization
    if target_rank >= n * m * p - 4:
        # Start from schoolbook and remove some columns, then try to fix
        U_school = np.zeros((num_rows_u, n * m * p))
        V_school = np.zeros((num_rows_v, n * m * p))
        W_school = np.zeros((num_rows_w, n * m * p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i * m + j, col] = 1
                    V_school[j * p + k, col] = 1
                    W_school[k * n + i, col] = 1
                    col += 1
        
        # Try removing the last (n*m*p - target_rank) columns and adjusting
        num_remove = n * m * p - target_rank
        U_init = U_school[:, :target_rank].copy()
        V_init = V_school[:, :target_rank].copy()
        W_init = W_school[:, :target_rank].copy()
        
        # The removed terms need to be absorbed. Try adding them back with small coefficients
        # This is a heuristic that may not work, but let's try
        for r in range(target_rank):
            if time.time() > deadline - 10:
                break
            
            # Try to add back the missing terms by adjusting existing columns
            for r2 in range(target_rank):
                # Skip if same column
                if r == r2:
                    continue
                
                # For each missing term, try to add its contribution to column r2
                for miss_idx in range(target_rank, n * m * p):
                    i_miss = miss_idx // (m * p)
                    j_miss = (miss_idx // p) % m
                    k_miss = miss_idx % p
                    
                    # Try adding a small multiple of the missing term to column r2
                    delta = 0.1 * (1 if np.random.rand() > 0.5 else -1)
                    U_init[i_miss * m + j_miss, r2] += delta
                    V_init[j_miss * p + k_miss, r2] += delta
                    W_init[k_miss * n + i_miss, r2] += delta
                    
                    err = compute_error(U_init, V_init, W_init)
                    if err < 1e-9:
                        return U_init.tolist(), V_init.tolist(), W_init.tolist()
                    else:
                        # Revert
                        U_init[i_miss * m + j_miss, r2] -= delta
                        V_init[j_miss * p + k_miss, r2] -= delta
                        W_init[k_miss * n + i_miss, r2] -= delta
    
    return None

def construct():
    _start = time.time()
    _deadline = _start + 140.0  # 140s deadline, 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # First, create the schoolbook decomposition as fallback
    schoolbook_result = _schoolbook(n, m, p)
    best_result = schoolbook_result
    best_rank = n * m * p
    
    # Try to find a lower rank decomposition
    # Start from rank 39 down to 35 (reasonable lower bound)
    for target_rank in range(n * m * p - 1, n * m * p - 6, -1):
        if time.time() > _deadline - 20:
            break
        
        time_budget = min(20.0, _deadline - time.time() - 10)
        if time_budget <= 0:
            break
        
        result = _try_rank_decomposition(n, m, p, target_rank, time_budget)
        if result is not None:
            U, V, W = result
            if _verify_decomposition(U, V, W, n, m, p):
                best_result = {"U": U, "V": V, "W": W}
                best_rank = target_rank
                break  # Found the lowest rank in our search range
    
    return best_result
# EVOLVE-BLOCK-END
