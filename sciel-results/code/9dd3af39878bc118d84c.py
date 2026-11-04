import time
import random
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def _validate(U, V, W):
    """
    Validate that U, V, W form an exact decomposition of the matmul(3,5,6) tensor.
    Returns True if valid, False otherwise.
    """
    n, m, p = DIMENSIONS
    if not U or not V or not W:
        return False
    r = len(U[0])
    
    # Check dimensions
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if not all(len(row) == r for row in U):
        return False
    if not all(len(row) == r for row in V):
        return False
    if not all(len(row) == r for row in W):
        return False
    if r == 0:
        return False
    
    # Convert to numpy for efficient computation
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # For each (i,j,k), check if sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_np[i*m + j]
                v_val = V_np[j*p + k]
                w_val = W_np[k*n + i]
                # Element-wise product and sum
                val = np.sum(u_val * v_val * w_val)
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _compute_error_fast(U, V, W, n, m, p):
    """Compute the decomposition error using vectorized operations."""
    errors = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m + j], V[j*p + k] * W[k*n + i])
                errors.append((val - 1.0) ** 2)
    return np.sqrt(np.sum(errors))

def _structured_init(rank, n, m, p, seed):
    """
    Create a structured initial decomposition that exploits the matmul structure.
    Uses Kronecker-product-like patterns.
    """
    np.random.seed(seed)
    
    # Initialize with structured patterns
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strategy: use a mix of structured and random components
    # For each column, assign it to cover certain (i,j,k) tuples
    
    # Divide rank into groups
    per_group = max(1, rank // max(n, m, p))
    
    for c in range(rank):
        # Assign this column to cover a specific pattern
        # Use a hash-like function to determine coverage
        i_idx = c % n
        j_idx = (c // n) % m
        k_idx = (c // (n * m)) % p
        
        # Primary assignment
        U[i_idx * m + j_idx, c] = 1.0
        V[j_idx * p + k_idx, c] = 1.0
        W[k_idx * n + i_idx, c] = 1.0
        
        # Add some randomization to avoid degenerate cases
        if c < rank // 2:
            # Add small random perturbations
            noise = 0.1 * np.random.randn(3)
            U[i_idx * m + j_idx, c] += noise[0]
            V[j_idx * p + k_idx, c] += noise[1]
            W[k_idx * n + i_idx, c] += noise[2]
    
    # Add random components for remaining columns
    for c in range(rank // 2, rank):
        U[:, c] = np.random.randn(n * m) * 0.5
        V[:, c] = np.random.randn(m * p) * 0.5
        W[:, c] = np.random.randn(p * n) * 0.5
    
    return U, V, W

def _local_search(U, V, W, n, m, p, max_iters=200, tol=1e-10, deadline=None):
    """
    Perform local search by perturbing individual entries.
    """
    U = U.copy()
    V = V.copy()
    W = W.copy()
    rank = U.shape[1]
    
    best_error = _compute_error_fast(U, V, W, n, m, p)
    
    for iteration in range(max_iters):
        if deadline is not None and time.time() > deadline - 5:
            break
        
        if best_error < tol:
            break
        
        # Try perturbing random entries
        improved = False
        
        # Perturb U
        for _ in range(5):
            i = np.random.randint(n * m)
            c = np.random.randint(rank)
            old_val = U[i, c]
            U[i, c] = old_val + np.random.randn() * 0.05
            new_error = _compute_error_fast(U, V, W, n, m, p)
            if new_error < best_error:
                best_error = new_error
                improved = True
            else:
                U[i, c] = old_val
        
        # Perturb V
        for _ in range(5):
            i = np.random.randint(m * p)
            c = np.random.randint(rank)
            old_val = V[i, c]
            V[i, c] = old_val + np.random.randn() * 0.05
            new_error = _compute_error_fast(U, V, W, n, m, p)
            if new_error < best_error:
                best_error = new_error
                improved = True
            else:
                V[i, c] = old_val
        
        # Perturb W
        for _ in range(5):
            i = np.random.randint(p * n)
            c = np.random.randint(rank)
            old_val = W[i, c]
            W[i, c] = old_val + np.random.randn() * 0.05
            new_error = _compute_error_fast(U, V, W, n, m, p)
            if new_error < best_error:
                best_error = new_error
                improved = True
            else:
                W[i, c] = old_val
        
        if not improved and iteration > 50:
            # Try larger steps
            step_size = 0.1 + 0.01 * iteration
            for _ in range(3):
                i = np.random.randint(n * m)
                c = np.random.randint(rank)
                old_val = U[i, c]
                U[i, c] = old_val + np.random.randn() * step_size
                new_error = _compute_error_fast(U, V, W, n, m, p)
                if new_error < best_error:
                    best_error = new_error
                    improved = True
                else:
                    U[i, c] = old_val
    
    if best_error < tol:
        return U, V, W, best_error
    return None, None, None, best_error

def _alternating_min(U, V, W, n, m, p, max_iters=100, tol=1e-12, deadline=None):
    """
    Alternating minimization to refine the decomposition.
    """
    U = U.copy()
    V = V.copy()
    W = W.copy()
    rank = U.shape[1]
    
    for iteration in range(max_iters):
        if deadline is not None and time.time() > deadline - 5:
            break
        
        # Update U
        for i in range(n):
            for j in range(m):
                M = np.zeros((p, rank))
                for k in range(p):
                    M[k, :] = V[j*p + k] * W[k*n + i]
                try:
                    u_new = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                except:
                    continue
                U[i*m + j] = u_new
        
        # Update V
        for j in range(m):
            for k in range(p):
                M = np.zeros((n, rank))
                for i in range(n):
                    M[i, :] = U[i*m + j] * W[k*n + i]
                try:
                    v_new = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
                except:
                    continue
                V[j*p + k] = v_new
        
        # Update W
        for k in range(p):
            for i in range(n):
                M = np.zeros((m, rank))
                for j in range(m):
                    M[j, :] = U[i*m + j] * V[j*p + k]
                try:
                    w_new = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
                except:
                    continue
                W[k*n + i] = w_new
        
        error = _compute_error_fast(U, V, W, n, m, p)
        if error < tol:
            break
    
    error = _compute_error_fast(U, V, W, n, m, p)
    if error < tol:
        return U, V, W, error
    return None, None, None, error

def _round_to_half_integers(arr):
    """Round array to nearest half-integers."""
    return np.round(arr * 2) / 2

def _try_rank_structured(rank, n, m, p, deadline, max_attempts=10):
    """
    Try to find a valid decomposition using structured initialization + local search.
    """
    for attempt in range(max_attempts):
        if time.time() > deadline - 10:
            break
        
        seed = attempt * 1009 + rank * 7
        U, V, W = _structured_init(rank, n, m, p, seed)
        
        # First pass: alternating minimization
        U2, V2, W2, err = _alternating_min(U, V, W, n, m, p, max_iters=50, tol=1e-10, deadline=deadline)
        if U2 is not None:
            U, V, W = U2, V2, W2
        
        # Second pass: local search
        U3, V3, W3, err = _local_search(U, V, W, n, m, p, max_iters=100, tol=1e-12, deadline=deadline)
        if U3 is not None:
            U, V, W = U3, V3, W3
            err = _compute_error_fast(U, V, W, n, m, p)
        
        if err < 1e-10:
            # Round to half-integers and validate
            U_rounded = _round_to_half_integers(U)
            V_rounded = _round_to_half_integers(V)
            W_rounded = _round_to_half_integers(W)
            
            U_list = U_rounded.tolist()
            V_list = V_rounded.tolist()
            W_list = W_rounded.tolist()
            
            if _validate(U_list, V_list, W_list):
                return U_list, V_list, W_list
    
    return None

def _try_rank_random(rank, n, m, p, deadline, max_attempts=5):
    """
    Fallback: try with random initialization + alternating minimization.
    """
    for attempt in range(max_attempts):
        if time.time() > deadline - 10:
            break
        
        np.random.seed(attempt * 7919 + rank * 104729)
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        
        U2, V2, W2, err = _alternating_min(U, V, W, n, m, p, max_iters=100, tol=1e-10, deadline=deadline)
        if U2 is not None:
            U, V, W = U2, V2, W2
        
        U3, V3, W3, err = _local_search(U, V, W, n, m, p, max_iters=50, tol=1e-12, deadline=deadline)
        if U3 is not None:
            U, V, W = U3, V3, W3
            err = _compute_error_fast(U, V, W, n, m, p)
        
        if err < 1e-10:
            U_rounded = _round_to_half_integers(U)
            V_rounded = _round_to_half_integers(V)
            W_rounded = _round_to_half_integers(W)
            
            U_list = U_rounded.tolist()
            V_list = V_rounded.tolist()
            W_list = W_rounded.tolist()
            
            if _validate(U_list, V_list, W_list):
                return U_list, V_list, W_list
    
    return None

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' containing lists of lists.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Start with schoolbook as fallback
    schoolbook_rank = n * m * p
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j][col] = 1
                V_sb[j*p + k][col] = 1
                W_sb[k*n + i][col] = 1
                col += 1
    
    best_U, best_V, best_W = [row[:] for row in U_sb], [row[:] for row in V_sb], [row[:] for row in W_sb]
    best_rank = schoolbook_rank
    
    # Known lower bound: max(n*m, m*p, p*n) = max(15, 30, 18) = 30
    lower_bound = max(n * m, m * p, p * n)
    
    # Try ranks from lower_bound upward
    ranks_to_try = list(range(lower_bound, min(lower_bound + 20, schoolbook_rank)))
    
    for rank in ranks_to_try:
        if time.time() > deadline - 15:
            break
        
        # First try structured approach
        result = _try_rank_structured(rank, n, m, p, deadline, max_attempts=8)
        
        if result is None:
            # Fallback to random approach
            result = _try_rank_random(rank, n, m, p, deadline, max_attempts=4)
        
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = rank
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
