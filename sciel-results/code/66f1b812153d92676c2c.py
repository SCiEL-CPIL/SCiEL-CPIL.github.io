import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify if U, V, W form an exact decomposition of the matrix multiplication tensor.
    U: (n*m, r), V: (m*p, r), W: (p*n, r)
    
    The decomposition must satisfy:
    For all i, j, k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    For all (a, b, c) not corresponding to valid (i, j, k): sum_c U[a, c] * V[b, c] * W[c, c] = 0
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    
    # Check 1s: for all i, j, k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U[a] * V[b], W[c])
                if abs(val - 1.0) > tol:
                    return False
                    
    # Check 0s: for all (a, b, c) that are NOT valid
    valid_indices = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                valid_indices.add((a, b, c))
                
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                if (a, b, c) not in valid_indices:
                    val = np.dot(U[a] * V[b], W[c])
                    if abs(val) > tol:
                        return False
                        
    return True

def _try_rank_r_milp_heuristic(n, m, p, r, max_attempts=500, seed=42):
    """
    Heuristic search for rank-r decomposition.
    We fix U and V with small integer entries, then solve for W.
    We use a smarter initialization and verification.
    """
    np.random.seed(seed)
    
    # Pre-compute valid index mappings for fast checking
    valid_ones = []
    valid_zeros = []
    all_a = range(n * m)
    all_b = range(m * p)
    all_c = range(p * n)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                valid_ones.append((a, b, c))
    
    valid_set = set(valid_ones)
    for a in all_a:
        for b in all_b:
            for c in all_c:
                if (a, b, c) not in valid_set:
                    valid_zeros.append((a, b, c))

    for attempt in range(max_attempts):
        # Strategy: Use structured sparsity
        # Assign each column c a "role"
        # We want U[i*m+j, c] * V[j*p+k, c] to be non-zero only for specific (i,j,k)
        
        # Randomly assign each column to a subset of (i, j) and (j, k)
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        
        for c in range(r):
            # Pick a random j
            j = np.random.randint(m)
            
            # Pick a subset of i's and k's
            # To allow cancellation, we might need multiple (i, k) pairs
            # But for minimal rank, we want efficiency.
            # Let's try: one (i, k) pair per column, or a few.
            num_pairs = np.random.randint(1, 4)
            for _ in range(num_pairs):
                i = np.random.randint(n)
                k = np.random.randint(p)
                
                # Assign values
                # Use small integers {-1, 1} or {-1, 0, 1}
                u_val = np.random.choice([-1, 1])
                v_val = np.random.choice([-1, 1])
                
                U[i * m + j, c] = u_val
                V[j * p + k, c] = v_val
        
        # Now solve for W
        W = np.zeros((p * n, r))
        valid = True
        
        for i in range(n):
            for k in range(p):
                A_mat = np.zeros((m, r))
                for j in range(m):
                    A_mat[j, :] = U[i * m + j, :] * V[j * p + k, :]
                
                # We need A_mat @ w_vec = ones(m)
                # Check if system is consistent
                try:
                    # Use least squares
                    w_vec, residuals, rank, s = np.linalg.lstsq(A_mat, np.ones(m), rcond=None)
                except np.linalg.LinAlgError:
                    valid = False
                    break
                
                # Check if integer solution exists
                w_int = np.round(w_vec)
                if np.linalg.norm(A_mat @ w_int - np.ones(m)) > 1e-9:
                    valid = False
                    break
                
                W[k * n + i, :] = w_int
        
        if not valid:
            continue
        
        # Verify zeros
        # This is the expensive part. Optimize it.
        # We can check if any zero constraint is violated.
        # val = sum_c U[a, c] * V[b, c] * W[c, c]
        # We can precompute U[a] * V[b] for all a, b? No, too big.
        # But we can iterate sparsely.
        
        is_valid = True
        for a, b, c in valid_zeros:
            # Compute dot product
            # If U[a] or V[b] or W[c] is sparse, this is fast.
            # For now, just do the dot product.
            val = np.dot(U[a] * V[b], W[c])
            if abs(val) > 1e-9:
                is_valid = False
                break
        
        if is_valid:
            return U, V, W
            
    return None

def _try_rank_r_exhaustive_small(n, m, p, r, max_attempts=100, seed=100):
    """
    Try a more structured approach:
    Assume U and V have entries in {-1, 0, 1}.
    Use random restarts with different densities.
    """
    np.random.seed(seed)
    
    for attempt in range(max_attempts):
        density = 0.1 + 0.5 * (attempt % 5) / 5.0
        U = np.random.randint(-1, 2, size=(n * m, r))
        V = np.random.randint(-1, 2, size=(m * p, r))
        
        mask_u = np.random.rand(n * m, r) < density
        mask_v = np.random.rand(m * p, r) < density
        U = U * mask_u
        V = V * mask_v
        
        W = np.zeros((p * n, r))
        valid = True
        
        for i in range(n):
            for k in range(p):
                A_mat = np.zeros((m, r))
                for j in range(m):
                    A_mat[j, :] = U[i * m + j, :] * V[j * p + k, :]
                
                try:
                    w_vec, residuals, rank, s = np.linalg.lstsq(A_mat, np.ones(m), rcond=None)
                except np.linalg.LinAlgError:
                    valid = False
                    break
                
                w_int = np.round(w_vec)
                if np.linalg.norm(A_mat @ w_int - np.ones(m)) > 1e-9:
                    valid = False
                    break
                
                W[k * n + i, :] = w_int
        
        if not valid:
            continue
            
        if _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
            return U, V, W
            
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin
    
    # Trivial solution fallback
    rank_trivial = n * m * p
    U_triv = np.zeros((n * m, rank_trivial))
    V_triv = np.zeros((m * p, rank_trivial))
    W_triv = np.zeros((p * n, rank_trivial))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_triv[i * m + j, col] = 1.0
                V_triv[j * p + k, col] = 1.0
                W_triv[k * n + i, col] = 1.0
                col += 1
                
    best_rank = rank_trivial
    best_U, best_V, best_W = U_triv, V_triv, W_triv
    
    # Lower bound
    lb = max(n * m, m * p, p * n)
    
    # Search from lb upwards
    for r in range(lb, rank_trivial + 1):
        if time.time() > deadline:
            break
            
        found = False
        
        # Approach 1: MILP heuristic
        if not found:
            res = _try_rank_r_milp_heuristic(n, m, p, r, max_attempts=200, seed=r)
            if res is not None:
                best_rank = r
                best_U, best_V, best_W = res
                found = True
        
        # Approach 2: Exhaustive small
        if not found and time.time() < deadline - 5:
            res = _try_rank_r_exhaustive_small(n, m, p, r, max_attempts=100, seed=r + 1000)
            if res is not None:
                best_rank = r
                best_U, best_V, best_W = res
                found = True
        
        if found:
            break
            
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
