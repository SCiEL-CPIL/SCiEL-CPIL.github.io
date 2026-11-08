"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    # Check: for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # Verify with multiple random test matrices
    rng = np.random.RandomState(42)
    for trial in range(3):
        A = rng.randn(n, m)
        B = rng.randn(m, p)
        C = A @ B
        C_prime = np.zeros((n, p))
        for c in range(rank):
            uc = U[:, c].reshape(n, m)
            vc = V[:, c].reshape(m, p)
            wc = W[:, c].reshape(p, n)
            wc_mat = wc.reshape(p, n)
            C_prime += wc_mat.T * (uc @ vc)
        if not np.allclose(C_prime, C, atol=1e-8):
            return False
    return True

def _solve_rank_decomposition(n, m, p, target_rank, max_time=120, deadline=145.0):
    """
    Try to find a rank-`target_rank` decomposition of the matrix multiplication tensor.
    
    Strategy: parameterize U, V, W with small integer entries, build the linear system,
    solve via SVD, and verify.
    
    The tensor equation is:
      sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1  for all i in [0,n), j in [0,m), k in [0,p)
    
    This is nonlinear in the parameters, but we can use the following approach:
    Fix U and V, then the system becomes linear in W:
      sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    For each (i,k) pair, this gives:
      sum_j sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    Actually, let's think differently. The standard approach:
    
    For a rank-r decomposition, we have r triples (u_c, v_c, w_c) where:
      u_c in R^(n*m), v_c in R^(m*p), w_c in R^(p*n)
    
    And the constraint is: for all (i,j,k):
      sum_c u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i] = 1
    
    This is a system of n*m*p = 54 equations in 3*r unknowns (with r columns each).
    
    Approach: Use the fact that we can normalize. Set W entries to be in {-1, 0, 1}
    (or small integers), then solve for U and V.
    
    Actually, a cleaner approach: 
    1. Randomly initialize U and V with small integer entries
    2. The system in W becomes: for each (i,k): sum_{j,c} U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
       This is a linear system in the W variables!
       There are p*n = 18 W variables per column... no, W has p*n*rank entries total.
       
    Let me re-think. W has shape (p*n, rank) = (18, r). So there are 18*r W variables.
    The equations are: for each (i,j,k) in [n]x[m]x[p]:
      sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    If U and V are fixed, this is a linear system in the W entries.
    Number of equations: n*m*p = 54
    Number of unknowns: p*n*r = 18*r
    For r >= 3, we have more unknowns than equations, so solutions exist (generically).
    
    So the strategy is:
    1. Try random small-integer U and V matrices
    2. Solve the linear system for W
    3. Check if W has nice (integer/half-integer) values
    """
    start = time.time()
    rng = np.random.RandomState(42)
    
    # Precompute index mappings
    # Equation index: eq_idx = (i * m + j) * p + k for equation (i,j,k)
    # W variable index: w_idx = (k * n + i) * r + c for W[k*n+i, c]
    
    n, m, p = n, m, p
    r = target_rank
    n_eq = n * m * p  # 54
    n_w = p * n * r   # 18 * r
    
    # Build the coefficient matrix A such that A @ w_vec = 1
    # A[eq_idx, w_idx] = U[i*m+j, c] * V[j*p+k, c]
    # where eq_idx corresponds to (i,j,k) and w_idx corresponds to (k,i,c)
    
    # We'll try several random U, V patterns
    max_attempts = 5000
    best_result = None
    
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
        
        # Generate random U and V with small integer entries
        # Use entries from {-1, 0, 1} or {-2, -1, 0, 1, 2}
        scale = 1 if attempt < max_attempts // 2 else 2
        U_mat = rng.randint(-scale, scale + 1, size=(n * m, r))
        V_mat = rng.randint(-scale, scale + 1, size=(m * p, r))
        
        # Build linear system: A @ w_vec = ones(n_eq)
        # A is (n*m*p, p*n*r)
        # For equation (i,j,k) at index i*m*p + j*p + k:
        #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        # W[k*n+i, c] is at flat index (k*n+i)*r + c
        
        # Build A efficiently
        A = np.zeros((n_eq, n_w), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq_idx = (i * m + j) * p + k
                    for c in range(r):
                        w_idx = (k * n + i) * r + c
                        A[eq_idx, w_idx] = U_mat[i * m + j, c] * V_mat[j * p + k, c]
        
        # Solve A @ w_vec = 1
        b = np.ones(n_eq)
        
        try:
            # Use least squares (system is overdetermined or square/underdetermined)
            if n_w >= n_eq:
                # Under-determined or square: use pinv or lstsq
                w_vec, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                # Check residual
                if residuals.size > 0 and residuals[0] > 1e-8:
                    continue
                # Verify the solution actually works
                if not np.allclose(A @ w_vec, b, atol=1e-8):
                    continue
            else:
                # Over-determined: use lstsq
                w_vec, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                if not np.allclose(A @ w_vec, b, atol=1e-8):
                    continue
        except np.linalg.LinAlgError:
            continue
        
        # Check if W entries are close to integers or half-integers
        W_mat = w_vec.reshape(p * n, r)
        # Check proximity to integers
        W_rounded = np.round(W_mat)
        W_half = np.round(W_mat * 2) / 2.0
        
        int_err = np.max(np.abs(W_mat - W_rounded))
        half_err = np.max(np.abs(W_mat - W_half))
        
        if min(int_err, half_err) < 1e-6:
            # Great! We found a valid decomposition
            if int_err < half_err:
                W_final = W_rounded
            else:
                W_final = W_half
            
            # Verify the full decomposition
            if _verify_decomposition(U_mat.tolist(), V_mat.tolist(), W_final.tolist(), n, m, p):
                return U_mat.tolist(), V_mat.tolist(), W_final.tolist()
        
        # Also check if U and V are already nice (they are integers by construction)
        # and W is close enough
        if min(int_err, half_err) < 1e-4:
            W_final = np.round(W_mat * 2) / 2.0 if half_err < int_err else np.round(W_mat)
            if _verify_decomposition(U_mat.tolist(), V_mat.tolist(), W_final.tolist(), n, m, p):
                return U_mat.tolist(), V_mat.tolist(), W_final.tolist()
    
    return None

def _structured_search(n, m, p, target_rank, deadline=145.0):
    """
    Structured search for rank decomposition using algebraic methods.
    
    Key insight: For matrix multiplication, we can use the "Laser Method" or
    structured patterns. Let's try patterns where U, V, W have specific structures.
    """
    start = time.time()
    rng = np.random.RandomState(7)
    
    # Approach 1: Try to build from the schoolbook and merge terms
    # The schoolbook has rank n*m*p = 54.
    # We want to find linear dependencies among the rank-1 tensors.
    
    # Approach 2: Use the fact that for 3x3x6, we can potentially use
    # Kronecker product structures or other algebraic patterns.
    
    # Approach 3: Direct search with structured U, V
    # Try U and V with entries in {-1, 0, 1} with specific patterns
    
    # Let's try: U has a block structure, V has a block structure
    # For 3x3x6: n=3, m=3, p=6
    
    # Try different structured patterns
    patterns_tried = 0
    max_patterns = 10000
    
    for pattern_idx in range(max_patterns):
        if time.time() > deadline:
            break
        patterns_tried += 1
        
        r = target_rank
        
        # Generate structured U: each row has a specific pattern
        # Try: U[i*m+j, c] depends on (i, j, c) in a structured way
        U_mat = np.zeros((n * m, r), dtype=np.int8)
        V_mat = np.zeros((m * p, r), dtype=np.int8)
        
        # Pattern: use modular arithmetic
        for c in range(r):
            for i in range(n):
                for j in range(m):
                    # U[i*m+j, c] = some function of i, j, c
                    U_mat[i * m + j, c] = ((i + j + c) % 3) - 1  # values in {-1, 0, 1}
            
            for j in range(m):
                for k in range(p):
                    # V[j*p+k, c] = some function of j, k, c
                    V_mat[j * p + k, c] = ((j + k + c) % 3) - 1
        
        # Build linear system for W
        n_eq = n * m * p
        n_w = p * n * r
        A = np.zeros((n_eq, n_w), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq_idx = (i * m + j) * p + k
                    for c in range(r):
                        w_idx = (k * n + i) * r + c
                        A[eq_idx, w_idx] = U_mat[i * m + j, c] * V_mat[j * p + k, c]
        
        b = np.ones(n_eq)
        
        try:
            w_vec, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
            if not np.allclose(A @ w_vec, b, atol=1e-8):
                continue
        except np.linalg.LinAlgError:
            continue
        
        W_mat = w_vec.reshape(p * n, r)
        W_rounded = np.round(W_mat)
        W_half = np.round(W_mat * 2) / 2.0
        int_err = np.max(np.abs(W_mat - W_rounded))
        half_err = np.max(np.abs(W_mat - W_half))
        
        if min(int_err, half_err) < 1e-6:
            W_final = W_rounded if int_err < half_err else W_half
            if _verify_decomposition(U_mat.tolist(), V_mat.tolist(), W_final.tolist(), n, m, p):
                return U_mat.tolist(), V_mat.tolist(), W_final.tolist()
    
    return None

def _random_search_with_solve(n, m, p, target_rank, deadline=145.0):
    """
    Random search: pick random integer U, V, solve linear system for W, check if W is nice.
    """
    start = time.time()
    rng = np.random.RandomState(99)
    
    n_eq = n * m * p
    r = target_rank
    n_w = p * n * r
    
    max_attempts = 20000
    
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
        
        # Random U, V with small integer entries
        U_mat = rng.randint(-1, 2, size=(n * m, r))  # values in {-1, 0, 1}
        V_mat = rng.randint(-1, 2, size=(m * p, r))  # values in {-1, 0, 1}
        
        # Build A
        A = np.zeros((n_eq, n_w), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq_idx = (i * m + j) * p + k
                    for c in range(r):
                        w_idx = (k * n + i) * r + c
                        A[eq_idx, w_idx] = U_mat[i * m + j, c] * V_mat[j * p + k, c]
        
        b = np.ones(n_eq)
        
        try:
            w_vec, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
            if not np.allclose(A @ w_vec, b, atol=1e-8):
                continue
        except np.linalg.LinAlgError:
            continue
        
        W_mat = w_vec.reshape(p * n, r)
        W_rounded = np.round(W_mat)
        W_half = np.round(W_mat * 2) / 2.0
        int_err = np.max(np.abs(W_mat - W_rounded))
        half_err = np.max(np.abs(W_mat - W_half))
        
        if min(int_err, half_err) < 1e-6:
            W_final = W_rounded if int_err < half_err else W_half
            if _verify_decomposition(U_mat.tolist(), V_mat.tolist(), W_final.tolist(), n, m, p):
                return U_mat.tolist(), V_mat.tolist(), W_final.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145.0  # Safety margin before 150s kill
    
    # Start with schoolbook (rank 54)
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank_schoolbook
    
    # Try to find lower rank decompositions
    # Known bounds: rank(3x3x6) is between ~40 and 54
    # Let's try ranks from 53 down to 40
    for target_rank in range(rank_schoolbook - 1, 39, -1):
        if time.time() > deadline:
            break
        
        # Try multiple methods
        result = None
        
        # Method 1: Structured search
        if time.time() < deadline - 5:
            result = _structured_search(n, m, p, target_rank, deadline=deadline)
        
        # Method 2: Random search with linear solve
        if result is None and time.time() < deadline - 5:
            result = _random_search_with_solve(n, m, p, target_rank, deadline=deadline)
        
        if result is not None:
            U, V, W = result
            best_U, best_V, best_W = U, V, W
            best_rank = target_rank
            # Found a decomposition at this rank; try to go lower
            continue
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
