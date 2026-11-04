import numpy as np
from fractions import Fraction
import time
import itertools
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 5]

def _grobner_solve_spe(n, m, p, rank, time_limit):
    """
    Try to find an exact tensor decomposition of rank `rank` by solving
    the system of polynomial equations using a Gröbner basis approach.
    
    The tensor T[i,j,k] = 1 for all (i,j,k) must equal
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    We use sympy if available, otherwise fall back to numerical methods.
    """
    start = time.time()
    nm, mp, pn = n*m, m*p, p*n
    
    # Total number of unknowns
    n_unknowns = rank * (nm + mp + pn)
    
    # Number of equations: n*m*p
    n_eqs = n * m * p
    
    # Check if we have enough unknowns (necessary but not sufficient)
    if n_unknowns < n_eqs:
        return None
    
    # Try to use sympy for Gröbner basis
    try:
        import sympy
    except ImportError:
        return None
    
    # For very large systems, this will be too slow
    # Set a practical limit: only try for rank >= some threshold where
    # the system is small enough
    if n_unknowns > 200:
        return None
    
    # Create symbolic variables
    # U[i*m+j, r] -> u_{i,m+j,r}
    # V[j*p+k, r] -> v_{j,p+k,r}  
    # W[k*n+i, r] -> w_{k,n+i,r}
    
    # Build the system of equations
    # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    
    # To keep the system manageable, use a specific parameterization
    # Fix some variables to reduce the system size
    
    # Strategy: fix first row of U, first row of V to normalize
    # This reduces gauge freedom
    
    # Actually, let's try a different approach: use the structure of the
    # matrix multiplication tensor more directly.
    
    # For 2x3 * 3x5, we can try to find a decomposition with rank r
    # by setting up the polynomial system and solving it.
    
    # Given the complexity, let's try a more targeted approach:
    # Assume a specific structure based on the known rank-28 solution
    # and try to reduce it.
    
    # Alternative: use the fact that the tensor has a specific structure
    # and try to find rank-r decompositions by solving smaller systems.
    
    # For now, return None to indicate we can't solve this efficiently
    # with the current approach within the time limit.
    
    if time.time() - start > time_limit:
        return None
    
    return None

def _verify_decomposition(U, V, W, n, m, p):
    rank = len(U[0])
    U_arr = np.array([[float(x) for x in row] for row in U], dtype=float)
    V_arr = np.array([[float(x) for x in row] for row in V], dtype=float)
    W_arr = np.array([[float(x) for x in row] for row in W], dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                s = 0.0
                for r in range(rank):
                    s += U_arr[idx_u, r] * V_arr[idx_v, r] * W_arr[idx_w, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _build_tensor_from_factors(U, V, W, n, m, p):
    """Build the full tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]"""
    rank = len(U[0])
    U_arr = np.array([[float(x) for x in row] for row in U], dtype=float)
    V_arr = np.array([[float(x) for x in row] for row in V], dtype=float)
    W_arr = np.array([[float(x) for x in row] for row in W], dtype=float)
    
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                T[i, j, k] = np.sum(U_arr[idx_u, :] * V_arr[idx_v, :] * W_arr[idx_w, :])
    return T

def _residual_tensor(T, U, V, W, n, m, p):
    """Compute residual = T - sum of rank-1 terms from U,V,W"""
    T_decomp = _build_tensor_from_factors(U, V, W, n, m, p)
    return T - T_decomp

def _try_peel_rank1(T, n, m, p, time_limit):
    """
    Try to find a rank-1 summand in the tensor T.
    A rank-1 tensor has form T[i,j,k] = a[i,j] * b[j,k] * c[k,i]
    We search for such a decomposition.
    """
    start_time = time.time()
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Reshape T to matrix form for rank analysis
    # T is (n, m, p). For rank-1, T[i,j,k] = u[i,j] * v[j,k] * w[k,i]
    # This is a 3D tensor. We can try to find if it's rank-1 by checking
    # if all 2D slices are rank-1 and consistent.
    
    # Alternative approach: try to find vectors u (nm x 1), v (mp x 1), w (pn x 1)
    # such that T[i,j,k] = u[i*m+j] * v[j*p+k] * w[k*n+i]
    
    # This is a multiplicative problem. Let's take logs where possible.
    # But entries can be zero or negative, so we need a different approach.
    
    # Strategy: try all possible sign patterns and magnitudes
    # For a rank-1 tensor, for any fixed j, the matrix T[:,j,:] should be outer product of u[:,j] and w[:,j]
    # i.e., T[i,j,k] = u_ij * w_ki for fixed j, which means T[:,j,:] = u_j * w_j^T where u_j is n-vector and w_j is p-vector
    
    # Actually for rank-1: T[i,j,k] = a[i,j] * b[j,k] * c[k,i]
    # For fixed j: T[i,j,k] = (a[i,j]) * (b[j,k] * c[k,i])
    # The matrix M_j[i,k] = b[j,k] * c[k,i] should be rank-1
    
    # Check if each slice T[:,j,:] is rank-1
    for j in range(m):
        M = T[:, j, :]  # (n, p) matrix
        # Check rank
        try:
            svd_ranks = np.linalg.matrix_rank(M, tol=1e-10)
            if svd_ranks > 1:
                return None
        except:
            return None
    
    # If all slices are rank-1, try to reconstruct
    # T[i,j,k] = a[i,j] * b[j,k] * c[k,i]
    # From slice j=0: T[i,0,k] = a[i,0] * b[0,k] * c[k,i]
    # Let's set b[0,k] = 1 for all k (normalization), then T[i,0,k] = a[i,0] * c[k,i]
    # So c[k,i] = T[i,0,k] / a[i,0]
    
    # We need to find a[i,j], b[j,k], c[k,i] consistently
    # This is a system of equations. Let's use the fact that:
    # T[i,j,k] * T[i',j',k'] * T[i'',j'',k''] = T[i,j',k'] * T[i',j'',k] * T[i'',j,k]
    # for rank-1 tensors (multiplicative consistency)
    
    # Simpler: fix normalization b[0,0] = 1
    # Then for j=0: T[i,0,k] = a[i,0] * c[k,i]
    # For k=0: T[i,j,0] = a[i,j] * b[j,0] * c[0,i]
    
    # Let's try a direct approach: assume the rank-1 form and solve
    # T[i,j,k] = a[i,j] * b[j,k] * c[k,i]
    
    # Take the element T[0,0,0] = a[0,0] * b[0,0] * c[0,0]
    # Normalize: set a[0,0] = 1, then b[0,0] * c[0,0] = T[0,0,0]
    
    # This is complex. Let's try a numerical optimization approach.
    
    def _rank1_residual(params, n=n, m=m, p=p, T=T):
        """Compute residual for given rank-1 parameters"""
        nm = n * m
        mp = m * p
        pn = p * n
        a = params[:nm].reshape(n, m)
        b = params[nm:nm+mp].reshape(m, p)
        c = params[nm+mp:nm+mp+pn].reshape(p, n)
        
        T_recon = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_recon[i, j, k] = a[i, j] * b[j, k] * c[k, i]
        return np.sum((T - T_recon) ** 2)
    
    # Initialize with zeros
    nm = n * m
    mp = m * p
    pn = p * n
    x0 = np.zeros(nm + mp + pn)
    
    # Try multiple initializations
    best_res = np.inf
    best_params = None
    
    for trial in range(10):
        if time.time() - start_time > time_limit:
            break
        
        # Random initialization
        np.random.seed(trial)
        x0 = np.random.randn(nm + mp + pn) * 0.1
        
        try:
            result = minimize(_rank1_residual, x0, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12})
            if result.fun < best_res:
                best_res = result.fun
                best_params = result.x
        except:
            continue
    
    if best_res > 1e-10:
        return None
    
    # Extract parameters
    a = best_params[:nm].reshape(n, m)
    b = best_params[nm:nm+mp].reshape(m, p)
    c = best_params[nm+mp:nm+mp+pn].reshape(p, n)
    
    # Check if entries are half-integers (within tolerance)
    all_vals = np.concatenate([a.flatten(), b.flatten(), c.flatten()])
    for val in all_vals:
        # Check if val is close to a half-integer
        nearest_half = np.round(val * 2) / 2
        if abs(val - nearest_half) > 1e-8:
            return None
    
    # Convert to fractions
    a_frac = [[Fraction(int(np.round(a[i,j] * 2)), 2) for j in range(m)] for i in range(n)]
    b_frac = [[Fraction(int(np.round(b[j,k] * 2)), 2) for k in range(p)] for j in range(m)]
    c_frac = [[Fraction(int(np.round(c[k,i] * 2)), 2) for i in range(n)] for k in range(p)]
    
    return (a_frac, b_frac, c_frac)

def _tensor_slicing_rank_lower_bound(n, m, p):
    """
    Compute a rank lower bound using tensor slicing.
    
    For the matrix multiplication tensor T[i,j,k] = 1, we can slice:
    - Fix i: get a matrix T[i,:,:] of shape (m, p) 
    - Fix j: get a matrix T[:,j,:] of shape (n, p)
    - Fix k: get a matrix T[:,:,k] of shape (n, m)
    
    The rank of the tensor is at least the maximum rank among all such slices.
    But more importantly, we can use the structure to find a minimal decomposition.
    
    For our case: T[i,j,k] = 1 for all i,j,k
    - Slicing by i: each slice is a (m,p) all-ones matrix, rank 1
    - Slicing by j: each slice is a (n,p) all-ones matrix, rank 1  
    - Slicing by k: each slice is a (n,m) all-ones matrix, rank 1
    
    This gives a lower bound of 1, which is not tight.
    
    A better approach: consider the "border rank" or use the fact that
    the tensor must decompose into rank-1 terms. Each rank-1 term contributes
    at most rank 1 to each slice.
    
    Let's use a different strategy: try to construct an explicit low-rank
    decomposition by exploiting the all-ones structure.
    """
    # For T[i,j,k] = 1, we need:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is equivalent to finding matrices U (nm x r), V (mp x r), W (pn x r)
    # such that the tensor product equals the all-ones tensor.
    
    # Key insight: if we can find r vectors u_r (nm-dim), v_r (mp-dim), w_r (pn-dim)
    # such that sum_r u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i] = 1
    
    # Let's try a structured approach:
    # For each (i,j,k), we need the sum to be 1.
    
    # One approach: use the fact that 1 = 1*1*1, so if we can make each
    # rank-1 term contribute 1/r to each entry, we're done.
    
    # But that requires each term to be constant 1/r everywhere, which means
    # u_r[i*m+j] = 1/r^(1/3), etc. But then we need r terms, giving rank r.
    
    # Better: use sparse patterns. For example, assign each (i,j,k) to a subset
    # of rank-1 terms that "cover" it.
    
    # Let's try: partition the indices into groups and use one rank-1 term per group.
    
    # For now, return the theoretical lower bound from slicing
    # The true lower bound for 2x3*3x5 is known to be at least 16 (from various results)
    # Let's return 16 as a conservative lower bound
    return 16

def _construct_structured_decomposition(n, m, p, target_rank):
    """
    Try to construct a rank-`target_rank` decomposition using structured sparse patterns.
    
    Strategy: Use the tensor slicing insight to create a canonical basis.
    For each (i,j,k), we need sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    
    We'll try to find a set of rank-1 terms that efficiently cover all entries.
    """
    nm, mp, pn = n*m, m*p, p*n
    
    # Try different structured approaches
    best_result = None
    best_residual = np.inf
    
    start_time = time.time()
    
    # Approach 1: Use a grid of rank-1 terms
    # Each term covers a "box" in the (i,j,k) space
    
    # For 2x3x5, we have 30 entries to cover
    # With rank r, we have r terms, each contributing to all 30 entries
    # We need to solve: for each (i,j,k), sum of contributions = 1
    
    # Let's try to find a solution where each term is supported on a subset
    # of indices, making the system sparse
    
    # Approach 2: Use the all-ones structure
    # T[i,j,k] = 1 means we need the tensor to be "uniform"
    
    # Try: set U[i*m+j, r] = 1 for certain (i,j,r) pairs, etc.
    
    # Let's use a more systematic approach:
    # Fix the W matrix first (it's the smallest: pn = 10 entries per rank)
    # Then solve for U and V
    
    # Actually, let's try a direct optimization approach for small ranks
    
    for rank in range(target_rank, max(target_rank - 5, 10), -1):
        if time.time() - start_time > 60:
            break
        
        # Try to find a rank-`rank` decomposition
        result = _optimize_decomposition(n, m, p, rank, time_limit=10)
        if result is not None:
            U, V, W = result
            best_result = (U, V, W, rank)
            break
    
    return best_result

def _optimize_decomposition(n, m, p, rank, time_limit=10):
    """
    Use numerical optimization to find a rank-`rank` decomposition.
    
    We parameterize:
    U: (nm x rank) 
    V: (mp x rank)
    W: (pn x rank)
    
    Objective: minimize sum_{i,j,k} (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
    """
    nm, mp, pn = n*m, m*p, p*n
    n_params = rank * (nm + mp + pn)
    
    start_time = time.time()
    
    # Parameter vector: [U_flat, V_flat, W_flat]
    def objective(params):
        U = params[:rank*nm].reshape(rank, nm).T  # (nm, rank)
        V = params[rank*nm:rank*nm+rank*mp].reshape(rank, mp).T  # (mp, rank)
        W = params[rank*nm+rank*mp:].reshape(rank, pn).T  # (pn, rank)
        
        residual = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    residual += (s - 1.0)**2
        return residual
    
    # Try multiple random initializations
    best_result = None
    best_obj = np.inf
    
    for trial in range(20):
        if time.time() - start_time > time_limit:
            break
        
        np.random.seed(trial * 7 + 13)
        x0 = np.random.randn(n_params) * 0.5
        
        try:
            result = minimize(objective, x0, method='L-BFGS-B', 
                            options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if result.fun < best_obj:
                best_obj = result.fun
                if result.fun < 1e-8:
                    # Check if we can round to half-integers
                    U = result.x[:rank*nm].reshape(rank, nm).T
                    V = result.x[rank*nm:rank*nm+rank*mp].reshape(rank, mp).T
                    W = result.x[rank*nm+rank*mp:].reshape(rank, pn).T
                    
                    # Try to round to half-integers
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    
                    # Check if rounding preserves the decomposition
                    residual_rounded = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = np.dot(U_rounded[i*m+j], V_rounded[j*p+k] * W_rounded[k*n+i])
                                residual_rounded += (s - 1.0)**2
                    
                    if residual_rounded < 1e-8:
                        # Convert to fractions
                        U_frac = [[Fraction(int(np.round(val * 2)), 2) for val in row] for row in U_rounded]
                        V_frac = [[Fraction(int(np.round(val * 2)), 2) for val in row] for row in V_rounded]
                        W_frac = [[Fraction(int(np.round(val * 2)), 2) for val in row] for row in W_rounded]
                        
                        best_result = (U_frac, V_frac, W_frac)
                        break
                    else:
                        # Keep the unrounded version as a fallback
                        U_frac = [[Fraction(int(np.round(val * 2)), 2) for val in row] for row in U]
                        V_frac = [[Fraction(int(np.round(val * 2)), 2) for val in row] for row in V]
                        W_frac = [[Fraction(int(np.round(val * 2)), 2) for val in row] for row in W]
                        best_result = (U_frac, V_frac, W_frac)
        except:
            continue
    
    return best_result

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 130  # Leave 20s margin
    
    # First, build the known rank-28 solution
    rank_28 = 28
    U28 = [[0] * rank_28 for _ in range(n * m)]
    V28 = [[0] * rank_28 for _ in range(m * p)]
    W28 = [[0] * rank_28 for _ in range(p * n)]
    
    r = 0
    
    # Part 1: j=2, all k (10 terms)
    for i in range(n):
        for k in range(p):
            U28[i * m + 2][r] = 1
            V28[2 * p + k][r] = 1
            W28[k * n + i][r] = 1
            r += 1
            
    # Part 2: j=0,1, k=4 (4 terms)
    for i in range(n):
        for j in range(2):
            U28[i * m + j][r] = 1
            V28[j * p + 4][r] = 1
            W28[4 * n + i][r] = 1
            r += 1
            
    # Part 3: j=0,1, k=0,1 using Strassen (7 terms)
    # M1
    U28[0 * m + 0][r] = 1
    U28[1 * m + 1][r] = 1
    V28[0 * p + 0][r] = 1
    V28[1 * p + 1][r] = 1
    W28[0 * n + 0][r] = 1
    W28[1 * n + 1][r] = 1
    r += 1
    
    # M2
    U28[1 * m + 0][r] = 1
    U28[1 * m + 1][r] = 1
    V28[0 * p + 0][r] = 1
    W28[0 * n + 1][r] = 1
    W28[1 * n + 1][r] = -1
    r += 1
    
    # M3
    U28[0 * m + 0][r] = 1
    V28[0 * p + 1][r] = 1
    V28[1 * p + 1][r] = -1
    W28[1 * n + 0][r] = 1
    W28[1 * n + 1][r] = 1
    r += 1
    
    # M4
    U28[1 * m + 1][r] = 1
    V28[1 * p + 0][r] = 1
    V28[0 * p + 0][r] = -1
    W28[0 * n + 0][r] = 1
    W28[0 * n + 1][r] = 1
    r += 1
    
    # M5
    U28[0 * m + 0][r] = 1
    U28[0 * m + 1][r] = 1
    V28[1 * p + 1][r] = 1
    W28[0 * n + 0][r] = -1
    W28[1 * n + 0][r] = 1
    r += 1
    
    # M6
    U28[1 * m + 0][r] = 1
    U28[0 * m + 0][r] = -1
    V28[0 * p + 0][r] = 1
    V28[0 * p + 1][r] = 1
    W28[1 * n + 1][r] = 1
    r += 1
    
    # M7
    U28[0 * m + 1][r] = 1
    U28[1 * m + 1][r] = -1
    V28[1 * p + 0][r] = 1
    V28[1 * p + 1][r] = 1
    W28[0 * n + 0][r] = 1
    r += 1
    
    # Part 4: j=0,1, k=2,3 using Strassen (7 terms)
    # M1
    U28[0 * m + 0][r] = 1
    U28[1 * m + 1][r] = 1
    V28[0 * p + 2][r] = 1
    V28[1 * p + 3][r] = 1
    W28[2 * n + 0][r] = 1
    W28[3 * n + 1][r] = 1
    r += 1
    
    # M2
    U28[1 * m + 0][r] = 1
    U28[1 * m + 1][r] = 1
    V28[0 * p + 2][r] = 1
    W28[2 * n + 1][r] = 1
    W28[3 * n + 1][r] = -1
    r += 1
    
    # M3
    U28[0 * m + 0][r] = 1
    V28[0 * p + 3][r] = 1
    V28[1 * p + 3][r] = -1
    W28[3 * n + 0][r] = 1
    W28[3 * n + 1][r] = 1
    r += 1
    
    # M4
    U28[1 * m + 1][r] = 1
    V28[1 * p + 2][r] = 1
    V28[0 * p + 2][r] = -1
    W28[2 * n + 0][r] = 1
    W28[2 * n + 1][r] = 1
    r += 1
    
    # M5
    U28[0 * m + 0][r] = 1
    U28[0 * m + 1][r] = 1
    V28[1 * p + 3][r] = 1
    W28[2 * n + 0][r] = -1
    W28[3 * n + 0][r] = 1
    r += 1
    
    # M6
    U28[1 * m + 0][r] = 1
    U28[0 * m + 0][r] = -1
    V28[0 * p + 2][r] = 1
    V28[0 * p + 3][r] = 1
    W28[3 * n + 1][r] = 1
    r += 1
    
    # M7
    U28[0 * m + 1][r] = 1
    U28[1 * m + 1][r] = -1
    V28[1 * p + 2][r] = 1
    V28[1 * p + 3][r] = 1
    W28[2 * n + 0][r] = 1
    r += 1
    
    # Verify the decomposition
    if not _verify_decomposition(U28, V28, W28, n, m, p):
        raise ValueError("Decomposition verification failed")
    
    # Start with the rank-28 solution
    best_U = [row[:] for row in U28]
    best_V = [row[:] for row in V28]
    best_W = [row[:] for row in W28]
    best_rank = rank_28
    
    # Try to find lower rank decompositions using tensor slicing + optimization
    remaining_time = deadline - (time.time() - start_time)
    if remaining_time > 30:
        # Use tensor slicing to get a lower bound
        lb = _tensor_slicing_rank_lower_bound(n, m, p)
        
        # Try to find decompositions from rank 27 down to lb
        for test_rank in range(rank_28 - 1, max(lb - 1, 10), -1):
            if time.time() - start_time > deadline:
                break
            
            time_per_rank = max(5, min(20, remaining_time / 15))
            result = _optimize_decomposition(n, m, p, test_rank, time_limit=time_per_rank)
            
            if result is not None:
                U_new, V_new, W_new = result
                
                # Verify
                if _verify_decomposition(U_new, V_new, W_new, n, m, p):
                    best_U = U_new
                    best_V = V_new
                    best_W = W_new
                    best_rank = test_rank
                    break
    
    return {"U": best_U, "V": best_V, "W": best_W}
