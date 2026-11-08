"""Matrix multiplication 2x4 by 4x7 rank-minimal decomposition search."""
import numpy as np
import time
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
    return T

def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    if not U or not V or not W:
        return False
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Reconstruct tensor
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for r in range(rank):
        T += np.einsum('i,j,k->ijk', U_arr[:, r], V_arr[:, r], W_arr[:, r])
    
    T_true = _build_tensor(n, m, p)
    return np.allclose(T, T_true, atol=tol)

def _check_half_integer(arr, tol=1e-9):
    """Check if all entries are integer or half-integer."""
    arr = np.array(arr)
    vals = arr.ravel()
    return np.all(np.abs(2 * vals - np.round(2 * vals)) <= tol)

def _to_list(arr):
    """Convert numpy array to list of lists with rounded half-integer values."""
    arr = np.array(arr)
    rounded = np.round(arr * 2) / 2.0
    return rounded.tolist()

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition (rank = n*m*p)."""
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
    return U, V, W

def _rank1_check(C, tol=1e-10):
    """Check if matrix C is rank 1. Returns (is_rank1, u, v) if so."""
    svals = np.linalg.svd(C, compute_uv=False)
    if len(svals) == 0:
        return True, None, None
    if svals[0] < 1e-12:
        return True, None, None
    if len(svals) > 1 and svals[1] > tol * max(svals[0], 1e-12):
        return False, None, None
    U_sv, S_sv, Vt_sv = np.linalg.svd(C)
    u = U_sv[:, 0] * np.sqrt(S_sv[0])
    v = Vt_sv[0, :] * np.sqrt(S_sv[0])
    return True, u, v

def _polynomial_w_patterns(n, m, p, r, deadline):
    """
    Generate W matrices with polynomial/cyclotomic structure.
    
    For the 2x4x7 problem, W has shape (14, r). We try patterns based on:
    1. Roots of unity (cyclotomic)
    2. Sparse polynomial evaluation
    3. Block-circulant structures
    """
    pn = p * n  # 14
    patterns = []
    
    # Pattern 1: Cyclotomic - use 7th roots of unity structure
    # W[k*n+i, j] where j indexes the "frequency"
    # For n=2, p=7: rows are indexed by (k, i) with k in 0..6, i in 0..1
    # Try W[k*n+i, j] = exp(2*pi*i*k*j/7) for appropriate j
    
    # Simplified: use real parts of roots of unity
    for shift in range(7):
        W_cand = np.zeros((pn, r))
        for k in range(p):
            for i in range(n):
                row = k * n + i
                for j in range(min(r, 7)):
                    angle = 2 * np.pi * (k + i * 0.5) * (j + shift) / 7
                    W_cand[row, j] = np.cos(angle)
                    if j + 1 < r:
                        W_cand[row, j + 1] = np.sin(angle)
        # Normalize and check rank
        if np.linalg.matrix_rank(W_cand) >= min(r, pn):
            patterns.append(W_cand)
    
    # Pattern 2: Sparse binary with structured sparsity
    np.random.seed(42)
    for attempt in range(50):
        if time.time() > deadline:
            break
        W_cand = np.zeros((pn, r))
        for row in range(pn):
            # Each row has 1-3 nonzeros
            nnz = min(r, np.random.choice([1, 2, 3], p=[0.3, 0.5, 0.2]))
            cols = np.random.choice(r, size=nnz, replace=False)
            W_cand[row, cols] = np.random.choice([-1, 1], size=nnz)
        if np.linalg.matrix_rank(W_cand) >= min(r, pn):
            patterns.append(W_cand)
    
    # Pattern 3: Block structure based on the 4x7 factorization
    # Group rows by k (0..6), each group has n=2 rows
    for attempt in range(30):
        if time.time() > deadline:
            break
        W_cand = np.zeros((pn, r))
        # Assign columns to k-groups
        cols_per_k = r // p + (1 if r % p > 0 else 0)
        col_offset = 0
        for k in range(p):
            # Each k-group gets cols_per_k columns
            for i in range(n):
                row = k * n + i
                for j in range(min(cols_per_k, r - col_offset)):
                    if np.random.random() < 0.7:
                        W_cand[row, col_offset + j] = np.random.choice([-1, 1])
            col_offset += cols_per_k
        if np.linalg.matrix_rank(W_cand) >= min(r, pn):
            patterns.append(W_cand)
    
    return patterns

def _try_w_based_solve(W_cand, T, nm, mp, pn, deadline):
    """
    Given a candidate W matrix, try to solve for U and V.
    Returns (U, V) if successful, else None.
    """
    r = W_cand.shape[1]
    
    # Check if W has full column rank
    if np.linalg.matrix_rank(W_cand) < r:
        return None
    
    try:
        WtW = W_cand.T @ W_cand
        WtW_inv = np.linalg.inv(WtW)
    except np.linalg.LinAlgError:
        return None
    
    # Contract T with W: TW[i,j,r] = sum_k T[i,j,k] * W[k,r]
    TW = np.einsum('ijk,kr->ijr', T, W_cand)
    
    # For each r, X_r = sum_s WtW_inv[r,s] * TW[:,:,s] should be rank-1
    U_sol = np.zeros((nm, r))
    V_sol = np.zeros((mp, r))
    
    for ri in range(r):
        if time.time() > deadline:
            return None
        X_ri = np.einsum('ijr,rs->ijs', TW, WtW_inv[ri, :])
        # Actually: X_ri[i,j] = sum_s WtW_inv[ri,s] * TW[i,j,s]
        X_ri = np.einsum('is,ijr->ij', WtW_inv[ri, :], TW)
        
        ok, u, v = _rank1_check(X_ri, tol=1e-8)
        if not ok:
            return None
        if u is not None:
            U_sol[:, ri] = u
            V_sol[:, ri] = v
    
    # Verify the decomposition
    T_recon = np.zeros((nm, mp, pn))
    for ri in range(r):
        T_recon += np.einsum('i,j,k->ijk', U_sol[:, ri], V_sol[:, ri], W_cand[:, ri])
    
    if not np.allclose(T_recon, T, atol=1e-8):
        return None
    
    # Check half-integer constraint
    if _check_half_integer(U_sol) and _check_half_integer(V_sol) and _check_half_integer(W_cand):
        return U_sol, V_sol
    
    # Try rounding
    Ur = np.round(U_sol * 2) / 2
    Vr = np.round(V_sol * 2) / 2
    Wr = np.round(W_cand * 2) / 2
    
    T_recon2 = np.zeros((nm, mp, pn))
    for ri in range(r):
        T_recon2 += np.einsum('i,j,k->ijk', Ur[:, ri], Vr[:, ri], Wr[:, ri])
    
    if np.allclose(T_recon2, T, atol=1e-8):
        return Ur, Vr, Wr
    
    return None

def _try_structured_W(n, m, p, time_limit=120.0):
    """
    Systematic search over structured W matrices using polynomial/cyclotomic patterns.
    """
    start_time = time.time()
    deadline = start_time + time_limit
    T = _build_tensor(n, m, p)
    pn = p * n  # 14
    nm = n * m  # 8
    mp = m * p  # 28
    
    # Compute matricization lower bounds
    T_mat1 = T.reshape(nm, mp * pn)
    T_mat2 = T.reshape(nm * mp, pn)
    T_mat3 = T.reshape(nm * pn, mp)
    rank_lb = max(np.linalg.matrix_rank(T_mat1), 
                  np.linalg.matrix_rank(T_mat2),
                  np.linalg.matrix_rank(T_mat3))
    
    best_rank = n * m * p
    best_result = None
    
    # Try ranks from a reasonable lower bound up to schoolbook
    # Start from the lower bound and go up
    start_r = max(rank_lb, 20)
    
    for r in range(start_r, min(n * m * p + 1, 60)):
        if time.time() > deadline:
            break
        
        if r >= best_rank:
            break
        
        found = False
        
        # Generate structured W patterns
        W_patterns = _polynomial_w_patterns(n, m, p, r, deadline)
        
        for W_cand in W_patterns:
            if time.time() > deadline:
                break
            
            result = _try_w_based_solve(W_cand, T, nm, mp, pn, deadline)
            if result is not None:
                U_sol, V_sol = result
                best_rank = r
                best_result = (U_sol.copy(), V_sol.copy(), W_cand.copy())
                found = True
                break
        
        if found:
            continue
    
    if best_result is not None:
        U, V, W = best_result
        return _to_list(U), _to_list(V), _to_list(W)
    return None

def _try_direct_search(n, m, p, time_limit=120.0):
    """
    Direct search: try to find decompositions at various ranks.
    """
    start_time = time.time()
    deadline = start_time + time_limit
    T = _build_tensor(n, m, p)
    pn = p * n
    nm = n * m
    mp = m * p
    
    best_rank = n * m * p
    best_result = None
    
    # Try ranks near schoolbook first (more likely to succeed)
    for target_rank in range(n * m * p - 10, n * m * p + 1):
        if time.time() > deadline:
            break
        
        if target_rank >= best_rank:
            break
        
        r = target_rank
        np.random.seed(123 + target_rank)
        
        for attempt in range(30):
            if time.time() > deadline:
                break
            
            # Generate W with structured sparsity
            W_cand = np.zeros((pn, r))
            for i in range(pn):
                nnz = min(r, np.random.choice([1, 2, 3], p=[0.4, 0.4, 0.2]))
                cols = np.random.choice(r, size=nnz, replace=False)
                W_cand[i, cols] = np.random.choice([-1, 1], size=nnz)
            
            result = _try_w_based_solve(W_cand, T, nm, mp, pn, deadline)
            if result is not None:
                U_sol, V_sol = result
                best_rank = r
                best_result = (U_sol.copy(), V_sol.copy(), W_cand.copy())
                break
    
    if best_result is not None:
        U, V, W = best_result
        return _to_list(U), _to_list(V), _to_list(W)
    return None

def _try_polish(n, m, p, time_limit=120.0):
    """
    Polish: start from schoolbook and try to reduce rank by finding
    linear dependencies among the terms.
    """
    start_time = time.time()
    deadline = start_time + time_limit
    T = _build_tensor(n, m, p)
    pn = p * n
    nm = n * m
    mp = m * p
    
    best_rank = n * m * p
    best_result = None
    
    # Get schoolbook decomposition
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    U_arr = np.array(U_sb, dtype=np.float64)
    V_arr = np.array(V_sb, dtype=np.float64)
    W_arr = np.array(W_sb, dtype=np.float64)
    rank = n * m * p
    
    # Try to find a subset of columns that spans the same tensor
    # This is equivalent to finding a smaller set of rank-1 terms
    # that sum to the same tensor
    
    # Use SVD on the matricization to find a low-rank approximation
    T_mat = T.reshape(nm, mp * pn)
    U_svd, S_svd, Vt_svd = np.linalg.svd(T_mat, full_matrices=False)
    
    # The rank of the matricization gives a lower bound
    # Try to construct a decomposition from the top singular vectors
    
    for target_r in range(max(20, int(S_svd[0] * 0.1)), min(rank, 50)):
        if time.time() > deadline:
            break
        
        if target_r >= best_rank:
            break
        
        # Try to find U, V, W of rank target_r
        # Use the SVD to initialize
        r = target_r
        
        # Initialize W from the right singular vectors
        V_init = Vt_svd[:r, :].T  # shape (mp*pn, r)
        
        # This is complex; instead, just try random W with the right structure
        np.random.seed(42 + target_r)
        
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            W_cand = np.zeros((pn, r))
            for i in range(pn):
                nnz = min(r, np.random.choice([1, 2, 3], p=[0.3, 0.5, 0.2]))
                cols = np.random.choice(r, size=nnz, replace=False)
                W_cand[i, cols] = np.random.choice([-1, 1], size=nnz)
            
            result = _try_w_based_solve(W_cand, T, nm, mp, pn, deadline)
            if result is not None:
                U_sol, V_sol = result
                best_rank = r
                best_result = (U_sol.copy(), V_sol.copy(), W_cand.copy())
                break
    
    if best_result is not None:
        U, V, W = best_result
        return _to_list(U), _to_list(V), _to_list(W)
    return None

# EVOLVE-BLOCK-END

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Try structured W search with polynomial patterns
    try:
        remaining = deadline - time.time()
        if remaining > 5.0:
            result = _try_structured_W(n, m, p, time_limit=min(remaining, 60.0))
            if result is not None:
                U, V, W = result
                if _verify_decomposition(U, V, W, n, m, p):
                    return {"U": U, "V": V, "W": W}
    except:
        pass
    
    # Try direct search
    try:
        remaining = deadline - time.time()
        if remaining > 5.0:
            result = _try_direct_search(n, m, p, time_limit=min(remaining, 40.0))
            if result is not None:
                U, V, W = result
                if _verify_decomposition(U, V, W, n, m, p):
                    return {"U": U, "V": V, "W": W}
    except:
        pass
    
    # Try polish
    try:
        remaining = deadline - time.time()
        if remaining > 5.0:
            result = _try_polish(n, m, p, time_limit=min(remaining, 30.0))
            if result is not None:
                U, V, W = result
                if _verify_decomposition(U, V, W, n, m, p):
                    return {"U": U, "V": V, "W": W}
    except:
        pass
    
    # Fall back to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
