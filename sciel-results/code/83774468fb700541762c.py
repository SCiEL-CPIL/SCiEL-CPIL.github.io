"""Matrix multiplication 2x4 by 4x7 rank-minimal decomposition search."""
import numpy as np
import time
from fractions import Fraction

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

def _try_structured_W(n, m, p, time_limit=120.0):
    """
    Systematic search over structured W matrices.
    
    Key insight: For the 2x4x7 problem, W has shape (14, r).
    We try W matrices with specific algebraic structures that might
    lead to rank-1 X_r matrices.
    
    Structures tried:
    1. Sparse binary patterns
    2. Vandermonde-like (polynomial) patterns  
    3. Block-diagonal patterns
    4. Random with constrained sparsity
    """
    start_time = time.time()
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
    
    # Use a time-based seed to vary search between runs if needed, 
    # but for reproducibility in evaluation, a fixed seed is often preferred 
    # unless we want to explore different paths. 
    # However, if the fixed seed path fails, it always fails.
    # Let's use a seed that allows some variation but is still deterministic per run.
    np.random.seed(42)
    
    # Try ranks from a reasonable lower bound
    start_r = max(rank_lb + 1, 28)
    
    for r in range(start_r, n * m * p + 1):
        if time.time() - start_time > time_limit:
            break
        
        # If we already have a result at this rank, skip
        if r >= best_rank:
            break
        
        found = False
        
        # --- Structure 1: Random sparse {-1,0,1} ---
        for attempt in range(30):
            if time.time() - start_time > time_limit:
                break
            W_cand = np.random.choice([-1, 0, 1], size=(pn, r))
            if np.linalg.matrix_rank(W_cand) < r:
                continue
            try:
                WtW = W_cand.T @ W_cand
                WtW_inv = np.linalg.inv(WtW)
            except np.linalg.LinAlgError:
                continue
            
            TW = np.einsum('abc,cr->abr', T, W_cand)
            X_vec = np.einsum('abr,rs->abs', TW, WtW_inv)
            
            U_sol = np.zeros((nm, r))
            V_sol = np.zeros((mp, r))
            all_ok = True
            
            for ri in range(r):
                ok, u, v = _rank1_check(X_vec[:, :, ri])
                if not ok:
                    all_ok = False
                    break
                if u is not None:
                    U_sol[:, ri] = u
                    V_sol[:, ri] = v
            
            if not all_ok:
                continue
            
            # Verify
            T_recon = np.zeros((nm, mp, pn))
            for ri in range(r):
                T_recon += np.einsum('i,j,k->ijk', U_sol[:, ri], V_sol[:, ri], W_cand[:, ri])
            
            if np.allclose(T_recon, T, atol=1e-8):
                # Check half-integer
                if _check_half_integer(U_sol) and _check_half_integer(V_sol) and _check_half_integer(W_cand):
                    best_rank = r
                    best_result = (U_sol.copy(), V_sol.copy(), W_cand.copy())
                    found = True
                    break
                else:
                    Ur = np.round(U_sol * 2) / 2
                    Vr = np.round(V_sol * 2) / 2
                    Wr = np.round(W_cand * 2) / 2
                    T_recon2 = np.zeros((nm, mp, pn))
                    for ri in range(r):
                        T_recon2 += np.einsum('i,j,k->ijk', Ur[:, ri], Vr[:, ri], Wr[:, ri])
                    if np.allclose(T_recon2, T, atol=1e-8):
                        best_rank = r
                        best_result = (Ur, Vr, Wr)
                        found = True
                        break
        
        if found:
            continue
        
        # --- Structure 2: Sparse with 2 nonzeros per row ---
        for attempt in range(20):
            if time.time() - start_time > time_limit:
                break
            W_cand = np.zeros((pn, r))
            for i in range(pn):
                cols = np.random.choice(r, size=min(2, r), replace=False)
                W_cand[i, cols] = np.random.choice([-1, 1], size=len(cols))
            if np.linalg.matrix_rank(W_cand) < r:
                continue
            try:
                WtW = W_cand.T @ W_cand
                WtW_inv = np.linalg.inv(WtW)
            except np.linalg.LinAlgError:
                continue
            
            TW = np.einsum('abc,cr->abr', T, W_cand)
            X_vec = np.einsum('abr,rs->abs', TW, WtW_inv)
            
            U_sol = np.zeros((nm, r))
            V_sol = np.zeros((mp, r))
            all_ok = True
            
            for ri in range(r):
                ok, u, v = _rank1_check(X_vec[:, :, ri])
                if not ok:
                    all_ok = False
                    break
                if u is not None:
                    U_sol[:, ri] = u
                    V_sol[:, ri] = v
            
            if not all_ok:
                continue
            
            T_recon = np.zeros((nm, mp, pn))
            for ri in range(r):
                T_recon += np.einsum('i,j,k->ijk', U_sol[:, ri], V_sol[:, ri], W_cand[:, ri])
            
            if np.allclose(T_recon, T, atol=1e-8):
                if _check_half_integer(U_sol) and _check_half_integer(V_sol) and _check_half_integer(W_cand):
                    best_rank = r
                    best_result = (U_sol.copy(), V_sol.copy(), W_cand.copy())
                    found = True
                    break
                else:
                    Ur = np.round(U_sol * 2) / 2
                    Vr = np.round(V_sol * 2) / 2
                    Wr = np.round(W_cand * 2) / 2
                    T_recon2 = np.zeros((nm, mp, pn))
                    for ri in range(r):
                        T_recon2 += np.einsum('i,j,k->ijk', Ur[:, ri], Vr[:, ri], Wr[:, ri])
                    if np.allclose(T_recon2, T, atol=1e-8):
                        best_rank = r
                        best_result = (Ur, Vr, Wr)
                        found = True
                        break
        
        if found:
            continue
        
        # --- Structure 3: Vandermonde-like ---
        for attempt in range(10):
            if time.time() - start_time > time_limit:
                break
            # Use polynomial basis: W[i, j] = x_i^j for some x_i in {-1, 0, 1}
            x_vals = np.random.choice([-1, 0, 1], size=pn)
            W_cand = np.zeros((pn, r))
            for i in range(pn):
                for j in range(r):
                    W_cand[i, j] = x_vals[i] ** j if x_vals[i] != 0 else (0 if j > 0 else 1)
            if np.linalg.matrix_rank(W_cand) < r:
                continue
            try:
                WtW = W_cand.T @ W_cand
                WtW_inv = np.linalg.inv(WtW)
            except np.linalg.LinAlgError:
                continue
            
            TW = np.einsum('abc,cr->abr', T, W_cand)
            X_vec = np.einsum('abr,rs->abs', TW, WtW_inv)
            
            U_sol = np.zeros((nm, r))
            V_sol = np.zeros((mp, r))
            all_ok = True
            
            for ri in range(r):
                ok, u, v = _rank1_check(X_vec[:, :, ri])
                if not ok:
                    all_ok = False
                    break
                if u is not None:
                    U_sol[:, ri] = u
                    V_sol[:, ri] = v
            
            if not all_ok:
                continue
            
            T_recon = np.zeros((nm, mp, pn))
            for ri in range(r):
                T_recon += np.einsum('i,j,k->ijk', U_sol[:, ri], V_sol[:, ri], W_cand[:, ri])
            
            if np.allclose(T_recon, T, atol=1e-8):
                if _check_half_integer(U_sol) and _check_half_integer(V_sol) and _check_half_integer(W_cand):
                    best_rank = r
                    best_result = (U_sol.copy(), V_sol.copy(), W_cand.copy())
                    found = True
                    break
                else:
                    Ur = np.round(U_sol * 2) / 2
                    Vr = np.round(V_sol * 2) / 2
                    Wr = np.round(W_cand * 2) / 2
                    T_recon2 = np.zeros((nm, mp, pn))
                    for ri in range(r):
                        T_recon2 += np.einsum('i,j,k->ijk', Ur[:, ri], Vr[:, ri], Wr[:, ri])
                    if np.allclose(T_recon2, T, atol=1e-8):
                        best_rank = r
                        best_result = (Ur, Vr, Wr)
                        found = True
                        break
        
        if found:
            continue
        
        # --- Structure 4: Block structure ---
        for attempt in range(10):
            if time.time() - start_time > time_limit:
                break
            W_cand = np.zeros((pn, r))
            # Assign each row to a small set of columns
            for i in range(pn):
                n_nonzero = np.random.choice([1, 2, 3], p=[0.3, 0.5, 0.2])
                if n_nonzero > r:
                    n_nonzero = r
                cols = np.random.choice(r, size=n_nonzero, replace=False)
                W_cand[i, cols] = np.random.choice([-1, 1], size=n_nonzero)
            if np.linalg.matrix_rank(W_cand) < r:
                continue
            try:
                WtW = W_cand.T @ W_cand
                WtW_inv = np.linalg.inv(WtW)
            except np.linalg.LinAlgError:
                continue
            
            TW = np.einsum('abc,cr->abr', T, W_cand)
            X_vec = np.einsum('abr,rs->abs', TW, WtW_inv)
            
            U_sol = np.zeros((nm, r))
            V_sol = np.zeros((mp, r))
            all_ok = True
            
            for ri in range(r):
                ok, u, v = _rank1_check(X_vec[:, :, ri])
                if not ok:
                    all_ok = False
                    break
                if u is not None:
                    U_sol[:, ri] = u
                    V_sol[:, ri] = v
            
            if not all_ok:
                continue
            
            T_recon = np.zeros((nm, mp, pn))
            for ri in range(r):
                T_recon += np.einsum('i,j,k->ijk', U_sol[:, ri], V_sol[:, ri], W_cand[:, ri])
            
            if np.allclose(T_recon, T, atol=1e-8):
                if _check_half_integer(U_sol) and _check_half_integer(V_sol) and _check_half_integer(W_cand):
                    best_rank = r
                    best_result = (U_sol.copy(), V_sol.copy(), W_cand.copy())
                    found = True
                    break
                else:
                    Ur = np.round(U_sol * 2) / 2
                    Vr = np.round(V_sol * 2) / 2
                    Wr = np.round(W_cand * 2) / 2
                    T_recon2 = np.zeros((nm, mp, pn))
                    for ri in range(r):
                        T_recon2 += np.einsum('i,j,k->ijk', Ur[:, ri], Vr[:, ri], Wr[:, ri])
                    if np.allclose(T_recon2, T, atol=1e-8):
                        best_rank = r
                        best_result = (Ur, Vr, Wr)
                        found = True
                        break
    
    if best_result is not None:
        U, V, W = best_result
        return _to_list(U), _to_list(V), _to_list(W)
    return None

def _try_direct_search(n, m, p, time_limit=120.0):
    """
    Direct search: enumerate small W patterns and check if they yield valid decompositions.
    
    For very small ranks, we can be more exhaustive.
    Also try a "polish" approach: start from schoolbook and try to merge terms.
    """
    start_time = time.time()
    T = _build_tensor(n, m, p)
    pn = p * n
    nm = n * m
    mp = m * p
    
    best_rank = n * m * p
    best_result = None
    
    # Try merging schoolbook terms
    # Start with schoolbook and try to combine pairs
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    U_arr = np.array(U_sb, dtype=np.float64)
    V_arr = np.array(V_sb, dtype=np.float64)
    W_arr = np.array(W_sb, dtype=np.float64)
    rank = n * m * p
    
    # Try to find pairs of terms that can be merged
    # Two terms (r1, r2) can be merged if there exist U', V', W' such that
    # U'⊗V'⊗W' = U_r1⊗V_r1⊗W_r1 + U_r2⊗V_r2⊗W_r2
    # This is generally not possible, but let's try a simpler approach
    
    # Alternative: try rank = rank - 1, rank - 2, ... by searching for W
    for target_rank in range(max(30, rank - 5), rank):
        if time.time() - start_time > time_limit:
            break
        
        r = target_rank
        for attempt in range(40):
            if time.time() - start_time > time_limit:
                break
            
            # Generate W with structured sparsity
            W_cand = np.zeros((pn, r))
            # Each row gets 1-3 nonzeros
            for i in range(pn):
                nnz = np.random.choice([1, 2, 3], p=[0.4, 0.4, 0.2])
                if nnz > r:
                    nnz = r
                cols = np.random.choice(r, size=nnz, replace=False)
                W_cand[i, cols] = np.random.choice([-1, 1], size=nnz)
            
            if np.linalg.matrix_rank(W_cand) < r:
                continue
            
            try:
                WtW = W_cand.T @ W_cand
                WtW_inv = np.linalg.inv(WtW)
            except np.linalg.LinAlgError:
                continue
            
            TW = np.einsum('abc,cr->abr', T, W_cand)
            X_vec = np.einsum('abr,rs->abs', TW, WtW_inv)
            
            U_sol = np.zeros((nm, r))
            V_sol = np.zeros((mp, r))
            all_ok = True
            
            for ri in range(r):
                ok, u, v = _rank1_check(X_vec[:, :, ri])
                if not ok:
                    all_ok = False
                    break
                if u is not None:
                    U_sol[:, ri] = u
                    V_sol[:, ri] = v
            
            if not all_ok:
                continue
            
            T_recon = np.zeros((nm, mp, pn))
            for ri in range(r):
                T_recon += np.einsum('i,j,k->ijk', U_sol[:, ri], V_sol[:, ri], W_cand[:, ri])
            
            if np.allclose(T_recon, T, atol=1e-8):
                if _check_half_integer(U_sol) and _check_half_integer(V_sol) and _check_half_integer(W_cand):
                    best_rank = r
                    best_result = (U_sol.copy(), V_sol.copy(), W_cand.copy())
                    break
                else:
                    Ur = np.round(U_sol * 2) / 2
                    Vr = np.round(V_sol * 2) / 2
                    Wr = np.round(W_cand * 2) / 2
                    T_recon2 = np.zeros((nm, mp, pn))
                    for ri in range(r):
                        T_recon2 += np.einsum('i,j,k->ijk', Ur[:, ri], Vr[:, ri], Wr[:, ri])
                    if np.allclose(T_recon2, T, atol=1e-8):
                        best_rank = r
                        best_result = (Ur, Vr, Wr)
                        break
        
        if best_rank == r:
            continue
    
    if best_result is not None:
        U, V, W = best_result
        return _to_list(U), _to_list(V), _to_list(W)
    return None

def _try_block_decomposition(n, m, p, time_limit=120.0):
    """
    Try to exploit block structure of the matrix multiplication.
    
    For 2x4 * 4x7, we can split B into two 2x7 blocks:
    A * B = A * [B1; B2] = A1*B1 + A2*B2 where A1, A2 are 2x2 blocks of A.
    
    This might allow us to find a lower rank decomposition by
    treating each block product separately and then combining.
    """
    start_time = time.time()
    T = _build_tensor(n, m, p)
    pn = p * n
    nm = n * m
    mp = m * p
    
    best_rank = n * m * p
    best_result = None
    
    # Try to find a decomposition by splitting the middle dimension
    # into blocks of size 2
    block_size = 2
    n_blocks = m // block_size  # 2 blocks for m=4
    
    # For each block, we have a 2x2 * 2x7 product
    # The tensor for 2x2 * 2x7 has rank at most 2*2*7 = 28
    # But we can potentially do better using Strassen-like ideas
    
    # Let's try to find a rank-r decomposition where r < n*m*p
    # by using the block structure
    
    # Strategy: try ranks from a lower bound up
    # Lower bound: at least max(n*m, m*p, p*n) = 28? No, that's not a valid lower bound.
    # Actually, the rank must be at least the rank of any matricization.
    
    T_mat1 = T.reshape(nm, mp * pn)
    T_mat2 = T.reshape(nm * mp, pn)
    T_mat3 = T.reshape(nm * pn, mp)
    rank_lb = max(np.linalg.matrix_rank(T_mat1), 
                  np.linalg.matrix_rank(T_mat2),
                  np.linalg.matrix_rank(T_mat3))
    
    np.random.seed(123)
    
    for r in range(max(rank_lb, 30), n * m * p + 1):
        if time.time() - start_time > time_limit:
            break
        if r >= best_rank:
            break
        
        found = False
        
        # Try random W with block structure
        for attempt in range(20):
            if time.time() - start_time > time_limit:
                break
            
            # Create W with block structure: each row corresponds to a (k, i) pair
            # W[k*n + i, r] should have some structure
            W_cand = np.zeros((pn, r))
            for k in range(p):
                for i in range(n):
                    # Assign a few nonzeros to each row
                    nnz = np.random.choice([1, 2], p=[0.5, 0.5])
                    if nnz > r:
                        nnz = r
                    cols = np.random.choice(r, size=nnz, replace=False)
                    W_cand[k * n + i, cols] = np.random.choice([-1, 1], size=nnz)
            
            if np.linalg.matrix_rank(W_cand) < r:
                continue
            
            try:
                WtW = W_cand.T @ W_cand
                WtW_inv = np.linalg.inv(WtW)
            except np.linalg.LinAlgError:
                continue
            
            TW = np.einsum('abc,cr->abr', T, W_cand)
            X_vec = np.einsum('abr,rs->abs', TW, WtW_inv)
            
            U_sol = np.zeros((nm, r))
            V_sol = np.zeros((mp, r))
            all_ok = True
            
            for ri in range(r):
                ok, u, v = _rank1_check(X_vec[:, :, ri])
                if not ok:
                    all_ok = False
                    break
                if u is not None:
                    U_sol[:, ri] = u
                    V_sol[:, ri] = v
            
            if not all_ok:
                continue
            
            T_recon = np.zeros((nm, mp, pn))
            for ri in range(r):
                T_recon += np.einsum('i,j,k->ijk', U_sol[:, ri], V_sol[:, ri], W_cand[:, ri])
            
            if np.allclose(T_recon, T, atol=1e-8):
                if _check_half_integer(U_sol) and _check_half_integer(V_sol) and _check_half_integer(W_cand):
                    best_rank = r
                    best_result = (U_sol.copy(), V_sol.copy(), W_cand.copy())
                    found = True
                    break
                else:
                    Ur = np.round(U_sol * 2) / 2
                    Vr = np.round(V_sol * 2) / 2
                    Wr = np.round(W_cand * 2) / 2
                    T_recon2 = np.zeros((nm, mp, pn))
                    for ri in range(r):
                        T_recon2 += np.einsum('i,j,k->ijk', Ur[:, ri], Vr[:, ri], Wr[:, ri])
                    if np.allclose(T_recon2, T, atol=1e-8):
                        best_rank = r
                        best_result = (Ur, Vr, Wr)
                        found = True
                        break
        
        if found:
            continue
    
    if best_result is not None:
        U, V, W = best_result
        return _to_list(U), _to_list(V), _to_list(W)
    return None

# EVOLVE-BLOCK-END

def construct():
    n, m, p = DIMENSIONS
    # Try to find a better decomposition than schoolbook.
    # Use a time budget to ensure we finish within the 120s limit.
    # If no better decomposition is found, fall back to schoolbook.
    
    deadline = time.time() + 100.0
    
    # Try structured search
    result = _try_structured_W(n, m, p, time_limit=deadline - time.time())
    if result:
        return result
        
    # Try direct search if time remains
    if time.time() < deadline:
        result = _try_direct_search(n, m, p, time_limit=deadline - time.time())
        if result:
            return result
    
    # Try block decomposition if time remains
    if time.time() < deadline:
        result = _try_block_decomposition(n, m, p, time_limit=deadline - time.time())
        if result:
            return result
            
    # Fallback to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
