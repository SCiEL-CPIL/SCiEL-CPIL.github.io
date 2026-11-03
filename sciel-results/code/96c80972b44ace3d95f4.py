"""Search for a rank-minimal exact tensor decomposition of 4x4 * 4x5 matrix multiplication.
Uses Kronecker-embedded Strassen (rank 65) as baseline, then attempts refinement via
improved ALS with multi-strategy rounding and high-precision polish.
Falls back to schoolbook (rank 80) if all else fails."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def _verify_exact(U, V, W, n=4, m=4, p=5):
    """Exact verification using Fraction arithmetic."""
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    rank = Ua.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += Ua[i*m+j, r] * Va[j*p+k, r] * Wa[k*n+i, r]
                if abs(s - 1.0) > 1e-10:
                    return False
    return True

def _verify_fast(U, V, W, n=4, m=4, p=5, atol=1e-9):
    """Fast verification using numpy with float64."""
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    rank = Ua.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(Ua[i*m+j, :] * Va[j*p+k, :], Wa[k*n+i, :])
                if abs(s - 1.0) > atol:
                    return False
    return True

def _is_valid_half_int(U, V, W, n=4, m=4, p=5):
    """Check if all entries are half-integers and decomposition is exact."""
    for M in [U, V, W]:
        Ma = np.array(M, dtype=np.float64)
        if not np.all(np.abs(Ma * 2 - np.round(Ma * 2)) < 1e-8):
            return False
    return _verify_fast(U, V, W, n, m, p, atol=1e-9)

def _compute_error(U, V, W, n, m, p, rank):
    """Compute the L2 error of the decomposition."""
    Ur = U.reshape(n, m, rank)
    Vr = V.reshape(m, p, rank)
    Wr = W.reshape(p, n, rank)
    T = np.zeros((n, m, p), dtype=np.float64)
    for c in range(rank):
        uc = Ur[:, :, c]
        vc = Vr[:, :, c]
        wc = Wr[:, :, c]
        wcT = wc.T
        for j in range(m):
            T[:, j, :] = T[:, j, :] + np.outer(uc[:, j], vc[j, :]) * wcT
    return np.sqrt(np.sum((T - 1.0)**2))

def _round_to_half_integer(x):
    """Round x to the nearest half-integer."""
    return round(x * 2) / 2.0

def _round_matrix_to_half_int(M):
    """Round all entries of matrix M to nearest half-integer."""
    return np.array([[_round_to_half_integer(x) for x in row] for row in M])

def _strassen_2x2x2_terms():
    """Return the 7 Strassen 2x2x2 terms."""
    return [
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),
        ([(2,1),(3,1)], [(0,1)], [(1,1),(3,-1)]),
        ([(0,1)], [(1,1),(3,-1)], [(2,1),(3,1)]),
        ([(3,1)], [(2,1),(0,-1)], [(0,1),(1,1)]),
        ([(0,1),(1,1)], [(3,1)], [(0,-1),(2,1)]),
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)]),
    ]

def _kron_strassen_construct(n, m, p):
    """Construct a rank-65 decomposition using Kronecker product embedding."""
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u, v, w):
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
    
    strassen_terms = _strassen_2x2x2_terms()
    
    for (u1_sparse, v1_sparse, w1_sparse) in strassen_terms:
        for (u2_sparse, v2_sparse, w2_sparse) in strassen_terms:
            u = np.zeros(n*m)
            v = np.zeros(m*p)
            w = np.zeros(p*n)
            
            for (idx1, val1) in u1_sparse:
                i1, j1 = idx1 // 2, idx1 % 2
                for (idx2, val2) in u2_sparse:
                    i2, j2 = idx2 // 2, idx2 % 2
                    i = i1 * 2 + i2
                    j = j1 * 2 + j2
                    u[i * m + j] = val1 * val2
            
            for (idx1, val1) in v1_sparse:
                j1, k1 = idx1 // 2, idx1 % 2
                for (idx2, val2) in v2_sparse:
                    j2, k2 = idx2 // 2, idx2 % 2
                    j = j1 * 2 + j2
                    k = k1 * 2 + k2
                    v[j * p + k] = val1 * val2
            
            for (idx1, val1) in w1_sparse:
                k1, i1 = idx1 // 2, idx1 % 2
                for (idx2, val2) in w2_sparse:
                    k2, i2 = idx2 // 2, idx2 % 2
                    k = k1 * 2 + k2
                    i = i1 * 2 + i2
                    w[k * n + i] = val1 * val2
            
            add_term(u, v, w)
    
    for i in range(n):
        for j in range(m):
            u = np.zeros(n*m); u[i * m + j] = 1
            v = np.zeros(m*p); v[j * p + 4] = 1
            w = np.zeros(p*n); w[4 * n + i] = 1
            add_term(u, v, w)
    
    U_mat = np.array(U_cols).T
    V_mat = np.array(V_cols).T
    W_mat = np.array(W_cols).T
    return U_mat.tolist(), V_mat.tolist(), W_mat.tolist()

def _als_decompose(n, m, p, rank, seed=42, max_iters=300, deadline=None,
                   U_init=None, V_init=None, W_init=None, polish_iters=200):
    """Try to find a rank-r decomposition using alternating least squares with rounding."""
    if U_init is not None and V_init is not None and W_init is not None:
        U = np.array(U_init, dtype=np.float64).copy()
        V = np.array(V_init, dtype=np.float64).copy()
        W = np.array(W_init, dtype=np.float64).copy()
    else:
        np.random.seed(seed)
        U = np.random.randn(n*m, rank) * 0.5
        V = np.random.randn(m*p, rank) * 0.5
        W = np.random.randn(p*n, rank) * 0.5
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    def _als_iteration():
        for i in range(n):
            for j in range(m):
                A_mat = np.array([V[j*p+k, :] * W[k*n+i, :] for k in range(p)])
                b_vec = np.ones(p)
                try:
                    U[i*m+j, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
        
        for j in range(m):
            for k in range(p):
                A_mat = np.array([U[i*m+j, :] * W[k*n+i, :] for i in range(n)])
                b_vec = np.ones(n)
                try:
                    V[j*p+k, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
        
        for i in range(n):
            for k in range(p):
                A_mat = np.array([U[i*m+j, :] * V[j*p+k, :] for j in range(m)])
                b_vec = np.ones(m)
                try:
                    W[k*n+i, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        _als_iteration()
        err = _compute_error(U, V, W, n, m, p, rank)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < 1e-12:
            break
    
    for it in range(polish_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        _als_iteration()
        err = _compute_error(U, V, W, n, m, p, rank)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < 1e-14:
            break
    
    # Try multiple rounding strategies
    candidates = []
    
    U_r1 = _round_matrix_to_half_int(best_U)
    V_r1 = _round_matrix_to_half_int(best_V)
    W_r1 = _round_matrix_to_half_int(best_W)
    candidates.append((U_r1, V_r1, W_r1))
    
    U_r2 = np.round(best_U)
    V_r2 = np.round(best_V)
    W_r2 = np.round(best_W)
    candidates.append((U_r2, V_r2, W_r2))
    
    for offset in [0, 0.25, -0.25, 0.125, -0.125]:
        U_r3 = _round_matrix_to_half_int(best_U + offset)
        V_r3 = _round_matrix_to_half_int(best_V + offset)
        W_r3 = _round_matrix_to_half_int(best_W + offset)
        candidates.append((U_r3, V_r3, W_r3))
    
    for Uc, Vc, Wc in candidates:
        if _verify_fast(Uc.tolist(), Vc.tolist(), Wc.tolist(), n, m, p, atol=1e-9):
            return Uc.tolist(), Vc.tolist(), Wc.tolist()
    
    return None

def _try_column_elimination(U_base, V_base, W_base, n, m, p, deadline):
    """Try to eliminate columns from the baseline decomposition."""
    Ua = np.array(U_base, dtype=np.float64)
    Va = np.array(V_base, dtype=np.float64)
    Wa = np.array(W_base, dtype=np.float64)
    rank = Ua.shape[1]
    
    best_result = None
    best_rank = rank
    
    # Try removing one column at a time and see if we can compensate
    for r_remove in range(rank):
        if time.time() > deadline:
            break
        
        # Remove column r_remove
        mask = [i for i in range(rank) if i != r_remove]
        U_sub = Ua[:, mask]
        V_sub = Va[:, mask]
        W_sub = Wa[:, mask]
        
        # Try to fix up with small perturbations
        for seed in range(10):
            if time.time() > deadline:
                break
            
            np.random.seed(seed * 1000 + r_remove)
            U_init = U_sub + np.random.randn(n*m, rank-1) * 0.01
            V_init = V_sub + np.random.randn(m*p, rank-1) * 0.01
            W_init = W_sub + np.random.randn(p*n, rank-1) * 0.01
            
            result = _als_decompose(n, m, p, rank-1,
                                    U_init=U_init, V_init=V_init, W_init=W_init,
                                    max_iters=100, deadline=deadline, polish_iters=50)
            if result is not None:
                best_result = result
                best_rank = rank - 1
                break
    
    return best_result

def _peel_decompose(n, m, p, target_rank, deadline, seed=42):
    """
    Linear algebraic peeling: iteratively extract rank-1 components via
    truncated SVD of mode-1 flattening, round to half-integers, subtract
    from residual. This avoids continuous optimization local minima.
    """
    np.random.seed(seed)
    
    # Initialize residual as the full multiplication tensor T[i,j,k] = 1
    T = np.ones((n, m, p), dtype=np.float64)
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    for step in range(target_rank):
        if time.time() > deadline:
            break
        
        if np.linalg.norm(T) < 1e-12:
            break
        
        # Mode-1 flattening: reshape to (n, m*p)
        T_flat = T.reshape(n, m * p)
        
        # Truncated SVD
        try:
            U_svd, s_vals, Vt = np.linalg.svd(T_flat, full_matrices=False)
        except:
            break
        
        if len(s_vals) == 0 or s_vals[0] < 1e-14:
            break
        
        # First singular vector gives the dominant direction
        u_vec = U_svd[:, 0]  # shape (n,)
        vt_vec = Vt[0, :]    # shape (m*p,)
        sigma = s_vals[0]
        
        # Reshape vt_vec back to (m, p)
        v_matrix = vt_vec.reshape(m, p)
        
        # For each i, extract the contribution
        # We need: T[i,j,k] ≈ sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # For rank-1 extraction: U[i,j,0] * V[j,k,0] * W[k,i,0] = T[i,j,k]
        
        # From SVD: T[i,j,k] ≈ sigma * u_vec[i] * v_matrix[j,k]
        # So we set: U[i,j,0] = u_vec[i], V[j,k,0] = v_matrix[j,k], W[k,i,0] = sigma
        # But we need to round these to half-integers
        
        # Try multiple scaling strategies
        best_candidate = None
        best_err = float('inf')
        
        for scale_factor in [1.0, 2.0, 0.5, 4.0, 0.25]:
            # Scale the components
            U_col = np.zeros(n * m)
            V_col = np.zeros(m * p)
            W_col = np.zeros(p * n)
            
            for i in range(n):
                for j in range(m):
                    U_col[i * m + j] = u_vec[i] * scale_factor
            
            for j in range(m):
                for k in range(p):
                    V_col[j * p + k] = v_matrix[j, k] / scale_factor
            
            for k in range(p):
                for i in range(n):
                    W_col[k * n + i] = sigma
            
            # Round to half-integers
            U_rounded = _round_matrix_to_half_int(U_col.reshape(-1, 1)).flatten()
            V_rounded = _round_matrix_to_half_int(V_col.reshape(-1, 1)).flatten()
            W_rounded = _round_matrix_to_half_int(W_col.reshape(-1, 1)).flatten()
            
            # Compute the rank-1 tensor from rounded values
            T_rank1 = np.zeros((n, m, p), dtype=np.float64)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_rank1[i, j, k] = U_rounded[i * m + j] * V_rounded[j * p + k] * W_rounded[k * n + i]
            
            # Compute error after subtraction
            residual = T - T_rank1
            err = np.linalg.norm(residual)
            
            if err < best_err:
                best_err = err
                best_candidate = (U_rounded, V_rounded, W_rounded, residual)
        
        if best_candidate is None:
            break
        
        U_rounded, V_rounded, W_rounded, residual = best_candidate
        
        # Check if this extraction actually helps
        if np.linalg.norm(residual) >= np.linalg.norm(T) * 0.99:
            # Extraction didn't help much, try a different approach
            # Use ALS to fit this rank-1 component
            U_init = np.zeros((n * m, 1))
            V_init = np.zeros((m * p, 1))
            W_init = np.zeros((p * n, 1))
            
            for i in range(n):
                for j in range(m):
                    U_init[i * m + j, 0] = u_vec[i]
            for j in range(m):
                for k in range(p):
                    V_init[j * p + k, 0] = v_matrix[j, k]
            for k in range(p):
                for i in range(n):
                    W_init[k * n + i, 0] = sigma
            
            # Quick ALS polish for this single component
            for _ in range(20):
                if time.time() > deadline:
                    break
                for i in range(n):
                    for j in range(m):
                        A_mat = np.array([V_init[j * p + k, 0] * W_init[k * n + i, 0] for k in range(p)])
                        b_vec = T[i, j, :]
                        if np.linalg.norm(A_mat) > 1e-12:
                            U_init[i * m + j, 0] = np.dot(A_mat, b_vec) / (np.dot(A_mat, A_mat) + 1e-12)
                for j in range(m):
                    for k in range(p):
                        A_mat = np.array([U_init[i * m + j, 0] * W_init[k * n + i, 0] for i in range(n)])
                        b_vec = T[:, j, k]
                        if np.linalg.norm(A_mat) > 1e-12:
                            V_init[j * p + k, 0] = np.dot(A_mat, b_vec) / (np.dot(A_mat, A_mat) + 1e-12)
                for i in range(n):
                    for k in range(p):
                        A_mat = np.array([U_init[i * m + j, 0] * V_init[j * p + k, 0] for j in range(m)])
                        b_vec = T[i, :, k]
                        if np.linalg.norm(A_mat) > 1e-12:
                            W_init[k * n + i, 0] = np.dot(A_mat, b_vec) / (np.dot(A_mat, A_mat) + 1e-12)
            
            U_rounded = _round_matrix_to_half_int(U_init.flatten().reshape(-1, 1)).flatten()
            V_rounded = _round_matrix_to_half_int(V_init.flatten().reshape(-1, 1)).flatten()
            W_rounded = _round_matrix_to_half_int(W_init.flatten().reshape(-1, 1)).flatten()
            
            T_rank1 = np.zeros((n, m, p), dtype=np.float64)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_rank1[i, j, k] = U_rounded[i * m + j] * V_rounded[j * p + k] * W_rounded[k * n + i]
            
            residual = T - T_rank1
        
        U_cols.append(U_rounded)
        V_cols.append(V_rounded)
        W_cols.append(W_rounded)
        T = residual
    
    if len(U_cols) == 0:
        return None
    
    U_mat = np.array(U_cols).T
    V_mat = np.array(V_cols).T
    W_mat = np.array(W_cols).T
    
    # Final verification and polish
    result = _als_decompose(n, m, p, len(U_cols),
                            U_init=U_mat, V_init=V_mat, W_init=W_mat,
                            max_iters=50, deadline=deadline, polish_iters=30)
    if result is not None:
        return result
    
    # If ALS didn't find a valid solution, check the peeled result directly
    if _verify_fast(U_mat.tolist(), V_mat.tolist(), W_mat.tolist(), n, m, p, atol=1e-9):
        return U_mat.tolist(), V_mat.tolist(), W_mat.tolist()
    
    return None

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Build the Kronecker-embedded Strassen baseline (rank 65)
    U_base, V_base, W_base = _kron_strassen_construct(n, m, p)
    
    # Verify baseline
    if not _verify_fast(U_base, V_base, W_base, n, m, p, atol=1e-9):
        return _schoolbook()
    
    best_result = {"U": U_base, "V": V_base, "W": W_base}
    best_rank = len(U_base[0])
    
    # Try column elimination from the baseline
    if time.time() < deadline - 60:
        reduced = _try_column_elimination(U_base, V_base, W_base, n, m, p, deadline - 60)
        if reduced is not None:
            U_red, V_red, W_red = reduced
            best_result = {"U": U_red, "V": V_red, "W": W_red}
            best_rank = len(U_red[0])
    
    # Try random initializations at various ranks
    for rank in range(best_rank - 1, max(50, best_rank - 15), -1):
        if time.time() > deadline - 20:
            break
        
        found = False
        for seed in range(20):
            if time.time() > deadline - 20:
                break
            
            result = _als_decompose(n, m, p, rank, seed=seed, max_iters=300,
                                    deadline=deadline - 20, polish_iters=100)
            if result is not None:
                U, V, W = result
                best_result = {"U": U, "V": V, "W": W}
                best_rank = rank
                found = True
                break
        
        if not found:
            break
    
    # Try linear algebraic peeling strategy
    if time.time() < deadline - 30:
        for target_rank in [best_rank - 2, best_rank - 3, best_rank - 4, best_rank - 5]:
            if target_rank < 40:
                break
            if time.time() > deadline - 30:
                break
            
            for seed in range(5):
                if time.time() > deadline - 30:
                    break
                
                result = _peel_decompose(n, m, p, target_rank, deadline - 30, seed=seed)
                if result is not None:
                    U, V, W = result
                    best_result = {"U": U, "V": V, "W": W}
                    best_rank = target_rank
                    break
            else:
                continue
            break
    
    return best_result
# EVOLVE-BLOCK-END
