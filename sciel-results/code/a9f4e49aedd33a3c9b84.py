"""Matrix multiplication 2,2,11 - rank-minimal decomposition via structured search."""
import time
import numpy as np

DIMENSIONS = [2, 2, 11]

def _build_target_tensor():
    """Build the multiplication tensor T of shape (n*m, m*p, p*n)."""
    n, m, p = 2, 2, 11
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _verify(U, V, W, T=None):
    """Verify that the decomposition exactly reproduces the multiplication tensor."""
    if T is None:
        T = _build_target_tensor()
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T, T_decomp, atol=1e-9)

def _verify_fast(U, V, W, T):
    """Fast verification using precomputed tensor structure."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T, T_decomp, atol=1e-9)

def _tensor_to_vec(U, V, W):
    """Flatten all rank-1 tensors into a 2D matrix of shape (prod_dims, rank)."""
    dim_a, dim_b, dim_c = U.shape[0], V.shape[0], W.shape[0]
    rank = U.shape[1]
    M = np.zeros((dim_a * dim_b * dim_c, rank), dtype=np.float64)
    for r in range(rank):
        M[:, r] = np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r]).flatten()
    return M

def _build_strassen_decomposition():
    """
    Build a decomposition using Strassen for 2x2 blocks.
    For 2x2 * 2x11: use Strassen on pairs of columns.
    
    11 columns = 5 pairs + 1 single column
    Each Strassen block (2x2 * 2x2) has rank 7
    Each single column (2x2 * 2x1) has rank 4 (schoolbook for that column)
    
    Total: 5*7 + 4 = 39
    """
    n, m, p = 2, 2, 11
    num_pairs = p // 2  # 5 pairs
    remainder = p % 2   # 1 remaining column
    
    rank = num_pairs * 7 + remainder * 4
    U = np.zeros((n * m, rank), dtype=np.float64)  # (4, rank)
    V = np.zeros((m * p, rank), dtype=np.float64)  # (22, rank)
    W = np.zeros((p * n, rank), dtype=np.float64)  # (22, rank)
    
    col = 0
    for b in range(num_pairs):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        v_e = k0
        v_f = k1
        v_g = p + k0
        v_h = p + k1
        
        w_c00 = k0 * n + 0
        w_c01 = k1 * n + 0
        w_c10 = k0 * n + 1
        w_c11 = k1 * n + 1
        
        u_a = 0
        u_b = 1
        u_c = 2
        u_d = 3
        
        # m1 = (a+d)(e+h)
        U[u_a, col] = 1.0; U[u_d, col] = 1.0
        V[v_e, col] = 1.0; V[v_h, col] = 1.0
        W[w_c00, col] = 1.0; W[w_c11, col] = 1.0
        col += 1
        
        # m2 = (c+d)e
        U[u_c, col] = 1.0; U[u_d, col] = 1.0
        V[v_e, col] = 1.0
        W[w_c10, col] = 1.0; W[w_c11, col] = -1.0
        col += 1
        
        # m3 = a(f-h)
        U[u_a, col] = 1.0
        V[v_f, col] = 1.0; V[v_h, col] = -1.0
        W[w_c01, col] = 1.0; W[w_c11, col] = 1.0
        col += 1
        
        # m4 = d(g-e)
        U[u_d, col] = 1.0
        V[v_g, col] = 1.0; V[v_e, col] = -1.0
        W[w_c00, col] = 1.0; W[w_c10, col] = 1.0
        col += 1
        
        # m5 = (a+b)h
        U[u_a, col] = 1.0; U[u_b, col] = 1.0
        V[v_h, col] = 1.0
        W[w_c00, col] = -1.0; W[w_c01, col] = 1.0
        col += 1
        
        # m6 = (c-a)(e+f)
        U[u_c, col] = 1.0; U[u_a, col] = -1.0
        V[v_e, col] = 1.0; V[v_f, col] = 1.0
        W[w_c11, col] = 1.0
        col += 1
        
        # m7 = (b-d)(g+h)
        U[u_b, col] = 1.0; U[u_d, col] = -1.0
        V[v_g, col] = 1.0; V[v_h, col] = 1.0
        W[w_c00, col] = 1.0
        col += 1
    
    if remainder == 1:
        k2 = num_pairs * 2  # = 10
        
        v_e2 = k2
        v_g2 = p + k2
        w_c02 = k2 * n + 0
        w_c12 = k2 * n + 1
        
        U[0, col] = 1.0
        V[v_e2, col] = 1.0
        W[w_c02, col] = 1.0
        col += 1
        
        U[1, col] = 1.0
        V[v_g2, col] = 1.0
        W[w_c02, col] = 1.0
        col += 1
        
        U[2, col] = 1.0
        V[v_e2, col] = 1.0
        W[w_c12, col] = 1.0
        col += 1
        
        U[3, col] = 1.0
        V[v_g2, col] = 1.0
        W[w_c12, col] = 1.0
        col += 1
    
    return U, V, W

def _build_schoolbook():
    """Build the schoolbook rank-44 decomposition."""
    n, m, p = 2, 2, 11
    rank = n * m * p  # 44
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    return U, V, W

def _fast_rank_reduction(U, V, W, T, max_time=100):
    """
    Efficient rank reduction using linear algebra on the tensor matricization.
    Uses QR with column pivoting to identify and remove dependent terms.
    """
    start_time = time.time()
    current_rank = U.shape[1]
    
    if current_rank <= 1:
        return U, V, W
    
    # Build the tensor matricization matrix
    M = _tensor_to_vec(U, V, W)
    dim_total, rank = M.shape
    
    # QR with column pivoting to find a minimal spanning set
    try:
        Q, R, P = np.linalg.qr(M.T, mode='complete')
        diag_R = np.abs(np.diag(R))
        tol = max(M.shape) * np.finfo(float).eps * diag_R[0]
        num_rank = np.sum(diag_R > tol)
    except Exception:
        return U, V, W
    
    if num_rank >= current_rank:
        return U, V, W
    
    # Get the pivot columns (first num_rank columns of R correspond to independent terms)
    pivot_indices = np.argsort(np.argsort(P.T[np.arange(rank)]))[:num_rank]
    pivot_indices = np.sort(pivot_indices)
    
    U_new = U[:, pivot_indices]
    V_new = V[:, pivot_indices]
    W_new = W[:, pivot_indices]
    
    if _verify_fast(U_new, V_new, W_new, T) and U_new.shape[1] < current_rank:
        return U_new, V_new, W_new
    
    # Try greedy single-column removal using residual check
    for attempt in range(current_rank):
        if time.time() - start_time > max_time:
            break
        removed = False
        for r in range(current_rank):
            if time.time() - start_time > max_time:
                break
            mask = [i for i in range(current_rank) if i != r]
            U_test = U[:, mask]
            V_test = V[:, mask]
            W_test = W[:, mask]
            if _verify_fast(U_test, V_test, W_test, T):
                U, V, W = U_test, V_test, W_test
                current_rank -= 1
                removed = True
                break
        if not removed:
            break
    
    return U, V, W

def _try_rank_reduction(U, V, W, T, max_time=60):
    """
    Attempt to reduce rank by finding linearly dependent terms.
    Uses efficient QR-based approach then greedy removal.
    """
    start_time = time.time()
    current_rank = U.shape[1]
    
    if current_rank <= 1:
        return U, V, W
    
    # Use the fast rank reduction
    U, V, W = _fast_rank_reduction(U, V, W, T, max_time=max_time)
    current_rank = U.shape[1]
    
    # Try removing pairs of columns (only if we have time)
    remaining = max_time - (time.time() - start_time)
    if remaining > 10 and current_rank > 2:
        for attempt in range(current_rank // 2):
            if time.time() - start_time > max_time:
                break
            removed = False
            for r1 in range(current_rank):
                if time.time() - start_time > max_time:
                    break
                for r2 in range(r1 + 1, current_rank):
                    if time.time() - start_time > max_time:
                        break
                    mask = [i for i in range(current_rank) if i != r1 and i != r2]
                    U_test = U[:, mask]
                    V_test = V[:, mask]
                    W_test = W[:, mask]
                    if _verify_fast(U_test, V_test, W_test, T):
                        U, V, W = U_test, V_test, W_test
                        current_rank -= 2
                        removed = True
                        break
                if removed:
                    break
            if not removed:
                break
    
    return U, V, W

def _cross_block_reduction(U, V, W, T, max_time=60):
    """
    Attempt to find cross-block linear dependencies by treating the entire
    decomposition as a single matrix and performing SVD-based rank analysis.
    This can detect non-obvious dependencies that block-local analysis misses.
    """
    start_time = time.time()
    current_rank = U.shape[1]
    
    if current_rank <= 2:
        return U, V, W
    
    # Build full tensor matricization
    M = _tensor_to_vec(U, V, W)
    dim_total, rank = M.shape
    
    # SVD to find numerical rank
    try:
        s = np.linalg.svd(M, compute_uv=False)
        tol = max(M.shape) * np.finfo(float).eps * s[0]
        num_rank = np.sum(s > tol)
    except Exception:
        return U, V, W
    
    if num_rank >= current_rank:
        return U, V, W
    
    # Try to find a subset of columns that spans the same space
    # Use greedy forward selection
    selected = []
    residual = M.copy()
    
    for _ in range(min(num_rank, current_rank)):
        if time.time() - start_time > max_time:
            break
        # Find column with largest norm contribution
        col_norms = np.linalg.norm(M[:, :], axis=0)
        # Weight by how much it reduces residual
        best_col = -1
        best_reduction = -np.inf
        for c in range(rank):
            if c in selected:
                continue
            # Compute reduction in residual norm
            test_residual = residual - M[:, c:c+1] * np.dot(M[:, c], residual) / max(np.dot(M[:, c], M[:, c]), 1e-15)
            reduction = np.linalg.norm(residual) - np.linalg.norm(test_residual)
            if reduction > best_reduction:
                best_reduction = reduction
                best_col = c
        if best_col == -1:
            break
        selected.append(best_col)
        # Update residual
        c = best_col
        if np.dot(M[:, c], M[:, c]) > 1e-15:
            residual = residual - M[:, c:c+1] * np.dot(M[:, c], residual) / np.dot(M[:, c], M[:, c])
    
    if len(selected) < current_rank:
        U_new = U[:, selected]
        V_new = V[:, selected]
        W_new = W[:, selected]
        if _verify_fast(U_new, V_new, W_new, T):
            return U_new, V_new, W_new
    
    # Try random subsets of size num_rank
    np.random.seed(42)
    for _ in range(100):
        if time.time() - start_time > max_time:
            break
        subset = np.random.choice(rank, size=num_rank, replace=False)
        U_new = U[:, subset]
        V_new = V[:, subset]
        W_new = W[:, subset]
        if _verify_fast(U_new, V_new, W_new, T):
            return U_new, V_new, W_new
    
    return U, V, W

def construct():
    start_time = time.time()
    DEADLINE = 140  # seconds, leave margin before 150s limit
    
    T = _build_target_tensor()
    
    # Try the Strassen-based construction (rank 39)
    U, V, W = _build_strassen_decomposition()
    
    if not _verify(U, V, W, T):
        # Fall back to schoolbook rank 44
        U, V, W = _build_schoolbook()
        if _verify(U, V, W, T):
            return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
        raise RuntimeError("No valid decomposition found")
    
    # Try rank reduction with multiple strategies
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed - 3
    
    if remaining > 10:
        # First: standard rank reduction (QR + greedy)
        U_red, V_red, W_red = _try_rank_reduction(U, V, W, T, max_time=min(remaining * 0.4, 40))
        if U_red.shape[1] < U.shape[1]:
            U, V, W = U_red, V_red, W_red
            elapsed = time.time() - start_time
            remaining = DEADLINE - elapsed - 3
        
        # Second: cross-block SVD-based reduction
        if remaining > 10:
            U_cross, V_cross, W_cross = _cross_block_reduction(U, V, W, T, max_time=min(remaining * 0.5, 50))
            if U_cross.shape[1] < U.shape[1]:
                U, V, W = U_cross, V_cross, W_cross
                elapsed = time.time() - start_time
                remaining = DEADLINE - elapsed - 3
        
        # Third: try pair removal if still time
        if remaining > 5:
            current_rank = U.shape[1]
            for attempt in range(current_rank // 2):
                if time.time() - start_time > DEADLINE - 3:
                    break
                removed = False
                for r1 in range(current_rank):
                    if time.time() - start_time > DEADLINE - 3:
                        break
                    for r2 in range(r1 + 1, current_rank):
                        if time.time() - start_time > DEADLINE - 3:
                            break
                        mask = [i for i in range(current_rank) if i != r1 and i != r2]
                        U_test = U[:, mask]
                        V_test = V[:, mask]
                        W_test = W[:, mask]
                        if _verify_fast(U_test, V_test, W_test, T):
                            U, V, W = U_test, V_test, W_test
                            current_rank -= 2
                            removed = True
                            break
                    if removed:
                        break
                if not removed:
                    break
    
    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
