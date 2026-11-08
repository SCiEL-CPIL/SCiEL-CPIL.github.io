import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 135  # Safety margin before 150s kill
    
    # Build the rank-71 solution: 4 Strassen blocks + schoolbook
    blocks = [(0, 0, 0), (0, 2, 0), (0, 0, 2), (0, 2, 2)]
    
    U71, V71, W71 = _build_rank71(n, m, p, blocks)
    rank71 = U71.shape[1]
    
    # Try to find a rank-70 solution via structured search
    if time.time() - start_time < time_limit - 30:
        res70 = _try_rank70(n, m, p, U71, V71, W71, start_time, time_limit)
        if res70 is not None:
            return res70
    
    # Fallback: return rank-71
    return {"U": U71.tolist(), "V": V71.tolist(), "W": W71.tolist()}


def _build_rank71(n, m, p, blocks):
    total_strassen_rank = len(blocks) * 7
    total_terms = n * m * p
    strassen_terms_count = len(blocks) * 8
    schoolbook_terms_count = total_terms - strassen_terms_count
    total_rank = total_strassen_rank + schoolbook_terms_count
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    for i0, j0, k0 in blocks:
        _add_strassen_block(U, V, W, i0, j0, k0, col, n, m, p)
        col += 7
    
    covered_3d = [[[False] * p for _ in range(m)] for _ in range(n)]
    for i0, j0, k0 in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    if i0+di < n and j0+dj < m and k0+dk < p:
                        covered_3d[i0+di][j0+dj][k0+dk] = True
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not covered_3d[i][j][k]:
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
    
    return U, V, W


def _add_strassen_block(U, V, W, i0, j0, k0, col_offset, n, m, p):
    # M1 = (A00+A11)(B00+B11)
    c = col_offset + 0
    U[i0*m + j0, c] += 1
    U[(i0+1)*m + (j0+1), c] += 1
    V[j0*p + k0, c] += 1
    V[(j0+1)*p + (k0+1), c] += 1
    W[k0*n + i0, c] += 1
    W[(k0+1)*n + (i0+1), c] += 1
    
    # M2 = (A10+A11)B00
    c = col_offset + 1
    U[(i0+1)*m + j0, c] += 1
    U[(i0+1)*m + (j0+1), c] += 1
    V[j0*p + k0, c] += 1
    W[k0*n + (i0+1), c] += 1
    W[(k0+1)*n + (i0+1), c] += -1
    
    # M3 = A00(B01-B11)
    c = col_offset + 2
    U[i0*m + j0, c] += 1
    V[j0*p + (k0+1), c] += 1
    V[(j0+1)*p + (k0+1), c] += -1
    W[(k0+1)*n + i0, c] += 1
    W[(k0+1)*n + (i0+1), c] += 1
    
    # M4 = A11(B10-B00)
    c = col_offset + 3
    U[(i0+1)*m + (j0+1), c] += 1
    V[(j0+1)*p + k0, c] += 1
    V[j0*p + k0, c] += -1
    W[k0*n + i0, c] += 1
    W[k0*n + (i0+1), c] += 1
    
    # M5 = (A00+A01)B11
    c = col_offset + 4
    U[i0*m + j0, c] += 1
    U[i0*m + (j0+1), c] += 1
    V[(j0+1)*p + (k0+1), c] += 1
    W[k0*n + i0, c] += -1
    W[(k0+1)*n + i0, c] += 1
    
    # M6 = (A10-A00)(B00+B01)
    c = col_offset + 5
    U[(i0+1)*m + j0, c] += 1
    U[i0*m + j0, c] += -1
    V[j0*p + k0, c] += 1
    V[j0*p + (k0+1), c] += 1
    W[(k0+1)*n + (i0+1), c] += 1
    
    # M7 = (A01-A11)(B10+B11)
    c = col_offset + 6
    U[i0*m + (j0+1), c] += 1
    U[(i0+1)*m + (j0+1), c] += -1
    V[(j0+1)*p + k0, c] += 1
    V[(j0+1)*p + (k0+1), c] += 1
    W[k0*n + i0, c] += 1


def _compute_error(U, V, W, n, m, p):
    """Compute sum of squared errors for the tensor decomposition."""
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                error += (val - 1.0) ** 2
    return error


def _try_rank70(n, m, p, U71, V71, W71, start_time, time_limit):
    """Try to find a rank-70 solution using alternating minimization."""
    rank = 70
    n_m = n * m
    m_p = m * p
    p_n = p * n
    total_rank_71 = U71.shape[1]
    
    # Precompute index arrays for speed
    idx_i = np.repeat(np.arange(n), m * p)
    idx_j = np.tile(np.repeat(np.arange(m), p), n)
    idx_k = np.tile(np.arange(p), n * m)
    
    def fast_error(U, V, W):
        """Vectorized error computation."""
        # For each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # Reshape for vectorized dot products
        vals = np.zeros(len(idx_i))
        for c in range(rank):
            vals += U[idx_i * m + idx_j, c] * V[idx_j * p + idx_k, c] * W[idx_k * n + idx_i, c]
        return np.sum((vals - 1.0) ** 2)
    
    def alternating_minimize(U, V, W, max_iters, tol=1e-10):
        """Run alternating least squares."""
        prev_err = fast_error(U, V, W)
        
        for it in range(max_iters):
            if time.time() - start_time > time_limit - 10:
                break
            
            # Update U: for each (i,j), solve A*x = b
            # A[k,c] = V[j*p+k,c] * W[k*n+i,c], b[k] = 1
            for i in range(n):
                for j in range(m):
                    A = V[j*p:(j+1)*p, :] * W[np.arange(p)*n + i, :].T
                    b = np.ones(p)
                    try:
                        U[i*m+j, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                    except:
                        pass
            
            # Update V: for each (j,k), solve A*x = b
            # A[i,c] = U[i*m+j,c] * W[k*n+i,c], b[i] = 1
            for j in range(m):
                for k in range(p):
                    A = U[:, j].reshape(n, -1).T * W[k*n:(k+1)*n, :].T
                    # A[i,c] = U[i*m+j,c] * W[k*n+i,c]
                    A = np.zeros((n, rank))
                    for i in range(n):
                        A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    b = np.ones(n)
                    try:
                        V[j*p+k, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                    except:
                        pass
            
            # Update W: for each (k,i), solve A*x = b
            # A[j,c] = U[i*m+j,c] * V[j*p+k,c], b[j] = 1
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, rank))
                    for j in range(m):
                        A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    b = np.ones(m)
                    try:
                        W[k*n+i, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                    except:
                        pass
            
            curr_err = fast_error(U, V, W)
            if curr_err < tol:
                break
            if curr_err > prev_err * 1.01:
                # Diverging, stop
                break
            prev_err = curr_err
        
        return U, V, W, prev_err
    
    def round_and_verify(U, V, W):
        """Round to half-integers and verify."""
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        # Verify
        err = fast_error(U_r, V_r, W_r)
        if err < 1e-6:
            # Verify half-integer constraint
            if (np.allclose(U_r * 2, np.round(U_r * 2), atol=1e-9) and
                np.allclose(V_r * 2, np.round(V_r * 2), atol=1e-9) and
                np.allclose(W_r * 2, np.round(W_r * 2), atol=1e-9)):
                return U_r, V_r, W_r
        return None
    
    # Strategy 1: Drop one schoolbook column from rank-71, perturb, and ALS
    schoolbook_start = len([(0,0,0),(0,2,0),(0,0,2),(0,2,2)]) * 7
    
    for trial in range(30):
        if time.time() - start_time > time_limit - 15:
            break
        
        np.random.seed(1000 + trial)
        
        if trial < 10:
            # Drop a random schoolbook column
            num_schoolbook = total_rank_71 - schoolbook_start
            drop_col = schoolbook_start + np.random.randint(num_schoolbook)
            keep = [c for c in range(total_rank_71) if c != drop_col]
            U = U71[:, keep].copy()
            V = V71[:, keep].copy()
            W = W71[:, keep].copy()
            U += np.random.randn(n_m, rank) * 0.02
            V += np.random.randn(m_p, rank) * 0.02
            W += np.random.randn(p_n, rank) * 0.02
        elif trial < 20:
            # Random init
            U = np.random.randn(n_m, rank) * 0.5
            V = np.random.randn(m_p, rank) * 0.5
            W = np.random.randn(p_n, rank) * 0.5
        else:
            # Drop last column, larger perturbation
            U = U71[:, :total_rank_71-1].copy()
            V = V71[:, :total_rank_71-1].copy()
            W = W71[:, :total_rank_71-1].copy()
            U += np.random.randn(n_m, rank) * 0.1
            V += np.random.randn(m_p, rank) * 0.1
            W += np.random.randn(p_n, rank) * 0.1
        
        U, V, W, err = alternating_minimize(U, V, W, max_iters=50)
        
        if err < 1e-8:
            result = round_and_verify(U, V, W)
            if result is not None:
                U_r, V_r, W_r = result
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
    
    # Strategy 2: Try different block configurations
    # Maybe a different set of Strassen blocks gives better coverage
    alt_blocks = [
        [(0, 0, 0), (0, 2, 0), (0, 0, 2), (0, 2, 2)],  # original
        [(0, 0, 0), (0, 2, 0), (1, 0, 2), (1, 2, 2)],  # shift last two
        [(0, 0, 0), (0, 2, 0), (0, 0, 2), (1, 2, 2)],  # mixed
    ]
    
    for block_set in alt_blocks:
        if time.time() - start_time > time_limit - 20:
            break
        
        # Check if this block set gives rank < 71
        U_alt, V_alt, W_alt = _build_rank71(n, m, p, block_set)
        if U_alt.shape[1] < rank71:
            # This is already better!
            return {"U": U_alt.tolist(), "V": V_alt.tolist(), "W": W_alt.tolist()}
        
        # Try to reduce further
        for trial in range(5):
            if time.time() - start_time > time_limit - 15:
                break
            np.random.seed(5000 + len(block_set) * 100 + trial)
            
            U = U_alt[:, :rank].copy() if U_alt.shape[1] >= rank else np.hstack([U_alt, np.random.randn(n_m, rank - U_alt.shape[1])])
            V = V_alt[:, :rank].copy() if V_alt.shape[1] >= rank else np.hstack([V_alt, np.random.randn(m_p, rank - V_alt.shape[1])])
            W = W_alt[:, :rank].copy() if W_alt.shape[1] >= rank else np.hstack([W_alt, np.random.randn(p_n, rank - W_alt.shape[1])])
            
            U += np.random.randn(n_m, rank) * 0.05
            V += np.random.randn(m_p, rank) * 0.05
            W += np.random.randn(p_n, rank) * 0.05
            
            U, V, W, err = alternating_minimize(U, V, W, max_iters=40)
            
            if err < 1e-8:
                result = round_and_verify(U, V, W)
                if result is not None:
                    U_r, V_r, W_r = result
                    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
    
    return None
