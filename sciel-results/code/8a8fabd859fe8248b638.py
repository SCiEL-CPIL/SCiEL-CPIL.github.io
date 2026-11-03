"""
Program for rank-minimal exact tensor decomposition of 3x5x6 matrix multiplication.

Uses block decomposition with Strassen for 2x2x2 blocks as the baseline (rank 84),
then attempts to find lower-rank decompositions via sparse alternating least-squares.
"""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START

def _build_target_tensor(n, m, p):
    """Build the multiplication tensor T of shape (n*m, m*p, p*n)."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _verify_decomp(n, m, p, U, V, W):
    """Verify that U, V, W exactly decompose the multiplication tensor."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    r = U.shape[1]
    if V.shape[1] != r or W.shape[1] != r:
        return False
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    T_comp = np.einsum('ak,bk,ck->abc', U, V, W, optimize=True)
    T_target = _build_target_tensor(n, m, p)
    return np.allclose(T_comp, T_target, atol=1e-9)

def _schoolbook(n_sub, m_sub, p_sub):
    """Return schoolbook decomposition for n_sub x m_sub x p_sub."""
    rank = n_sub * m_sub * p_sub
    U = np.zeros((n_sub * m_sub, rank), dtype=np.float64)
    V = np.zeros((m_sub * p_sub, rank), dtype=np.float64)
    W = np.zeros((p_sub * n_sub, rank), dtype=np.float64)
    col = 0
    for i in range(n_sub):
        for j in range(m_sub):
            for k in range(p_sub):
                U[i * m_sub + j, col] = 1.0
                V[j * p_sub + k, col] = 1.0
                W[k * n_sub + i, col] = 1.0
                col += 1
    return U, V, W

def _strassen_222():
    """
    Return Strassen decomposition for 2x2x2 matrix multiplication.
    Rank 7. Returns (U, V, W) where U is 4x7, V is 4x7, W is 4x7.
    Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    W convention: W[k*n+i] where n=2, so W[k*2+i] for C[i,k]
    """
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    
    # A[0,0]=U[0], A[0,1]=U[1], A[1,0]=U[2], A[1,1]=U[3]
    # B[0,0]=V[0], B[0,1]=V[1], B[1,0]=V[2], B[1,1]=V[3]
    # C[0,0]=W[0], C[0,1]=W[2], C[1,0]=W[1], C[1,1]=W[3]
    
    # M1 = (A00+A11)*(B00+B11)
    # C00 += M1, C11 += M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)*B00
    # C10 += M2, C11 -= M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # M3 = A00*(B01-B11)
    # C01 += M3, C11 += M3
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # M4 = A11*(B10-B00)
    # C00 += M4, C10 += M4
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5 = (A00+A01)*B11
    # C00 -= M5, C01 += M5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6 = (A10-A00)*(B00+B01)
    # C11 += M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A01-A11)*(B10+B11)
    # C00 += M7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _embed_sub_decomp(U_sub, V_sub, W_sub, n_sub, m_sub, p_sub,
                      i_off, j_off, k_off, n, m, p):
    """
    Embed a sub-decomposition into the full 3x5x6 problem.
    """
    rank = U_sub.shape[1]
    U_full = np.zeros((n * m, rank), dtype=np.float64)
    V_full = np.zeros((m * p, rank), dtype=np.float64)
    W_full = np.zeros((p * n, rank), dtype=np.float64)
    
    for idx in range(n_sub * m_sub):
        i_s = idx // m_sub
        j_s = idx % m_sub
        i_f = i_off + i_s
        j_f = j_off + j_s
        U_full[i_f * m + j_f, :] = U_sub[idx, :]
    
    for idx in range(m_sub * p_sub):
        j_s = idx // p_sub
        k_s = idx % p_sub
        j_f = j_off + j_s
        k_f = k_off + k_s
        V_full[j_f * p + k_f, :] = V_sub[idx, :]
    
    for idx in range(p_sub * n_sub):
        k_s = idx // n_sub
        i_s = idx % n_sub
        k_f = k_off + k_s
        i_f = i_off + i_s
        W_full[k_f * n + i_f, :] = W_sub[idx, :]
    
    return U_full, V_full, W_full

def _block_decomposition_356():
    """
    Construct a block decomposition for 3x5x6 with rank 84.
    Split: n=[2,1], m=[2,2,1], p=[2,2,2]
    Use Strassen (rank 7) for 2x2x2 blocks, schoolbook for others.
    """
    n, m, p = 3, 5, 6
    
    n_splits = [2, 1]
    m_splits = [2, 2, 1]
    p_splits = [2, 2, 2]
    
    U_total = np.zeros((n * m, 0), dtype=np.float64)
    V_total = np.zeros((m * p, 0), dtype=np.float64)
    W_total = np.zeros((p * n, 0), dtype=np.float64)
    
    i_off = 0
    for i_s, n_sub in enumerate(n_splits):
        j_off = 0
        for j_s, m_sub in enumerate(m_splits):
            k_off = 0
            for k_s, p_sub in enumerate(p_splits):
                if n_sub == 2 and m_sub == 2 and p_sub == 2:
                    U_sub, V_sub, W_sub = _strassen_222()
                else:
                    U_sub, V_sub, W_sub = _schoolbook(n_sub, m_sub, p_sub)
                
                U_full, V_full, W_full = _embed_sub_decomp(
                    U_sub, V_sub, W_sub, n_sub, m_sub, p_sub,
                    i_off, j_off, k_off, n, m, p
                )
                
                if U_total.shape[1] == 0:
                    U_total = U_full
                    V_total = V_full
                    W_total = W_full
                else:
                    U_total = np.hstack([U_total, U_full])
                    V_total = np.hstack([V_total, V_full])
                    W_total = np.hstack([W_total, W_full])
                
                k_off += p_sub
            j_off += m_sub
        i_off += n_sub
    
    return U_total, V_total, W_total

def _sparse_alt_least_squares(n, m, p, rank, U_init, V_init, W_init, max_iters=200, deadline=None):
    """
    Sparse Alternating Least Squares for tensor decomposition.
    Tries to minimize the number of non-zero entries while maintaining
    the tensor decomposition property.
    """
    T_target = _build_target_tensor(n, m, p)
    nm, mp, pn = n * m, m * p, p * n
    
    U = U_init.copy()
    V = V_init.copy()
    W = W_init.copy()
    
    # Precompute the set of (a,b,c) indices where T_target is non-zero
    nonzero_indices = np.argwhere(T_target != 0)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline - 5:
            break
            
        # Update U
        for a in range(nm):
            # Solve for U[a, :] given V and W
            # T[a, b, c] = sum_k U[a,k] * V[b,k] * W[c,k]
            # For fixed a: T[a, :, :] = U[a, :] @ (V * W)^T in some sense
            # More precisely: T[a, b, c] = sum_k U[a,k] * (V[b,k] * W[c,k])
            # Let M[a, b, c] = V[b,k] * W[c,k] summed over k... no.
            # T[a, b, c] = sum_k U[a,k] * V[b,k] * W[c,k]
            # This is a trilinear form. For fixed a, we want:
            # sum_k U[a,k] * V[b,k] * W[c,k] = T[a, b, c] for all b, c
            # This is: U[a, :] @ (V * W^T) ... not quite.
            # Let's reshape: for fixed a, define vector t_a of length mp*pn
            # t_a[b*pn+c] = T[a, b, c]
            # And matrix A_a where A_a[b*pn+c, k] = V[b,k] * W[c,k]
            # Then t_a = A_a @ U[a, :]
            # So U[a, :] = A_a^+ @ t_a (pseudo-inverse)
            
            t_a = T_target[a, :, :].flatten()
            # Build A_a: shape (mp*pn, rank)
            # A_a[b*pn+c, k] = V[b,k] * W[c,k]
            VW = np.einsum('bk,ck->bck', V, W).reshape(mp * pn, rank)
            # Solve least squares: VW @ U[a,:] = t_a
            # Use lstsq
            try:
                U_new, residuals, rank_a, sv = np.linalg.lstsq(VW, t_a, rcond=None)
                U[a, :] = U_new
            except:
                pass
        
        # Update V
        for b in range(mp):
            t_b = T_target[:, b, :].reshape(nm, pn).T.flatten()  # shape (nm*pn,)
            # T[a, b, c] = sum_k U[a,k] * V[b,k] * W[c,k]
            # For fixed b: T[:, b, :] = V[b, :] @ (U * W) in some sense
            # t_b[a*pn+c] = T[a, b, c] = sum_k U[a,k] * W[c,k] * V[b,k]
            # A_b[a*pn+c, k] = U[a,k] * W[c,k]
            UW = np.einsum('ak,ck->ack', U, W).reshape(nm * pn, rank)
            try:
                V_new, residuals, rank_b, sv = np.linalg.lstsq(UW, t_b, rcond=None)
                V[b, :] = V_new
            except:
                pass
        
        # Update W
        for c in range(pn):
            t_c = T_target[:, :, c].reshape(nm, mp).T.flatten()  # shape (mp*nm,)
            # T[a, b, c] = sum_k U[a,k] * V[b,k] * W[c,k]
            # For fixed c: T[:, :, c] = W[c, :] @ (U * V) in some sense
            # t_c[b*nm+a] = T[a, b, c] = sum_k U[a,k] * V[b,k] * W[c,k]
            # A_c[b*nm+a, k] = V[b,k] * U[a,k]
            VU = np.einsum('bk,ak->bak', V, U).reshape(mp * nm, rank)
            try:
                W_new, residuals, rank_c, sv = np.linalg.lstsq(VU, t_c, rcond=None)
                W[c, :] = W_new
            except:
                pass
        
        # Sparsification step: zero out small entries
        # This encourages sparse solutions
        threshold = 1e-6
        U[np.abs(U) < threshold] = 0.0
        V[np.abs(V) < threshold] = 0.0
        W[np.abs(W) < threshold] = 0.0
        
        # Check convergence
        if it % 10 == 0:
            T_comp = np.einsum('ak,bk,ck->abc', U, V, W, optimize=True)
            err = np.max(np.abs(T_comp - T_target))
            if err < 1e-9:
                break
    
    return U, V, W

def _try_sparse_search(n, m, p, deadline, start_rank=50, end_rank=84):
    """
    Try to find a decomposition with rank < 84 using sparse ALS.
    Initialize with the block decomposition and try to reduce rank.
    """
    best = None
    best_rank = 84
    
    U_init, V_init, W_init = _block_decomposition_356()
    
    # Try ranks from start_rank to end_rank-1
    for r in range(start_rank, end_rank):
        if time.time() > deadline - 10:
            break
            
        # Initialize with random half-integer matrices
        rng = np.random.default_rng(r)
        vals = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
        
        for trial in range(5):  # 5 trials per rank
            if time.time() > deadline - 5:
                break
            
            U_rand = rng.choice(vals, size=(n * m, r))
            V_rand = rng.choice(vals, size=(m * p, r))
            W_rand = rng.choice(vals, size=(p * n, r))
            
            U_opt, V_opt, W_opt = _sparse_alt_least_squares(
                n, m, p, r, U_rand, V_rand, W_rand,
                max_iters=50, deadline=deadline
            )
            
            if _verify_decomp(n, m, p, U_opt, V_opt, W_opt):
                best = (U_opt, V_opt, W_opt)
                best_rank = r
                break  # Found rank r, stop
        
        if best_rank <= r:
            break
    
    return best, best_rank

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110.0  # 110s budget, 10s safety margin
    
    # Start with block decomposition (rank 84)
    U, V, W = _block_decomposition_356()
    
    if not _verify_decomp(n, m, p, U, V, W):
        # Fallback to schoolbook
        U, V, W = _schoolbook(n, m, p)
    
    best_rank = U.shape[1]
    
    # Try sparse search for lower rank
    if time.time() < deadline - 30:
        better, better_rank = _try_sparse_search(n, m, p, deadline, start_rank=50, end_rank=84)
        if better is not None and better_rank < best_rank:
            U, V, W = better
            best_rank = better_rank
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

# EVOLVE-BLOCK-END
