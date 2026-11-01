"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]
import time
import itertools
import random
import numpy as np

# EVOLVE-BLOCK-START
def _construct_rank23():
    """
    Construct the rank-23 decomposition for 3x3 matrix multiplication.
    
    Uses Strassen's 7-term algorithm for the 2x2 core (top-left),
    plus 16 terms for cross-interactions with the third row/column.
    
    Convention: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    so C[i,k] = sum_j A[i,j]*B[j,k] = sum_r U[i*3+j,r]*V[j*3+k,r]*W[k*3+i,r]
    """
    n, m, p = 3, 3, 3
    rank = 23
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Helper: set entries
    def set_entry(mat, row, col, val):
        mat[row, col] += val
    
    # Strassen 2x2 core (indices 0,1 for rows/cols)
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    r = 0
    set_entry(U, 0, r, 1)   # A[0,0]
    set_entry(U, 4, r, 1)   # A[1,1]
    set_entry(V, 0, r, 1)   # B[0,0]
    set_entry(V, 4, r, 1)   # B[1,1]
    set_entry(W, 0, r, 1)   # C[0,0]: k=0,i=0
    set_entry(W, 4, r, 1)   # C[1,1]: k=1,i=1
    
    # M2 = (A10+A11)(B00) -> C10 += M2
    r = 1
    set_entry(U, 3, r, 1)   # A[1,0]
    set_entry(U, 4, r, 1)   # A[1,1]
    set_entry(V, 0, r, 1)   # B[0,0]
    set_entry(W, 1, r, 1)   # C[1,0]: k=0,i=1
    
    # M3 = A00(B10-B11) -> C01 += M3
    r = 2
    set_entry(U, 0, r, 1)   # A[0,0]
    set_entry(V, 3, r, 1)   # B[1,0]
    set_entry(V, 4, r, -1)  # B[1,1]
    set_entry(W, 3, r, 1)   # C[0,1]: k=1,i=0
    
    # M4 = A11(B01-B00) -> C00 -= M4, C10 += M4
    r = 3
    set_entry(U, 4, r, 1)   # A[1,1]
    set_entry(V, 1, r, 1)   # B[0,1]
    set_entry(V, 0, r, -1)  # B[0,0]
    set_entry(W, 0, r, -1)  # C[0,0]: k=0,i=0
    set_entry(W, 1, r, 1)   # C[1,0]: k=0,i=1
    
    # M5 = (A00+A01)B11 -> C01 += M5, C11 += M5
    r = 4
    set_entry(U, 0, r, 1)   # A[0,0]
    set_entry(U, 1, r, 1)   # A[0,1]
    set_entry(V, 4, r, 1)   # B[1,1]
    set_entry(W, 3, r, 1)   # C[0,1]: k=1,i=0
    set_entry(W, 4, r, 1)   # C[1,1]: k=1,i=1
    
    # M6 = (A10-A00)(B00+B01) -> C00 += M6, C10 += M6
    r = 5
    set_entry(U, 3, r, 1)   # A[1,0]
    set_entry(U, 0, r, -1)  # A[0,0]
    set_entry(V, 0, r, 1)   # B[0,0]
    set_entry(V, 1, r, 1)   # B[0,1]
    set_entry(W, 0, r, 1)   # C[0,0]: k=0,i=0
    set_entry(W, 1, r, 1)   # C[1,0]: k=0,i=1
    
    # M7 = (A01-A11)(B10+B11) -> C00 += M7, C11 -= M7
    r = 6
    set_entry(U, 1, r, 1)   # A[0,1]
    set_entry(U, 4, r, -1)  # A[1,1]
    set_entry(V, 3, r, 1)   # B[1,0]
    set_entry(V, 4, r, 1)   # B[1,1]
    set_entry(W, 0, r, 1)   # C[0,0]: k=0,i=0
    set_entry(W, 4, r, -1)  # C[1,1]: k=1,i=1
    
    # Now the 16 cross terms for interactions with row/col 2
    
    # Row 0 of C (i=0), interacting with B column 2 (k=2): C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
    # We need 3 terms for these
    
    # Term 7: A[0,0]*B[0,2] -> C[0,2]
    r = 7
    set_entry(U, 0, r, 1)   # A[0,0]
    set_entry(V, 2, r, 1)   # B[0,2]
    set_entry(W, 6, r, 1)   # C[0,2]: k=2,i=0
    
    # Term 8: A[0,1]*B[1,2] -> C[0,2]
    r = 8
    set_entry(U, 1, r, 1)   # A[0,1]
    set_entry(V, 5, r, 1)   # B[1,2]
    set_entry(W, 6, r, 1)   # C[0,2]: k=2,i=0
    
    # Term 9: A[0,2]*B[2,2] -> C[0,2]
    r = 9
    set_entry(U, 2, r, 1)   # A[0,2]
    set_entry(V, 8, r, 1)   # B[2,2]
    set_entry(W, 6, r, 1)   # C[0,2]: k=2,i=0
    
    # Row 1 of C (i=1), interacting with B column 2 (k=2): C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2]
    
    # Term 10: A[1,0]*B[0,2] -> C[1,2]
    r = 10
    set_entry(U, 3, r, 1)   # A[1,0]
    set_entry(V, 2, r, 1)   # B[0,2]
    set_entry(W, 7, r, 1)   # C[1,2]: k=2,i=1
    
    # Term 11: A[1,1]*B[1,2] -> C[1,2]
    r = 11
    set_entry(U, 4, r, 1)   # A[1,1]
    set_entry(V, 5, r, 1)   # B[1,2]
    set_entry(W, 7, r, 1)   # C[1,2]: k=2,i=1
    
    # Term 12: A[1,2]*B[2,2] -> C[1,2]
    r = 12
    set_entry(U, 5, r, 1)   # A[1,2]
    set_entry(V, 8, r, 1)   # B[2,2]
    set_entry(W, 7, r, 1)   # C[1,2]: k=2,i=1
    
    # Row 2 of C (i=2), interacting with B column 0 (k=0): C[2,0] = A[2,0]*B[0,0] + A[2,1]*B[1,0] + A[2,2]*B[2,0]
    
    # Term 13: A[2,0]*B[0,0] -> C[2,0]
    r = 13
    set_entry(U, 6, r, 1)   # A[2,0]
    set_entry(V, 0, r, 1)   # B[0,0]
    set_entry(W, 2, r, 1)   # C[2,0]: k=0,i=2
    
    # Term 14: A[2,1]*B[1,0] -> C[2,0]
    r = 14
    set_entry(U, 7, r, 1)   # A[2,1]
    set_entry(V, 3, r, 1)   # B[1,0]
    set_entry(W, 2, r, 1)   # C[2,0]: k=0,i=2
    
    # Term 15: A[2,2]*B[2,0] -> C[2,0]
    r = 15
    set_entry(U, 8, r, 1)   # A[2,2]
    set_entry(V, 6, r, 1)   # B[2,0]
    set_entry(W, 2, r, 1)   # C[2,0]: k=0,i=2
    
    # Row 2 of C (i=2), interacting with B column 1 (k=1): C[2,1] = A[2,0]*B[0,1] + A[2,1]*B[1,1] + A[2,2]*B[2,1]
    
    # Term 16: A[2,0]*B[0,1] -> C[2,1]
    r = 16
    set_entry(U, 6, r, 1)   # A[2,0]
    set_entry(V, 1, r, 1)   # B[0,1]
    set_entry(W, 5, r, 1)   # C[2,1]: k=1,i=2
    
    # Term 17: A[2,1]*B[1,1] -> C[2,1]
    r = 17
    set_entry(U, 7, r, 1)   # A[2,1]
    set_entry(V, 4, r, 1)   # B[1,1]
    set_entry(W, 5, r, 1)   # C[2,1]: k=1,i=2
    
    # Term 18: A[2,2]*B[2,1] -> C[2,1]
    r = 18
    set_entry(U, 8, r, 1)   # A[2,2]
    set_entry(V, 7, r, 1)   # B[2,1]
    set_entry(W, 5, r, 1)   # C[2,1]: k=1,i=2
    
    # Row 0 of C (i=0), interacting with B column 0 (k=0): already covered by Strassen
    # Row 0 of C (i=0), interacting with B column 1 (k=1): already covered by Strassen
    
    # Row 2 of C (i=2), interacting with B column 2 (k=2): C[2,2] = A[2,0]*B[0,2] + A[2,1]*B[1,2] + A[2,2]*B[2,2]
    
    # Term 19: A[2,0]*B[0,2] -> C[2,2]
    r = 19
    set_entry(U, 6, r, 1)   # A[2,0]
    set_entry(V, 2, r, 1)   # B[0,2]
    set_entry(W, 8, r, 1)   # C[2,2]: k=2,i=2
    
    # Term 20: A[2,1]*B[1,2] -> C[2,2]
    r = 20
    set_entry(U, 7, r, 1)   # A[2,1]
    set_entry(V, 5, r, 1)   # B[1,2]
    set_entry(W, 8, r, 1)   # C[2,2]: k=2,i=2
    
    # Term 21: A[2,2]*B[2,2] -> C[2,2]
    r = 21
    set_entry(U, 8, r, 1)   # A[2,2]
    set_entry(V, 8, r, 1)   # B[2,2]
    set_entry(W, 8, r, 1)   # C[2,2]: k=2,i=2
    
    # Now we need to handle: C[0,2] has A[0,0]*B[0,2], A[0,1]*B[1,2], A[0,2]*B[2,2] - done (terms 7,8,9)
    # C[1,2] has A[1,0]*B[0,2], A[1,1]*B[1,2], A[1,2]*B[2,2] - done (terms 10,11,12)
    # C[2,0] has A[2,0]*B[0,0], A[2,1]*B[1,0], A[2,2]*B[2,0] - done (terms 13,14,15)
    # C[2,1] has A[2,0]*B[0,1], A[2,1]*B[1,1], A[2,2]*B[2,1] - done (terms 16,17,18)
    # C[2,2] has A[2,0]*B[0,2], A[2,1]*B[1,2], A[2,2]*B[2,2] - done (terms 19,20,21)
    
    # But wait - we also need to handle the "missing" interactions:
    # The Strassen terms only cover C[0,0], C[0,1], C[1,0], C[1,1] for j in {0,1}
    # We need C[0,2], C[1,2], C[2,0], C[2,1], C[2,2] which involve j=2 or k=2 or i=2
    
    # Actually, let me reconsider. The Strassen terms handle:
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]  (j=0,1 only, missing A[0,2]*B[2,0])
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]  (j=0,1 only, missing A[0,2]*B[2,1])
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]  (j=0,1 only, missing A[1,2]*B[2,0])
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]  (j=0,1 only, missing A[1,2]*B[2,1])
    
    # So we ALSO need:
    # C[0,0] += A[0,2]*B[2,0]
    # C[0,1] += A[0,2]*B[2,1]
    # C[1,0] += A[1,2]*B[2,0]
    # C[1,1] += A[1,2]*B[2,1]
    
    # Term 22: A[0,2]*B[2,0] -> C[0,0]
    r = 22
    set_entry(U, 2, r, 1)   # A[0,2]
    set_entry(V, 6, r, 1)   # B[2,0]
    set_entry(W, 0, r, 1)   # C[0,0]: k=0,i=0
    
    # Term 23: A[0,2]*B[2,1] -> C[0,1]
    r = 23  # Wait, this would be the 24th term (index 23)
    
    # Hmm, I have 24 terms here. Let me recount.
    # Terms 0-6: Strassen (7 terms)
    # Terms 7-9: C[0,2] (3 terms)
    # Terms 10-12: C[1,2] (3 terms)
    # Terms 13-15: C[2,0] (3 terms)
    # Terms 16-18: C[2,1] (3 terms)
    # Terms 19-21: C[2,2] (3 terms)
    # Terms 22-23: C[0,0] and C[0,1] extra (2 terms)
    # Total: 7+3+3+3+3+3+2 = 24 terms
    
    # That's 24, not 23. I need to reduce by 1.
    
    # Actually, I realize I can combine some terms. Let me reconsider.
    
    # The issue is that I'm using separate terms for each (i,j,k) triple in the cross terms.
    # But some of these can be combined.
    
    # Let me restart with a cleaner approach.
    # 
    # Total needed terms: 27 (one for each i,j,k)
    # Strassen saves 4 terms in the 2x2 core (27 - 4 = 23 if we could save 4 more elsewhere)
    # But Strassen only applies to the 2x2 core, giving 7 terms instead of 4*2=8... 
    # Wait, the 2x2 core has 2*2*2 = 8 schoolbook terms, Strassen uses 7, saving 1.
    # The remaining 19 terms (27-8) are for the cross interactions.
    # 7 + 19 = 26, not 23.
    
    # So I need to save 3 more terms in the cross interactions.
    
    # Let me think about this differently. The known rank is 23, so there must be a way.
    
    # One approach: use the fact that some cross terms can be combined with Strassen terms.
    
    # Actually, I think the correct construction is more subtle. Let me use a numerical approach
    # to find the rank-23 decomposition, starting from the schoolbook and using ALS.
    
    return None  # Signal to use numerical approach


def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify the decomposition."""
    r = U.shape[1]
    T = np.einsum('ar,br,cr->abc', U, V, W)
    T_exp = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_exp[i*m+j, j*p+k, k*n+i] = 1.0
    return np.allclose(T, T_exp, atol=tol, rtol=0)


def _als_decompose(T, rank, n, m, p, max_iters=200, tol=1e-10, init_U=None, init_V=None, init_W=None, deadline=None):
    """ALS for tensor decomposition."""
    dim_a, dim_b, dim_c = n*m, m*p, p*n
    
    if init_U is None:
        U = np.random.randn(dim_a, rank)
        V = np.random.randn(dim_b, rank)
        W = np.random.randn(dim_c, rank)
    else:
        U = init_U.copy()
        V = init_V.copy()
        W = init_W.copy()
        
    for r in range(rank):
        nu = np.linalg.norm(U[:, r])
        nv = np.linalg.norm(V[:, r])
        nw = np.linalg.norm(W[:, r])
        if nu > 1e-10 and nv > 1e-10 and nw > 1e-10:
            U[:, r] /= nu
            V[:, r] /= nv
            W[:, r] /= nw
            
    prev_error = float('inf')
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            return None
            
        M = np.einsum('br,cr->bcr', V, W).reshape(dim_b * dim_c, rank)
        for a in range(dim_a):
            T_flat = T[a, :, :].reshape(-1)
            u_a, _, _, _ = np.linalg.lstsq(M, T_flat, rcond=None)
            U[a, :] = u_a
            
        M = np.einsum('ar,cr->acr', U, W).reshape(dim_a * dim_c, rank)
        for b in range(dim_b):
            T_flat = T[:, b, :].reshape(-1)
            v_b, _, _, _ = np.linalg.lstsq(M, T_flat, rcond=None)
            V[b, :] = v_b
            
        M = np.einsum('ar,br->abr', U, V).reshape(dim_a * dim_b, rank)
        for c in range(dim_c):
            T_flat = T[:, :, c].reshape(-1)
            w_c, _, _, _ = np.linalg.lstsq(M, T_flat, rcond=None)
            W[c, :] = w_c
            
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        error = np.linalg.norm(T_approx - T)
        
        if abs(prev_error - error) < tol:
            if error < 1e-7:
                return U, V, W
            else:
                return None
        prev_error = error
        
    T_approx = np.einsum('ar,br,cr->abc', U, V, W)
    error = np.linalg.norm(T_approx - T)
    if error < 1e-7:
        return U, V, W
    return None


def _try_rank(target_rank, n, m, p, deadline, max_attempts=50, seed=42):
    """Try to find a decomposition of given rank using ALS."""
    dim_a, dim_b, dim_c = n*m, m*p, p*n
    T = np.zeros((dim_a, dim_b, dim_c))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
                
    U0, V0, W0 = _schoolbook(n, m, p)
    total_terms = n*m*p
    n_drop = total_terms - target_rank
    
    if n_drop <= 0:
        return U0, V0, W0
    
    all_indices = list(range(total_terms))
    rng = random.Random(seed)
    
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
            
        drop_indices = rng.sample(all_indices, n_drop)
        keep_indices = [i for i in all_indices if i not in drop_indices]
        
        init_U = U0[:, keep_indices].copy()
        init_V = V0[:, keep_indices].copy()
        init_W = W0[:, keep_indices].copy()
        
        perturb_scale = rng.uniform(0.05, 0.3)
        for r_idx in range(len(keep_indices)):
            mask_u = init_U[:, r_idx] != 0
            mask_v = init_V[:, r_idx] != 0
            mask_w = init_W[:, r_idx] != 0
            if mask_u.any():
                init_U[mask_u, r_idx] += rng.uniform(-perturb_scale, perturb_scale)
            if mask_v.any():
                init_V[mask_v, r_idx] += rng.uniform(-perturb_scale, perturb_scale)
            if mask_w.any():
                init_W[mask_w, r_idx] += rng.uniform(-perturb_scale, perturb_scale)
            
        result = _als_decompose(T, target_rank, n, m, p, max_iters=200, tol=1e-10,
                                init_U=init_U, init_V=init_V, init_W=init_W,
                                deadline=deadline)
        
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, n, m, p, tol=1e-8):
                return U, V, W
                
    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0
    
    # Try rank 23 first
    for target_rank in [23, 24, 25, 26]:
        if time.time() > deadline:
            break
        result = _try_rank(target_rank, n, m, p, deadline, max_attempts=40, seed=42)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
