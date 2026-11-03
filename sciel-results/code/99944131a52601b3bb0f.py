"""
Combined approach:
1. Deterministic Rank-65 Construction via Kronecker-embedded Strassen (4x4x4) + Naive (4x4x1).
2. Attempt rank reduction via ALS initialized from the Rank-65 structure.
"""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START

def _verify_fast(U, V, W, n=4, m=4, p=5, atol=1e-9):
    """Fast verification using numpy with float64."""
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    rank = Ua.shape[1]
    if Ua.shape[0] != n*m or Va.shape[0] != m*p or Wa.shape[0] != p*n:
        return False
    
    Ur = Ua.reshape(n, m, rank)
    Vr = Va.reshape(m, p, rank)
    Wr = Wa.reshape(p, n, rank)
    
    T = np.zeros((n, m, p), dtype=np.float64)
    # Efficient accumulation
    # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # We can loop over r and add outer products
    for c in range(rank):
        uc = Ur[:, :, c]
        vc = Vr[:, :, c]
        wc = Wr[:, :, c]
        # For each j, T[:, j, :] += outer(U[:, j, c], V[j, :, c]) * W[:, j, c] (broadcasting W along i? No, W is k,i)
        # Let's do it explicitly but vectorized over i, k for fixed j
        for j in range(m):
            # uc[:, j] is vector of size n
            # vc[j, :] is vector of size p
            # wc[:, j] is vector of size p (since Wr is p x n, Wr[k, i], so Wr[:, j] is k-axis for fixed i=j? No, W index is k*n+i. 
            # Wait, W is shape (p*n, rank). Reshaped to (p, n, rank).
            # So Wr[k, i, r].
            # We need W[k, i] for fixed i? No, the term is U[i,j,r] * V[j,k,r] * W[k,i,r].
            # For fixed j and r:
            # T[i, j, k] += U[i, j, r] * V[j, k, r] * W[k, i, r]
            # Let u_vec = U[:, j, r] (size n)
            # Let v_vec = V[j, :, r] (size p)
            # Let w_mat = W[:, j, r] (size p) -- WAIT. W is (p, n, r). So W[:, i, r] is size p.
            # The index in W is k*n + i. So W[k, i].
            # So for fixed i, W[:, i, r] is the vector over k.
            # This structure is tricky to vectorize simply without einsum.
            # Let's use einsum for speed if available, or just loop i.
            pass
    
    # Simpler verification for correctness check (slower but safe)
    # T = np.einsum('ijr,jkr,kir->ijk', Ur, Vr, Wr)
    # return np.allclose(T, 1.0, atol=atol)
    
    # Manual loop for compatibility and clarity, optimized slightly
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += Ur[i, j, r] * Vr[j, k, r] * Wr[k, i, r]
                if abs(val - 1.0) > atol:
                    return False
    return True

def _round_to_half_integer(x):
    return round(x * 2) / 2.0

def _round_matrix_to_half_int(M):
    """M is a numpy array or list of lists. Returns list of lists of floats."""
    if isinstance(M, np.ndarray):
        M = M.tolist()
    return [[_round_to_half_integer(x) for x in row] for row in M]

def _strassen_2x2x2_terms():
    """
    Returns 7 terms for 2x2x2 Strassen.
    Each term is (u_sparse, v_sparse, w_sparse) where sparse is list of (index, value).
    U index: i*2+j
    V index: j*2+k
    W index: k*2+i
    """
    return [
        # M1 = (a+d)(e+h)
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),
        # M2 = (c+d)e
        ([(2,1),(3,1)], [(0,1)], [(1,1),(3,-1)]),
        # M3 = a(f-h)
        ([(0,1)], [(1,1),(3,-1)], [(2,1),(3,1)]),
        # M4 = d(g-e)
        ([(3,1)], [(2,1),(0,-1)], [(0,1),(1,1)]),
        # M5 = (a+b)h
        ([(0,1),(1,1)], [(3,1)], [(0,-1),(2,1)]),
        # M6 = (c-a)(e+f)
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),
        # M7 = (b-d)(g+h)
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)]),
    ]

def _construct_rank_65(n=4, m=4, p=5):
    """
    Constructs a rank-65 decomposition.
    49 terms from 4x4x4 Kronecker Strassen.
    16 terms from 4x4x1 Naive.
    """
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u, v, w):
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
    
    def get_u_idx(i, j): return i * m + j
    def get_v_idx(j, k): return j * p + k
    def get_w_idx(k, i): return k * n + i

    strassen_terms = _strassen_2x2x2_terms()
    
    # 1. 4x4x4 part (k=0..3)
    # Kronecker product of two 2x2x2 Strassen decompositions
    for (u1_sparse, v1_sparse, w1_sparse) in strassen_terms:
        for (u2_sparse, v2_sparse, w2_sparse) in strassen_terms:
            u = np.zeros(n*m)
            v = np.zeros(m*p)
            w = np.zeros(p*n)
            
            # U: i = i1*2 + i2, j = j1*2 + j2
            for (idx1, val1) in u1_sparse:
                i1, j1 = idx1 // 2, idx1 % 2
                for (idx2, val2) in u2_sparse:
                    i2, j2 = idx2 // 2, idx2 % 2
                    i = i1 * 2 + i2
                    j = j1 * 2 + j2
                    u[get_u_idx(i, j)] = val1 * val2
            
            # V: j = j1*2 + j2, k = k1*2 + k2
            for (idx1, val1) in v1_sparse:
                j1, k1 = idx1 // 2, idx1 % 2
                for (idx2, val2) in v2_sparse:
                    j2, k2 = idx2 // 2, idx2 % 2
                    j = j1 * 2 + j2
                    k = k1 * 2 + k2
                    v[get_v_idx(j, k)] = val1 * val2
            
            # W: k = k1*2 + k2, i = i1*2 + i2
            for (idx1, val1) in w1_sparse:
                k1, i1 = idx1 // 2, idx1 % 2
                for (idx2, val2) in w2_sparse:
                    k2, i2 = idx2 // 2, idx2 % 2
                    k = k1 * 2 + k2
                    i = i1 * 2 + i2
                    w[get_w_idx(k, i)] = val1 * val2
            
            add_term(u, v, w)
            
    # 2. 4x4x1 part (k=4)
    # Naive: A[i,j] * B[j,4] -> C[4,i]
    for i in range(n):
        for j in range(m):
            u = np.zeros(n*m); u[get_u_idx(i, j)] = 1
            v = np.zeros(m*p); v[get_v_idx(j, 4)] = 1
            w = np.zeros(p*n); w[get_w_idx(4, i)] = 1
            add_term(u, v, w)
    
    U_mat = np.array(U_cols).T
    V_mat = np.array(V_cols).T
    W_mat = np.array(W_cols).T
    return U_mat.tolist(), V_mat.tolist(), W_mat.tolist()

def _als_decompress(n, m, p, U_init, V_init, W_init, target_rank, max_iters=200, deadline=None, seed=42):
    """
    Attempt to reduce rank by optimizing U, V, W initialized from a higher rank decomposition.
    We project the initial high-rank solution onto a lower-rank subspace or just use it as init
    if dimensions match, otherwise we need a smarter init.
    
    For now, if target_rank == init_rank, we just polish.
    If target_rank < init_rank, we can try random projection of the init factors.
    """
    rng = np.random.RandomState(seed)
    
    U0 = np.array(U_init, dtype=float)
    V0 = np.array(V_init, dtype=float)
    W0 = np.array(W_init, dtype=float)
    init_rank = U0.shape[1]
    
    if target_rank > init_rank:
        # Pad with zeros
        U = np.zeros((n*m, target_rank))
        V = np.zeros((m*p, target_rank))
        W = np.zeros((p*n, target_rank))
        U[:, :init_rank] = U0
        V[:, :init_rank] = V0
        W[:, :init_rank] = W0
    elif target_rank == init_rank:
        U = U0.copy()
        V = V0.copy()
        W = W0.copy()
    else:
        # Random projection from init_rank to target_rank
        # U_new = U0 @ P_U
        P_U = rng.randn(init_rank, target_rank)
        P_V = rng.randn(init_rank, target_rank)
        P_W = rng.randn(init_rank, target_rank)
        # This changes the basis, so we need to re-optimize
        U = U0 @ P_U
        V = V0 @ P_V
        W = W0 @ P_W
        
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
            
        # Update U
        for i in range(n):
            for j in range(m):
                num = np.zeros(target_rank)
                den = np.zeros(target_rank)
                for k in range(p):
                    num += V[j*p+k, :] * W[k*n+i, :]
                    den += V[j*p+k, :]**2 * W[k*n+i, :]**2
                mask = den > 1e-10
                if np.any(mask):
                    U[i*m+j, mask] = num[mask] / den[mask]
        
        # Update V
        for j in range(m):
            for k in range(p):
                num = np.zeros(target_rank)
                den = np.zeros(target_rank)
                for i in range(n):
                    num += U[i*m+j, :] * W[k*n+i, :]
                    den += U[i*m+j, :]**2 * W[k*n+i, :]**2
                mask = den > 1e-10
                if np.any(mask):
                    V[j*p+k, mask] = num[mask] / den[mask]
        
        # Update W
        for k in range(p):
            for i in range(n):
                num = np.zeros(target_rank)
                den = np.zeros(target_rank)
                for j in range(m):
                    num += U[i*m+j, :] * V[j*p+k, :]
                    den += U[i*m+j, :]**2 * V[j*p+k, :]**2
                mask = den > 1e-10
                if np.any(mask):
                    W[k*n+i, mask] = num[mask] / den[mask]
                    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110 # Safety margin
    
    # 1. Generate Rank 65 Baseline
    U65, V65, W65 = _construct_rank_65(n, m, p)
    
    # Verify baseline (should be exact integers/half-integers)
    if not _verify_fast(U65, V65, W65, n, m, p):
        # Fallback to schoolbook if construction is buggy (should not happen)
        rank_schoolbook = n * m * p
        U_school = np.zeros((n*m, rank_schoolbook))
        V_school = np.zeros((m*p, rank_schoolbook))
        W_school = np.zeros((p*n, rank_schoolbook))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i*m+j, col] = 1
                    V_school[j*p+k, col] = 1
                    W_school[k*n+i, col] = 1
                    col += 1
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    best_U, best_V, best_W = U65, V65, W65
    best_rank = 65
    
    # 2. Attempt to reduce rank
    # Try ranks 64 down to 50
    for target_rank in range(64, 49, -1):
        if time.time() > deadline:
            break
            
        # Try a few seeds for random projection if needed
        for seed in range(3):
            if time.time() > deadline:
                break
                
            # Use rank 65 as init
            U_opt, V_opt, W_opt = _als_decompress(
                n, m, p, U65, V65, W65, target_rank, 
                max_iters=100, deadline=deadline, seed=seed
            )
            
            # Round to half-integers
            U_r = _round_matrix_to_half_int(U_opt)
            V_r = _round_matrix_to_half_int(V_opt)
            W_r = _round_matrix_to_half_int(W_opt)
            
            if _verify_fast(U_r, V_r, W_r, n, m, p):
                best_U, best_V, best_W = U_r, V_r, W_r
                best_rank = target_rank
                break # Found a valid decomposition for this rank, try lower
                
    return {"U": best_U, "V": best_V, "W": best_W}

# EVOLVE-BLOCK-END
