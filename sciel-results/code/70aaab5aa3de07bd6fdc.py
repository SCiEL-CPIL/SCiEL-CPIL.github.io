import numpy as np
import time

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _round_to_half_integer(x):
    """Round x to the nearest half-integer."""
    return round(x * 2) / 2.0

def _round_matrix_to_half_int(M):
    """Round all entries of matrix M to nearest half-integer."""
    return np.array([[_round_to_half_integer(x) for x in row] for row in M])

def _strassen_construct(n, m, p):
    """Construct the Strassen-based decomposition (rank 72 for 4x4x5)."""
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u, v, w):
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
        
    def get_u_idx(gi, gj):
        return gi * m + gj
        
    def get_v_idx(gj, gk):
        return gj * p + gk
        
    def get_w_idx(gk, gi):
        return gk * n + gi

    def add_strassen_2x2_2(ai, aj, bj, bk):
        # M1 = (a+d)(e+h)
        u = np.zeros(n*m); u[get_u_idx(ai, aj)] = 1; u[get_u_idx(ai+1, aj+1)] = 1
        v = np.zeros(m*p); v[get_v_idx(bj, bk)] = 1; v[get_v_idx(bj+1, bk+1)] = 1
        w = np.zeros(p*n); w[get_w_idx(bk, ai)] = 1; w[get_w_idx(bk+1, ai+1)] = 1
        add_term(u, v, w)
        
        # M2 = (c+d)e
        u = np.zeros(n*m); u[get_u_idx(ai+1, aj)] = 1; u[get_u_idx(ai+1, aj+1)] = 1
        v = np.zeros(m*p); v[get_v_idx(bj, bk)] = 1
        w = np.zeros(p*n); w[get_w_idx(bk, ai+1)] = 1; w[get_w_idx(bk+1, ai+1)] = -1
        add_term(u, v, w)
        
        # M3 = a(f-h)
        u = np.zeros(n*m); u[get_u_idx(ai, aj)] = 1
        v = np.zeros(m*p); v[get_v_idx(bj, bk+1)] = 1; v[get_v_idx(bj+1, bk+1)] = -1
        w = np.zeros(p*n); w[get_w_idx(bk+1, ai)] = 1; w[get_w_idx(bk+1, ai+1)] = 1
        add_term(u, v, w)
        
        # M4 = d(g-e)
        u = np.zeros(n*m); u[get_u_idx(ai+1, aj+1)] = 1
        v = np.zeros(m*p); v[get_v_idx(bj+1, bk)] = 1; v[get_v_idx(bj, bk)] = -1
        w = np.zeros(p*n); w[get_w_idx(bk, ai)] = 1; w[get_w_idx(bk+1, ai)] = 1
        add_term(u, v, w)
        
        # M5 = (a+b)h
        u = np.zeros(n*m); u[get_u_idx(ai, aj)] = 1; u[get_u_idx(ai, aj+1)] = 1
        v = np.zeros(m*p); v[get_v_idx(bj+1, bk+1)] = 1
        w = np.zeros(p*n); w[get_w_idx(bk, ai)] = -1; w[get_w_idx(bk+1, ai)] = 1
        add_term(u, v, w)
        
        # M6 = (c-a)(e+f)
        u = np.zeros(n*m); u[get_u_idx(ai+1, aj)] = 1; u[get_u_idx(ai, aj)] = -1
        v = np.zeros(m*p); v[get_v_idx(bj, bk)] = 1; v[get_v_idx(bj, bk+1)] = 1
        w = np.zeros(p*n); w[get_w_idx(bk+1, ai+1)] = 1
        add_term(u, v, w)
        
        # M7 = (b-d)(g+h)
        u = np.zeros(n*m); u[get_u_idx(ai, aj+1)] = 1; u[get_u_idx(ai+1, aj+1)] = -1
        v = np.zeros(m*p); v[get_v_idx(bj+1, bk)] = 1; v[get_v_idx(bj+1, bk+1)] = 1
        w = np.zeros(p*n); w[get_w_idx(bk, ai)] = 1
        add_term(u, v, w)

    def add_2x2x3(ai, aj, bj, bk):
        add_strassen_2x2_2(ai, aj, bj, bk)
        # Naive for col bk+2
        for di in range(2):
            for dj in range(2):
                u = np.zeros(n*m); u[get_u_idx(ai+di, aj+dj)] = 1
                v = np.zeros(m*p); v[get_v_idx(bj+dj, bk+2)] = 1
                w = np.zeros(p*n); w[get_w_idx(bk+2, ai+di)] = 1
                add_term(u, v, w)

    # C00 = A00 B00 + A01 B10
    add_strassen_2x2_2(0, 0, 0, 0)
    add_strassen_2x2_2(0, 2, 2, 0)
    # C01 = A00 B01 + A01 B11
    add_2x2x3(0, 0, 0, 2)
    add_2x2x3(0, 2, 2, 2)
    # C10 = A10 B00 + A11 B10
    add_strassen_2x2_2(2, 0, 0, 0)
    add_strassen_2x2_2(2, 2, 2, 0)
    # C11 = A10 B01 + A11 B11
    add_2x2x3(2, 0, 0, 2)
    add_2x2x3(2, 2, 2, 2)

    U_mat = np.array(U_cols).T
    V_mat = np.array(V_cols).T
    W_mat = np.array(W_cols).T
    return U_mat.tolist(), V_mat.tolist(), W_mat.tolist()

def _combinatorial_pruning(U, V, W, n, m, p, tol=1e-9):
    """
    Attempt to reduce rank by finding linear dependencies among the rank-1 terms.
    For each term r, check if the vector (U[:,r] ⊗ V[:,r] ⊗ W[:,r]) is in the span
    of the other terms. If so, we can potentially remove it.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    if rank <= 1:
        return U, V, W
    
    # Build the tensor flattening: each term r contributes a vector of size n*m*p
    # where entry (i,j,k) is U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    nm = n * m
    mp = m * p
    pn = p * n
    
    # For each term, compute its contribution vector
    term_vectors = []
    for r in range(rank):
        vec = np.zeros(n * m * p)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    vec[i * m * p + j * p + k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        term_vectors.append(vec)
    
    term_vectors = np.array(term_vectors)  # shape (rank, n*m*p)
    
    # We want to find a subset of terms that still spans the required tensor.
    # The required tensor has 1 at each (i,j,k) position.
    target = np.zeros(n * m * p)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i * m * p + j * p + k] = 1.0
    
    # Greedy approach: try to remove terms one by one
    # For each term, check if it's linearly dependent on the others
    keep = list(range(rank))
    
    for r in range(rank):
        if r not in keep:
            continue
        if len(keep) <= 1:
            break
        
        # Check if term r is in the span of the other kept terms
        others = [k for k in keep if k != r]
        A = term_vectors[others]  # shape (len(others), n*m*p)
        b = term_vectors[r]
        
        # Check if b is in the column space of A (treating A's rows as vectors)
        # i.e., is b in the row space of A?
        # Use SVD to check
        try:
            U_svd, s, Vt = np.linalg.svd(A, full_matrices=False)
            # Project b onto the row space of A
            if len(s) > 0:
                # Row space basis is Vt rows with non-zero singular values
                rank_A = np.sum(s > tol)
                if rank_A > 0:
                    proj = (b @ Vt[:rank_A].T) @ Vt[:rank_A]
                    if np.linalg.norm(b - proj) < tol:
                        # b is in the span of others, remove it
                        keep.remove(r)
        except:
            pass
    
    if len(keep) < rank:
        U_new = U[:, keep]
        V_new = V[:, keep]
        W_new = W[:, keep]
        if _verify_decomposition(U_new.tolist(), V_new.tolist(), W_new.tolist(), n, m, p, tol):
            return U_new, V_new, W_new
    
    return U, V, W

def _als_decompose(n, m, p, rank, seed=42, max_iters=500):
    """Try to find a rank-r decomposition using alternating least squares."""
    np.random.seed(seed)
    
    # Initialize with small random values
    U = np.random.randn(n*m, rank) * 0.1
    V = np.random.randn(m*p, rank) * 0.1
    W = np.random.randn(p*n, rank) * 0.1
    
    for it in range(max_iters):
        # Optimize U
        for i in range(n):
            for j in range(m):
                A_mat = np.array([V[j*p+k, :] * W[k*n+i, :] for k in range(p)])
                b_vec = np.ones(p)
                try:
                    U[i*m+j, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
        
        # Optimize V
        for j in range(m):
            for k in range(p):
                A_mat = np.array([U[i*m+j, :] * W[k*n+i, :] for i in range(n)])
                b_vec = np.ones(n)
                try:
                    V[j*p+k, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
        
        # Optimize W
        for i in range(n):
            for k in range(p):
                A_mat = np.array([U[i*m+j, :] * V[j*p+k, :] for j in range(m)])
                b_vec = np.ones(m)
                try:
                    W[k*n+i, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
    
    # Round to half-integers
    U_r = _round_matrix_to_half_int(U)
    V_r = _round_matrix_to_half_int(V)
    W_r = _round_matrix_to_half_int(W)
    
    # Verify
    if _verify_decomposition(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p):
        return U_r.tolist(), V_r.tolist(), W_r.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Get the Strassen-based decomposition as baseline (rank 72)
    U_base, V_base, W_base = _strassen_construct(n, m, p)
    best_result = {"U": U_base, "V": V_base, "W": W_base}
    best_rank = len(U_base[0])
    
    # Try combinatorial pruning on the Strassen decomposition
    if time.time() < deadline - 60:
        U_pruned, V_pruned, W_pruned = _combinatorial_pruning(
            np.array(U_base), np.array(V_base), np.array(W_base), n, m, p
        )
        if U_pruned.shape[1] < best_rank:
            best_result = {"U": U_pruned.tolist(), "V": V_pruned.tolist(), "W": W_pruned.tolist()}
            best_rank = U_pruned.shape[1]
    
    # Try to find lower-rank decompositions using ALS
    # Focus on the likely optimal range (39-50)
    for rank in range(39, best_rank):
        if time.time() > deadline - 30:
            break
        
        found = False
        for seed in range(10):
            if time.time() > deadline - 30:
                break
            
            result = _als_decompose(n, m, p, rank, seed=seed, max_iters=300)
            if result is not None:
                U, V, W = result
                # Try pruning on the ALS result too
                U_arr, V_arr, W_arr = np.array(U), np.array(V), np.array(W)
                U_pruned, V_pruned, W_pruned = _combinatorial_pruning(
                    U_arr, V_arr, W_arr, n, m, p
                )
                if U_pruned.shape[1] < rank:
                    best_result = {"U": U_pruned.tolist(), "V": V_pruned.tolist(), "W": W_pruned.tolist()}
                    best_rank = U_pruned.shape[1]
                else:
                    best_result = {"U": U, "V": V, "W": W}
                    best_rank = rank
                found = True
                break
        
        if found:
            continue
    
    return best_result
# EVOLVE-BLOCK-END
