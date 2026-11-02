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
    return np.array([[ _round_to_half_integer(x) for x in row] for row in M])

def _strassen_construct(n, m, p):
    """Construct the Strassen-based decomposition (rank 64 for 4x4x5)."""
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
        w = np.zeros(p*n); w[get_w_idx(bk, ai)] = 1; w[get_w_idx(bk, ai+1)] = 1
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
    return U_mat, V_mat, W_mat

def _remove_zero_columns(U, V, W):
    """Remove columns that are all zeros in any of U, V, or W."""
    rank = U.shape[1]
    keep = []
    for r in range(rank):
        if np.any(U[:, r] != 0) and np.any(V[:, r] != 0) and np.any(W[:, r] != 0):
            keep.append(r)
    U = U[:, keep]
    V = V[:, keep]
    W = W[:, keep]
    return U, V, W

def _eliminate_dependent_terms(U, V, W, n, m, p, tol=1e-6):
    """
    Attempts to reduce the rank by finding linear dependencies among the rank-1 terms.
    Returns new U, V, W if a reduction is found, else None.
    """
    rank = U.shape[1]
    if rank <= 1:
        return None
    
    # Flatten terms into vectors
    # Term r is U[:,r] o V[:,r] o W[:,r]
    # We want to find beta such that sum_r beta_r * Term_r = 0
    # This is equivalent to finding the null space of the matrix formed by these vectors.
    
    # Construct the matrix A where column r is the flattened term T_r
    # Shape: (n*m*p, rank)
    A = np.zeros((n*m*p, rank))
    for r in range(rank):
        # U[:,r] is (n*m, 1), V[:,r] is (m*p, 1), W[:,r] is (p*n, 1)
        # Outer product
        term = np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
        A[:,r] = term.flatten()
        
    # Find null space of A
    # If null space is non-trivial, we have a dependency.
    # Use SVD to find null space
    try:
        U_svd, s, Vt = np.linalg.svd(A, full_matrices=False)
    except:
        return None
        
    # The null space vectors are the last (rank - len(s)) rows of Vt
    # If s has a very small value, it indicates a dependency.
    # We look for a dependency vector beta.
    
    # The vector 1 (ones) is in the row space? No.
    # We know sum(T_r) = T_mm.
    # We want to modify the coefficients.
    # Let c be the current coefficient vector (all 1s).
    # If we find a null vector beta, we can update c -> c + alpha * beta.
    # We want to set one component of c to 0.
    # c_i + alpha * beta_i = 0 => alpha = -c_i / beta_i = -1 / beta_i.
    # We need the new c to be "nice" (integers/half-integers).
    
    # Let's check for exact integer dependencies first.
    # If the entries of A are integers, the null space might have rational vectors.
    
    # Iterate through potential elimination indices
    # We try to eliminate the term with the largest magnitude in beta to keep numbers small?
    # Or just try to find any valid elimination.
    
    for r_elim in range(rank):
        # Find a null vector beta with beta[r_elim] != 0
        # We can use the last few columns of Vt
        # Let's check the smallest singular values
        
        # If rank(A) < rank, there is a null space.
        # Let's get the null space basis.
        # The number of singular values > tol is the rank.
        rank_A = np.sum(s > tol)
        if rank_A == rank:
            continue # Full rank, no dependency
        
        # The null space is spanned by Vt[rank_A:, :]
        # We want a combination of these null vectors that has a large component at r_elim.
        # Actually, we can just pick one null vector from the basis.
        # Let's try each null vector in the basis.
        
        # Get null space basis vectors
        null_vecs = Vt[rank_A:, :]
        
        for nv_idx in range(null_vecs.shape[0]):
            beta = null_vecs[nv_idx, :]
            if abs(beta[r_elim]) < tol:
                continue
            
            # Calculate alpha
            # Current coeff c = ones
            # New coeff c_new = c + alpha * beta
            # We want c_new[r_elim] = 0
            # 1 + alpha * beta[r_elim] = 0
            alpha = -1.0 / beta[r_elim]
            
            c_new = np.ones(rank) + alpha * beta
            
            # Check if c_new is close to half-integers
            # c_new should be close to k/2
            c_new_scaled = c_new * 2
            c_new_rounded = np.round(c_new_scaled)
            
            if np.allclose(c_new_scaled, c_new_rounded, atol=1e-4):
                # It's a valid half-integer solution
                # Apply the new coefficients
                U_new = U * c_new[np.newaxis, :]
                V_new = V * c_new[np.newaxis, :]
                W_new = W * c_new[np.newaxis, :]
                
                # Remove the eliminated column
                keep = [i for i in range(rank) if i != r_elim]
                U_new = U_new[:, keep]
                V_new = V_new[:, keep]
                W_new = W_new[:, keep]
                
                # Round to half-integers to clean up floating point errors
                U_new = _round_matrix_to_half_int(U_new)
                V_new = _round_matrix_to_half_int(V_new)
                W_new = _round_matrix_to_half_int(W_new)
                
                # Verify
                if _verify_decomposition(U_new, V_new, W_new, n, m, p):
                    return U_new, V_new, W_new
                    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Get the Strassen-based decomposition as baseline (rank 64)
    U_base, V_base, W_base = _strassen_construct(n, m, p)
    
    # Initial cleanup
    U_base, V_base, W_base = _remove_zero_columns(U_base, V_base, W_base)
    
    best_U, best_V, best_W = U_base, V_base, W_base
    best_rank = U_base.shape[1]
    
    # Attempt elimination
    # We can try multiple times, as eliminating one term might reveal another
    for _ in range(10):
        if time.time() > deadline - 10:
            break
        
        result = _eliminate_dependent_terms(best_U, best_V, best_W, n, m, p)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = best_U.shape[1]
        else:
            break
            
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
