"""Matrix multiplication 4,4,5 tensor decomposition with integer-aware rank reduction."""
import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START

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
                    u[get_u_idx(i, j)] = val1 * val2
            
            for (idx1, val1) in v1_sparse:
                j1, k1 = idx1 // 2, idx1 % 2
                for (idx2, val2) in v2_sparse:
                    j2, k2 = idx2 // 2, idx2 % 2
                    j = j1 * 2 + j2
                    k = k1 * 2 + k2
                    v[get_v_idx(j, k)] = val1 * val2
            
            for (idx1, val1) in w1_sparse:
                k1, i1 = idx1 // 2, idx1 % 2
                for (idx2, val2) in w2_sparse:
                    k2, i2 = idx2 // 2, idx2 % 2
                    k = k1 * 2 + k2
                    i = i1 * 2 + i2
                    w[get_w_idx(k, i)] = val1 * val2
            
            add_term(u, v, w)
            
    # 2. 4x4x1 part (k=4)
    for i in range(n):
        for j in range(m):
            u = np.zeros(n*m); u[get_u_idx(i, j)] = 1
            v = np.zeros(m*p); v[get_v_idx(j, 4)] = 1
            w = np.zeros(p*n); w[get_w_idx(4, i)] = 1
            add_term(u, v, w)
    
    U_mat = np.array(U_cols).T
    V_mat = np.array(V_cols).T
    W_mat = np.array(W_cols).T
    return U_mat, V_mat, W_mat

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check if sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k."""
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # Vectorized verification
    U_reshaped = U_np.reshape(n, m, -1)  # (n, m, rank)
    V_reshaped = V_np.reshape(m, p, -1)  # (m, p, rank)
    W_reshaped = W_np.reshape(p, n, -1)  # (p, n, rank)
    
    rank = U_np.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_reshaped[i, j, :]
                v_val = V_reshaped[j, k, :]
                w_val = W_reshaped[k, i, :]
                total = np.dot(u_val * v_val, w_val)
                if abs(total - 1.0) > tol:
                    return False
    return True

def _verify_integer_half_integer(U, V, W, tol=1e-6):
    """Check that all entries are integer or half-integer (multiple of 0.5)."""
    for M in [U, V, W]:
        for val in M.flat:
            if abs(val * 2 - round(val * 2)) > tol:
                return False
    return True

def _build_schoolbook():
    """Fallback schoolbook decomposition with rank n*m*p = 80."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, column] = 1
                V[j*p+k, column] = 1
                W[k*n+i, column] = 1
                column += 1
    return U, V, W

def _compute_term_vectors(U, V, W, n, m, p):
    """Compute the flattened tensor vector for each rank-1 term using vectorized operations."""
    rank = U.shape[1]
    num_entries = n * m * p
    
    # Reshape for efficient computation
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # For each term r, we need: U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Build the full tensor contribution using broadcasting
    U_reshaped = U.reshape(n, m, rank)  # (n, m, rank)
    V_reshaped = V.reshape(m, p, rank)  # (m, p, rank)
    W_reshaped = W.reshape(p, n, rank)  # (p, n, rank)
    
    # For each (i,j,k), the value is U[i,j,r] * V[j,k,r] * W[k,i,r]
    # We want to compute this for all r simultaneously
    # Result shape: (n, m, p, rank)
    
    # U[i,j,:] * V[j,k,:] * W[k,i,:] for all i,j,k
    # This is a tensor contraction
    
    vecs = np.zeros((n, m, p, rank))
    for i in range(n):
        for j in range(m):
            u_ij = U_reshaped[i, j, :]  # (rank,)
            for k in range(p):
                v_jk = V_reshaped[j, k, :]  # (rank,)
                w_ki = W_reshaped[k, i, :]  # (rank,)
                vecs[i, j, k, :] = u_ij * v_jk * w_ki
    
    return vecs.reshape(num_entries, rank)

def _is_half_integer_vector(vec, tol=1e-5):
    """Check if a vector contains only half-integers."""
    doubled = vec * 2
    rounded = np.round(doubled)
    if np.any(np.abs(doubled - rounded) > tol):
        return False
    return True

def _try_remove_term(U, V, W, term_vecs, target_flat, r_remove, n, m, p, tol=1e-8):
    """Try to remove term r_remove by solving for scaling of remaining terms."""
    rank = U.shape[1]
    remaining = [x for x in range(rank) if x != r_remove]
    if len(remaining) == 0:
        return None
    
    M_remaining = term_vecs[:, remaining]
    
    try:
        c, residuals, rank_m, sv = np.linalg.lstsq(M_remaining, target_flat, rcond=None)
    except Exception:
        return None
    
    if not residuals or len(residuals) == 0 or residuals[0] > tol:
        return None
    
    # Check if scaling results in half-integer entries
    valid = True
    new_U = np.zeros((n*m, len(remaining)))
    new_V = np.zeros((m*p, len(remaining)))
    new_W = np.zeros((p*n, len(remaining)))
    
    for idx, r_idx in enumerate(remaining):
        u_scaled = U[:, r_idx] * c[idx]
        if not _is_half_integer_vector(u_scaled, tol=1e-5):
            valid = False
            break
        new_U[:, idx] = np.round(u_scaled * 2) / 2.0
        
        v_scaled = V[:, r_idx] * c[idx]
        if not _is_half_integer_vector(v_scaled, tol=1e-5):
            valid = False
            break
        new_V[:, idx] = np.round(v_scaled * 2) / 2.0
        
        w_scaled = W[:, r_idx] * c[idx]
        if not _is_half_integer_vector(w_scaled, tol=1e-5):
            valid = False
            break
        new_W[:, idx] = np.round(w_scaled * 2) / 2.0
    
    if not valid:
        return None
    
    # Final verification
    if _verify_decomposition(new_U, new_V, new_W, n, m, p, tol=1e-7):
        return new_U, new_V, new_W
    
    return None

def _try_remove_two_terms(U, V, W, term_vecs, target_flat, r1, r2, n, m, p, tol=1e-8):
    """Try to remove terms r1 and r2 simultaneously."""
    rank = U.shape[1]
    remaining = [x for x in range(rank) if x != r1 and x != r2]
    if len(remaining) == 0:
        return None
    
    M_remaining = term_vecs[:, remaining]
    
    try:
        c, residuals, rank_m, sv = np.linalg.lstsq(M_remaining, target_flat, rcond=None)
    except Exception:
        return None
    
    if not residuals or len(residuals) == 0 or residuals[0] > tol:
        return None
    
    valid = True
    new_U = np.zeros((n*m, len(remaining)))
    new_V = np.zeros((m*p, len(remaining)))
    new_W = np.zeros((p*n, len(remaining)))
    
    for idx, r_idx in enumerate(remaining):
        u_scaled = U[:, r_idx] * c[idx]
        if not _is_half_integer_vector(u_scaled, tol=1e-5):
            valid = False
            break
        new_U[:, idx] = np.round(u_scaled * 2) / 2.0
        
        v_scaled = V[:, r_idx] * c[idx]
        if not _is_half_integer_vector(v_scaled, tol=1e-5):
            valid = False
            break
        new_V[:, idx] = np.round(v_scaled * 2) / 2.0
        
        w_scaled = W[:, r_idx] * c[idx]
        if not _is_half_integer_vector(w_scaled, tol=1e-5):
            valid = False
            break
        new_W[:, idx] = np.round(w_scaled * 2) / 2.0
    
    if not valid:
        return None
    
    if _verify_decomposition(new_U, new_V, new_W, n, m, p, tol=1e-7):
        return new_U, new_V, new_W
    
    return None

def _iterative_rank_reduction(U, V, W, n, m, p, max_time=80):
    """
    Iteratively try to reduce rank by removing terms.
    Uses SVD-based null-space detection for efficient elimination.
    """
    start_time = time.time()
    deadline = start_time + max_time
    target_flat = np.ones((n, m, p)).flatten()
    
    current_U, current_V, current_W = U.copy(), V.copy(), W.copy()
    current_rank = current_U.shape[1]
    
    # Pre-compute term vectors
    term_vecs = _compute_term_vectors(current_U, current_V, current_W, n, m, p)
    
    made_progress = True
    while made_progress and time.time() < deadline - 5:
        made_progress = False
        current_rank = current_U.shape[1]
        
        # First: check if any term is linearly dependent using SVD
        # If term r is in the span of others, we can potentially eliminate it
        for r_remove in range(current_rank):
            if time.time() > deadline - 5:
                break
            
            remaining = [x for x in range(current_rank) if x != r_remove]
            if len(remaining) < 2:
                continue
                
            M_remaining = term_vecs[:, remaining]
            
            # Check if target is in the span of remaining terms
            try:
                # Use least squares to see if we can represent target with remaining terms
                c, residuals, rank_m, sv = np.linalg.lstsq(M_remaining, target_flat, rcond=1e-10)
            except Exception:
                continue
            
            if not residuals or len(residuals) == 0:
                continue
                
            # If residual is small, the target can be represented
            if residuals[0] > 1e-8:
                continue
            
            # Now check if scaling produces valid half-integer entries
            valid = True
            new_U = np.zeros((n*m, len(remaining)))
            new_V = np.zeros((m*p, len(remaining)))
            new_W = np.zeros((p*n, len(remaining)))
            
            for idx, r_idx in enumerate(remaining):
                u_scaled = current_U[:, r_idx] * c[idx]
                if not _is_half_integer_vector(u_scaled, tol=1e-5):
                    valid = False
                    break
                new_U[:, idx] = np.round(u_scaled * 2) / 2.0
                
                v_scaled = current_V[:, r_idx] * c[idx]
                if not _is_half_integer_vector(v_scaled, tol=1e-5):
                    valid = False
                    break
                new_V[:, idx] = np.round(v_scaled * 2) / 2.0
                
                w_scaled = current_W[:, r_idx] * c[idx]
                if not _is_half_integer_vector(w_scaled, tol=1e-5):
                    valid = False
                    break
                new_W[:, idx] = np.round(w_scaled * 2) / 2.0
            
            if valid:
                if _verify_decomposition(new_U, new_V, new_W, n, m, p, tol=1e-7):
                    current_U, current_V, current_W = new_U, new_V, new_W
                    current_rank -= 1
                    made_progress = True
                    term_vecs = _compute_term_vectors(current_U, current_V, current_W, n, m, p)
                    break
        
        # If no single removal worked, try SVD-based null space detection
        if not made_progress and current_rank >= 3:
            # Compute SVD of term vectors to find null space
            try:
                U_svd, S, Vt = np.linalg.svd(term_vecs, full_matrices=False)
            except Exception:
                U_svd, S, Vt = None, None, None
            
            if U_svd is not None and S is not None and Vt is not None:
                # Find singular values that are very small
                tol_svd = 1e-8
                small_idx = np.where(S < tol_svd)[0]
                
                # For each small singular value, the corresponding right singular vector
                # gives a linear combination of terms that sums to zero
                for si in small_idx:
                    if time.time() > deadline - 5:
                        break
                    
                    # Right singular vector gives coefficients
                    coeffs = Vt[si, :]
                    
                    # Find the term with largest absolute coefficient to remove
                    max_coeff_idx = np.argmax(np.abs(coeffs))
                    r_remove = int(max_coeff_idx)
                    
                    if abs(coeffs[r_remove]) < 1e-6:
                        continue
                    
                    # Try to remove this term
                    remaining = [x for x in range(current_rank) if x != r_remove]
                    M_remaining = term_vecs[:, remaining]
                    
                    try:
                        c, residuals, rank_m, sv = np.linalg.lstsq(M_remaining, target_flat, rcond=1e-10)
                    except Exception:
                        continue
                    
                    if not residuals or len(residuals) == 0 or residuals[0] > 1e-8:
                        continue
                    
                    valid = True
                    new_U = np.zeros((n*m, len(remaining)))
                    new_V = np.zeros((m*p, len(remaining)))
                    new_W = np.zeros((p*n, len(remaining)))
                    
                    for idx, r_idx in enumerate(remaining):
                        u_scaled = current_U[:, r_idx] * c[idx]
                        if not _is_half_integer_vector(u_scaled, tol=1e-5):
                            valid = False
                            break
                        new_U[:, idx] = np.round(u_scaled * 2) / 2.0
                        
                        v_scaled = current_V[:, r_idx] * c[idx]
                        if not _is_half_integer_vector(v_scaled, tol=1e-5):
                            valid = False
                            break
                        new_V[:, idx] = np.round(v_scaled * 2) / 2.0
                        
                        w_scaled = current_W[:, r_idx] * c[idx]
                        if not _is_half_integer_vector(w_scaled, tol=1e-5):
                            valid = False
                            break
                        new_W[:, idx] = np.round(w_scaled * 2) / 2.0
                    
                    if valid and _verify_decomposition(new_U, new_V, new_W, n, m, p, tol=1e-7):
                        current_U, current_V, current_W = new_U, new_V, new_W
                        current_rank -= 1
                        made_progress = True
                        term_vecs = _compute_term_vectors(current_U, current_V, current_W, n, m, p)
                        break
    
    if current_rank < U.shape[1]:
        return current_U, current_V, current_W
    
    return None

def _try_structured_merge(U, V, W, n, m, p, max_time=30):
    """
    Try to find two terms that can be merged into one.
    This is a heuristic: if two terms have similar support patterns,
    maybe their sum can be represented as a single rank-1 term.
    """
    start_time = time.time()
    deadline = start_time + max_time
    target_flat = np.ones((n, m, p)).flatten()
    rank = U.shape[1]
    
    # For each pair of terms, check if their combined contribution
    # can be represented as a single rank-1 term with half-integer entries
    for r1 in range(rank):
        for r2 in range(r1 + 1, rank):
            if time.time() > deadline:
                return None
            
            # Compute the combined contribution of terms r1 and r2
            combined = np.zeros(n * m * p)
            for i in range(n):
                for j in range(m):
                    u1 = U[i*m+j, r1]
                    u2 = U[i*m+j, r2]
                    for k in range(p):
                        v1 = V[j*p+k, r1]
                        v2 = V[j*p+k, r2]
                        w1 = W[k*n+i, r1]
                        w2 = W[k*n+i, r2]
                        combined[(i*m+j)*p+k] = u1*v1*w1 + u2*v2*w2
            
            # Try to factor combined as u_new ⊗ v_new ⊗ w_new
            # This is generally hard, so we use a heuristic:
            # Check if combined has a special structure
            
            # Skip this for now - too expensive
            pass
    
    return None

def _final_polish(U, V, W, n, m, p, max_time=15):
    """
    High-precision polish: snap to half-integers and try more aggressive reduction.
    Uses SVD to find better elimination candidates.
    """
    start_time = time.time()
    deadline = start_time + max_time
    target_flat = np.ones((n, m, p)).flatten()
    
    # Snap to nearest half-integer
    U_snapped = np.round(U * 2) / 2.0
    V_snapped = np.round(V * 2) / 2.0
    W_snapped = np.round(W * 2) / 2.0
    
    if not _verify_decomposition(U_snapped, V_snapped, W_snapped, n, m, p, tol=1e-9):
        U_snapped, V_snapped, W_snapped = U.copy(), V.copy(), W.copy()
    
    # Try more aggressive reduction with tighter tolerance
    term_vecs = _compute_term_vectors(U_snapped, V_snapped, W_snapped, n, m, p)
    current_rank = U_snapped.shape[1]
    
    for _ in range(20):  # Multiple passes
        if time.time() > deadline:
            break
        
        progress = False
        
        # First try direct removal
        for r_remove in range(current_rank):
            if time.time() > deadline:
                break
            
            remaining = [x for x in range(current_rank) if x != r_remove]
            if not remaining:
                break
            
            M_remaining = term_vecs[:, remaining]
            
            try:
                c, residuals, rank_m, sv = np.linalg.lstsq(M_remaining, target_flat, rcond=1e-12)
            except Exception:
                continue
            
            if not residuals or len(residuals) == 0 or residuals[0] > 1e-10:
                continue
            
            valid = True
            new_U = np.zeros((n*m, len(remaining)))
            new_V = np.zeros((m*p, len(remaining)))
            new_W = np.zeros((p*n, len(remaining)))
            
            for idx, r_idx in enumerate(remaining):
                u_scaled = U_snapped[:, r_idx] * c[idx]
                if not _is_half_integer_vector(u_scaled, tol=1e-8):
                    valid = False
                    break
                new_U[:, idx] = np.round(u_scaled * 2) / 2.0
                
                v_scaled = V_snapped[:, r_idx] * c[idx]
                if not _is_half_integer_vector(v_scaled, tol=1e-8):
                    valid = False
                    break
                new_V[:, idx] = np.round(v_scaled * 2) / 2.0
                
                w_scaled = W_snapped[:, r_idx] * c[idx]
                if not _is_half_integer_vector(w_scaled, tol=1e-8):
                    valid = False
                    break
                new_W[:, idx] = np.round(w_scaled * 2) / 2.0
            
            if valid and _verify_decomposition(new_U, new_V, new_W, n, m, p, tol=1e-9):
                U_snapped, V_snapped, W_snapped = new_U, new_V, new_W
                current_rank = len(remaining)
                term_vecs = _compute_term_vectors(U_snapped, V_snapped, W_snapped, n, m, p)
                progress = True
                break
        
        # If no direct removal, try SVD-based approach
        if not progress and current_rank >= 4:
            try:
                U_svd, S, Vt = np.linalg.svd(term_vecs, full_matrices=False)
            except Exception:
                U_svd, S, Vt = None, None, None
            
            if U_svd is not None and S is not None and Vt is not None:
                tol_svd = 1e-10
                small_idx = np.where(S < tol_svd)[0]
                
                for si in small_idx:
                    if time.time() > deadline:
                        break
                    
                    coeffs = Vt[si, :]
                    max_coeff_idx = np.argmax(np.abs(coeffs))
                    r_remove = int(max_coeff_idx)
                    
                    if abs(coeffs[r_remove]) < 1e-8:
                        continue
                    
                    remaining = [x for x in range(current_rank) if x != r_remove]
                    M_remaining = term_vecs[:, remaining]
                    
                    try:
                        c, residuals, rank_m, sv = np.linalg.lstsq(M_remaining, target_flat, rcond=1e-12)
                    except Exception:
                        continue
                    
                    if not residuals or len(residuals) == 0 or residuals[0] > 1e-10:
                        continue
                    
                    valid = True
                    new_U = np.zeros((n*m, len(remaining)))
                    new_V = np.zeros((m*p, len(remaining)))
                    new_W = np.zeros((p*n, len(remaining)))
                    
                    for idx, r_idx in enumerate(remaining):
                        u_scaled = U_snapped[:, r_idx] * c[idx]
                        if not _is_half_integer_vector(u_scaled, tol=1e-8):
                            valid = False
                            break
                        new_U[:, idx] = np.round(u_scaled * 2) / 2.0
                        
                        v_scaled = V_snapped[:, r_idx] * c[idx]
                        if not _is_half_integer_vector(v_scaled, tol=1e-8):
                            valid = False
                            break
                        new_V[:, idx] = np.round(v_scaled * 2) / 2.0
                        
                        w_scaled = W_snapped[:, r_idx] * c[idx]
                        if not _is_half_integer_vector(w_scaled, tol=1e-8):
                            valid = False
                            break
                        new_W[:, idx] = np.round(w_scaled * 2) / 2.0
                    
                    if valid and _verify_decomposition(new_U, new_V, new_W, n, m, p, tol=1e-9):
                        U_snapped, V_snapped, W_snapped = new_U, new_V, new_W
                        current_rank = len(remaining)
                        term_vecs = _compute_term_vectors(U_snapped, V_snapped, W_snapped, n, m, p)
                        progress = True
                        break
        
        if not progress:
            break
    
    return U_snapped, V_snapped, W_snapped

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4 @ 4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    n, m, p = DIMENSIONS
    
    # Step 1: Construct initial rank-65 decomposition
    U, V, W = _construct_rank_65(n, m, p)
    if not _verify_decomposition(U, V, W, n, m, p):
        U, V, W = _build_schoolbook()
        return {"U": [[float(x) for x in row] for row in U],
                "V": [[float(x) for x in row] for row in V],
                "W": [[float(x) for x in row] for row in W]}
    
    current_rank = U.shape[1]
    
    # Step 2: Iterative rank reduction with SVD-based null-space detection
    if time.time() < deadline - 25:
        try:
            reduced = _iterative_rank_reduction(U, V, W, n, m, p, 
                                                 max_time=min(100, deadline - time.time() - 20))
            if reduced is not None:
                U, V, W = reduced
                current_rank = U.shape[1]
        except Exception:
            pass
    
    # Step 3: Final polish with tighter tolerance and SVD
    if time.time() < deadline - 5:
        try:
            U, V, W = _final_polish(U, V, W, n, m, p, 
                                     max_time=min(20, deadline - time.time() - 3))
            current_rank = U.shape[1]
        except Exception:
            pass
    
    # Convert to lists of floats
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
