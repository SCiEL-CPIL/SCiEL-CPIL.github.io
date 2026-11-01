import time
import numpy as np

DIMENSIONS = [2, 4, 7]

def _build_target_tensor(n, m, p):
    """Build the matrix multiplication tensor with the trace convention."""
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1.0
    return T

def _verify_decomposition(U, V, W, T, tol=1e-9):
    """Verify that the decomposition is valid."""
    T_comp = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T_comp, T, atol=tol)

def _build_trivial(n, m, p):
    """Construct the trivial rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                U[a_idx, r] = 1.0
                V[b_idx, r] = 1.0
                W[c_idx, r] = 1.0
                r += 1
    return U, V, W

def _make_strassen_2x2x2_factors():
    """
    Returns U, V, W for 2x2x2 matrix multiplication with rank 7 (Strassen).
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # r=0: M1 = (A11 + A22) * (B11 + B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # r=1: M2 = (A21 + A22) * B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # r=2: M3 = A11 * (B12 - B22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = -1
    
    # r=3: M4 = A22 * (B21 - B11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = -1; W[2, 3] = 1
    
    # r=4: M5 = (A11 + A12) * B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # r=5: M6 = (A21 - A11) * (B11 + B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # r=6: M7 = (A12 - A22) * (B21 + B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _make_trivial_2x2x1_factors():
    """
    Returns U, V, W for 2x2x1 matrix multiplication with rank 4.
    """
    U = np.zeros((4, 4))
    V = np.zeros((2, 4))
    W = np.zeros((2, 4))
    
    U[0, 0] = 1
    V[0, 0] = 1
    W[0, 0] = 1
    
    U[1, 1] = 1
    V[1, 1] = 1
    W[0, 1] = 1
    
    U[2, 2] = 1
    V[0, 2] = 1
    W[1, 2] = 1
    
    U[3, 3] = 1
    V[1, 3] = 1
    W[1, 3] = 1
    
    return U, V, W

def _build_block_decomposition(n, m, p):
    """Build the block decomposition (Strassen + schoolbook) for 2x4x7."""
    # Build 2x2x7 sub-decomposition
    U_sub = np.zeros((4, 25))
    V_sub = np.zeros((14, 25))
    W_sub = np.zeros((14, 25))
    
    r_offset = 0
    
    def add_strassen_2x2x2(U, V, W, r_off, k_start):
        U_s, V_s, W_s = _make_strassen_2x2x2_factors()
        U[:, r_off:r_off+7] = U_s
        V[k_start, r_off:r_off+7] = V_s[0, :]
        V[k_start+1, r_off:r_off+7] = V_s[1, :]
        V[7+k_start, r_off:r_off+7] = V_s[2, :]
        V[7+k_start+1, r_off:r_off+7] = V_s[3, :]
        W[k_start*2, r_off:r_off+7] = W_s[0, :]
        W[k_start*2+1, r_off:r_off+7] = W_s[1, :]
        W[(k_start+1)*2, r_off:r_off+7] = W_s[2, :]
        W[(k_start+1)*2+1, r_off:r_off+7] = W_s[3, :]
        return 7
    
    def add_trivial_2x2x1(U, V, W, r_off, k_start):
        U_t, V_t, W_t = _make_trivial_2x2x1_factors()
        U[:, r_off:r_off+4] = U_t
        V[k_start, r_off:r_off+4] = V_t[0, :]
        V[7+k_start, r_off:r_off+4] = V_t[1, :]
        W[k_start*2, r_off:r_off+4] = W_t[0, :]
        W[k_start*2+1, r_off:r_off+4] = W_t[1, :]
        return 4
    
    r_offset += add_strassen_2x2x2(U_sub, V_sub, W_sub, r_offset, 0)
    r_offset += add_strassen_2x2x2(U_sub, V_sub, W_sub, r_offset, 2)
    r_offset += add_strassen_2x2x2(U_sub, V_sub, W_sub, r_offset, 4)
    r_offset += add_trivial_2x2x1(U_sub, V_sub, W_sub, r_offset, 6)
    
    # Build full 2x4x7
    U_full = np.zeros((8, 50))
    V_full = np.zeros((28, 50))
    W_full = np.zeros((14, 50))
    
    # Block 1: A cols 0,1
    U_full[0:4, 0:25] = U_sub
    V_full[0:14, 0:25] = V_sub
    W_full[0:14, 0:25] = W_sub
    
    # Block 2: A cols 2,3
    U_full[2, 25:50] = U_sub[0, :]
    U_full[3, 25:50] = U_sub[1, :]
    U_full[6, 25:50] = U_sub[2, :]
    U_full[7, 25:50] = U_sub[3, :]
    V_full[14:21, 25:50] = V_sub[0:7, :]
    V_full[21:28, 25:50] = V_sub[7:14, :]
    W_full[0:14, 25:50] = W_sub
    
    return U_full, V_full, W_full

def _try_merge_two_terms(U, V, W, T, r1, r2, tol=1e-9):
    """
    Try to replace two rank-1 terms (r1, r2) with a single rank-1 term.
    This is possible if the difference tensor D = term_r1 + term_r2 can be
    expressed as a single rank-1 tensor, i.e., D[a,b,c] = u_a * v_b * w_c.
    
    We check this by verifying that D has tensor rank 1.
    """
    # Compute the two rank-1 terms
    term1 = np.einsum('a,b,c->abc', U[:, r1], V[:, r1], W[:, r1])
    term2 = np.einsum('a,b,c->abc', U[:, r2], V[:, r2], W[:, r2])
    D = term1 + term2
    
    # Check if D has rank 1: all 2x2 minors of any matrix unfolding must be zero.
    # More efficiently: check if D can be written as outer product of three vectors.
    # A tensor has rank 1 iff all its 2x2 minors in every unfolding are zero.
    # We check the a-b unfolding (fix c), a-c unfolding (fix b), b-c unfolding (fix a).
    
    na, nb, nc = D.shape
    
    # Check a-b unfolding for each c
    for c in range(nc):
        mat = D[:, :, c]  # na x nb
        if mat.size == 0:
            continue
        # Check if rank(mat) <= 1
        if mat.shape[0] >= 2 and mat.shape[1] >= 2:
            # Use SVD to check rank
            try:
                s = np.linalg.svd(mat, compute_uv=False)
                if len(s) >= 2 and s[1] > tol * max(s[0] if len(s) > 0 else 1, 1):
                    return None
            except:
                return None
    
    # Check a-c unfolding for each b
    for b in range(nb):
        mat = D[:, b, :]  # na x nc
        if mat.shape[0] >= 2 and mat.shape[1] >= 2:
            try:
                s = np.linalg.svd(mat, compute_uv=False)
                if len(s) >= 2 and s[1] > tol * max(s[0] if len(s) > 0 else 1, 1):
                    return None
            except:
                return None
    
    # Check b-c unfolding for each a
    for a in range(na):
        mat = D[a, :, :]  # nb x nc
        if mat.shape[0] >= 2 and mat.shape[1] >= 2:
            try:
                s = np.linalg.svd(mat, compute_uv=False)
                if len(s) >= 2 and s[1] > tol * max(s[0] if len(s) > 0 else 1, 1):
                    return None
            except:
                return None
    
    # If we get here, D appears to have rank 1. Extract the vectors.
    # Find the non-zero entries to extract u, v, w.
    # D[a,b,c] = u_a * v_b * w_c
    # Find a, b, c where D is non-zero.
    nonzero = np.nonzero(np.abs(D) > tol)
    if len(nonzero[0]) == 0:
        # D is zero, so we can just remove both terms (replace with zero term)
        return np.zeros(na), np.zeros(nb), np.zeros(nc)
    
    a0, b0, c0 = nonzero[0][0], nonzero[1][0], nonzero[2][0]
    D0 = D[a0, b0, c0]
    
    # Extract vectors: u_a = D[a,b0,c0] / D0, v_b = D[a0,b,c0] / D0, w_c = D[a0,b0,c] / D0
    u = D[:, b0, c0] / D0
    v = D[a0, :, c0] / D0
    w = D[a0, b0, :] / D0
    
    # Verify
    D_check = np.einsum('a,b,c->abc', u, v, w)
    if not np.allclose(D_check, D, atol=tol):
        return None
    
    return u, v, w

def _iterative_rank_reduction(U, V, W, T, max_iterations=100, deadline=None):
    """
    Iteratively try to reduce the rank by merging pairs of terms.
    """
    current_U, current_V, current_W = U.copy(), V.copy(), W.copy()
    current_rank = current_U.shape[1]
    
    for iteration in range(max_iterations):
        if deadline is not None and time.time() > deadline:
            break
        
        if current_rank <= 1:
            break
        
        merged = False
        # Try all pairs of terms
        for r1 in range(current_rank):
            if deadline is not None and time.time() > deadline:
                break
            for r2 in range(r1 + 1, current_rank):
                if deadline is not None and time.time() > deadline:
                    break
                
                result = _try_merge_two_terms(current_U, current_V, current_W, T, r1, r2)
                if result is not None:
                    u_new, v_new, w_new = result
                    # Remove terms r1 and r2, add the new term
                    # Create new arrays with rank-1
                    new_rank = current_rank - 1
                    new_U = np.zeros((current_U.shape[0], new_rank), dtype=float)
                    new_V = np.zeros((current_V.shape[0], new_rank), dtype=float)
                    new_W = np.zeros((current_W.shape[0], new_rank), dtype=float)
                    
                    # Copy terms excluding r1 and r2
                    keep = [r for r in range(current_rank) if r != r1 and r != r2]
                    for new_idx, old_idx in enumerate(keep):
                        new_U[:, new_idx] = current_U[:, old_idx]
                        new_V[:, new_idx] = current_V[:, old_idx]
                        new_W[:, new_idx] = current_W[:, old_idx]
                    
                    # Add the new merged term at the end
                    new_U[:, new_rank - 1] = u_new
                    new_V[:, new_rank - 1] = v_new
                    new_W[:, new_rank - 1] = w_new
                    
                    # Verify
                    if _verify_decomposition(new_U, new_V, new_W, T):
                        current_U, current_V, current_W = new_U, new_V, new_W
                        current_rank = new_rank
                        merged = True
                        break
            if merged:
                break
        
        if not merged:
            break
    
    return current_U, current_V, current_W

def _random_rank_reduction_search(U, V, W, T, max_trials=50, deadline=None):
    """
    Randomized search for rank reduction: randomly perturb and try to find
    a lower-rank solution by solving a system of equations.
    """
    current_U, current_V, current_W = U.copy(), V.copy(), W.copy()
    current_rank = current_U.shape[1]
    best_rank = current_rank
    best_U, best_V, best_W = current_U.copy(), current_V.copy(), current_W.copy()
    
    rng = np.random.default_rng(12345)
    
    for trial in range(max_trials):
        if deadline is not None and time.time() > deadline:
            break
        
        # Try to reduce by 1
        target_rank = current_rank - 1
        if target_rank < 1:
            break
        
        # Randomly select which terms to keep and which to merge
        # We'll try a different approach: randomly initialize a rank-target_rank
        # solution and try to fit it.
        
        # For now, just try the iterative merge approach with random restarts
        # by shuffling the term order
        perm = rng.permutation(current_rank)
        shuffled_U = current_U[:, perm]
        shuffled_V = current_V[:, perm]
        shuffled_W = current_W[:, perm]
        
        result_U, result_V, result_W = _iterative_rank_reduction(
            shuffled_U, shuffled_V, shuffled_W, T, 
            max_iterations=20, deadline=deadline
        )
        
        if result_U.shape[1] < best_rank:
            best_rank = result_U.shape[1]
            best_U, best_V, best_W = result_U.copy(), result_V.copy(), result_W.copy()
        
        if best_rank <= 1:
            break
    
    return best_U, best_V, best_W

def construct():
    start_time = time.time()
    deadline = 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    T = _build_target_tensor(n, m, p)
    
    # Build the block decomposition (rank 50)
    U, V, W = _build_block_decomposition(n, m, p)
    
    if _verify_decomposition(U, V, W, T):
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_rank = U.shape[1]
    else:
        # Fallback to trivial
        U, V, W = _build_trivial(n, m, p)
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_rank = U.shape[1]
    
    # Try iterative rank reduction
    elapsed = time.time() - start_time
    if elapsed < deadline - 10 and best_rank > 1:
        result_U, result_V, result_W = _iterative_rank_reduction(
            best_U, best_V, best_W, T,
            max_iterations=100, deadline=deadline
        )
        if result_U.shape[1] < best_rank and _verify_decomposition(result_U, result_V, result_W, T):
            best_U, best_V, best_W = result_U.copy(), result_V.copy(), result_W.copy()
            best_rank = result_U.shape[1]
    
    # Try randomized search if time permits
    elapsed = time.time() - start_time
    if elapsed < deadline - 5 and best_rank > 1:
        result_U, result_V, result_W = _random_rank_reduction_search(
            best_U, best_V, best_W, T,
            max_trials=30, deadline=deadline
        )
        if result_U.shape[1] < best_rank and _verify_decomposition(result_U, result_V, result_W, T):
            best_U, best_V, best_W = result_U.copy(), result_V.copy(), result_W.copy()
            best_rank = result_U.shape[1]
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    
    T = np.einsum('ar,br,cr->abc', U, V, W)
    Expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                Expected[i*m + j, j*p + k, k*n + i] = 1
    
    if np.allclose(T, Expected, atol=1e-9):
        print(f"Valid decomposition with rank {U.shape[1]}")
    else:
        print("Invalid decomposition")
        diff = np.abs(T - Expected)
        print(f"Max diff: {diff.max()}")
