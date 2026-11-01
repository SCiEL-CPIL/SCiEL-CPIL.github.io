import time
import numpy as np

DIMENSIONS = [2, 2, 10]
_DEADLINE = 140.0  # Safety margin

def _build_strassen_35():
    """Return the known rank-35 Strassen decomposition as numpy arrays."""
    n, m, p = 2, 2, 10
    rank = 35
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen terms for 2x2 matrix multiplication
    # U vectors (4x1), V vectors (4x1), W vectors (4x1)
    # Note: The indices in the prompt's seed program were:
    # U: n*m = 4 entries. Index i = row*m + col? No, standard flattening.
    # The prompt's _build_strassen_35 uses:
    # U[:, col] = a_vec (length 4)
    # V[0*p + 2*k ...]
    # This corresponds to A being 2x2, B being 2x2 (for one block), C being 2x2.
    # a_vec corresponds to A entries [A00, A01, A10, A11]
    # b_vec corresponds to B entries [B00, B01, B10, B11]
    # c_vec corresponds to C entries [C00, C01, C10, C11]
    
    strassen_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    
    col = 0
    for k in range(p // 2):
        for a_vec, b_vec, c_vec in strassen_terms:
            U[:, col] = a_vec
            # V indices: j = row * p + col_in_block?
            # B is 2x10. Row 0: cols 0..9. Row 1: cols 0..9.
            # For block k (cols 2k, 2k+1):
            # B[0, 2k] is at index 0*p + 2k = 2k
            # B[0, 2k+1] is at index 0*p + 2k+1 = 2k+1
            # B[1, 2k] is at index 1*p + 2k = 10 + 2k
            # B[1, 2k+1] is at index 1*p + 2k+1 = 10 + 2k+1
            V[0 * p + 2 * k, col] = b_vec[0]
            V[0 * p + 2 * k + 1, col] = b_vec[1]
            V[1 * p + 2 * k, col] = b_vec[2]
            V[1 * p + 2 * k + 1, col] = b_vec[3]
            
            # W indices: k_index = col * n + row?
            # C is 2x10.
            # C[0, 2k] is at index 2k * 2 + 0 = 4k
            # C[0, 2k+1] is at index (2k+1) * 2 + 0 = 4k+2
            # C[1, 2k] is at index 2k * 2 + 1 = 4k+1
            # C[1, 2k+1] is at index (2k+1) * 2 + 1 = 4k+3
            W[2 * k * n + 0, col] = c_vec[0]
            W[2 * k * n + 1, col] = c_vec[1]
            W[(2 * k + 1) * n + 0, col] = c_vec[2]
            W[(2 * k + 1) * n + 1, col] = c_vec[3]
            col += 1
    return U, V, W

def _build_target_tensor(n=2, m=2, p=10):
    """Build the target tensor T as a list of (i, j, k, value) for non-zero entries."""
    entries = []
    for i_row in range(n):
        for i_col in range(m):
            i = i_row * m + i_col
            for j_row in range(m):
                for j_col in range(p):
                    j = j_row * p + j_col
                    for k_row in range(n):
                        for k_col in range(p):
                            k = k_col * n + k_row
                            if i_row == k_row and i_col == j_row:
                                entries.append((i, j, k, 1.0))
    return entries

def _residual(U, V, W, target_entries):
    """Compute the residual of the tensor equation."""
    residual = 0.0
    for i, j, k, val in target_entries:
        s = np.dot(U[i, :], V[j, :] * W[k, :])
        residual += (s - val) ** 2
    return residual

def _solve_for_U(V, W, n=2, m=2, p=10):
    R = V.shape[1]
    U = np.zeros((n * m, R))
    for i in range(n * m):
        rows = []
        b = []
        i_row = i // m
        i_col = i % m
        for j in range(m * p):
            j_row = j // p
            j_col = j % p
            for k in range(p * n):
                k_row = k % n
                k_col = k // n
                if i_row == k_row and i_col == j_row:
                    rows.append(V[j, :] * W[k, :])
                    b.append(1.0)
        if not rows:
            continue
        A = np.array(rows)
        b = np.array(b)
        U[i, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return U

def _solve_for_V(U, W, n=2, m=2, p=10):
    R = U.shape[1]
    V = np.zeros((m * p, R))
    for j in range(m * p):
        rows = []
        b = []
        j_row = j // p
        j_col = j % p
        for i in range(n * m):
            i_row = i // m
            i_col = i % m
            for k in range(p * n):
                k_row = k % n
                k_col = k // n
                if i_row == k_row and i_col == j_row:
                    rows.append(U[i, :] * W[k, :])
                    b.append(1.0)
        if not rows:
            continue
        A = np.array(rows)
        b = np.array(b)
        V[j, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return V

def _solve_for_W(U, V, n=2, m=2, p=10):
    R = U.shape[1]
    W = np.zeros((p * n, R))
    for k in range(p * n):
        rows = []
        b = []
        k_row = k % n
        k_col = k // n
        for i in range(n * m):
            i_row = i // m
            i_col = i % m
            for j in range(m * p):
                j_row = j // p
                j_col = j % p
                if i_row == k_row and i_col == j_row:
                    rows.append(U[i, :] * V[j, :])
                    b.append(1.0)
        if not rows:
            continue
        A = np.array(rows)
        b = np.array(b)
        W[k, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return W

def _try_rank_34(start_time):
    n, m, p = 2, 2, 10
    target_entries = _build_target_tensor(n, m, p)
    U35, V35, W35 = _build_strassen_35()
    
    best_residual = float('inf')
    best_U, best_V, best_W = None, None, None
    
    # Try dropping each term
    for drop_idx in range(35):
        if time.time() - start_time > _DEADLINE - 10:
            break
        
        U = U35[:, [r for r in range(35) if r != drop_idx]]
        V = V35[:, [r for r in range(35) if r != drop_idx]]
        W = W35[:, [r for r in range(35) if r != drop_idx]]
        
        # Try a few random perturbations
        for restart in range(3):
            if time.time() - start_time > _DEADLINE - 5:
                break
            
            # Perturb
            rng = np.random.default_rng(drop_idx * 100 + restart)
            U_pert = U + rng.normal(0, 0.1, U.shape)
            V_pert = V + rng.normal(0, 0.1, V.shape)
            W_pert = W + rng.normal(0, 0.1, W.shape)
            
            U_curr, V_curr, W_curr = U_pert, V_pert, W_pert
            
            for iteration in range(20):
                if time.time() - start_time > _DEADLINE - 5:
                    break
                U_curr = _solve_for_U(V_curr, W_curr, n, m, p)
                V_curr = _solve_for_V(U_curr, W_curr, n, m, p)
                W_curr = _solve_for_W(U_curr, V_curr, n, m, p)
            
            res = _residual(U_curr, V_curr, W_curr, target_entries)
            if res < best_residual:
                best_residual = res
                best_U, best_V, best_W = U_curr.copy(), V_curr.copy(), W_curr.copy()
            
            if best_residual < 1e-6:
                break
        if best_residual < 1e-6:
            break

    if best_residual < 1e-6:
        # Verify
        rng = np.random.default_rng(999)
        A_test = rng.integers(-3, 4, size=(n, m)).astype(float)
        B_test = rng.integers(-3, 4, size=(m, p)).astype(float)
        C_expected = A_test @ B_test
        
        A_flat = A_test.flatten()
        B_flat = B_test.flatten()
        
        C_computed = np.zeros(p * n)
        for r in range(34):
            uA = np.dot(best_U[:, r], A_flat)
            vB = np.dot(best_V[:, r], B_flat)
            C_computed += uA * vB * best_W[:, r]
        
        C_expected_reordered = np.zeros(p * n)
        for i in range(n):
            for k in range(p):
                C_expected_reordered[k * n + i] = C_expected[i, k]
        
        if np.allclose(C_computed, C_expected_reordered, atol=1e-6):
            return best_U, best_V, best_W
    
    return None

def construct():
    start_time = time.time()
    n, m, p = 2, 2, 10
    
    # Start with rank-35 Strassen
    U35, V35, W35 = _build_strassen_35()
    
    # Try to find rank-34
    result_34 = _try_rank_34(start_time)
    
    if result_34 is not None:
        U, V, W = result_34
    else:
        U, V, W = U35, V35, W35
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
