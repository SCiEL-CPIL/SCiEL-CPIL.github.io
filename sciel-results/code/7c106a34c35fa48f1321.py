"""Matrix multiplication 2,2,11 - rank-minimal decomposition via structured search."""
import time
import numpy as np
from fractions import Fraction

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


def _build_strassen_decomposition():
    """
    Build a decomposition using Strassen for 2x2 blocks.
    For 2x2 * 2x11: use Strassen on pairs of columns.
    11 columns = 5 pairs + 1 single column
    Total: 5*7 + 4 = 39
    """
    n, m, p = 2, 2, 11
    num_pairs = p // 2
    remainder = p % 2
    rank = num_pairs * 7 + remainder * 4
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)

    col = 0
    for b in range(num_pairs):
        k0 = 2 * b
        k1 = 2 * b + 1
        v_e = k0; v_f = k1; v_g = p + k0; v_h = p + k1
        w_c00 = k0 * n + 0; w_c01 = k1 * n + 0; w_c10 = k0 * n + 1; w_c11 = k1 * n + 1
        u_a = 0; u_b = 1; u_c = 2; u_d = 3

        U[u_a, col] = 1.0; U[u_d, col] = 1.0
        V[v_e, col] = 1.0; V[v_h, col] = 1.0
        W[w_c00, col] = 1.0; W[w_c11, col] = 1.0
        col += 1

        U[u_c, col] = 1.0; U[u_d, col] = 1.0
        V[v_e, col] = 1.0
        W[w_c10, col] = 1.0; W[w_c11, col] = -1.0
        col += 1

        U[u_a, col] = 1.0
        V[v_f, col] = 1.0; V[v_h, col] = -1.0
        W[w_c01, col] = 1.0; W[w_c11, col] = 1.0
        col += 1

        U[u_d, col] = 1.0
        V[v_g, col] = 1.0; V[v_e, col] = -1.0
        W[w_c00, col] = 1.0; W[w_c10, col] = 1.0
        col += 1

        U[u_a, col] = 1.0; U[u_b, col] = 1.0
        V[v_h, col] = 1.0
        W[w_c00, col] = -1.0; W[w_c01, col] = 1.0
        col += 1

        U[u_c, col] = 1.0; U[u_a, col] = -1.0
        V[v_e, col] = 1.0; V[f] if False else None
        V[v_e, col] = 1.0; V[v_f, col] = 1.0
        W[w_c11, col] = 1.0
        col += 1

        U[u_b, col] = 1.0; U[u_d, col] = -1.0
        V[v_g, col] = 1.0; V[v_h, col] = 1.0
        W[w_c00, col] = 1.0
        col += 1

    if remainder == 1:
        k2 = num_pairs * 2
        v_e2 = k2; v_g2 = p + k2
        w_c02 = k2 * n + 0; w_c12 = k2 * n + 1
        U[0, col] = 1.0; V[v_e2, col] = 1.0; W[w_c02, col] = 1.0; col += 1
        U[1, col] = 1.0; V[v_g2, col] = 1.0; W[w_c02, col] = 1.0; col += 1
        U[2, col] = 1.0; V[v_e2, col] = 1.0; W[w_c12, col] = 1.0; col += 1
        U[3, col] = 1.0; V[v_g2, col] = 1.0; W[w_c12, col] = 1.0; col += 1

    return U, V, W


def _build_subspace_splitting_decomposition():
    """
    Subspace splitting: 5 Strassen blocks (35) + 3 terms for last column = 38.
    """
    n, m, p = 2, 2, 11
    num_pairs = p // 2
    remainder = p % 2
    rank = num_pairs * 7 + 3
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)

    col = 0
    for b in range(num_pairs):
        k0 = 2 * b
        k1 = 2 * b + 1
        v_e = k0; v_f = k1; v_g = p + k0; v_h = p + k1
        w_c00 = k0 * n + 0; w_c01 = k1 * n + 0; w_c10 = k0 * n + 1; w_c11 = k1 * n + 1
        u_a = 0; u_b = 1; u_c = 2; u_d = 3

        U[u_a, col] = 1.0; U[u_d, col] = 1.0
        V[v_e, col] = 1.0; V[v_h, col] = 1.0
        W[w_c00, col] = 1.0; W[w_c11, col] = 1.0
        col += 1

        U[u_c, col] = 1.0; U[u_d, col] = 1.0
        V[v_e, col] = 1.0
        W[w_c10, col] = 1.0; W[w_c11, col] = -1.0
        col += 1

        U[u_a, col] = 1.0
        V[v_f, col] = 1.0; V[v_h, col] = -1.0
        W[w_c01, col] = 1.0; W[w_c11, col] = 1.0
        col += 1

        U[u_d, col] = 1.0
        V[v_g, col] = 1.0; V[v_e, col] = -1.0
        W[w_c00, col] = 1.0; W[w_c10, col] = 1.0
        col += 1

        U[u_a, col] = 1.0; U[u_b, col] = 1.0
        V[v_h, col] = 1.0
        W[w_c00, col] = -1.0; W[w_c01, col] = 1.0
        col += 1

        U[u_c, col] = 1.0; U[u_a, col] = -1.0
        V[v_e, col] = 1.0; V[v_f, col] = 1.0
        W[w_c11, col] = 1.0
        col += 1

        U[u_b, col] = 1.0; U[u_d, col] = -1.0
        V[v_g, col] = 1.0; V[v_h, col] = 1.0
        W[w_c00, col] = 1.0
        col += 1

    if remainder == 1:
        k2 = num_pairs * 2
        v_e2 = k2; v_g2 = p + k2
        w_c02 = k2 * n + 0; w_c12 = k2 * n + 1
        U[0, col] = 1.0; V[v_e2, col] = 1.0; W[w_c02, col] = 1.0; col += 1
        U[1, col] = 1.0; V[v_g2, col] = 1.0; W[w_c02, col] = 1.0; col += 1
        U[2, col] = 1.0; U[3, col] = 1.0
        V[v_e2, col] = 1.0; V[v_g2, col] = 1.0
        W[w_c12, col] = 1.0; col += 1

    return U, V, W


def _build_schoolbook():
    """Build the schoolbook rank-44 decomposition."""
    n, m, p = 2, 2, 11
    rank = n * m * p
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


def _randomized_search(U0, V0, W0, T, target_rank, max_time=30, seed=42):
    """
    Randomized search: try to find a rank-`target_rank` decomposition by
    taking random linear combinations of the original terms and solving
    for the remaining degrees of freedom.
    
    Strategy: 
    1. Start with U0 (4 x r0). Pick a random subset of `target_rank` columns
       or random linear combinations.
    2. Fix U to have rank at most 4 (it's already 4 x r).
    3. For a given U, the equations are linear in V and W.
    4. Use the structure: for each output position (c_idx), the equation is
       sum_r U[a_idx, r] * V[b_idx, r] * W[c_idx, r] = T[a_idx, b_idx, c_idx]
    
    Actually, let's use a different approach:
    - The tensor has 4*22*22 = 1936 entries, but only 44 are nonzero.
    - For rank r, we have 4r + 22r + 22r = 48r variables.
    - The equations are: for each (a,b,c) with T[a,b,c]=1, sum_r U[a,r]*V[b,r]*W[c,r] = 1
      and for each (a,b,c) with T[a,b,c]=0, sum_r U[a,r]*V[b,r]*W[c,r] = 0
    
    This is a system of polynomial equations. Let's use a random initialization
    and iterative refinement.
    """
    rng = np.random.default_rng(seed)
    start_time = time.time()
    n, m, p = 2, 2, 11
    dim_a, dim_b, dim_c = 4, 22, 22
    r = target_rank
    
    # Get the list of nonzero positions
    nonzero_positions = []
    zero_positions = []
    for a in range(dim_a):
        for b in range(dim_b):
            for c in range(dim_c):
                if T[a, b, c] != 0:
                    nonzero_positions.append((a, b, c, 1.0))
                else:
                    zero_positions.append((a, b, c, 0.0))
    
    # For efficiency, only check nonzero positions during search
    # (zero positions are more numerous but less critical for finding a solution)
    
    best_U, best_V, best_W = None, None, None
    best_error = float('inf')
    
    for attempt in range(50):
        if time.time() - start_time > max_time:
            break
        
        # Random initialization
        U = rng.standard_normal((dim_a, r))
        V = rng.standard_normal((dim_b, r))
        W = rng.standard_normal((dim_c, r))
        
        # Iterative refinement: alternate fixing two matrices and solving for the third
        for iteration in range(100):
            if time.time() - start_time > max_time:
                break
            
            # Fix U and V, solve for W
            # For each c: W[c, r] should satisfy
            # sum_r U[a, r] * V[b, r] * W[c, r] = T[a, b, c] for all (a,b)
            # This is a linear system in W[:, c] for each c
            # Actually, for fixed U, V, the equation for each (a,b,c) is:
            # sum_r (U[a, r] * V[b, r]) * W[c, r] = T[a, b, c]
            # Let M[a, b, r] = U[a, r] * V[b, r]
            # Then for each c: M[:, :, :] contracted with W[:, c] should give T[:, :, c]
            # i.e., np.einsum('abr,cr->abc', M, W) should equal T
            # For fixed M, this is linear in W: for each c, W[:, c] is a vector of length r
            # and the equation is: sum_r M[a, b, r] * W[c, r] = T[a, b, c]
            # This is: (M reshaped) @ W[c, :]^T = T[:, :, c] flattened
            # M reshaped is (dim_a * dim_b, r), T[:, :, c] flattened is (dim_a * dim_b,)
            # So: M_reshaped @ W[c, :]^T = T_flat_c
            # This is a linear system: M_reshaped (4*22 x r) @ W[c, :]^T (r x 1) = T_flat_c (4*22 x 1)
            
            M = U[:, :, None] * V[None, :, :]  # shape (4, 22, r)
            M_reshaped = M.transpose(2, 0, 1).reshape(r, dim_a * dim_b)  # (r, 44)
            
            # For each c, solve: M_reshaped.T @ W[c, :] = T[:, :, c].flatten()
            # M_reshaped.T is (44, r), W[c, :] is (r,), T[:, :, c].flatten() is (44,)
            # This is an overdetermined system (44 equations, r unknowns) if r < 44
            # Use least squares
            T_flat = T.reshape(dim_a * dim_b, dim_c)  # (44, 22)
            
            W_new = np.zeros((dim_c, r))
            for c in range(dim_c):
                # Solve M_reshaped.T @ W[c, :] = T_flat[:, c]
                # M_reshaped.T is (44, r)
                sol, residuals, rank_ls, sv = np.linalg.lstsq(M_reshaped.T, T_flat[:, c], rcond=None)
                W_new[c, :] = sol
            
            # Now fix U and W, solve for V
            # Similarly: for each b, solve for V[b, :]
            M2 = U[:, :, None] * W[None, :, :]  # shape (4, 22, r) but W is (22, r) so this is (4, 22, r)
            # Wait, W is (dim_c, r) = (22, r), and we need M2[a, b, r] = U[a, r] * W[c, r] for each c
            # Actually, the equation is: sum_r U[a, r] * V[b, r] * W[c, r] = T[a, b, c]
            # For fixed U, W: sum_r (U[a, r] * W[c, r]) * V[b, r] = T[a, b, c]
            # For each b: V[b, :] is (r,), and the equation for all (a, c) is:
            # sum_r V[b, r] * U[a, r] * W[c, r] = T[a, b, c]
            # Let N[a, c, r] = U[a, r] * W[c, r]
            # Then: N[:, :, :] contracted with V[b, :] should give T[:, b, :]
            # i.e., np.einsum('acr,br->abc', N, V) should equal T
            # For fixed N: for each b, N_reshaped @ V[b, :]^T = T[:, b, :].flatten()
            # N_reshaped: (r, dim_a * dim_c) = (r, 88)
            # T[:, b, :].flatten(): (88,)
            
            N = U[:, None, :] * W[None, :, :]  # (4, 22, r)
            N_reshaped = N.transpose(2, 0, 1).reshape(r, dim_a * dim_c)  # (r, 88)
            
            V_new = np.zeros((dim_b, r))
            for b in range(dim_b):
                T_flat_b = T[:, b, :].flatten()  # (88,)
                sol, residuals, rank_ls, sv = np.linalg.lstsq(N_reshaped.T, T_flat_b, rcond=None)
                V_new[b, :] = sol
            
            # Check error
            T_decomp = np.einsum('ar,br,cr->abc', U, V_new, W_new)
            error = np.linalg.norm(T_decomp - T)
            
            if error < best_error:
                best_error = error
                best_U, best_V, best_W = U.copy(), V_new.copy(), W_new.copy()
            
            if error < 1e-6:
                break
            
            # Update V and W
            V, W = V_new, W_new
        
        if best_error < 1e-6:
            break
    
    if best_U is not None and best_error < 1e-6:
        return best_U, best_V, best_W, best_error
    
    return None, None, None, best_error


def _try_rank_reduction(U, V, W, T, max_time=30):
    """
    Attempt to reduce rank by finding linearly dependent terms.
    """
    start_time = time.time()
    current_rank = U.shape[1]

    if current_rank <= 1:
        return U, V, W

    # Flatten each rank-1 tensor into a vector
    dim_a, dim_b, dim_c = U.shape[0], V.shape[0], W.shape[0]
    M = np.zeros((dim_a * dim_b * dim_c, current_rank), dtype=np.float64)
    for r in range(current_rank):
        M[:, r] = np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r]).flatten()

    # Use SVD to find the numerical rank
    try:
        U_svd, s_vals, Vt = np.linalg.svd(M, full_matrices=False)
        tol = max(M.shape) * np.finfo(float).eps * s_vals[0]
        num_rank = np.sum(s_vals > tol)
    except Exception:
        num_rank = current_rank

    if num_rank < current_rank:
        try:
            Q, R = np.linalg.qr(M, mode='complete')
            diag_R = np.abs(np.diag(R))
            pivot_cols = np.argsort(diag_R)[::-1][:num_rank]
            pivot_cols = np.sort(pivot_cols)
            U_new = U[:, pivot_cols]
            V_new = V[:, pivot_cols]
            W_new = W[:, pivot_cols]
            if _verify(U_new, V_new, W_new, T) and U_new.shape[1] < current_rank:
                return U_new, V_new, W_new
        except Exception:
            pass

    # Greedy: remove one column at a time
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


def _try_lower_rank_search(U0, V0, W0, T, target_rank, max_time=30):
    """
    Try to find a decomposition of exactly `target_rank` by using the structure
    of the problem. Since the A-space is only 4-dimensional, we can parameterize
    U as a 4 x r matrix and solve for V and W.
    """
    start_time = time.time()
    n, m, p = 2, 2, 11
    dim_a, dim_b, dim_c = 4, 22, 22
    r = target_rank
    
    # The key insight: for a given U (4 x r), the problem becomes linear in V and W.
    # Specifically, for each output position c, we need:
    #   sum_{a,b} T[a,b,c] * (sum_r U[a,r] * V[b,r] * W[c,r]) = 0 for all (a,b) where T=0
    #   sum_r U[a,r] * V[b,r] * W[c,r] = 1 for all (a,b) where T=1
    
    # Let's fix U to be a random 4 x r matrix with rank r (or less).
    # Then for each c, the equation is:
    #   For all (a,b): sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
    # This can be written as: (U * V^T element-wise) @ W[c,:]^T = T[:, :, c].flatten()
    # where U * V^T is (4, 22, r) and we contract over r.
    
    # Let M[a,b,r] = U[a,r] * V[b,r]. Then for each c:
    #   sum_r M[a,b,r] * W[c,r] = T[a,b,c]
    # This is a linear system in W[:, c] for each c.
    
    # Strategy: iterate over random U matrices, solve for V and W.
    
    rng = np.random.default_rng(123)
    best_error = float('inf')
    best_solution = None
    
    for attempt in range(30):
        if time.time() - start_time > max_time:
            break
        
        # Random U with structure: try to make it have low rank
        U = rng.standard_normal((dim_a, r))
        
        # For this U, we need to find V (22 x r) and W (22 x r) such that
        # sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c] for all (a,b,c)
        
        # This is a bilinear system. Let's use an alternating least squares approach.
        V = rng.standard_normal((dim_b, r))
        W = rng.standard_normal((dim_c, r))
        
        for iteration in range(50):
            if time.time() - start_time > max_time:
                break
            
            # Fix U, V, solve for W
            # For each c: sum_r (U[a,r] * V[b,r]) * W[c,r] = T[a,b,c] for all (a,b)
            # Let M[a,b,r] = U[a,r] * V[b,r]
            # Then: M_reshaped @ W[c,:]^T = T[:,:,c].flatten()
            # M_reshaped is (r, 4*22) = (r, 88)
            M = U[:, :, None] * V[None, :, :]  # (4, 22, r)
            M_reshaped = M.transpose(2, 0, 1).reshape(r, dim_a * dim_b)  # (r, 88)
            
            W_new = np.zeros((dim_c, r))
            T_flat = T.reshape(dim_a * dim_b, dim_c)  # (88, 22)
            
            for c in range(dim_c):
                sol, _, _, _ = np.linalg.lstsq(M_reshaped.T, T_flat[:, c], rcond=None)
                W_new[c, :] = sol
            
            # Fix U, W, solve for V
            # For each b: sum_r (U[a,r] * W[c,r]) * V[b,r] = T[a,b,c] for all (a,c)
            N = U[:, None, :] * W[None, :, :]  # (4, 22, r)
            N_reshaped = N.transpose(2, 0, 1).reshape(r, dim_a * dim_c)  # (r, 88)
            
            V_new = np.zeros((dim_b, r))
            for b in range(dim_b):
                T_flat_b = T[:, b, :].flatten()  # (88,)
                sol, _, _, _ = np.linalg.lstsq(N_reshaped.T, T_flat_b, rcond=None)
                V_new[b, :] = sol
            
            # Check error
            T_decomp = np.einsum('ar,br,cr->abc', U, V_new, W_new)
            error = np.linalg.norm(T_decomp - T)
            
            if error < best_error:
                best_error = error
                best_solution = (U.copy(), V_new.copy(), W_new.copy())
            
            if error < 1e-8:
                break
            
            V, W = V_new, W_new
        
        if best_error < 1e-8:
            break
    
    if best_solution is not None and best_error < 1e-8:
        return best_solution
    
    return None


def construct():
    start_time = time.time()
    DEADLINE = 140  # seconds, leave margin before 150s limit

    T = _build_target_tensor()

    best_U, best_V, best_W = None, None, None
    best_rank = float('inf')

    # Try the subspace splitting construction (rank 38)
    try:
        U, V, W = _build_subspace_splitting_decomposition()
        if _verify(U, V, W, T):
            best_U, best_V, best_W = U, V, W
            best_rank = U.shape[1]
    except Exception:
        pass

    # Try the Strassen-based construction (rank 39)
    try:
        U, V, W = _build_strassen_decomposition()
        if _verify(U, V, W, T):
            if U.shape[1] < best_rank:
                best_U, best_V, best_W = U, V, W
                best_rank = U.shape[1]
    except Exception:
        pass

    # Use the best construction found so far
    if best_U is not None:
        U, V, W = best_U, best_V, best_W

        # Try rank reduction via SVD/QR
        remaining_time = DEADLINE - (time.time() - start_time) - 5
        if remaining_time > 10:
            try:
                U_red, V_red, W_red = _try_rank_reduction(
                    U, V, W, T, max_time=min(remaining_time, 30)
                )
                if U_red.shape[1] < U.shape[1]:
                    U, V, W = U_red, V_red, W_red
            except Exception:
                pass

        # Try to find lower rank solutions using the algebraic search
        # Try ranks from best_rank-1 down to 30
        for target_rank in range(best_rank - 1, 29, -1):
            remaining_time = DEADLINE - (time.time() - start_time) - 5
            if remaining_time < 5:
                break
            
            try:
                result = _try_lower_rank_search(U, V, W, T, target_rank, max_time=min(remaining_time, 20))
                if result is not None:
                    U_new, V_new, W_new = result
                    if _verify(U_new, V_new, W_new, T):
                        U, V, W = U_new, V_new, W_new
                        best_rank = target_rank
                        break  # Found a solution at this rank, stop
            except Exception:
                pass
        
        # Final verification before returning
        if _verify(U, V, W, T):
            return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}

    # Fall back to schoolbook rank 44
    U, V, W = _build_schoolbook()
    if _verify(U, V, W, T):
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}

    # Should not reach here
    raise RuntimeError("No valid decomposition found")
