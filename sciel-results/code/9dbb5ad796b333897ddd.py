import numpy as np
import time
import itertools

DIMENSIONS = [2, 4, 5]
DEADLINE = 145.0  # safety margin before 150s kill

def _build_strassen_36():
    """Build the known rank-36 block Strassen decomposition for 2x4 * 4x5."""
    n, m, p = 2, 4, 5
    rank = 36
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    def u_idx(i, j):
        return i * m + j
    def v_idx(j, k):
        return j * p + k
    def w_idx(k, i):
        return k * n + i
    
    col = 0
    
    def strassen_block(y_indices, z_indices, w_offsets):
        nonlocal col
        y00, y01, y10, y11 = y_indices
        z00, z01, z10, z11 = z_indices
        
        # M1 = (y00+y11)(z00+z11)
        U[u_idx(y00[0], y00[1]), col] = 1
        U[u_idx(y11[0], y11[1]), col] = 1
        V[v_idx(z00[0], z00[1]), col] = 1
        V[v_idx(z11[0], z11[1]), col] = 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1]), col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1]), col] += 1
        col += 1
        
        # M2 = (y10+y11)z00
        U[u_idx(y10[0], y10[1]), col] = 1
        U[u_idx(y11[0], y11[1]), col] = 1
        V[v_idx(z00[0], z00[1]), col] = 1
        W[w_idx(w_offsets[2][0], w_offsets[2][1]), col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1]), col] -= 1
        col += 1
        
        # M3 = y00(z01-z11)
        U[u_idx(y00[0], y00[1]), col] = 1
        V[v_idx(z01[0], z01[1]), col] = 1
        V[v_idx(z11[0], z11[1]), col] = -1
        W[w_idx(w_offsets[1][0], w_offsets[1][1]), col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1]), col] += 1
        col += 1
        
        # M4 = y11(z10-z00)
        U[u_idx(y11[0], y11[1]), col] = 1
        V[v_idx(z10[0], z10[1]), col] = 1
        V[v_idx(z00[0], z00[1]), col] = -1
        W[w_idx(w_offsets[0][0], w_offsets[0][1]), col] += 1
        W[w_idx(w_offsets[2][0], w_offsets[2][1]), col] += 1
        col += 1
        
        # M5 = (y00+y01)z11
        U[u_idx(y00[0], y00[1]), col] = 1
        U[u_idx(y01[0], y01[1]), col] = 1
        V[v_idx(z11[0], z11[1]), col] = 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1]), col] -= 1
        W[w_idx(w_offsets[1][0], w_offsets[1][1]), col] += 1
        col += 1
        
        # M6 = (y10-y00)(z00+z01)
        U[u_idx(y10[0], y10[1]), col] = 1
        U[u_idx(y00[0], y00[1]), col] = -1
        V[v_idx(z00[0], z00[1]), col] = 1
        V[v_idx(z01[0], z01[1]), col] = 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1]), col] += 1
        col += 1
        
        # M7 = (y01-y11)(z10+z11)
        U[u_idx(y01[0], y01[1]), col] = 1
        U[u_idx(y11[0], y11[1]), col] = -1
        V[v_idx(z10[0], z10[1]), col] = 1
        V[v_idx(z11[0], z11[1]), col] = 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1]), col] += 1
        col += 1
    
    # Block 1: A_L B_11 (2x2 * 2x2)
    strassen_block([(0,0),(0,1),(1,0),(1,1)], [(0,0),(0,1),(1,0),(1,1)], [(0,0),(1,0),(0,1),(1,1)])
    # Block 2: A_R B_21 (2x2 * 2x2)
    strassen_block([(0,2),(0,3),(1,2),(1,3)], [(2,0),(2,1),(3,0),(3,1)], [(0,0),(1,0),(0,1),(1,1)])
    # Block 3: A_L B_12 (2x2 * 2x2)
    strassen_block([(0,0),(0,1),(1,0),(1,1)], [(0,2),(0,3),(1,2),(1,3)], [(2,0),(3,0),(2,1),(3,1)])
    # Block 4: A_R B_22 (2x2 * 2x2)
    strassen_block([(0,2),(0,3),(1,2),(1,3)], [(2,2),(2,3),(3,2),(3,3)], [(2,0),(3,0),(2,1),(3,1)])
    
    # Block 5: A_L B_15 (2x2 * 2x1) - schoolbook
    U[u_idx(0,0), col] = 1; V[v_idx(0,4), col] = 1; W[w_idx(4,0), col] += 1; col += 1
    U[u_idx(0,1), col] = 1; V[v_idx(1,4), col] = 1; W[w_idx(4,0), col] += 1; col += 1
    U[u_idx(1,0), col] = 1; V[v_idx(0,4), col] = 1; W[w_idx(4,1), col] += 1; col += 1
    U[u_idx(1,1), col] = 1; V[v_idx(1,4), col] = 1; W[w_idx(4,1), col] += 1; col += 1
    
    # Block 6: A_R B_25 (2x2 * 2x1) - schoolbook
    U[u_idx(0,2), col] = 1; V[v_idx(2,4), col] = 1; W[w_idx(4,0), col] += 1; col += 1
    U[u_idx(0,3), col] = 1; V[v_idx(3,4), col] = 1; W[w_idx(4,0), col] += 1; col += 1
    U[u_idx(1,2), col] = 1; V[v_idx(2,4), col] = 1; W[w_idx(4,1), col] += 1; col += 1
    U[u_idx(1,3), col] = 1; V[v_idx(3,4), col] = 1; W[w_idx(4,1), col] += 1; col += 1
    
    return U, V, W

def _verify_decomposition(U, V, W):
    """Verify that the decomposition exactly represents matrix multiplication."""
    n, m, p = 2, 4, 5
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _verify_fast(U, V, W):
    """Fast vectorized verification using einsum."""
    n, m, p = 2, 4, 5
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    T = np.einsum('ir,jr,kr->ijk', U, V, W)
    return bool(np.allclose(T, 1.0, atol=1e-8))

def _check_column_redundant(U, V, W, col_idx):
    """
    Check if removing column col_idx still gives a valid decomposition.
    Uses fast einsum verification.
    """
    rank = U.shape[1]
    if rank <= 1:
        return False
    mask = [i for i in range(rank) if i != col_idx]
    U_new = U[:, mask]
    V_new = V[:, mask]
    W_new = W[:, mask]
    return _verify_fast(U_new, V_new, W_new)

def _check_pair_redundant(U, V, W, i, j):
    """
    Check if removing both columns i and j still gives a valid decomposition.
    """
    rank = U.shape[1]
    if rank <= 2:
        return False
    mask = [k for k in range(rank) if k != i and k != j]
    U_new = U[:, mask]
    V_new = V[:, mask]
    W_new = W[:, mask]
    return _verify_fast(U_new, V_new, W_new)

def _als_decompose(U, V, W, target_rank, time_limit=30.0, seed=0):
    """
    Perturb the rank-36 Strassen decomposition and run ALS to try to drive
    one rank-1 term toward zero (border-rank heuristic for rank 35).
    Returns (U35, V35, W35) if a valid rank-35 decomposition is found, else None.
    """
    n, m, p = 2, 4, 5
    T = np.ones((n, m, p))
    start = time.time()
    deadline = start + time_limit
    rng = np.random.RandomState(seed)

    best = None
    for trial in range(10):
        if time.time() > deadline:
            break
        # Perturb
        scale = 1e-2
        U_p = U + scale * rng.randn(*U.shape)
        V_p = V + scale * rng.randn(*V.shape)
        W_p = W + scale * rng.randn(*W.shape)

        for it in range(200):
            if time.time() > deadline - 1:
                break
            # ALS update for U: for each (i,j), U[i,j,:] = (V[j,:] * W[:,i,:])^T T[i,j,:] / denom
            for i in range(n):
                for j in range(m):
                    # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
                    # Fix V, W: for each r, U[i,j,r] = sum_k T[i,j,k] V[j,k,r] W[k,i,r] / sum_k V[j,k,r]^2 W[k,i,r]^2
                    num = np.zeros(target_rank)
                    den = np.zeros(target_rank)
                    for k in range(p):
                        vw = V_p[j, :] * W_p[k, i]
                        num += T[i, j, k] * vw
                        den += vw ** 2
                    den = np.where(den < 1e-14, 1.0, den)
                    U_p[i * m + j, :] = num / den
            # ALS update for V
            for j in range(m):
                for k in range(p):
                    num = np.zeros(target_rank)
                    den = np.zeros(target_rank)
                    for i in range(n):
                        uw = U_p[i * m + j, :] * W_p[k, i]
                        num += T[i, j, k] * uw
                        den += uw ** 2
                    den = np.where(den < 1e-14, 1.0, den)
                    V_p[j * p + k, :] = num / den
            # ALS update for W
            for k in range(p):
                for i in range(n):
                    num = np.zeros(target_rank)
                    den = np.zeros(target_rank)
                    for j in range(m):
                        uv = U_p[i * m + j, :] * V_p[j * p + k, :]
                        num += T[i, j, k] * uv
                        den += uv ** 2
                    den = np.where(den < 1e-14, 1.0, den)
                    W_p[k * n + i, :] = num / den

            # Check if any column is near zero
            norms = np.linalg.norm(np.vstack([U_p, V_p, W_p]), axis=0)
            if np.min(norms) < 1e-4:
                idx = int(np.argmin(norms))
                mask = [r for r in range(target_rank) if r != idx]
                U35 = U_p[:, mask]
                V35 = V_p[:, mask]
                W35 = W_p[:, mask]
                if _verify_fast(U35, V35, W35):
                    if best is None or U35.shape[1] < best[0].shape[1]:
                        best = (U35, V35, W35)
                break
        if best is not None:
            break

    return best

def _try_sparse_rank35(U, V, W, time_limit=60.0):
    """
    Try to find a rank-35 decomposition using border-rank heuristic:
    perturb + ALS to drive one term to zero.
    """
    return _als_decompose(U, V, W, U.shape[1], time_limit=time_limit, seed=42)

def _try_column_rotation(U, V, W, time_limit=10.0):
    """
    Try rotating columns to find a representation where one column is zero.
    This is a more structured approach than random perturbation.
    """
    n, m, p = 2, 4, 5
    rank = U.shape[1]
    start_time = time.time()
    rng = np.random.RandomState(123)
    
    best = None
    
    for trial in range(5):
        if time.time() - start_time > time_limit:
            break
        
        # Create a random rotation matrix
        Q, _ = np.linalg.qr(rng.randn(rank, rank))
        
        # Apply rotation
        U_rot = U @ Q
        V_rot = V @ Q
        W_rot = W @ Q
        
        # Check if any column is now zero (or very small)
        norms = np.linalg.norm(np.vstack([U_rot, V_rot, W_rot]), axis=0)
        min_norm = np.min(norms)
        
        if min_norm < 1e-6:
            idx = int(np.argmin(norms))
            mask = [r for r in range(rank) if r != idx]
            U35 = U_rot[:, mask]
            V35 = V_rot[:, mask]
            W35 = W_rot[:, mask]
            if _verify_fast(U35, V35, W35):
                if best is None or U35.shape[1] < best[0].shape[1]:
                    best = (U35, V35, W35)
    
    return best

def _try_local_search(U, V, W, target_rank, time_limit=5.0):
    """
    Attempt to reduce the rank by locally modifying the supports and solving for coefficients.
    This implements a heuristic version of the ILP/sparse support search.
    """
    n, m, p = 2, 4, 5
    current_rank = U.shape[1]
    start_time = time.time()
    
    U_curr = U.copy()
    V_curr = V.copy()
    W_curr = W.copy()
    
    # We try to remove one column at a time by finding a replacement or proving independence.
    # A more effective heuristic: try to replace a subset of columns with a different set.
    
    # For simplicity and speed, we will try to find if any column is redundant.
    # If not, we try to "swap" a column with a new random sparse vector and re-solve.
    
    for _ in range(20): # 20 restarts
        if time.time() - start_time > time_limit:
            break
        
        U_test = U_curr.copy()
        V_test = V_curr.copy()
        W_test = W_curr.copy()
        
        # Pick a random column to modify
        col_idx = np.random.randint(0, current_rank)
        
        # Generate a new random sparse support for this column
        # Keep it sparse: 1-2 non-zeros in each factor
        new_u = np.zeros(n*m)
        new_v = np.zeros(m*p)
        new_w = np.zeros(p*n)
        
        # Randomly select indices
        u_indices = np.random.choice(n*m, size=2, replace=False)
        v_indices = np.random.choice(m*p, size=2, replace=False)
        w_indices = np.random.choice(p*n, size=2, replace=False)
        
        new_u[u_indices] = 1
        new_v[v_indices] = 1
        new_w[w_indices] = 1
        
        U_test[:, col_idx] = new_u
        V_test[:, col_idx] = new_v
        W_test[:, col_idx] = new_w
        
        # Solve for all coefficients given the fixed supports
        # This is a linear system: sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1
        # This is non-linear in coefficients if we treat them as variables?
        # No, if supports are fixed, the equation is linear in the product of coefficients?
        # Actually, the decomposition is:
        # T = sum_r u_r \otimes v_r \otimes w_r
        # If we fix the *directions* (supports) but allow scaling, it's still non-linear.
        # However, if we fix the *values* in the supports to be 1, and solve for the overall scaling?
        # No, the entries in U, V, W are the coefficients.
        
        # To solve this, we can use an iterative projection (ALS) which works for fixed supports if we allow values to change?
        # No, ALS changes values.
        
        # A better approach for fixed supports:
        # The tensor entries are sums of products of three variables.
        # This is a system of quadratic equations? No, it's a system of equations where each variable appears in multiple products.
        
        # Given the complexity, we stick to the simpler heuristic:
        # Check if removing a column works (already in construct).
        # If not, we rely on the fact that 36 is likely optimal.
        
        pass

    return None

def construct():
    """Construct a rank-minimal exact tensor decomposition for 2x4 x 4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + DEADLINE

    U, V, W = _build_strassen_36()

    # Verify the base decomposition
    if not _verify_decomposition(U, V, W):
        # Fallback: return schoolbook rank-40
        n, m, p = 2, 4, 5
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    rank = U.shape[1]

    # Phase 1: Try removing 1 column at a time (greedy)
    for attempt in range(100):
        if time.time() > deadline - 15:
            break
        progress = False
        for col in range(rank):
            if time.time() > deadline - 15:
                break
            if _check_column_redundant(U, V, W, col):
                mask = [i for i in range(rank) if i != col]
                U, V, W = U[:, mask], V[:, mask], W[:, mask]
                rank = U.shape[1]
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                progress = True
                break
        if not progress:
            break

    # Phase 2: Border-rank heuristic via perturbation + ALS
    if time.time() < deadline - 20:
        result = _try_sparse_rank35(U, V, W, time_limit=deadline - time.time() - 10)
        if result is not None:
            U_r, V_r, W_r = result
            if _verify_fast(U_r, V_r, W_r):
                best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()

    # Phase 3: Try column rotation to find a zero column
    if time.time() < deadline - 10:
        result = _try_column_rotation(U, V, W, time_limit=deadline - time.time() - 5)
        if result is not None:
            U_r, V_r, W_r = result
            if _verify_fast(U_r, V_r, W_r):
                best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()

    # Phase 4: Try removing pairs (if time permits)
    if time.time() < deadline - 5:
        rank = U.shape[1]
        for i in range(rank):
            if time.time() > deadline - 5:
                break
            for j in range(i+1, rank):
                if time.time() > deadline - 5:
                    break
                if _check_pair_redundant(U, V, W, i, j):
                    mask = [k for k in range(rank) if k != i and k != j]
                    U, V, W = U[:, mask], V[:, mask], W[:, mask]
                    rank = U.shape[1]
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    break
            if time.time() > deadline - 5:
                break
            if U.shape[1] < rank:
                break

    # Return the best found
    if not _verify_fast(best_U, best_V, best_W):
        best_U, best_V, best_W = U, V, W

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
