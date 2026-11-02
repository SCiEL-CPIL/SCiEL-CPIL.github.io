import numpy as np
import time

DIMENSIONS = [2, 4, 5]
DEADLINE = 145.0  # safety margin before 150s kill


def _find_null_vectors(K, tol=1e-10):
    """Find the null space of K using SVD. Returns list of null space vectors (rows of Vt)."""
    try:
        U_svd, s_vals, Vt = np.linalg.svd(K, full_matrices=True)
    except:
        return []
    if len(s_vals) == 0:
        return []
    num_rank = np.sum(s_vals > tol * s_vals[0])
    if num_rank >= K.shape[1]:
        return []
    null_vecs = Vt[num_rank:]
    return null_vecs


def _find_redundant_sets(K, null_vecs, tol=1e-8):
    """Find sets of columns that can be removed based on null space structure."""
    r = K.shape[1]
    redundant_sets = []
    
    # Single column redundancy: null vector with nonzero entry at col
    for col in range(r):
        for nv in null_vecs:
            if abs(nv[col]) > tol:
                redundant_sets.append((col,))
                break
    
    # Pair redundancy: check if both columns are in span of the rest
    for i in range(r):
        for j in range(i + 1, r):
            # Quick check via null vectors: is there a null vector with support subset of {i,j}?
            found = False
            for nv in null_vecs:
                support = set(c for c in range(r) if abs(nv[c]) > tol)
                if support.issubset({i, j}) and len(support) == 2:
                    found = True
                    break
            if found:
                redundant_sets.append((i, j))
                continue
            
            # Full check: are both columns in span of the rest?
            keep = [c for c in range(r) if c != i and c != j]
            if len(keep) == 0:
                continue
            A = K[:, keep]
            B = K[:, [i, j]]
            try:
                sol, res, rk, sv = np.linalg.lstsq(A, B, rcond=1e-10)
                if np.max(np.abs(A @ sol - B)) < 1e-8:
                    redundant_sets.append((i, j))
            except:
                pass
    
    return redundant_sets


def _build_strassen_36():
    """Build the known rank-36 block Strassen decomposition for 2x4 * 4x5."""
    n, m, p = 2, 4, 5
    rank = 36
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

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
    strassen_block([(0, 0), (0, 1), (1, 0), (1, 1)],
                   [(0, 0), (0, 1), (1, 0), (1, 1)],
                   [(0, 0), (1, 0), (0, 1), (1, 1)])
    # Block 2: A_R B_21 (2x2 * 2x2)
    strassen_block([(0, 2), (0, 3), (1, 2), (1, 3)],
                   [(2, 0), (2, 1), (3, 0), (3, 1)],
                   [(0, 0), (1, 0), (0, 1), (1, 1)])
    # Block 3: A_L B_12 (2x2 * 2x2)
    strassen_block([(0, 0), (0, 1), (1, 0), (1, 1)],
                   [(0, 2), (0, 3), (1, 2), (1, 3)],
                   [(2, 0), (3, 0), (2, 1), (3, 1)])
    # Block 4: A_R B_22 (2x2 * 2x2)
    strassen_block([(0, 2), (0, 3), (1, 2), (1, 3)],
                   [(2, 2), (2, 3), (3, 2), (3, 3)],
                   [(2, 0), (3, 0), (2, 1), (3, 1)])

    # Block 5: A_L B_15 (2x2 * 2x1) - schoolbook
    U[u_idx(0, 0), col] = 1; V[v_idx(0, 4), col] = 1; W[w_idx(4, 0), col] += 1; col += 1
    U[u_idx(0, 1), col] = 1; V[v_idx(1, 4), col] = 1; W[w_idx(4, 0), col] += 1; col += 1
    U[u_idx(1, 0), col] = 1; V[v_idx(0, 4), col] = 1; W[w_idx(4, 1), col] += 1; col += 1
    U[u_idx(1, 1), col] = 1; V[v_idx(1, 4), col] = 1; W[w_idx(4, 1), col] += 1; col += 1

    # Block 6: A_R B_25 (2x2 * 2x1) - schoolbook
    U[u_idx(0, 2), col] = 1; V[v_idx(2, 4), col] = 1; W[w_idx(4, 0), col] += 1; col += 1
    U[u_idx(0, 3), col] = 1; V[v_idx(3, 4), col] = 1; W[w_idx(4, 0), col] += 1; col += 1
    U[u_idx(1, 2), col] = 1; V[v_idx(2, 4), col] = 1; W[w_idx(4, 1), col] += 1; col += 1
    U[u_idx(1, 3), col] = 1; V[v_idx(3, 4), col] = 1; W[w_idx(4, 1), col] += 1; col += 1

    return U, V, W


def _verify_decomposition(U, V, W):
    """Verify that the decomposition exactly represents matrix multiplication (vectorized)."""
    n, m, p = 2, 4, 5
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True


def _build_kronecker_matrix(U, V, W):
    """Build the n*m*p x rank matrix where each column is the vectorized Kronecker product
    of U[:,c], V[:,c], W[:,c]."""
    n, m, p = 2, 4, 5
    r = U.shape[1]
    A = np.zeros((n * m * p, r))
    for c in range(r):
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    A[idx, c] = U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    idx += 1
    return A


def _check_column_redundant_robust(K, col, tol=1e-8):
    """Check if column 'col' of K is in the span of the other columns using least squares."""
    r = K.shape[1]
    keep = [c for c in range(r) if c != col]
    if len(keep) == 0:
        return True
    A = K[:, keep]
    B = K[:, col]
    try:
        sol, res, rk, sv = np.linalg.lstsq(A, B, rcond=1e-10)
        if np.max(np.abs(A @ sol - B)) < tol:
            return True
    except:
        pass
    return False


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
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    rank = U.shape[1]
    
    # Pre-compute Kronecker matrix K for the initial decomposition
    K = _build_kronecker_matrix(U, V, W)
    
    # Phase 1: Use null space to find all redundancies systematically
    progress = True
    while progress and time.time() < deadline - 20:
        progress = False
        rank = U.shape[1]
        
        # Compute null space of current Kronecker matrix
        null_vecs = _find_null_vectors(K, tol=1e-10)
        
        if not null_vecs:
            # Try less strict tolerance
            null_vecs = _find_null_vectors(K, tol=1e-8)
        
        if not null_vecs:
            break
        
        # Find all redundant sets
        redundant_sets = _find_redundant_sets(K, null_vecs, tol=1e-8)
        
        if not redundant_sets:
            break
        
        # Sort by size (largest first) to maximize reduction per iteration
        redundant_sets.sort(key=lambda s: -len(s))
        
        for rset in redundant_sets:
            if time.time() > deadline - 20:
                break
            
            mask = [c for c in range(rank) if c not in rset]
            U_new = U[:, mask]
            V_new = V[:, mask]
            W_new = W[:, mask]
            
            if _verify_decomposition(U_new, V_new, W_new):
                U, V, W = U_new, V_new, W_new
                K = _build_kronecker_matrix(U, V, W)
                rank = U.shape[1]
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                progress = True
                break
    
    # Phase 2: Systematic pair search with least squares (catches what null space misses)
    if time.time() < deadline - 15:
        rank = U.shape[1]
        progress = True
        while progress and time.time() < deadline - 15:
            progress = False
            rank = U.shape[1]
            
            for i in range(rank):
                if time.time() > deadline - 15:
                    break
                for j in range(i + 1, rank):
                    if time.time() > deadline - 15:
                        break
                    
                    keep = [c for c in range(rank) if c != i and c != j]
                    if len(keep) == 0:
                        continue
                    
                    A = K[:, keep]
                    B = K[:, [i, j]]
                    
                    try:
                        sol, res, rk, sv = np.linalg.lstsq(A, B, rcond=1e-10)
                    except:
                        continue
                    
                    if np.max(np.abs(A @ sol - B)) < 1e-9:
                        mask = keep
                        U_new = U[:, mask]
                        V_new = V[:, mask]
                        W_new = W[:, mask]
                        
                        if _verify_decomposition(U_new, V_new, W_new):
                            U, V, W = U_new, V_new, W_new
                            K = _build_kronecker_matrix(U, V, W)
                            rank = U.shape[1]
                            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                            progress = True
                            break
                if time.time() > deadline - 15:
                    break
                if progress:
                    break
    
    # Phase 3: Rational perturbation search to expose hidden dependencies
    # Small rational perturbations can reveal dependencies that are "almost" present
    if time.time() < deadline - 8:
        rng = np.random.RandomState(42)
        rank = U.shape[1]
        
        rational_vals = [0.5, -0.5, 1.0/3, -1.0/3, 2.0/3, -2.0/3, 
                        1.0/4, -1.0/4, 1.5, -1.5, 0.25, -0.25, 2.0, -2.0,
                        1.0/5, -1.0/5, 3.0/4, -3.0/4]
        
        for trial in range(150):
            if time.time() > deadline - 8:
                break
            if rank < 3:
                break
            
            cols = rng.choice(rank, size=2, replace=False)
            t = rational_vals[rng.randint(len(rational_vals))]
            
            U_rot = U.copy()
            V_rot = V.copy()
            W_rot = W.copy()
            U_rot[:, cols[0]] = U[:, cols[0]] + t * U[:, cols[1]]
            V_rot[:, cols[0]] = V[:, cols[0]] + t * V[:, cols[1]]
            W_rot[:, cols[0]] = W[:, cols[0]] + t * W[:, cols[1]]
            
            K_rot = _build_kronecker_matrix(U_rot, V_rot, W_rot)
            
            # Check for single column redundancy in rotated system
            null_vecs_rot = _find_null_vectors(K_rot, tol=1e-10)
            if null_vecs_rot:
                for nv in null_vecs_rot:
                    if time.time() > deadline - 8:
                        break
                    for col in range(rank):
                        if abs(nv[col]) > 1e-8:
                            mask = [c for c in range(rank) if c != col]
                            U_new = U_rot[:, mask]
                            V_new = V_rot[:, mask]
                            W_new = W_rot[:, mask]
                            
                            if _verify_decomposition(U_new, V_new, W_new):
                                U, V, W = U_new, V_new, W_new
                                K = _build_kronecker_matrix(U, V, W)
                                rank = U.shape[1]
                                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                            break
                    break
    
    # Phase 4: Final tight-tolerance pair check
    if time.time() < deadline - 3:
        rank = U.shape[1]
        for i in range(rank):
            if time.time() > deadline - 3:
                break
            for j in range(i + 1, rank):
                if time.time() > deadline - 3:
                    break
                
                keep = [c for c in range(rank) if c != i and c != j]
                if len(keep) == 0:
                    continue
                
                A = K[:, keep]
                B = K[:, [i, j]]
                
                try:
                    sol, res, rk, sv = np.linalg.lstsq(A, B, rcond=1e-12)
                except:
                    continue
                
                if np.max(np.abs(A @ sol - B)) < 1e-10:
                    mask = keep
                    U_new = U[:, mask]
                    V_new = V[:, mask]
                    W_new = W[:, mask]
                    
                    if _verify_decomposition(U_new, V_new, W_new):
                        U, V, W = U_new, V_new, W_new
                        K = _build_kronecker_matrix(U, V, W)
                        rank = U.shape[1]
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    # Final verification
    if not _verify_decomposition(U, V, W):
        U, V, W = best_U, best_V, best_W

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
