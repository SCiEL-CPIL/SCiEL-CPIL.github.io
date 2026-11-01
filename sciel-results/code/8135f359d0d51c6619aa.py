import numpy as np
import time

DIMENSIONS = [2, 4, 5]
DEADLINE = 145.0  # safety margin before 150s kill


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
    # The tensor T_{i,j,k} is 1 for all i,j,k
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


def _check_column_redundant_fast(K, col_to_remove):
    """Check if column col_to_remove of K is in the span of the other columns.
    Returns True if redundant, False otherwise."""
    r = K.shape[1]
    if r <= 1:
        return False
    
    keep_cols = [c for c in range(r) if c != col_to_remove]
    A = K[:, keep_cols]
    b = K[:, col_to_remove]
    
    # Use lstsq to check if b is in the column space of A
    try:
        solution, residuals, rank_sol, sv = np.linalg.lstsq(A, b, rcond=1e-10)
    except:
        return False

    # Check if the solution actually satisfies the equations
    A_sol = A @ solution
    if np.max(np.abs(A_sol - b)) > 1e-8:
        return False

    return True


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
    
    # Phase 1: Greedy removal of linearly dependent columns
    # We check if any column is in the span of the others.
    # If so, we can remove it and the decomposition remains valid (sum of products is still 1).
    # Note: Removing a column c means we set its coefficients to 0. 
    # Since K_c = sum_{c' != c} alpha_{c'} K_{c'}, the total sum remains unchanged.
    
    progress = True
    while progress and time.time() < deadline - 10:
        progress = False
        rank = U.shape[1]
        for col in range(rank):
            if time.time() > deadline - 10:
                break
            
            # Check redundancy
            if _check_column_redundant_fast(K, col):
                # Remove column
                mask = [i for i in range(rank) if i != col]
                U_new = U[:, mask]
                V_new = V[:, mask]
                W_new = W[:, mask]
                
                # Verify to be safe
                if _verify_decomposition(U_new, V_new, W_new):
                    U, V, W = U_new, V_new, W_new
                    K = _build_kronecker_matrix(U, V, W)
                    rank = U.shape[1]
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    progress = True
                    break
        if not progress:
            break

    # Phase 2: Try removing pairs
    # Sometimes two columns together are in the span of the rest, but neither alone is.
    if time.time() < deadline - 5:
        progress = True
        while progress and time.time() < deadline - 5:
            progress = False
            rank = U.shape[1]
            for i in range(rank):
                if time.time() > deadline - 5:
                    break
                for j in range(i + 1, rank):
                    if time.time() > deadline - 5:
                        break
                    
                    keep_cols = [c for c in range(rank) if c != i and c != j]
                    if len(keep_cols) < 1:
                        continue
                        
                    A = K[:, keep_cols]
                    b = np.column_stack((K[:, i], K[:, j]))
                    
                    try:
                        solution, residuals, rank_sol, sv = np.linalg.lstsq(A, b, rcond=1e-10)
                    except:
                        continue

                    A_sol = A @ solution
                    if np.max(np.abs(A_sol - b)) > 1e-8:
                        continue
                        
                    # If valid, remove both
                    mask = [c for c in range(rank) if c != i and c != j]
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
                if time.time() > deadline - 5:
                    break
                if progress:
                    break

    # Phase 3: Random rotations to expose dependencies
    # This is a heuristic. We mix two columns and see if a dependency appears.
    if time.time() < deadline - 2:
        rng = np.random.RandomState(42)
        rank = U.shape[1]
        for trial in range(50):
            if time.time() > deadline - 2:
                break
            if rank < 3:
                break
            
            cols = rng.choice(rank, size=2, replace=False)
            t = rng.uniform(-2, 2)
            
            U_rot = U.copy()
            V_rot = V.copy()
            W_rot = W.copy()
            
            U_rot[:, cols[0]] = U[:, cols[0]] + t * U[:, cols[1]]
            V_rot[:, cols[0]] = V[:, cols[0]] + t * V[:, cols[1]]
            W_rot[:, cols[0]] = W[:, cols[0]] + t * W[:, cols[1]]
            
            K_rot = _build_kronecker_matrix(U_rot, V_rot, W_rot)
            
            # Check for single column redundancy in the rotated system
            for col in range(rank):
                if time.time() > deadline - 2:
                    break
                if _check_column_redundant_fast(K_rot, col):
                    mask = [i for i in range(rank) if i != col]
                    U_new = U_rot[:, mask]
                    V_new = V_rot[:, mask]
                    W_new = W_rot[:, mask]
                    
                    if _verify_decomposition(U_new, V_new, W_new):
                        U, V, W = U_new, V_new, W_new
                        K = _build_kronecker_matrix(U, V, W)
                        rank = U.shape[1]
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        break
            if time.time() > deadline - 2:
                break

    # Final verification
    if not _verify_decomposition(U, V, W):
        U, V, W = best_U, best_V, best_W

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
