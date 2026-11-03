import time
import random
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit

    n, m, p = DIMENSIONS  # 2, 3, 6

    # The 2x3x6 multiplication splits into two independent 2x3x3 multiplications:
    # B (3x6) = [B1 (3x3) | B2 (3x3)], C (2x6) = [A*B1 | A*B2].
    # So rank(2x3x6) <= 2 * rank(2x3x3).
    # Lower bound for 2x3x6 is max(6,18,12)=18, so rank(2x3x3) >= 9 (since 2*r >= 18).
    # Lower bound for 2x3x3 is max(6,9,6)=9.

    # First, build a guaranteed schoolbook decomposition for the full 2x3x6.
    best_result = _schoolbook_decomposition(n, m, p)
    best_rank = n * m * p  # 36

    # Try to find a low-rank 2x3x3 decomposition, then double it for 2x3x6.
    # Try ranks from the theoretical minimum upward.
    for target_233_rank in [9, 10, 11, 12, 13, 14, 15, 16, 17, 18]:
        if time.time() > deadline - 15:
            break
        result = _try_233_decomposition(target_233_rank, start_time, deadline)
        if result is not None:
            # result is (U233, V233, W233) for 2x3x3
            full = _embed_233_into_236(result, n, m, p)
            if full is not None:
                full_rank = target_233_rank * 2
                if full_rank < best_rank:
                    best_result = full
                    best_rank = full_rank
                break  # found the lowest achievable; stop

    return best_result

def _schoolbook_decomposition(n, m, p):
    """Return the standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _verify_233(U, V, W, R):
    """Verify 2x3x3 decomposition: T[i,j,k]=1 for all i,j,k."""
    n2, m2, p2 = 2, 3, 3
    for i in range(n2):
        for j in range(m2):
            for k in range(p2):
                s = 0.0
                for c in range(R):
                    s += U[i * m2 + j][c] * V[j * p2 + k][c] * W[k * n2 + i][c]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _verify_full(U, V, W, n, m, p):
    """Verify full decomposition: sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] == 1."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _is_half_integer(x):
    """Check if x is an integer or half-integer."""
    if x is None:
        return False
    y = x * 2
    return abs(y - round(y)) < 1e-9

def _round_half(x):
    return round(x * 2) / 2.0

def _try_233_decomposition(R, start_time, deadline):
    """Find a rank-R half-integer decomposition for 2x3x3 via ALS + rounding.
    Returns (U, V, W) as lists, or None."""
    n2, m2, p2 = 2, 3, 3
    rows_u = n2 * m2  # 6
    rows_v = m2 * p2  # 9
    rows_w = p2 * n2  # 6

    # Target tensor T[i, j, k] = 1 for all i,j,k (2x3x3)
    T = np.ones((rows_u, rows_v, rows_w), dtype=float)

    random.seed(12345)
    np.random.seed(12345)

    # Number of restarts scaled by time budget, but capped.
    # Each ALS run does ~30 iterations with 3 linear solves.
    max_restarts = 40

    for restart in range(max_restarts):
        if time.time() > deadline - 10:
            break

        # Initialize U, V, W
        scale = 0.5
        U = np.random.randn(rows_u, R) * scale
        V = np.random.randn(rows_v, R) * scale
        W = np.random.randn(rows_w, R) * scale

        converged = False
        for it in range(30):
            if time.time() > deadline - 10:
                break

            # Solve for U: for each i, T[i,:,:].flatten() = sum_c U[i,c] * (V[:,c]*W[:,c]).flatten()
            # Let Z[r, idx(j,k)] = V[j,r]*W[k,r]. Then T_flat[i,:] = U[i,:] @ Z.T
            U_new = _solve_factor(U, V, W, T, 'U', rows_u, rows_v, rows_w, R)
            if U_new is None:
                break
            V_new = _solve_factor(U_new, V, W, T, 'V', rows_u, rows_v, rows_w, R)
            if V_new is None:
                break
            W_new = _solve_factor(U_new, V_new, W, T, 'W', rows_u, rows_v, rows_w, R)
            if W_new is None:
                break

            diff = (np.max(np.abs(U_new - U)) + np.max(np.abs(V_new - V)) + np.max(np.abs(W_new - W)))
            U, V, W = U_new, V_new, W_new
            if diff < 1e-10:
                converged = True
                break

        if not converged:
            continue

        # Polish: do a few more iterations with tighter tolerance
        for it in range(20):
            if time.time() > deadline - 10:
                break
            U_new = _solve_factor(U, V, W, T, 'U', rows_u, rows_v, rows_w, R)
            if U_new is None:
                break
            V_new = _solve_factor(U_new, V, W, T, 'V', rows_u, rows_v, rows_w, R)
            if V_new is None:
                break
            W_new = _solve_factor(U_new, V_new, W, T, 'W', rows_u, rows_v, rows_w, R)
            if W_new is None:
                break
            U, V, W = U_new, V_new, W_new

        # Check if the continuous solution is close to a half-integer solution.
        # Round to nearest half-integer and verify.
        U_r = np.round(U * 2) / 2.0
        V_r = np.round(V * 2) / 2.0
        W_r = np.round(W * 2) / 2.0

        # Verify the rounded solution
        T_check = np.einsum('ir,jr,kr->ijk', U_r, V_r, W_r)
        if np.max(np.abs(T_check - T)) < 1e-9:
            # Convert to lists
            U_list = [[U_r[i, c] for c in range(R)] for i in range(rows_u)]
            V_list = [[V_r[j, c] for c in range(R)] for j in range(rows_v)]
            W_list = [[W_r[k, c] for c in range(R)] for k in range(rows_w)]
            # Double-check with exact verify
            if _verify_233(U_list, V_list, W_list, R):
                return (U_list, V_list, W_list)

        # Try a local search: for each factor, try small perturbations to reach half-integers.
        # This is expensive; only do it if we're close.
        resid = np.max(np.abs(T_check - T))
        if resid < 0.5:
            # Try rounding with a few different offsets / snapping strategies
            found = _local_half_integer_search(U, V, W, T, rows_u, rows_v, rows_w, R, start_time, deadline)
            if found is not None:
                return found

    return None

def _solve_factor(U, V, W, T, which, rows_u, rows_v, rows_w, R):
    """Solve for one factor given the other two, using least squares.
    T[i,j,k] = sum_c U[i,c] V[j,c] W[k,c]."""
    if which == 'U':
        # For each i: T[i,:,:].flatten() = sum_c U[i,c] * (V[:,c]*W[:,c]).flatten()
        # Z[c, idx(j,k)] = V[j,c]*W[k,c]
        Z = np.zeros((R, rows_v * rows_w))
        for c in range(R):
            Z[c, :] = np.outer(V[:, c], W[:, c]).flatten()
        T_flat = T.reshape(rows_u, rows_v * rows_w)
        # T_flat[i,:] = U[i,:] @ Z.T  =>  Z.T @ U[i,:].T = T_flat[i,:].T
        try:
            U_sol, _, _, _ = np.linalg.lstsq(Z.T, T_flat.T, rcond=None)
            return U_sol.T
        except np.linalg.LinAlgError:
            return None
    elif which == 'V':
        # For each j: T[:,j,:].flatten() = sum_c V[j,c] * (U[:,c]*W[:,c]).flatten()
        Z = np.zeros((R, rows_u * rows_w))
        for c in range(R):
            Z[c, :] = np.outer(U[:, c], W[:, c]).flatten()
        T_flat = np.transpose(T, (1, 0, 2)).reshape(rows_v, rows_u * rows_w)
        try:
            V_sol, _, _, _ = np.linalg.lstsq(Z.T, T_flat.T, rcond=None)
            return V_sol.T
        except np.linalg.LinAlgError:
            return None
    elif which == 'W':
        # For each k: T[:,:,k].flatten() = sum_c W[k,c] * (U[:,c]*V[:,c]).flatten()
        Z = np.zeros((R, rows_u * rows_v))
        for c in range(R):
            Z[c, :] = np.outer(U[:, c], V[:, c]).flatten()
        T_flat = np.transpose(T, (2, 0, 1)).reshape(rows_w, rows_u * rows_v)
        try:
            W_sol, _, _, _ = np.linalg.lstsq(Z.T, T_flat.T, rcond=None)
            return W_sol.T
        except np.linalg.LinAlgError:
            return None
    return None

def _local_half_integer_search(U, V, W, T, rows_u, rows_w, rows_v, R, start_time, deadline):
    """Try to snap U,V,W to half-integers by small local adjustments.
    Returns (U_list, V_list, W_list) or None."""
    # Simple approach: round each and check; if not, try rounding with a global sign flip
    # or a per-factor offset of 0.25 in the rounding lattice.
    for du in [0.0]:
        for dv in [0.0]:
            for dw in [0.0]:
                U_r = np.round((U + du) * 2) / 2.0
                V_r = np.round((V + dv) * 2) / 2.0
                W_r = np.round((W + dw) * 2) / 2.0
                T_check = np.einsum('ir,jr,kr->ijk', U_r, V_r, W_r)
                if np.max(np.abs(T_check - T)) < 1e-9:
                    U_list = [[U_r[i, c] for c in range(R)] for i in range(rows_u)]
                    V_list = [[V_r[j, c] for c in range(R)] for j in range(rows_v)]
                    W_list = [[W_r[k, c] for c in range(R)] for k in range(rows_w)]
                    if _verify_233(U_list, V_list, W_list, R):
                        return (U_list, V_list, W_list)
    return None

def _embed_233_into_236(result, n, m, p):
    """Embed a 2x3x3 decomposition (U233, V233, W233) into 2x3x6.
    The 2x3x6 multiplication: A (2x3) * B (3x6) -> C (2x6).
    Split B into B1 (3x3, cols 0-2) and B2 (3x3, cols 3-5).
    C[:, 0:3] = A * B1, C[:, 3:6] = A * B2.
    
    2x3x3 factor matrices (for n2=2, m2=3, p2=3):
    U233: shape (n2*m2, R) = (6, R), indexed [i*m2+j, c]
    V233: shape (m2*p2, R) = (9, R), indexed [j*p2+k, c]
    W233: shape (p2*n2, R) = (6, R), indexed [k*n2+i, c]
    
    For 2x3x6 (n=2, m=3, p=6):
    U: shape (n*m, 2R) = (6, 2R), indexed [i*m+j, c]
    V: shape (m*p, 2R) = (18, 2R), indexed [j*p+k, c]
    W: shape (p*n, 2R) = (12, 2R), indexed [k*n+i, c]
    
    For block 1 (k in 0..2, i.e. C columns 0-2):
    U1[i*m+j, c] = U233[i*m2+j, c] for c in 0..R-1
    V1[j*p+k, c] = V233[j*p2+k, c] for k in 0..2, c in 0..R-1
    W1[k*n+i, c] = W233[k*n2+i, c] for k in 0..2, c in 0..R-1
    
    For block 2 (k in 3..5, i.e. C columns 3-5):
    U2[i*m+j, c+R] = U233[i*m2+j, c] for c in 0..R-1
    V2[j*p+k, c+R] = V233[j*p2+(k-3), c] for k in 3..5, c in 0..R-1
    W2[k*n+i, c+R] = W233[(k-3)*n2+i, c] for k in 3..5, c in 0..R-1
    """
    U233, V233, W233 = result
    R = len(U233[0])
    n2, m2, p2 = 2, 3, 3
    total_R = 2 * R

    U = [[0] * total_R for _ in range(n * m)]
    V = [[0] * total_R for _ in range(m * p)]
    W = [[0] * total_R for _ in range(p * n)]

    for i in range(n):
        for j in range(m):
            for c in range(R):
                # Block 1
                U[i * m + j][c] = U233[i * m2 + j][c]
                # Block 2
                U[i * m + j][c + R] = U233[i * m2 + j][c]

    for j in range(m):
        for k in range(p):
            for c in range(R):
                if k < 3:
                    # Block 1
                    V[j * p + k][c] = V233[j * p2 + k][c]
                else:
                    # Block 2
                    V[j * p + k][c + R] = V233[j * p2 + (k - 3)][c]

    for k in range(p):
        for i in range(n):
            for c in range(R):
                if k < 3:
                    # Block 1
                    W[k * n + i][c] = W233[k * n2 + i][c]
                else:
                    # Block 2
                    W[k * n + i][c + R] = W233[(k - 3) * n2 + i][c]

    # Verify the full decomposition
    if _verify_full(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    return None
# EVOLVE-BLOCK-END
