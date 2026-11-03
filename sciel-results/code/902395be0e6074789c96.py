import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    best_result = _schoolbook_decomposition(n, m, p)
    best_rank = n * m * p  # 36

    # Theoretical lower bound: max(n*m, m*p, p*n) = max(6, 18, 12) = 18
    lower_bound = max(n * m, m * p, p * n)

    # Strategy: search for direct rank-r decomposition of 2x3x6
    # For each rank r, we need U (6xr), V (18xr), W (12xr) such that
    # for all i in [0,n), j in [0,m), k in [0,p):
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1

    # We'll use a structured approach: fix U and V, solve for W
    # Key insight: if we fix U and V, the constraint becomes linear in W
    # A[(i,k), c] = U[i*m+j, c] * V[j*p+k, c] for each (i,j,k)
    # But W appears as W[k*n+i, c], so for fixed (i,k), the constraint is:
    #   sum_c (U[i*m+j, c] * V[j*p+k, c]) * W[k*n+i, c] = 1  for each j
    # This is a linear system in W[k*n+i, :] for each (i,k)

    # Let's try ranks from lower_bound upward
    for r in range(lower_bound, best_rank + 1):
        if time.time() > deadline:
            break
        if r >= best_rank:
            break

        # Budget: more attempts for smaller ranks
        max_attempts = 300 if r <= 20 else 100
        found = _search_decomposition(n, m, p, r, max_attempts, deadline)

        if found is not None:
            best_result = found
            best_rank = r
            if best_rank <= lower_bound:
                break

    # Also try the block-repetition strategy as fallback
    if best_rank > lower_bound:
        n_sub, m_sub, p_sub = 2, 3, 2
        repeats = p // p_sub  # 3
        min_sub_rank = max(n_sub * m_sub, m_sub * p_sub, p_sub * n_sub)  # 6

        for sub_rank in range(min_sub_rank, best_rank // repeats + 1):
            if time.time() > deadline:
                break
            full_rank = sub_rank * repeats
            if full_rank >= best_rank:
                break

            max_attempts = 200 if sub_rank <= 8 else 100
            found = _search_sub_decomposition(n_sub, m_sub, p_sub, sub_rank, max_attempts, deadline)

            if found is not None:
                U_sub, V_sub, W_sub = found
                result = _expand_decomposition(U_sub, V_sub, W_sub, n, m, p, p_sub)
                if result is not None and full_rank < best_rank:
                    best_result = result
                    best_rank = full_rank
                    if best_rank <= lower_bound:
                        break

    return best_result


def _schoolbook_decomposition(n, m, p):
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


def _search_decomposition(n, m, p, r, max_attempts, deadline):
    """
    Search for rank-r decomposition of n×m × m×p matmul tensor.
    Returns dict with U, V, W or None.
    
    Strategy: randomly sample U and V with small integer entries,
    then solve the linear system for W and check if W has half-integer entries.
    """
    nm = n * m  # 6
    mp = m * p  # 18
    pn = p * n  # 12

    for attempt in range(max_attempts):
        if time.time() > deadline:
            break

        # Generate random U and V with small integer values
        # Use values from {-1, 0, 1} for sparsity
        U_rand = np.random.randint(-1, 2, size=(nm, r))
        V_rand = np.random.randint(-1, 2, size=(mp, r))

        # Build the linear system for W
        # For each (i, k) pair, we need:
        #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1  for each j in [0, m)
        # This gives m equations in r unknowns (W[k*n+i, 0], ..., W[k*n+i, r-1])
        # But we have n*p such (i,k) pairs, each giving m equations.
        # Total: n*p*m = n*m*p = 36 equations in p*n*r = 12*r unknowns.

        # Let's build the full system:
        # Variables: W[k*n+i, c] for k in [0,p), i in [0,n), c in [0,r)
        # For each (i, j, k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1

        # Build matrix B and vector d:
        # B[(i,j,k), (k', i', c)] = U[i*m+j, c] * V[j*p+k, c] if k=k' and i=i', else 0
        # d[(i,j,k)] = 1

        # This is a large sparse system. Let's build it efficiently.
        # Number of equations: n*m*p = 36
        # Number of unknowns: p*n*r = 12*r

        num_eq = n * m * p
        num_unknowns = p * n * r

        B = np.zeros((num_eq, num_unknowns))
        d = np.ones(num_eq)

        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for c in range(r):
                        unknown_idx = k * n + i  # W index for row
                        B[eq_idx, unknown_idx * r + c] = U_rand[i * m + j, c] * V_rand[j * p + k, c]
                    eq_idx += 1

        # Solve B @ w = d where w is vector of length num_unknowns
        try:
            # Use least squares
            w_sol, residuals, rank_B, sv = np.linalg.lstsq(B, d, rcond=1e-10)
        except np.linalg.LinAlgError:
            continue

        # Check if solution is exact
        if not np.allclose(B @ w_sol, d, atol=1e-9):
            continue

        # Reshape w_sol to W: shape (p*n, r)
        W_sol = w_sol.reshape(p * n, r)

        # Check if W_sol entries are close to half-integers
        W_rounded = np.round(W_sol * 2) / 2.0

        # Verify with rounded values
        if not np.allclose(B @ W_rounded.flatten(), d, atol=1e-9):
            continue

        # Full verification
        U_list = U_rand.tolist()
        V_list = V_rand.tolist()
        W_list = W_rounded.tolist()

        if _verify_decomposition(U_list, V_list, W_list, n, m, p):
            return {"U": U_list, "V": V_list, "W": W_list}

    return None


def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify: for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    """
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    r = U_np.shape[1]

    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U_np[i * m + j, c] * V_np[j * p + k, c] * W_np[k * n + i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def _search_sub_decomposition(n, m, p, r, max_attempts, deadline):
    """
    Search for a rank-r decomposition of n×m × m×p matmul tensor.
    Returns (U, V, W) as lists of lists, or None.
    """
    np_sub = n * p
    nm = n * m
    mp = m * p

    for attempt in range(max_attempts):
        if time.time() > deadline:
            break

        U_rand = np.random.randint(-1, 2, size=(nm, r))
        V_rand = np.random.randint(-1, 2, size=(mp, r))

        # Build A: shape (n*p, m*r)
        # A[(i,k), (j,c)] = U[i*m+j, c] * V[j*p+k, c]
        A = np.zeros((np_sub, m * r))
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for c in range(r):
                        A[i * p + k, j * r + c] = U_rand[i * m + j, c] * V_rand[j * p + k, c]

        try:
            rank_A = np.linalg.matrix_rank(A, tol=1e-9)
        except np.linalg.LinAlgError:
            continue

        if rank_A < np_sub:
            continue

        try:
            W_solve = np.linalg.lstsq(A, np.eye(np_sub), rcond=1e-12)[0]
        except np.linalg.LinAlgError:
            continue

        W_rounded = np.round(W_solve * 2) / 2.0

        if not np.allclose(A @ W_rounded, np.eye(np_sub), atol=1e-9):
            continue

        if _verify_decomposition(U_rand.tolist(), V_rand.tolist(), W_rounded.tolist(), n, m, p):
            return U_rand.tolist(), V_rand.tolist(), W_rounded.tolist()

    return None


def _expand_decomposition(U_sub, V_sub, W_sub, n, m, p, p_sub):
    """
    Expand a 2x3x2 decomposition to 2x3x6 by repeating it p/p_sub times.
    """
    r_sub = len(U_sub[0])
    repeats = p // p_sub
    r_total = r_sub * repeats

    U_full = [[0] * r_total for _ in range(n * m)]
    V_full = [[0] * r_total for _ in range(m * p)]
    W_full = [[0] * r_total for _ in range(p * n)]

    for block in range(repeats):
        offset = block * r_sub

        for i in range(n * m):
            for c in range(r_sub):
                U_full[i][offset + c] = U_sub[i][c]

        for j in range(m):
            for k in range(p_sub):
                for c in range(r_sub):
                    V_full[j * p + block * p_sub + k][offset + c] = V_sub[j * p_sub + k][c]

        for i in range(n):
            for k in range(p_sub):
                for c in range(r_sub):
                    W_full[(block * p_sub + k) * n + i][offset + c] = W_sub[k * n + i][c]

    if _verify_decomposition(U_full, V_full, W_full, n, m, p):
        return {"U": U_full, "V": V_full, "W": W_full}
    return None
# EVOLVE-BLOCK-END
