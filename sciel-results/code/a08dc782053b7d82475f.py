"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
import numpy as np

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START

def _build_t_tensor(n, m, p):
    """Build the target multiplication tensor T of shape (n*m, m*p, p*n) with 1s at active sites."""
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    T = np.zeros((dim_U, dim_V, dim_W), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T


def _tensor_contract(U, V, W):
    """Compute sum_r U[a,r] V[b,r] W[c,r] for all a,b,c using einsum."""
    return np.einsum('ar,br,cr->abc', U, V, W)


def _verify_exact(U, V, W, T, tol=1e-8):
    """Check if U, V, W exactly reproduce T within tolerance."""
    prod = _tensor_contract(U, V, W)
    return np.allclose(prod, T, atol=tol, rtol=0)


def _smooth_error(U, V, W, T):
    """Squared error between contraction and target."""
    prod = _tensor_contract(U, V, W)
    return float(np.sum((prod - T) ** 2))


def _smooth_grad(U, V, W, T):
    """
    Compute gradients of smooth error wrt U, V, W.
    E = ||U@V@W^T - T||^2 where contraction is sum_r U[a,r]V[b,r]W[c,r].
    dE/dU[a,r] = 2 * sum_{b,c} (prod[a,b,c] - T[a,b,c]) * V[b,r] * W[c,r]
    """
    prod = _tensor_contract(U, V, W)
    diff = prod - T  # shape (dim_U, dim_V, dim_W)
    # grad_U[a,r] = 2 * sum_{b,c} diff[a,b,c] * V[b,r] * W[c,r]
    # = 2 * (diff * V * W) contracted over b,c
    # Use einsum: 'abc,br,cr -> ar'
    grad_U = 2.0 * np.einsum('abc,br,cr->ar', diff, V, W)
    grad_V = 2.0 * np.einsum('abc,ar,cr->br', diff, U, W)
    grad_W = 2.0 * np.einsum('abc,ar,br->cr', diff, U, V)
    return grad_U, grad_V, grad_W


def _round_to_half(x):
    """Round to nearest half-integer (i.e., multiple of 0.5)."""
    return round(2.0 * x) / 2.0


def _try_round_solution(U, V, W, T):
    """
    Try to snap U, V, W to integer or half-integer values and verify.
    Returns (U_list, V_list, W_list, success).
    """
    # Strategy 1: round to nearest integer
    U_r = np.round(U).astype(np.float64)
    V_r = np.round(V).astype(np.float64)
    W_r = np.round(W).astype(np.float64)
    if _verify_exact(U_r, V_r, W_r, T, tol=1e-6):
        return U_r.tolist(), V_r.tolist(), W_r.tolist(), True

    # Strategy 2: round to nearest half-integer
    U_h = np.array([[_round_to_half(x) for x in row] for row in U], dtype=np.float64)
    V_h = np.array([[_round_to_half(x) for x in row] for row in V], dtype=np.float64)
    W_h = np.array([[_round_to_half(x) for x in row] for row in W], dtype=np.float64)
    if _verify_exact(U_h, V_h, W_h, T, tol=1e-6):
        return U_h.tolist(), V_h.tolist(), W_h.tolist(), True

    # Strategy 3: mixed — try each matrix rounded independently in 4 combos
    U_int = np.round(U)
    U_half = np.array([[_round_to_half(x) for x in row] for row in U], dtype=np.float64)
    V_int = np.round(V)
    V_half = np.array([[_round_to_half(x) for x in row] for row in V], dtype=np.float64)
    W_int = np.round(W)
    W_half = np.array([[_round_to_half(x) for x in row] for row in W], dtype=np.float64)
    for u_c in [U_int, U_half]:
        for v_c in [V_int, V_half]:
            for w_c in [W_int, W_half]:
                if _verify_exact(u_c, v_c, w_c, T, tol=1e-6):
                    return u_c.tolist(), v_c.tolist(), w_c.tolist(), True
    return None, None, None, False


def _local_search_gradient(U, V, W, T, r, max_iters=50000, time_budget=20.0):
    """
    Gradient-based local search in continuous space, followed by rounding.
    Uses adaptive step sizes and restarts.
    """
    start = time.time()
    U = U.copy().astype(np.float64)
    V = V.copy().astype(np.float64)
    W = W.copy().astype(np.float64)
    dim_U, dim_V, dim_W = U.shape[0], V.shape[0], W.shape[0]

    current_err = _smooth_error(U, V, W, T)
    best_err = current_err
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    # Adaptive step sizes per matrix
    step_U = 0.5
    step_V = 0.5
    step_W = 0.5
    min_step = 0.01
    max_step = 5.0
    patience = 5000

    for it in range(max_iters):
        if time.time() - start > time_budget:
            break
        if current_err < 1e-10:
            break

        gU, gV, gW = _smooth_grad(U, V, W, T)

        # Normalize gradients
        gU_norm = np.linalg.norm(gU)
        gV_norm = np.linalg.norm(gV)
        gW_norm = np.linalg.norm(gW)

        # Adaptive steps: scale inversely with gradient magnitude
        if gU_norm > 1e-8:
            sU = step_U * min(1.0, 5.0 / gU_norm)
        else:
            sU = 0.0
        if gV_norm > 1e-8:
            sV = step_V * min(1.0, 5.0 / gV_norm)
        else:
            sV = 0.0
        if gW_norm > 1e-8:
            sW = step_W * min(1.0, 5.0 / gW_norm)
        else:
            sW = 0.0

        # Add small random perturbation to escape plateaus
        noise = 0.1
        dU = -sU * gU / (gU_norm + 1e-8) + noise * np.random.randn(*U.shape) * step_U
        dV = -sV * gV / (gV_norm + 1e-8) + noise * np.random.randn(*V.shape) * step_V
        dW = -sW * gW / (gW_norm + 1e-8) + noise * np.random.randn(*W.shape) * step_W

        U_new = U + dU
        V_new = V + dV
        W_new = W + dW

        # Clip to reasonable range
        U_new = np.clip(U_new, -15, 15)
        V_new = np.clip(V_new, -15, 15)
        W_new = np.clip(W_new, -15, 15)

        new_err = _smooth_error(U_new, V_new, W_new, T)

        if new_err < current_err:
            U, V, W = U_new, V_new, W_new
            current_err = new_err
            if new_err < best_err:
                best_err = new_err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                patience = 5000
            patience -= 1
        else:
            patience -= 1
            # Reduce step sizes
            step_U = max(min_step, step_U * 0.95)
            step_V = max(min_step, step_V * 0.95)
            step_W = max(min_step, step_W * 0.95)

        if patience <= 0:
            # Restart with current best
            U, V, W = best_U.copy(), best_V.copy(), best_W.copy()
            current_err = best_err
            step_U = random.uniform(0.3, 2.0)
            step_V = random.uniform(0.3, 2.0)
            step_W = random.uniform(0.3, 2.0)
            patience = 5000

    return best_U, best_V, best_W, best_err


def _coordinate_descent(U, V, W, T, r, max_iters=100000, time_budget=15.0):
    """
    Coordinate descent on integer/half-integer grid.
    For each entry, try neighboring values and accept if error decreases.
    """
    start = time.time()
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)

    # Snap to half-integers initially
    U = np.array([[_round_to_half(x) for x in row] for row in U], dtype=np.float64)
    V = np.array([[_round_to_half(x) for x in row] for row in V], dtype=np.float64)
    W = np.array([[_round_to_half(x) for x in row] for row in W], dtype=np.float64)

    current_err = _smooth_error(U, V, W, T)
    best_err = current_err
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    # Precompute for speed: for each column r, we can compute partial contributions
    # But let's just do direct evaluation for correctness

    deltas = [-1.0, -0.5, 0.5, 1.0, -2.0, 2.0, -1.5, 1.5]

    no_improve = 0
    max_no_improve = 5000

    for it in range(max_iters):
        if time.time() - start > time_budget:
            break
        if current_err < 1e-10:
            break

        # Pick a random matrix, row, col
        mat_choice = random.randint(0, 2)
        if mat_choice == 0:
            mat = U
            nrows = U.shape[0]
        elif mat_choice == 1:
            mat = V
            nrows = V.shape[0]
        else:
            mat = W
            nrows = W.shape[0]

        row = random.randint(0, nrows - 1)
        col = random.randint(0, r - 1)

        old_val = mat[row, col]
        improved = False

        for d in deltas:
            new_val = old_val + d
            if abs(new_val) > 12:
                continue
            mat[row, col] = new_val
            new_err = _smooth_error(U, V, W, T)
            if new_err < current_err:
                current_err = new_err
                improved = True
                if new_err < best_err:
                    best_err = new_err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                break
            mat[row, col] = old_val

        if improved:
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > max_no_improve:
                # Random restart from best with perturbation
                U, V, W = best_U.copy(), best_V.copy(), best_W.copy()
                current_err = best_err
                # Perturb a few entries
                for _ in range(5):
                    mc = random.randint(0, 2)
                    if mc == 0:
                        mat = U; nr = U.shape[0]
                    elif mc == 1:
                        mat = V; nr = V.shape[0]
                    else:
                        mat = W; nr = W.shape[0]
                    rr = random.randint(0, nr - 1)
                    cc = random.randint(0, r - 1)
                    mat[rr, cc] += random.choice([-1.0, -0.5, 0.5, 1.0])
                current_err = _smooth_error(U, V, W, T)
                no_improve = 0

    return best_U, best_V, best_W, best_err


def _try_rank(r, T, n, m, p, time_budget=30.0):
    """
    Attempt to find a rank-r decomposition.
    Uses multiple initialization strategies and gradient-based local search.
    """
    start = time.time()
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n

    max_attempts = max(5, int((time_budget - (time.time() - start)) * 3))

    for attempt in range(max_attempts):
        if time.time() - start > time_budget:
            break

        # Choose initialization strategy
        strategy = attempt % 5

        if strategy == 0:
            # Sparse ±1 initialization
            U = np.zeros((dim_U, r), dtype=np.float64)
            V = np.zeros((dim_V, r), dtype=np.float64)
            W = np.zeros((dim_W, r), dtype=np.float64)
            for ci in range(r):
                # Each column: pick random nonzero pattern
                for a in range(dim_U):
                    if random.random() < 0.3:
                        U[a, ci] = random.choice([-1.0, 1.0])
                for b in range(dim_V):
                    if random.random() < 0.3:
                        V[b, ci] = random.choice([-1.0, 1.0])
                for c in range(dim_W):
                    if random.random() < 0.3:
                        W[c, ci] = random.choice([-1.0, 1.0])

        elif strategy == 1:
            # Dense small integer initialization
            U = np.random.randint(-2, 3, size=(dim_U, r)).astype(np.float64)
            V = np.random.randint(-2, 3, size=(dim_V, r)).astype(np.float64)
            W = np.random.randint(-2, 3, size=(dim_W, r)).astype(np.float64)

        elif strategy == 2:
            # Superposition of random schoolbook-like terms
            U = np.zeros((dim_U, r), dtype=np.float64)
            V = np.zeros((dim_V, r), dtype=np.float64)
            W = np.zeros((dim_W, r), dtype=np.float64)
            for ci in range(r):
                for _ in range(random.randint(1, 3)):
                    i = random.randint(0, n - 1)
                    j = random.randint(0, m - 1)
                    k = random.randint(0, p - 1)
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = random.choice([-1.0, 1.0])
                    U[a, ci] += s
                    V[b, ci] += s
                    W[c, ci] += s

        elif strategy == 3:
            # Gaussian initialization
            U = np.random.randn(dim_U, r) * 0.5
            V = np.random.randn(dim_V, r) * 0.5
            W = np.random.randn(dim_W, r) * 0.5

        else:
            # Structured: based on parity of indices
            U = np.zeros((dim_U, r), dtype=np.float64)
            V = np.zeros((dim_V, r), dtype=np.float64)
            W = np.zeros((dim_W, r), dtype=np.float64)
            for ci in range(r):
                for a in range(dim_U):
                    i, j = divmod(a, m)
                    U[a, ci] = (1.0 if (i + j + ci) % 2 == 0 else -1.0) * random.choice([0.0, 1.0])
                for b in range(dim_V):
                    j, k = divmod(b, p)
                    V[b, ci] = (1.0 if (j + k + ci) % 2 == 0 else -1.0) * random.choice([0.0, 1.0])
                for c in range(dim_W):
                    k, i = divmod(c, n)
                    W[c, ci] = (1.0 if (k + i + ci) % 2 == 0 else -1.0) * random.choice([0.0, 1.0])

        # Run gradient-based local search
        time_left = time_budget - (time.time() - start)
        if time_left < 2.0:
            continue

        U_gs, V_gs, W_gs, err_gs = _local_search_gradient(
            U, V, W, T, r,
            max_iters=min(50000, int(time_left * 2000)),
            time_budget=min(10.0, time_left * 0.6)
        )

        # Check if we have a solution
        if err_gs < 1e-8:
            U_l, V_l, W_l, ok = _try_round_solution(U_gs, V_gs, W_gs, T)
            if ok:
                return {"U": U_l, "V": V_l, "W": W_l}

        # Try coordinate descent on the rounded solution
        time_left = time_budget - (time.time() - start)
        if time_left < 3.0:
            continue

        # First try rounding the gradient search result
        U_cd = np.array([[_round_to_half(x) for x in row] for row in U_gs], dtype=np.float64)
        V_cd = np.array([[_round_to_half(x) for x in row] for row in V_gs], dtype=np.float64)
        W_cd = np.array([[_round_to_half(x) for x in row] for row in W_gs], dtype=np.float64)

        # Also try the original rounded
        U_cd2 = np.round(U)
        V_cd2 = np.round(V)
        W_cd2 = np.round(W)

        # Run coordinate descent on both
        for Uc, Vc, Wc in [(U_cd, V_cd, W_cd), (U_cd2, V_cd2, W_cd2)]:
            time_left = time_budget - (time.time() - start)
            if time_left < 2.0:
                break

            U_final, V_final, W_final, err_final = _coordinate_descent(
                Uc, Vc, Wc, T, r,
                max_iters=min(100000, int(time_left * 5000)),
                time_budget=min(5.0, time_left * 0.8)
            )

            if err_final < 1e-8:
                U_l, V_l, W_l, ok = _try_round_solution(U_final, V_final, W_final, T)
                if ok:
                    return {"U": U_l, "V": V_l, "W": W_l}

    return None


def _schoolbook(n, m, p):
    """Return the schoolbook (rank = n*m*p) decomposition."""
    rank = n * m * p
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    U = [[0] * rank for _ in range(dim_U)]
    V = [[0] * rank for _ in range(dim_V)]
    W = [[0] * rank for _ in range(dim_W)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a][col] = 1
                V[b][col] = 1
                W[c][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 * 3x4 matrix multiplication.
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 140s budget, 10s safety margin

    n, m, p = DIMENSIONS
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n

    T = _build_t_tensor(n, m, p)

    # Start with schoolbook as fallback
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p

    # Try ranks from low to high, stopping early if found
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(9, 12, 12) = 12
    # Known results suggest rank around 22-24 for this size
    target_ranks = list(range(18, best_rank))

    for r in target_ranks:
        if time.time() > deadline - 5.0:
            break

        time_left = deadline - time.time()
        if time_left < 5.0:
            break

        # Allocate more time for lower ranks (harder to find if they exist)
        if r <= 22:
            budget = min(time_left * 0.4, 30.0)
        elif r <= 26:
            budget = min(time_left * 0.3, 20.0)
        else:
            budget = min(time_left * 0.2, 10.0)

        result = _try_rank(r, T, n, m, p, time_budget=budget)
        if result is not None:
            best_result = result
            best_rank = r
            # Continue trying lower ranks for a while
            continue

    return best_result

# EVOLVE-BLOCK-END
