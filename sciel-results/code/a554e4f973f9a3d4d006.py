"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import numpy as np
from scipy.optimize import minimize, linprog
import time

N = 26

# EVOLVE-BLOCK-START

def _compute_radii(pos, n=N):
    """
    Fast vectorized LP to compute optimal radii for given positions.
    Returns (sum_r, radii_array).
    """
    # Boundary distances: min distance to each edge
    b = np.minimum(np.minimum(pos[:, 0], pos[:, 1]), 1.0 - pos[:, 0])
    b = np.minimum(b, 1.0 - pos[:, 1])
    b = np.maximum(b, 0.0)

    # Pairwise distances (vectorized)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))

    # Build LP: max sum(r) s.t. r_i + r_j <= d_ij, 0 <= r_i <= b_i
    c = -np.ones(n)

    # Inequality constraints: r_i + r_j <= d_ij for all i < j
    n_pairs = n * (n - 1) // 2
    A_ub = np.zeros((n_pairs, n))
    b_ub = np.zeros(n_pairs)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = dist[i, j]
            idx += 1

    bounds = [(0.0, b[i]) for i in range(n)]

    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            radii = np.clip(res.x, 0, None)
            return float(np.sum(radii)), radii
        else:
            raise RuntimeError("LP failed")
    except Exception:
        # Fallback: greedy iterative
        radii = np.zeros(n)
        for _ in range(100):
            changed = False
            for i in range(n):
                max_r = b[i]
                for j in range(n):
                    if i != j:
                        max_r = min(max_r, dist[i, j] - radii[j])
                max_r = max(max_r, 0.0)
                if max_r > radii[i] + 1e-9:
                    radii[i] = max_r
                    changed = True
            if not changed:
                break
        return float(np.sum(radii)), radii


def _objective_and_grad(params, n=N):
    """
    Objective: -sum(radii) (to minimize).
    Returns (value, gradient) using LP dual variables for subgradient.
    """
    pos = params.reshape(n, 2)

    # Boundary distances
    b = np.minimum(np.minimum(pos[:, 0], pos[:, 1]), 1.0 - pos[:, 0])
    b = np.minimum(b, 1.0 - pos[:, 1])
    b = np.maximum(b, 0.0)

    # Pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2) + 1e-15)  # avoid div by zero

    # Build LP
    c = -np.ones(n)
    n_pairs = n * (n - 1) // 2
    A_ub = np.zeros((n_pairs, n))
    b_ub = np.zeros(n_pairs)
    idx = 0
    pair_list = []
    for i in range(n):
        for j in range(i + 1, n):
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = dist[i, j]
            pair_list.append((i, j))
            idx += 1

    bounds = [(0.0, b[i]) for i in range(n)]

    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if not res.success:
            # Fallback: just compute radii, no gradient
            sum_r, _ = _compute_radii(pos, n)
            return sum_r, np.zeros(2 * n)

        radii = np.clip(res.x, 0, None)
        sum_r = float(np.sum(radii))

        # Get dual variables from HiGHS
        # res.ineqlin[0] gives duals for A_ub @ x <= b_ub
        # For minimization with A_ub x <= b_ub, the dual (shadow price) tells us
        # how the objective changes when b_ub changes.
        # d(sum_r)/d(dist_ij) = dual_ij (the shadow price)
        # d(sum_r)/d(pos) = sum over pairs involving i of dual_ij * d(dist_ij)/d(pos_i)

        # Extract duals
        try:
            duals = res.ineqlin[0]  # shape (n_pairs,)
        except (IndexError, AttributeError):
            duals = np.zeros(n_pairs)

        # Compute gradient of sum_r w.r.t. positions
        # For each pair (i,j) with dual value lam:
        #   d(dist_ij)/d(pos_i) = (pos_i - pos_j) / dist_ij
        #   d(dist_ij)/d(pos_j) = (pos_j - pos_i) / dist_ij
        #   d(sum_r)/d(pos_i) += lam * (pos_i - pos_j) / dist_ij
        #   d(sum_r)/d(pos_j) += lam * (pos_j - pos_i) / dist_ij

        grad_pos = np.zeros((n, 2))
        for k, (i, j) in enumerate(pair_list):
            lam = duals[k]
            if abs(lam) < 1e-12:
                continue
            d_ij = dist[i, j]
            if d_ij < 1e-10:
                continue
            direction = (pos[i] - pos[j]) / d_ij
            grad_pos[i] += lam * direction
            grad_pos[j] -= lam * direction

        # Also account for boundary constraints
        # If r_i is at its upper bound b_i, then d(b_i)/d(pos_i) contributes
        # b_i = min(x_i, y_i, 1-x_i, 1-y_i)
        # The active boundary determines the subgradient
        for i in range(n):
            if radii[i] >= b[i] - 1e-10 and b[i] > 1e-10:
                # Which boundary is active?
                x, y = pos[i]
                if x <= y and x <= 1 - x and x <= 1 - y:
                    # x is the min: d(b_i)/dx = -1, d(b_i)/dy = 0
                    grad_pos[i, 0] += 1.0  # since we minimize -sum_r, sign flips
                elif y <= x and y <= 1 - x and y <= 1 - y:
                    grad_pos[i, 1] += 1.0
                elif 1 - x <= x and 1 - x <= y and 1 - x <= 1 - y:
                    grad_pos[i, 0] -= 1.0
                elif 1 - y <= x and 1 - y <= y and 1 - y <= 1 - x:
                    grad_pos[i, 1] -= 1.0

        # The objective we minimize is -sum_r, so gradient is -grad_pos
        grad_params = -grad_pos.flatten()

        return sum_r, grad_params

    except Exception:
        sum_r, _ = _compute_radii(pos, n)
        return sum_r, np.zeros(2 * n)


def _penalty_objective(params, n=N):
    """
    Objective with soft penalty for boundary violations.
    """
    pos = params.reshape(n, 2)

    # Penalty for going outside [0,1]
    penalty = 0.0
    for i in range(n):
        x, y = pos[i]
        if x < 0: penalty += 100.0 * x * x
        if x > 1: penalty += 100.0 * (x - 1) ** 2
        if y < 0: penalty += 100.0 * y * y
        if y > 1: penalty += 100.0 * (y - 1) ** 2

    val, _ = _compute_radii(pos, n)
    return val + penalty


def _initial_grid(n=N):
    """Create a grid layout as initial guess."""
    rows = 5
    cols = 6
    positions = []
    for i in range(rows):
        for j in range(cols):
            if len(positions) >= n:
                break
            x = (j + 0.5) / cols
            y = (i + 0.5) / rows
            positions.append([x, y])

    if len(positions) < n:
        rng = np.random.RandomState(42)
        while len(positions) < n:
            positions.append(rng.uniform(0.1, 0.9, 2))

    return np.array(positions[:n])


def _initial_hexagonal(n=N):
    """Create a hexagonal packing initial guess."""
    positions = []
    # Hexagonal arrangement
    row_spacing = 0.866  # sqrt(3)/2
    col_spacing = 1.0
    for row in range(6):
        for col in range(6):
            if len(positions) >= n:
                break
            x = (col + 0.5) * col_spacing / 6.0
            y = (row + 0.5) * row_spacing / 6.0
            if row % 2 == 1:
                x += col_spacing / 12.0
            x = min(max(x, 0.05), 0.95)
            y = min(max(y, 0.05), 0.95)
            positions.append([x, y])

    if len(positions) < n:
        rng = np.random.RandomState(42)
        while len(positions) < n:
            positions.append(rng.uniform(0.1, 0.9, 2))

    return np.array(positions[:n])


def _initial_random(seed=0, n=N):
    """Random initial configuration."""
    rng = np.random.RandomState(seed)
    return rng.uniform(0.1, 0.9, (n, 2))


def _is_valid(pos, radii, n, tol=1e-7):
    """Check if packing is valid."""
    for i in range(n):
        x, y = pos[i]
        r = radii[i]
        if r < -tol:
            return False
        if x - r < -tol or y - r < -tol or x + r > 1 + tol or y + r > 1 + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pos[i] - pos[j])
            if d < radii[i] + radii[j] - tol:
                return False
    return True


def _fix_to_boundary(pos, radii, n):
    """Fix any small violations by reducing radii."""
    radii = radii.copy()

    for i in range(n):
        max_r = min(pos[i, 0], pos[i, 1], 1 - pos[i, 0], 1 - pos[i, 1])
        if radii[i] > max_r:
            radii[i] = max(max_r, 0.0)

    for _ in range(30):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(pos[i] - pos[j])
                if radii[i] + radii[j] > d:
                    excess = radii[i] + radii[j] - d
                    reduction = excess / 2.0 + 1e-8
                    radii[i] = max(radii[i] - reduction, 0.0)
                    radii[j] = max(radii[j] - reduction, 0.0)
                    changed = True
        if not changed:
            break

    return radii


def _local_search(pos, n=N, time_budget=10.0, use_grad=True):
    """
    Local search using L-BFGS-B with gradient or finite differences.
    """
    start_time = time.time()
    bounds = [(0.0, 1.0)] * (2 * n)

    best_val = np.inf
    best_pos = pos.copy()

    if use_grad:
        # Use gradient-based optimization
        try:
            res = minimize(
                lambda p: _objective_and_grad(p, n)[0],
                pos.flatten(),
                jac=lambda p: _objective_and_grad(p, n)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 500}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_pos = res.x.reshape(n, 2)
        except Exception:
            pass

        # Second pass with more iterations if time allows
        if time.time() - start_time < time_budget * 0.5:
            try:
                res2 = minimize(
                    lambda p: _objective_and_grad(p, n)[0],
                    best_pos.flatten(),
                    jac=lambda p: _objective_and_grad(p, n)[1],
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 800}
                )
                if res2.fun < best_val:
                    best_val = res2.fun
                    best_pos = res2.x.reshape(n, 2)
            except Exception:
                pass
    else:
        # Finite difference fallback
        try:
            res = minimize(
                lambda p: _penalty_objective(p, n),
                pos.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_pos = res.x.reshape(n, 2)
        except Exception:
            pass

    # Clip to [0,1]
    best_pos = np.clip(best_pos, 0.0, 1.0)
    return best_pos, best_val


def _simulated_annealing(pos, n=N, time_budget=15.0):
    """
    Simulated annealing for multi-start diversity.
    """
    start_time = time.time()
    current_pos = pos.copy()
    current_val, _ = _compute_radii(current_pos, n)
    best_pos = current_pos.copy()
    best_val = current_val

    T = 0.05  # initial temperature
    cooling = 0.97
    step_size = 0.05

    rng = np.random.RandomState(42)

    while time.time() - start_time < time_budget:
        # Propose a move
        idx = rng.randint(n)
        delta = rng.normal(0, step_size, 2)
        new_pos = current_pos.copy()
        new_pos[idx] = np.clip(new_pos[idx] + delta, 0.0, 1.0)

        new_val, _ = _compute_radii(new_pos, n)
        delta_val = new_val - current_val

        if delta_val < 0 or rng.random() < np.exp(-delta_val / max(T, 1e-10)):
            current_pos = new_pos
            current_val = new_val
            if current_val < best_val:
                best_val = current_val
                best_pos = current_pos.copy()

        T *= cooling
        if T < 0.001:
            T = 0.001

    return best_pos, best_val


def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    best_score = -np.inf
    best_result = None

    def _update_best(pos):
        nonlocal best_score, best_result
        sum_r, radii = _compute_radii(pos, N)
        radii = _fix_to_boundary(pos, radii, N)
        if _is_valid(pos, radii, N):
            score = float(np.sum(radii))
            if score > best_score:
                best_score = score
                result = np.zeros((N, 3))
                result[:, :2] = pos
                result[:, 2] = radii
                best_result = result
            return score
        return -np.inf

    # Initial configurations
    configs = [
        _initial_grid(N),
        _initial_hexagonal(N),
        _initial_random(0, N),
        _initial_random(123, N),
        _initial_random(999, N),
    ]

    # Phase 1: Local search on each config with gradient
    for ci, pos0 in enumerate(configs):
        if time.time() > deadline - 20:
            break
        budget = min(12.0, deadline - time.time() - 5)
        if budget < 2:
            break
        pos_opt, val = _local_search(pos0, N, time_budget=budget, use_grad=True)
        _update_best(pos_opt)

    # Phase 2: SA to find diverse starting points, then local search
    if time.time() < deadline - 25:
        sa_budget = min(15.0, deadline - time.time() - 25)
        if sa_budget > 3:
            pos_sa, _ = _simulated_annealing(_initial_grid(N), N, time_budget=sa_budget)
            _update_best(pos_sa)
            # Local search from SA result
            if time.time() < deadline - 15:
                pos_opt, _ = _local_search(pos_sa, N, time_budget=min(10.0, deadline - time.time() - 10), use_grad=True)
                _update_best(pos_opt)

    # Phase 3: If time remains, try more random restarts with local search
    while time.time() < deadline - 10:
        seed = int(time.time() * 1000) % 10000
        pos_rand = _initial_random(seed, N)
        budget = min(8.0, deadline - time.time() - 5)
        if budget < 2:
            break
        pos_opt, _ = _local_search(pos_rand, N, time_budget=budget, use_grad=True)
        _update_best(pos_opt)

    # Fallback
    if best_result is None:
        pos = _initial_grid(N)
        radii = np.full(N, 0.05)
        radii = _fix_to_boundary(pos, radii, N)
        best_result = np.zeros((N, 3))
        best_result[:, :2] = pos
        best_result[:, 2] = radii

    return best_result

# EVOLVE-BLOCK-END
