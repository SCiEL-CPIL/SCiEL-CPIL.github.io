# EVOLVE-BLOCK-START
import time
import numpy as np

_DEMANDS = [
    (0, 5, 3),
    (1, 7, 2),
    (2, 9, 4),
    (3, 6, 1),
    (0, 4, 5),
    (2, 8, 2),
    (5, 9, 3),
    (1, 4, 2),
    (6, 0, 4),
    (3, 8, 1),
    (7, 2, 3),
    (4, 9, 2),
    (8, 1, 5),
    (0, 6, 1),
    (5, 2, 2),
]


def _build_matrices():
    n = 10
    D = len(_DEMANDS)
    cw = np.zeros((D, n), dtype=np.int8)
    ccw = np.zeros((D, n), dtype=np.int8)
    for i, (s, t, w) in enumerate(_DEMANDS):
        e = s
        while e != t:
            cw[i, e] = 1
            e = (e + 1) % n
        ccw[i] = 1 - cw[i]
    return cw, ccw


def _compute_loads(routing, cw, ccw, weights, base_loads, diff_T):
    switch = weights * routing
    return base_loads + diff_T @ switch


def _local_search(routing, cw, ccw, weights, base_loads, diff_T, n, D):
    """Flip-based local search to minimize max edge load."""
    routing = routing.copy()
    current_loads = _compute_loads(routing, cw, ccw, weights, base_loads, diff_T)
    current_max = float(np.max(current_loads))

    improved = True
    max_iter = 100000
    it = 0
    while improved and it < max_iter:
        improved = False
        it += 1
        for i in range(D):
            # Try flipping demand i
            new_routing = routing.copy()
            new_routing[i] = 1.0 - new_routing[i]
            new_loads = _compute_loads(new_routing, cw, ccw, weights, base_loads, diff_T)
            new_max = float(np.max(new_loads))
            if new_max < current_max - 1e-12:
                routing = new_routing
                current_loads = new_loads
                current_max = new_max
                improved = True
    return routing, current_max


def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem on a ring of n=10 nodes.
    Uses branch-and-bound with LP relaxation + local search polish.
    """
    deadline = time.time() + 600.0
    n = 10
    D = len(_DEMANDS)

    cw, ccw = _build_matrices()
    weights = np.array([w for _, _, w in _DEMANDS], dtype=np.float64)
    base_loads = weights @ cw.astype(np.float64)
    diff_matrix = ccw.astype(np.float64) - cw.astype(np.float64)
    diff_T = diff_matrix.T  # (n, D)

    # --- Branch and Bound ---
    best_routing = np.zeros(D, dtype=np.float64)
    best_max = float('inf')

    # Lower bound via LP relaxation: minimize t s.t. base_loads + diff_T @ (w*x) <= t*1
    # where x in [0,1]^D.  Use scipy linprog.
    from scipy.optimize import linprog

    def lp_lower_bound(fixed):
        """
        fixed: array of length D, entries -1 (free), 0 (CW), 1 (CCW).
        Returns the LP optimal value of min max load with those fixes.
        """
        free = np.where(fixed == -1)[0]
        if len(free) == 0:
            routing = (fixed == 1).astype(np.float64)
            loads = _compute_loads(routing, cw, ccw, weights, base_loads, diff_T)
            return float(np.max(loads))

        # Variables: x_free (len(free)) and t (1 scalar)
        # Objective: minimize t
        c = np.zeros(len(free) + 1)
        c[-1] = 1.0

        # Constraints: base_loads + diff_T @ (w * x_full) <= t
        # For each edge e:
        #   base_loads[e] + sum_{i in free} diff_T[e,i]*w_i*x_i + fixed_part <= t
        # => sum_{i in free} diff_T[e,i]*w_i*x_i - t <= -base_loads[e] - fixed_part

        fixed_part = np.zeros(n)
        for i in range(D):
            if fixed[i] == 1:
                fixed_part += diff_T[:, i] * weights[i]
            # if fixed[i] == 0, contributes 0

        A = np.zeros((n, len(free) + 1))
        b = np.zeros(n)
        for e in range(n):
            for j, i in enumerate(free):
                A[e, j] = diff_T[e, i] * weights[i]
            A[e, -1] = -1.0
            b[e] = -base_loads[e] - fixed_part[e]

        bounds = [(0.0, 1.0)] * len(free) + [(0.0, None)]

        res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
        if res.status == 0:
            return float(res.fun)
        else:
            # Fallback: use a simple upper bound
            routing = (fixed == 1).astype(np.float64)
            # Set free to 0 for estimate
            for i in free:
                routing[i] = 0.0
            loads = _compute_loads(routing, cw, ccw, weights, base_loads, diff_T)
            return float(np.max(loads))

    # Order demands by some heuristic for better pruning
    # Sort by weight descending for branching
    order = np.argsort(-weights)

    stack = [(np.full(D, -1, dtype=np.int8), 0.0)]  # (fixed, current_lb)
    nodes_explored = 0
    max_nodes = 200000

    while stack:
        if time.time() > deadline - 30.0:
            break
        if nodes_explored > max_nodes:
            break

        fixed, _ = stack.pop()
        nodes_explored += 1

        # Count free variables
        free_count = np.sum(fixed == -1)

        if free_count == 0:
            routing = (fixed == 1).astype(np.float64)
            loads = _compute_loads(routing, cw, ccw, weights, base_loads, diff_T)
            mx = float(np.max(loads))
            if mx < best_max:
                best_max = mx
                best_routing = routing.copy()
            continue

        # LP lower bound
        lb = lp_lower_bound(fixed)
        if lb >= best_max - 1e-9:
            continue

        # Pick the next free variable to branch on
        next_idx = order[np.where(fixed == -1)[0][0]]

        # Branch: try CCW (1) first if it seems promising
        fixed_ccw = fixed.copy()
        fixed_ccw[next_idx] = 1
        stack.append((fixed_ccw, 0.0))

        fixed_cw = fixed.copy()
        fixed_cw[next_idx] = 0
        stack.append((fixed_cw, 0.0))

    # --- Polish with local search ---
    if time.time() < deadline - 10.0:
        best_routing, best_max = _local_search(
            best_routing, cw, ccw, weights, base_loads, diff_T, n, D
        )

    # Ensure output format: >=0.5 means CCW, <0.5 means CW
    result = best_routing.copy()
    result[result > 0.5] = 1.0
    result[result <= 0.5] = 0.0

    return result


# EVOLVE-BLOCK-END
