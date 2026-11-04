"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay


def _eval_packing(xy, r):
    """
    Evaluate a packing: returns (total_radius, is_valid).
    Valid means all circles inside [0,1]^2 and disjoint interiors.
    """
    n = len(xy)
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        if ri < -1e-9:
            return 0.0, False
        if x - ri < -1e-9 or 1.0 - x - ri < -1e-9 or y - ri < -1e-9 or 1.0 - y - ri < -1e-9:
            return 0.0, False
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-9:
                return 0.0, False
    return float(np.sum(r)), True


def _build_contact_graph(xy, r, tol=1e-6):
    """
    Build a contact graph: edge (i,j) if circles i and j are touching (dist ≈ r_i + r_j).
    Also add wall contacts: circle i touches wall if x_i ≈ r_i or 1-x_i ≈ r_i, etc.
    Returns:
        adj: adjacency matrix (n x n) for circle-circle contacts
        wall_contact: (n, 4) bool array: [left, right, bottom, top]
    """
    n = len(xy)
    adj = np.zeros((n, n), dtype=bool)
    wall_contact = np.zeros((n, 4), dtype=bool)

    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        if abs(x - ri) < tol:
            wall_contact[i, 0] = True
        if abs(1.0 - x - ri) < tol:
            wall_contact[i, 1] = True
        if abs(y - ri) < tol:
            wall_contact[i, 2] = True
        if abs(1.0 - y - ri) < tol:
            wall_contact[i, 3] = True

    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if abs(dist - (r[i] + r[j])) < tol:
                adj[i, j] = True
                adj[j, i] = True

    return adj, wall_contact


def _find_rigid_cliques(adj, wall_contact, min_clique_size=3):
    """
    Find rigid cliques: sets of circles that are all mutually touching (clique in contact graph)
    and at least one circle in the clique touches a wall. These are likely rigid and hard to improve.
    Returns list of clique index lists.
    """
    n = len(adj)
    # Simple approach: find all maximal cliques of size >= min_clique_size
    # For n=26, this is feasible with a simple backtracking
    cliques = []
    visited = [False] * n

    def expand(clique, candidates):
        if not candidates:
            if len(clique) >= min_clique_size:
                # Check if any circle in clique touches a wall
                if any(wall_contact[i].any() for i in clique):
                    cliques.append(list(clique))
            return
        for idx in range(len(candidates)):
            c = candidates[idx]
            # Check if c can join clique (must be adjacent to all in clique)
            can_join = all(adj[c][i] for i in clique)
            if can_join:
                new_clique = clique + [c]
                new_candidates = candidates[idx+1:]
                # Filter candidates to those adjacent to c
                new_candidates = [x for x in new_candidates if adj[c][x]]
                expand(new_clique, new_candidates)

    expand([], list(range(n)))
    return cliques


def _constrained_optimize(xy, r, free_indices, deadline):
    """
    Optimize positions and radii of free circles only, keeping fixed circles in place.
    Uses SLSQP with strict inequality constraints.
    """
    n = len(xy)
    m = len(free_indices)
    if m == 0:
        return xy.copy(), r.copy()

    fixed_indices = [i for i in range(n) if i not in free_indices]

    # Variables: for each free circle, (x, y, r)
    v0 = np.zeros(3 * m)
    for k, i in enumerate(free_indices):
        v0[3*k] = xy[i, 0]
        v0[3*k+1] = xy[i, 1]
        v0[3*k+2] = r[i]

    bounds = []
    for k in range(m):
        bounds.append((0.001, 0.999))  # x
        bounds.append((0.001, 0.999))  # y
        bounds.append((0.001, 0.499))  # r

    def unpack(v):
        x = np.zeros(n)
        y = np.zeros(n)
        rr = np.zeros(n)
        for i in fixed_indices:
            x[i] = xy[i, 0]
            y[i] = xy[i, 1]
            rr[i] = r[i]
        for k, i in enumerate(free_indices):
            x[i] = v[3*k]
            y[i] = v[3*k+1]
            rr[i] = v[3*k+2]
        return x, y, rr

    def objective(v):
        if time.time() > deadline:
            return float('inf')
        x, y, rr = unpack(v)
        return -np.sum(rr[free_indices])

    def objective_grad(v):
        if time.time() > deadline:
            return np.zeros(3 * m)
        g = np.zeros(3 * m)
        for k in range(m):
            g[3*k+2] = -1.0  # gradient of -r_i
        return g

    constraints = []

    # Wall constraints for free circles: x >= r, 1-x >= r, y >= r, 1-y >= r
    for k, i in enumerate(free_indices):
        # x - r >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, k=k: v[3*k] - v[3*k+2],
            'jac': lambda v, k=k: np.concatenate([np.eye(1)[0] - np.zeros(3*m), np.zeros(3*m), np.eye(1)[0] * -1])
        })
        # 1 - x - r >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, k=k: 1.0 - v[3*k] - v[3*k+2],
            'jac': lambda v, k=k: np.concatenate([np.eye(1)[0] * -1, np.zeros(3*m), np.eye(1)[0] * -1])
        })
        # y - r >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, k=k: v[3*k+1] - v[3*k+2],
            'jac': lambda v, k=k: np.concatenate([np.zeros(3*m), np.eye(1)[0] - np.zeros(3*m), np.eye(1)[0] * -1])
        })
        # 1 - y - r >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, k=k: 1.0 - v[3*k+1] - v[3*k+2],
            'jac': lambda v, k=k: np.concatenate([np.zeros(3*m), np.eye(1)[0] * -1, np.eye(1)[0] * -1])
        })

    # Pairwise constraints: for each pair (i, j), dist >= r_i + r_j
    # Only need to check pairs where at least one is free
    for i in range(n):
        for j in range(i + 1, n):
            if i not in free_indices and j not in free_indices:
                continue
            # dist(i,j) - r_i - r_j >= 0
            def pair_fun(v, i=i, j=j):
                x, y, rr = unpack(v)
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                return dist - rr[i] - rr[j]

            def pair_jac(v, i=i, j=j):
                x, y, rr = unpack(v)
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx * dx + dy * dy
                if dist_sq < 1e-20:
                    return np.zeros(3 * m)
                dist = np.sqrt(dist_sq)
                g = np.zeros(3 * m)
                # d(dist)/dx_i = dx/dist, d(dist)/dx_j = -dx/dist
                # d(dist)/dy_i = dy/dist, d(dist)/dy_j = -dy/dist
                # d(-r_i - r_j)/dr_i = -1, d(-r_i - r_j)/dr_j = -1
                if i in free_indices:
                    ki = free_indices.index(i)
                    g[3*ki] += dx / dist
                    g[3*ki+1] += dy / dist
                    g[3*ki+2] -= 1.0
                if j in free_indices:
                    kj = free_indices.index(j)
                    g[3*kj] -= dx / dist
                    g[3*kj+1] -= dy / dist
                    g[3*kj+2] -= 1.0
                return g

            constraints.append({
                'type': 'ineq',
                'fun': pair_fun,
                'jac': pair_jac
            })

    try:
        result = minimize(objective, v0, jac=objective_grad, method='SLSQP',
                          bounds=bounds, constraints=constraints,
                          options={'maxiter': 500, 'ftol': 1e-10, 'disp': False})
        if result.success:
            v_opt = result.x
            new_xy = xy.copy()
            new_r = r.copy()
            for k, i in enumerate(free_indices):
                new_xy[i, 0] = v_opt[3*k]
                new_xy[i, 1] = v_opt[3*k+1]
                new_r[i] = v_opt[3*k+2]
            return new_xy, new_r
    except Exception:
        pass

    return xy.copy(), r.copy()


def _grid_init(side):
    """Create a grid of up to N circles."""
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) < N:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 0.002
                positions.append([x, y, r])
    while len(positions) < N:
        positions.append([0.5, 0.5, 0.01])
    return np.array(positions[:N])


def _random_init(seed=42):
    """Random initialization with small circles."""
    rng = np.random.RandomState(seed)
    xy = rng.uniform(0.1, 0.9, size=(N, 2))
    r = rng.uniform(0.05, 0.15, size=N)
    return xy, r


def _seeded_init(n_seeds=26):
    """
    Seed circles using largest-empty-circle placement.
    Start with a few large circles, then fill in with largest empty circles.
    """
    xy_list = []
    r_list = []

    # Start with 4-6 large circles in corners and center
    initial_positions = [
        (0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75), (0.5, 0.5)
    ]
    for x, y in initial_positions[:min(5, n_seeds)]:
        r = 0.22
        xy_list.append([x, y])
        r_list.append(r)

    # Add remaining circles at largest empty spots
    for _ in range(len(initial_positions), n_seeds):
        xy = np.array(xy_list) if xy_list else np.zeros((0, 2))
        r = np.array(r_list) if r_list else np.zeros(0)
        # Simple largest empty circle: grid scan
        n_grid = 100
        gx = np.linspace(0.0, 1.0, n_grid)
        gy = np.linspace(0.0, 1.0, n_grid)
        GX, GY = np.meshgrid(gx, gy, indexing='ij')
        candidates = np.column_stack([GX.ravel(), GY.ravel()])
        M = len(candidates)

        wall_clear = np.minimum(np.minimum(candidates[:, 0], 1.0 - candidates[:, 0]),
                                np.minimum(candidates[:, 1], 1.0 - candidates[:, 1]))

        if len(xy) > 0:
            best_r = np.zeros(M)
            chunk = 10000
            for start in range(0, M, chunk):
                end = min(start + chunk, M)
                cx = candidates[start:end, 0:1]
                cy = candidates[start:end, 1:2]
                dx = cx - xy[:, 0][np.newaxis, :]
                dy = cy - xy[:, 1][np.newaxis, :]
                dists = np.sqrt(dx * dx + dy * dy)
                clear = np.min(dists - r[np.newaxis, :], axis=1)
                best_r[start:end] = np.minimum(wall_clear[start:end], clear)
        else:
            best_r = wall_clear.copy()

        idx = int(np.argmax(best_r))
        x, y = candidates[idx, 0], candidates[idx, 1]
        rad = max(best_r[idx], 0.02)
        xy_list.append([x, y])
        r_list.append(rad)

    return np.array(xy_list), np.array(r_list)


def _refine_with_constrained_opt(xy, r, deadline):
    """
    Refine a packing using constrained optimization on free circles.
    1. Build contact graph
    2. Find rigid cliques
    3. Mark circles in cliques as fixed
    4. Optimize free circles
    """
    if time.time() > deadline:
        return xy.copy(), r.copy()

    adj, wall_contact = _build_contact_graph(xy, r, tol=1e-5)
    cliques = _find_rigid_cliques(adj, wall_contact, min_clique_size=3)

    # Mark circles in cliques as fixed
    fixed_set = set()
    for clique in cliques:
        for i in clique:
            fixed_set.add(i)

    free_indices = [i for i in range(len(xy)) if i not in fixed_set]

    if len(free_indices) == 0:
        return xy.copy(), r.copy()

    return _constrained_optimize(xy, r, free_indices, deadline)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    best_xy = None
    best_r = None
    best_total = 0.0
    best_valid = False

    # Multiple initializations
    inits = []

    # Grid inits
    for side in [5, 6, 7]:
        grid = _grid_init(side)
        inits.append((grid[:, :2].copy(), grid[:, 2].copy()))

    # Random inits with different seeds
    for seed in [1, 2, 3, 4, 5]:
        xy, r = _random_init(seed)
        inits.append((xy, r))

    # Seeded init
    xy, r = _seeded_init(N)
    inits.append((xy, r))

    for init_xy, init_r in inits:
        if time.time() > deadline - 10:
            break

        # Quick validation and fix
        total, valid = _eval_packing(init_xy, init_r)
        if not valid:
            # Try to fix by shrinking radii
            for i in range(N):
                init_r[i] = min(init_r[i], init_xy[i, 0], 1.0 - init_xy[i, 0],
                               init_xy[i, 1], 1.0 - init_xy[i, 1])
                init_r[i] = max(init_r[i], 0.001)
            total, valid = _eval_packing(init_xy, init_r)
            if not valid:
                continue

        # Constrained optimization refinement
        refined_xy, refined_r = _refine_with_constrained_opt(init_xy, init_r, deadline - 5)
        total, valid = _eval_packing(refined_xy, refined_r)
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_xy = refined_xy.copy()
            best_r = refined_r.copy()

    # If we have a valid result, do a final polish with all circles free
    if best_valid and best_xy is not None and time.time() < deadline - 5:
        # Try optimizing all circles as free
        all_free = list(range(N))
        polished_xy, polished_r = _constrained_optimize(best_xy, best_r, all_free, deadline - 2)
        total, valid = _eval_packing(polished_xy, polished_r)
        if valid and total > best_total:
            best_total = total
            best_xy = polished_xy.copy()
            best_r = polished_r.copy()

    # Fallback
    if best_xy is None or not best_valid:
        side = int(np.ceil(np.sqrt(N)))
        grid = _grid_init(side)
        xy = grid[:, :2].copy()
        r = grid[:, 2].copy()
        total, valid = _eval_packing(xy, r)
        if valid:
            best_total = total
            best_valid = True
            best_xy = xy
            best_r = r
        else:
            result = np.zeros((N, 3))
            for i in range(N):
                gx = i % side
                gy = i // side
                result[i, 0] = (gx + 0.5) / side
                result[i, 1] = (gy + 0.5) / side
                result[i, 2] = 0.4 / side
            return result

    # Final validation and clipping
    n = len(best_xy)
    result = np.zeros((n, 3))
    for i in range(n):
        x, y, ri = best_xy[i, 0], best_xy[i, 1], best_r[i]
        ri = min(ri, x, 1.0 - x, y, 1.0 - y)
        ri = max(ri, 0.001)
        result[i, 2] = ri
        result[i, 0] = np.clip(x, ri, 1.0 - ri)
        result[i, 1] = np.clip(y, ri, 1.0 - ri)

    # Final check
    final_xy = result[:, :2]
    final_r = result[:, 2]
    total, valid = _eval_packing(final_xy, final_r)
    if not valid:
        side = int(np.ceil(np.sqrt(N)))
        result = np.zeros((N, 3))
        for i in range(N):
            gx = i % side
            gy = i // side
            result[i, 0] = (gx + 0.5) / side
            result[i, 1] = (gy + 0.5) / side
            result[i, 2] = 0.4 / side

    return result
# EVOLVE-BLOCK-END
