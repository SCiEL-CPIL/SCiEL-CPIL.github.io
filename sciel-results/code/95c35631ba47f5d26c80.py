"""Circle packing in unit square: maximize sum of radii for 26 disjoint circles.
Strategy: Global-constraint interior-point optimization (trust-constr).
We enforce all pairwise non-overlap constraints directly (not just Delaunay edges),
using scipy.optimize.minimize with method='trust-constr'. This avoids the
limitation of a fixed contact graph and allows the solver to find better packings
by considering all interactions simultaneously.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.sparse import lil_matrix

N = 26
EPS = 1e-9
DEADLINE = 140.0  # Safety margin for 150s limit


def _max_radius(pos, r):
    """Maximal radius for each circle given current positions and radii."""
    n = len(r)
    mr = np.empty(n)
    for i in range(n):
        dmin = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
        for j in range(n):
            if i != j:
                dmin = min(dmin, np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1]) - r[j])
        mr[i] = max(0.0, dmin)
    return mr


def _all_pairs():
    """Return all unique pairs (i, j) with i < j."""
    return [(i, j) for i in range(N) for j in range(i + 1, N)]


def _pair_constraints(x):
    """Inequality constraints: ||p_i - p_j|| - (r_i + r_j) >= 0 for all pairs."""
    p = x[:2 * N].reshape(N, 2)
    r = x[2 * N:]
    vals = np.empty(_NPAIRS)
    k = 0
    for i, j in _PAIRS:
        d = np.hypot(p[i, 0] - p[j, 0], p[i, 1] - p[j, 1])
        vals[k] = d - r[i] - r[j]
        k += 1
    return vals


def _pair_constraint_jac(x):
    """Jacobian for all-pair inequality constraints (sparse)."""
    p = x[:2 * N].reshape(N, 2)
    r = x[2 * N:]
    J = lil_matrix((_NPAIRS, 3 * N), dtype=float)
    k = 0
    for i, j in _PAIRS:
        diff = p[i] - p[j]
        d = np.hypot(diff[0], diff[1])
        if d < EPS:
            d = EPS
        ux, uy = diff[0] / d, diff[1] / d
        J[k, 2 * i] = ux
        J[k, 2 * i + 1] = uy
        J[k, 2 * j] = -ux
        J[k, 2 * j + 1] = -uy
        J[k, 2 * N + i] = -1.0
        J[k, 2 * N + j] = -1.0
        k += 1
    return J.tocsr()


def _objective(x):
    """Minimize negative sum of radii."""
    return -np.sum(x[2 * N:])


def _objective_grad(x):
    g = np.zeros_like(x)
    g[2 * N:] = -1.0
    return g


def _boundary_constraints(x):
    """Inequality constraints for boundary: x_i - r_i >= 0, 1 - x_i - r_i >= 0, etc."""
    p = x[:2 * N].reshape(N, 2)
    r = x[2 * N:]
    # For each circle i: x_i - r_i >= 0, 1 - x_i - r_i >= 0, y_i - r_i >= 0, 1 - y_i - r_i >= 0
    # Total 4*N constraints
    vals = np.empty(4 * N)
    for i in range(N):
        vals[4 * i] = p[i, 0] - r[i]
        vals[4 * i + 1] = 1.0 - p[i, 0] - r[i]
        vals[4 * i + 2] = p[i, 1] - r[i]
        vals[4 * i + 3] = 1.0 - p[i, 1] - r[i]
    return vals


def _boundary_constraint_jac(x):
    """Jacobian for boundary constraints (sparse)."""
    J = lil_matrix((4 * N, 3 * N), dtype=float)
    for i in range(N):
        # x_i - r_i >= 0
        J[4 * i, 2 * i] = 1.0
        J[4 * i, 2 * N + i] = -1.0
        # 1 - x_i - r_i >= 0
        J[4 * i + 1, 2 * i] = -1.0
        J[4 * i + 1, 2 * N + i] = -1.0
        # y_i - r_i >= 0
        J[4 * i + 2, 2 * i + 1] = 1.0
        J[4 * i + 2, 2 * N + i] = -1.0
        # 1 - y_i - r_i >= 0
        J[4 * i + 3, 2 * i + 1] = -1.0
        J[4 * i + 3, 2 * N + i] = -1.0
    return J.tocsr()


# Precompute pairs
_PAIRS = _all_pairs()
_NPAIRS = len(_PAIRS)


def _polish(cands):
    """Final polish: fix positions, maximize radii exactly."""
    if not cands or len(cands) < N:
        return cands
    arr = np.array(cands[:N])
    pos = arr[:, :2].copy()
    rad = arr[:, 2].copy()
    
    # Iterate: fix pos -> max rad
    for it in range(30):
        if time.time() > DEADLINE - 5:
            break
        rad_new = _max_radius(pos, rad)
        if np.allclose(rad_new, rad, atol=1e-10):
            rad = rad_new
            break
        rad = rad_new
    out = np.column_stack([pos, rad])
    return out.tolist()


def construct():
    t_start = time.time()
    rng = np.random.default_rng(42)
    best = None
    best_score = -1.0

    def consider(cands):
        nonlocal best, best_score
        # keep only valid (non-overlapping, in-square) circles
        valid = []
        for x, y, r in cands:
            if r <= 0:
                continue
            if x < r + EPS or x > 1 - r - EPS or y < r + EPS or y > 1 - r - EPS:
                continue
            ok = True
            for x2, y2, r2 in valid:
                if (x - x2) ** 2 + (y - y2) ** 2 < (r + r2) ** 2 - 1e-9:
                    ok = False
                    break
            if ok:
                valid.append((x, y, r))
        if len(valid) < N:
            return
        arr = np.array(valid[:N])
        s = float(arr[:, 2].sum())
        if s > best_score:
            best_score = s
            best = arr

    # ---- Phase A: Generate diverse initializations ----
    all_starts = []

    # 1. Grid initializations
    for side in range(4, 8):
        if time.time() > DEADLINE:
            break
        cells = []
        for i in range(side):
            for j in range(side):
                cells.append(((i + 0.5) / side, (j + 0.5) / side))
        cells = cells[:N]
        pos = np.array(cells, dtype=float)
        r = np.full(N, 0.5 / side - 1e-3)
        for _ in range(10):
            r = _max_radius(pos, r)
        all_starts.append(np.concatenate([pos, r]))

    # 2. Hexagonal packing initialization
    try:
        pos_hex = []
        side = int(np.ceil(np.sqrt(N)))
        for i in range(side):
            for j in range(side):
                if len(pos_hex) >= N:
                    break
                x = (j + 0.5 * (i % 2)) / side
                y = (i + 0.5) / side
                pos_hex.append((x, y))
        pos_hex = np.array(pos_hex[:N])
        pos_hex = np.clip(pos_hex, 0.1, 0.9)
        r_hex = np.full(N, 0.05)
        for _ in range(20):
            r_hex = _max_radius(pos_hex, r_hex)
        all_starts.append(np.concatenate([pos_hex, r_hex]))
    except Exception:
        pass

    # 3. Random perturbations of grid
    for side in [5, 6, 7]:
        if time.time() > DEADLINE:
            break
        cells = []
        for i in range(side):
            for j in range(side):
                cells.append(((i + 0.5) / side, (j + 0.5) / side))
        cells = cells[:N]
        pos = np.array(cells, dtype=float)
        r = np.full(N, 0.5 / side - 1e-3)
        for _ in range(10):
            r = _max_radius(pos, r)
        
        for k in range(8):
            if time.time() > DEADLINE:
                break
            p0 = pos + 0.05 * (rng.random((N, 2)) * 2 - 1)
            p0 = np.clip(p0, 0.05, 0.95)
            r0 = r.copy()
            r0 = _max_radius(p0, r0)
            all_starts.append(np.concatenate([p0, r0]))

    # 4. Pure random
    for k in range(8):
        if time.time() > DEADLINE:
            break
        p0 = rng.random((N, 2)) * 0.8 + 0.1
        r0 = np.full(N, 0.03)
        r0 = _max_radius(p0, r0)
        all_starts.append(np.concatenate([p0, r0]))

    if best is not None:
        all_starts.append(best.copy())

    # ---- Phase B: Trust-constr optimization with all-pair constraints ----
    # Bounds: positions in [0,1], radii in [0, 0.5]
    bounds = [(0, 1)] * (2 * N) + [(0, 0.5)] * N

    # We'll run trust-constr on each start, iterating a few times if needed
    for x0 in all_starts:
        if time.time() > DEADLINE - 20:
            break
        
        current_x = x0.copy()
        
        # Run trust-constr
        try:
            res = minimize(
                _objective,
                current_x,
                jac=_objective_grad,
                bounds=bounds,
                constraints=[
                    {'type': 'ineq', 'fun': _pair_constraints, 'jac': _pair_constraint_jac},
                    {'type': 'ineq', 'fun': _boundary_constraints, 'jac': _boundary_constraint_jac},
                ],
                method='trust-constr',
                options={'maxiter': 500, 'gtol': 1e-6, 'xtol': 1e-8, 'verbose': 0}
            )
            
            if res.fun < -np.sum(current_x[2 * N:]):
                current_x = res.x
        except Exception:
            pass

        # Second pass with tighter tolerances if time permits
        if time.time() < DEADLINE - 30:
            try:
                res2 = minimize(
                    _objective,
                    current_x,
                    jac=_objective_grad,
                    bounds=bounds,
                    constraints=[
                        {'type': 'ineq', 'fun': _pair_constraints, 'jac': _pair_constraint_jac},
                        {'type': 'ineq', 'fun': _boundary_constraints, 'jac': _boundary_constraint_jac},
                    ],
                    method='trust-constr',
                    options={'maxiter': 300, 'gtol': 1e-8, 'xtoL': 1e-10, 'verbose': 0}
                )
                if res2.fun < -np.sum(current_x[2 * N:]):
                    current_x = res2.x
            except Exception:
                pass

        # Convert to list and polish
        pos_final = current_x[:2 * N].reshape(N, 2)
        rad_final = current_x[2 * N:]
        cands = list(zip(pos_final[:, 0], pos_final[:, 1], rad_final))
        polished = _polish(cands)
        consider(polished)

    # Final polish on the best solution found
    if best is not None:
        cands = best.tolist()
        polished = _polish(cands)
        consider(polished)

    if best is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    return best


if __name__ == '__main__':
    out = construct()
    print("sum of radii:", out[:, 2].sum())
    print(out)
