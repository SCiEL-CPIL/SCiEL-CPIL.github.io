"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii."""
import time
import numpy as np
from scipy.optimize import minimize

N = 21
DEADLINE = time.time() + 140.0  # safety margin before 150s kill


def _validity_check(circles, tol=1e-7):
    """Check if circles are valid: disjoint interiors, inside bounding rect with perimeter <= 4."""
    n = len(circles)
    if n != N:
        return False
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    if np.any(rs <= 0):
        return False
    # Bounding rectangle
    xmin = np.min(xs - rs)
    xmax = np.max(xs + rs)
    ymin = np.min(ys - rs)
    ymax = np.max(ys + rs)
    W = xmax - xmin
    H = ymax - ymin
    perimeter = 2 * (W + H)
    if perimeter > 4.0 + tol:
        return False
    # Check pairwise non-overlap
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < rs[i] + rs[j] - tol:
                return False
    return True


def _sum_radii(circles):
    return float(np.sum(circles[:, 2]))


def _grid_init():
    """Create a grid-like initial placement."""
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) < N:
                cells.append([0.5 + i, 0.5 + j, 0.45])
    return np.array(cells, dtype=float)


def _random_init(seed):
    """Random initial placement in a 2x2 square (perimeter 8, will be scaled)."""
    rng = np.random.RandomState(seed)
    xs = rng.uniform(0, 2, N)
    ys = rng.uniform(0, 2, N)
    rs = rng.uniform(0.3, 0.5, N)
    return np.column_stack([xs, ys, rs])


def _optimize_from_init(init, max_iter=500):
    """
    Optimize: variables are x, y, r for each circle (3N vars).
    Objective: maximize sum(r) subject to:
      - r_i > 0
      - pairwise: dist(i,j) >= r_i + r_j
      - bounding box perimeter <= 4: 2*((max(x+r)-min(x-r)) + (max(y+r)-min(y-r))) <= 4
    We use a penalty approach with SLSQP.
    """
    n = len(init)
    x0 = init.flatten()

    def objective(v):
        if time.time() > DEADLINE - 5:
            return 0.0
        X = v.reshape(n, 3)
        rs = X[:, 2]
        # Penalty for negative radii
        pen = 100.0 * np.sum(np.maximum(0, -rs))
        # Penalty for overlaps
        xs, ys = X[:, 0], X[:, 1]
        for i in range(n):
            for j in range(i + 1, n):
                d = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                overlap = rs[i] + rs[j] - d
                if overlap > 0:
                    pen += 100.0 * overlap
        # Penalty for perimeter > 4
        xmin = np.min(xs - rs)
        xmax = np.max(xs + rs)
        ymin = np.min(ys - rs)
        ymax = np.max(ys + rs)
        perim = 2.0 * ((xmax - xmin) + (ymax - ymin))
        if perim > 4.0:
            pen += 100.0 * (perim - 4.0)
        return -np.sum(rs) + pen

    # Use a simpler approach: iterative relaxation
    X = init.copy()
    best_X = X.copy()
    best_val = _sum_radii(X) if _validity_check(X) else -1

    for it in range(max_iter):
        if time.time() > DEADLINE - 10:
            break
        # Repulsion: push overlapping circles apart
        converged = True
        for i in range(n):
            for j in range(i + 1, n):
                dx = X[i, 0] - X[j, 0]
                dy = X[i, 1] - X[j, 1]
                d = np.sqrt(dx*dx + dy*dy)
                min_d = X[i, 2] + X[j, 2]
                if d < min_d:
                    converged = False
                    if d < 1e-12:
                        dx, dy = 1e-6, 0.0
                        d = 1e-6
                    # Push apart
                    push = (min_d - d) / (2.0 * d)
                    X[i, 0] += push * dx
                    X[i, 1] += push * dy
                    X[j, 0] -= push * dx
                    X[j, 1] -= push * dy
        # Shrink to fit perimeter: scale positions toward center
        xs, ys, rs = X[:, 0], X[:, 1], X[:, 2]
        xmin = np.min(xs - rs)
        xmax = np.max(xs + rs)
        ymin = np.min(ys - rs)
        ymax = np.max(ys + rs)
        W = xmax - xmin
        H = ymax - ymin
        perim = 2.0 * (W + H)
        if perim > 4.0:
            scale = 4.0 / perim
            cx = (xmin + xmax) / 2.0
            cy = (ymin + ymax) / 2.0
            X[:, 0] = cx + (X[:, 0] - cx) * scale
            X[:, 1] = cy + (X[:, 1] - cy) * scale
            converged = False
        # Check validity and track best
        if _validity_check(X):
            val = _sum_radii(X)
            if val > best_val:
                best_val = val
                best_X = X.copy()
        if converged and _validity_check(X):
            val = _sum_radii(X)
            if val > best_val:
                best_val = val
                best_X = X.copy()
            break
    return best_X


def _slsqp_optimize(init, max_iter=200):
    """SLSQP-based optimization with explicit constraints."""
    n = len(init)
    x0 = init.flatten()

    def obj(v):
        X = v.reshape(n, 3)
        return -np.sum(X[:, 2])

    def grad_obj(v):
        g = np.zeros_like(v)
        g[2::3] = -1.0
        return g

    constraints = []
    # Perimeter constraint: 2*((max(x+r)-min(x-r)) + (max(y+r)-min(y-r))) <= 4
    # This is non-smooth; approximate with a soft constraint via penalty in objective
    # Instead, use bounds and a simpler approach:
    # We'll use the penalty method embedded in objective

    def obj_pen(v):
        X = v.reshape(n, 3)
        rs = X[:, 2]
        xs, ys = X[:, 0], X[:, 1]
        pen = 0.0
        # Negative radius penalty
        neg = np.maximum(0, -rs)
        pen += 1000.0 * np.sum(neg)
        # Overlap penalty
        for i in range(n):
            for j in range(i + 1, n):
                d = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                ov = rs[i] + rs[j] - d
                if ov > 0:
                    pen += 500.0 * ov
        # Perimeter penalty
        xmin = np.min(xs - rs)
        xmax = np.max(xs + rs)
        ymin = np.min(ys - rs)
        ymax = np.max(ys + rs)
        perim = 2.0 * ((xmax - xmin) + (ymax - ymin))
        if perim > 4.0:
            pen += 500.0 * (perim - 4.0)
        return -np.sum(rs) + pen

    # Bounds: radii in (0, 1], positions in [-1, 3]
    bounds = []
    for i in range(n):
        bounds.append((-2.0, 3.0))  # x
        bounds.append((-2.0, 3.0))  # y
        bounds.append((0.001, 1.0))  # r

    try:
        res = minimize(obj_pen, x0, method='SLSQP', bounds=bounds,
                       options={'maxiter': max_iter, 'ftol': 1e-8})
        if res.success or res.fun < 0:
            X = res.x.reshape(n, 3)
            if _validity_check(X):
                return X
    except Exception:
        pass
    return None


def construct():
    t_start = time.time()
    best_X = None
    best_val = -1

    # Multiple random restarts
    for seed in range(50):
        if time.time() > DEADLINE - 20:
            break
        # Mix of grid and random inits
        if seed == 0:
            init = _grid_init()
        else:
            init = _random_init(seed)

        # Relaxation phase
        X_relaxed = _optimize_from_init(init, max_iter=300)
        if _validity_check(X_relaxed):
            val = _sum_radii(X_relaxed)
            if val > best_val:
                best_val = val
                best_X = X_relaxed.copy()

        # SLSQP polish if time permits
        if time.time() < DEADLINE - 30:
            X_slsqp = _slsqp_optimize(X_relaxed, max_iter=100)
            if X_slsqp is not None:
                val2 = _sum_radii(X_slsqp)
                if val2 > best_val:
                    best_val = val2
                    best_X = X_slsqp.copy()

    # Final polish: try to squeeze more by iterative radius increase
    if best_X is not None and time.time() < DEADLINE - 15:
        X = best_X.copy()
        for _ in range(50):
            if time.time() > DEADLINE - 10:
                break
            improved = False
            for i in range(N):
                # Try increasing radius slightly
                old_r = X[i, 2]
                X[i, 2] = old_r * 1.005
                if _validity_check(X):
                    improved = True
                else:
                    X[i, 2] = old_r
            if not improved:
                break

    if best_X is None:
        # Fallback: return the simple grid
        return _grid_init()

    return best_X


# Quick self-test
if __name__ == "__main__":
    result = construct()
    print(f"Sum of radii: {_sum_radii(result)}")
    print(f"Valid: {_validity_check(result)}")
    xs, ys, rs = result[:,0], result[:,1], result[:,2]
    print(f"Perimeter: {2*(np.max(xs+rs)-np.min(xs-rs)+np.max(ys+rs)-np.min(ys-rs))}")
