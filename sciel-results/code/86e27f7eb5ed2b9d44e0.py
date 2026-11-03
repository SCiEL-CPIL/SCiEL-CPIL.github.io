"""Circle packing in unit square: pack N circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

DEADLINE = 140.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START
def _objective_and_grad(z, n):
    """z = [x0,y0,r0, x1,y1,r1, ...]. Minimize negative sum of radii.
    Returns (f, grad)."""
    z = np.asarray(z)
    xy = z[0::3]
    r = z[2::3]
    f = -np.sum(r)
    # gradient
    g = np.zeros_like(z)
    g[2::3] = -1.0
    return f, g


def _constraints(n):
    """Return list of constraint dicts (non-overlap + boundary) for SLSQP."""
    cons = []
    # boundary: r >= 0 handled implicitly; x-r>=0, x+r<=1, y-r>=0, y+r<=1
    for i in range(n):
        cons.append({'type': 'ineq', 'fun': lambda z, i=i: z[3*i] - z[3*i+2], 'jac': lambda z, i=i: np.array([1.0, 0.0, -1.0])})
        cons.append({'type': 'ineq', 'fun': lambda z, i=i: 1.0 - z[3*i] - z[3*i+2], 'jac': lambda z, i=i: np.array([-1.0, 0.0, -1.0])})
        cons.append({'type': 'ineq', 'fun': lambda z, i=i: z[3*i+1] - z[3*i+2], 'jac': lambda z, i=i: np.array([0.0, 1.0, -1.0])})
        cons.append({'type': 'ineq', 'fun': lambda z, i=i: 1.0 - z[3*i+1] - z[3*i+2], 'jac': lambda z, i=i: np.array([0.0, -1.0, -1.0])})
    # non-overlap: dist >= r_i + r_j  =>  (x_i-x_j)^2+(y_i-y_j)^2 - (r_i+r_j)^2 >= 0
    for i in range(n):
        for j in range(i+1, n):
            cons.append({'type': 'ineq',
                         'fun': lambda z, i=i, j=j: (z[3*i]-z[3*j])**2 + (z[3*i+1]-z[3*j+1])**2 - (z[3*i+2]+z[3*j+2])**2,
                         'jac': lambda z, i=i, j=j: _overlap_grad(z, i, j)})
    return cons


def _overlap_grad(z, i, j):
    g = np.zeros_like(z)
    dx = z[3*i] - z[3*j]
    dy = z[3*i+1] - z[3*j+1]
    g[3*i] = 2*dx
    g[3*i+1] = 2*dy
    g[3*j] = -2*dx
    g[3*j+1] = -2*dy
    g[3*i+2] = -2*(z[3*i+2]+z[3*j+2])
    g[3*j+2] = -2*(z[3*i+2]+z[3*j+2])
    return g


def _is_valid(arr, n, tol=1e-6):
    """Check disjoint interiors and inside unit square."""
    if arr.shape != (n, 3):
        return False
    xy = arr[:, :2]
    r = arr[:, 2]
    if np.any(r < -tol):
        return False
    # boundaries
    if np.any(xy[:, 0] - r < -tol) or np.any(xy[:, 0] + r > 1.0 + tol):
        return False
    if np.any(xy[:, 1] - r < -tol) or np.any(xy[:, 1] + r > 1.0 + tol):
        return False
    # non-overlap
    for i in range(n):
        for j in range(i+1, n):
            d = np.hypot(xy[i,0]-xy[j,0], xy[i,1]-xy[j,1])
            if d < r[i] + r[j] - tol:
                return False
    return True


def _repair(arr, n, tol=1e-6):
    """Make arrangement strictly valid by shrinking radii / nudging positions."""
    arr = arr.copy().astype(float)
    # clamp radii non-negative
    arr[:, 2] = np.clip(arr[:, 2], 0.0, None)
    # clamp centers so circles touch boundary
    for i in range(n):
        r = arr[i, 2]
        arr[i, 0] = np.clip(arr[i, 0], r, 1.0 - r)
        arr[i, 1] = np.clip(arr[i, 1], r, 1.0 - r)
    # shrink radii until no overlap (iterative)
    for _ in range(200):
        ok = True
        for i in range(n):
            for j in range(i+1, n):
                d = np.hypot(arr[i,0]-arr[j,0], arr[i,1]-arr[j,1])
                need = arr[i,2] + arr[j,2]
                if d < need:
                    # shrink both proportionally
                    excess = need - d
                    # reduce radii to leave a gap
                    scale = (d - 1e-7) / need if need > 0 else 0.0
                    arr[i, 2] *= scale
                    arr[j, 2] *= scale
                    ok = False
        if ok:
            break
    # final boundary clamp again
    for i in range(n):
        r = arr[i, 2]
        arr[i, 0] = np.clip(arr[i, 0], r, 1.0 - r)
        arr[i, 1] = np.clip(arr[i, 1], r, 1.0 - r)
    return arr


def _seed_grid(n):
    side = int(np.ceil(np.sqrt(n)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
             for i in range(side) for j in range(side)]
    return np.array(cells[:n])


def _random_start(n, rng):
    """Random feasible-ish start: small radii, random positions."""
    r = rng.uniform(0.02, 0.08, size=n)
    x = rng.uniform(r, 1.0 - r, size=n)
    y = rng.uniform(r, 1.0 - r, size=n)
    return np.column_stack([x, y, r])


def _start_to_z(arr):
    return arr.flatten()


def _z_to_arr(z):
    return z.reshape(-1, 3)


def construct():
    t0 = time.time()
    n = N
    cons = _constraints(n)
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * n

    # initial best: grid seed
    grid = _seed_grid(n)
    grid = _repair(grid, n)
    best_arr = grid
    best_val = float(np.sum(grid[:, 2]))

    rng = np.random.default_rng(12345)

    def remaining():
        return DEADLINE - (time.time() - t0)

    # multi-start: grid + random perturbations
    starts = [grid]
    for _ in range(12):
        starts.append(_random_start(n, rng))

    for s in starts:
        if remaining() < 5.0:
            break
        s = _repair(s, n)
        z0 = _start_to_z(s)
        try:
            res = minimize(
                _objective_and_grad, z0, args=(n,),
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                jac=None,
                options={'maxiter': 300, 'ftol': 1e-9}
            )
        except Exception:
            continue
        if res.x is None:
            continue
        arr = _z_to_arr(res.x)
        arr = _repair(arr, n)
        if _is_valid(arr, n):
            val = float(np.sum(arr[:, 2]))
            if val > best_val + 1e-9:
                best_val = val
                best_arr = arr

    return best_arr
# EVOLVE-BLOCK-END
