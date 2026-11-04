"""Circle packing in unit square: pack 26 circles, maximize sum of radii.
NEW STRATEGY: Lagrangian-dual / squared-radius convex-QP framework.
- Variables: centers c_i (2n) and squared radii rho_i = r_i^2 (n).
- For fixed centers, rho_i must satisfy:
    rho_i >= d_ij^2   for all j != i   (pair non-overlap, squared)
    rho_i >= d_wall_i^2  (wall clearance squared)
  and we maximize sum sqrt(rho_i)  (concave in rho) subject to rho_i >= lower_i.
  Since sqrt is concave, maximizing it over a box [lower_i, upper_i] is trivially
  at rho_i = upper_i? No -- we want rho as large as possible, but the upper bound
  comes from the fact that if rho_i > d_ij^2 for some j, circle i overlaps j.
  Actually the constraint is rho_i <= d_ij^2 for ALL j? No:
  Non-overlap: ||c_i - c_j|| >= r_i + r_j  =>  d_ij^2 >= (sqrt(rho_i)+sqrt(rho_j))^2
  This is NOT linear in rho. However, a SUFFICIENT condition is:
    d_ij^2 >= rho_i + rho_j + 2*sqrt(rho_i*rho_j)
  Still nonlinear.

  ALTERNATIVE: Use the fact that for FIXED centers, the optimal rho_i is
  rho_i = min_j d_ij^2  (and wall), because increasing rho_i beyond min_j d_ij^2
  would cause overlap with the nearest neighbor. So:
    rho_i* = min( min_{j!=i} d_ij^2, d_wall_i^2 )
  and sum sqrt(rho_i*) is the objective for fixed centers.
  This is a MAX-MIN problem: maximize min_j d_ij^2 for each i simultaneously.

  So the problem reduces to: find centers c_i that maximize
    f(c) = sum_i sqrt( min( min_{j!=i} ||c_i-c_j||^2, d_wall_i^2 ) )
  This is a smooth-ish (but non-smooth at ties) function of centers.
  We optimize it via gradient descent / L-BFGS with subgradient handling.

  This is MATERIALLY DIFFERENT: no radii variables in the optimizer, no contact
  graph, no penalty method on (x,y,r). The radii are DETERMINED by centers.
"""

N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _eval_packing(xy, r):
    """Evaluate a packing: returns (total_radius, is_valid)."""
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


def _compute_radii_from_centers(xy):
    """
    Given centers, compute the maximum valid radius for each circle:
    r_i = min( min_{j!=i} ||c_i - c_j||/2,  min wall clearance )
    Wait: non-overlap requires ||c_i - c_j|| >= r_i + r_j.
    If we set r_i = d_ij/2 for the nearest j, that's only valid if r_j <= d_ij/2 too.
    The correct max radius for circle i given all others is:
      r_i <= (d_ij - r_j) for all j, and r_i <= wall_clear_i.
    This is coupled. The standard approach: 
      r_i = min( wall_clear_i, min_{j != i} (d_ij - r_j) )
    This is a fixed-point / iterative problem. For the DUAL approach, we use:
      r_i = min( wall_clear_i, 0.5 * min_{j != i} d_ij )
    as a sufficient (but not always tight) condition. Actually 0.5*d_ij is the
    radius if both circles have equal radius. For unequal, it's conservative.
    
    Better: use the exact coupled condition via iterative relaxation:
      r_i^{(k+1)} = min( wall_clear_i, min_{j != i} (d_ij - r_j^{(k)}) )
    This converges monotonically from r=0.
    """
    n = len(xy)
    # Wall clearances
    wall = np.minimum(np.minimum(xy[:, 0], 1.0 - xy[:, 0]),
                      np.minimum(xy[:, 1], 1.0 - xy[:, 1]))
    
    # Pairwise distances
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dx = xy[i, 0] - xy[j, 0]
                dy = xy[i, 1] - xy[j, 1]
                D[i, j] = np.sqrt(dx * dx + dy * dy)
    
    # Iterative radius computation (monotone convergence)
    r = np.zeros(n)
    for _ in range(50):
        new_r = wall.copy()
        for i in range(n):
            for j in range(n):
                if i != j:
                    candidate = D[i, j] - r[j]
                    if candidate < new_r[i]:
                        new_r[i] = candidate
        new_r = np.maximum(new_r, 0.0)
        if np.allclose(new_r, r, atol=1e-12):
            r = new_r
            break
        r = new_r
    
    return r


def _objective_and_grad(xy):
    """
    Objective: sum_i r_i where r_i = _compute_radii_from_centers(xy)[i].
    Gradient via finite differences on the objective (since radii are non-smooth).
    We use a central difference with small step.
    """
    n = len(xy)
    r = _compute_radii_from_centers(xy)
    total = float(np.sum(r))
    
    # Numerical gradient
    h = 1e-5
    grad = np.zeros_like(xy)
    for i in range(n):
        for k in range(2):
            xy_save = xy[i, k].copy()
            xy[i, k] = xy_save + h
            r_plus = _compute_radii_from_centers(xy)
            xy[i, k] = xy_save - h
            r_minus = _compute_radii_from_centers(xy)
            xy[i, k] = xy_save
            grad[i, k] = (np.sum(r_plus) - np.sum(r_minus)) / (2 * h)
    
    return total, grad


def _objective_only(xy):
    """Just the objective value (for L-BFGS with numerical grad)."""
    r = _compute_radii_from_centers(xy)
    return -float(np.sum(r))  # minimize negative


def _optimize_centers(xy0, deadline, method='L-BFGS-B', max_iter=300):
    """
    Optimize centers to maximize sum of radii.
    Uses L-BFGS-B with numerical gradients (since radii are piecewise-smooth).
    """
    n = len(xy0)
    bounds = [(0.01, 0.99)] * (2 * n)
    
    def obj(v):
        if time.time() > deadline:
            return 1e6
        xy = v.reshape(n, 2)
        return _objective_only(xy)
    
    def grad(v):
        if time.time() > deadline:
            return np.zeros(2 * n)
        xy = v.reshape(n, 2)
        _, g = _objective_and_grad(xy)
        return -g.ravel()  # negative because we minimize
    
    try:
        result = minimize(obj, xy0.ravel(), jac=grad, method=method,
                          bounds=bounds,
                          options={'maxiter': max_iter, 'ftol': 1e-10,
                                   'gtol': 1e-6, 'maxfun': 20000})
        new_xy = result.x.reshape(n, 2)
    except Exception:
        new_xy = xy0.copy()
    
    return new_xy


def _grid_init(side):
    """Create a grid of up to N circles (centers only)."""
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) < N:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                positions.append([x, y])
    while len(positions) < N:
        positions.append([0.5, 0.5])
    return np.array(positions[:N])


def _hex_init(n):
    """Hexagonal lattice initialization."""
    positions = []
    row = 0
    while len(positions) < n:
        cols = int(np.ceil(n / (row + 1))) if row == 0 else int(np.ceil(n / 3))
        offset = 0.0 if row % 2 == 0 else 0.5 / max(cols, 1)
        for col in range(max(cols, 1)):
            if len(positions) >= n:
                break
            x = (col + 0.5) / max(cols, 1)
            y = (row + 0.5) / 5.0
            x = np.clip(x + offset * 0.3, 0.05, 0.95)
            y = np.clip(y, 0.05, 0.95)
            positions.append([x, y])
        row += 1
    return np.array(positions[:n])


def _random_init(n, seed=0):
    """Random initialization."""
    rng = np.random.RandomState(seed)
    return rng.uniform(0.1, 0.9, size=(n, 2))


def _refine_with_local_search(xy, deadline):
    """
    After center optimization, do a local refinement:
    For each circle, try to move it slightly to increase its radius,
    checking that it doesn't reduce others' radii too much.
    """
    n = len(xy)
    best_xy = xy.copy()
    best_r = _compute_radii_from_centers(xy)
    best_total = float(np.sum(best_r))
    
    # Try small perturbations of each center
    rng = np.random.RandomState(42)
    for _ in range(20):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            # Try random small moves
            for _ in range(10):
                dx = rng.uniform(-0.02, 0.02)
                dy = rng.uniform(-0.02, 0.02)
                new_xy = xy.copy()
                new_xy[i, 0] = np.clip(new_xy[i, 0] + dx, 0.01, 0.99)
                new_xy[i, 1] = np.clip(new_xy[i, 1] + dy, 0.01, 0.99)
                new_r = _compute_radii_from_centers(new_xy)
                new_total = float(np.sum(new_r))
                if new_total > best_total + 1e-10:
                    xy = new_xy
                    best_xy = new_xy
                    best_r = new_r
                    best_total = new_total
                    improved = True
                    break
        if not improved:
            break
    
    return best_xy, best_r, best_total


def _dual_gap_polish(xy, deadline):
    """
    Dual-gap polish: identify which constraints are active (tight),
    then do a more precise optimization focusing on active constraints.
    For each circle, find its binding constraint (nearest neighbor or wall),
    and do a 1D search along the direction that increases its radius.
    """
    n = len(xy)
    best_xy = xy.copy()
    best_r = _compute_radii_from_centers(xy)
    best_total = float(np.sum(best_r))
    
    for iteration in range(30):
        if time.time() > deadline:
            break
        
        r = _compute_radii_from_centers(best_xy)
        
        # For each circle, find the binding constraint
        wall = np.minimum(np.minimum(best_xy[:, 0], 1.0 - best_xy[:, 0]),
                          np.minimum(best_xy[:, 1], 1.0 - best_xy[:, 1]))
        
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            
            # Find nearest neighbor
            min_dist = np.inf
            nearest = -1
            for j in range(n):
                if i != j:
                    dx = best_xy[i, 0] - best_xy[j, 0]
                    dy = best_xy[i, 1] - best_xy[j, 1]
                    d = np.sqrt(dx * dx + dy * dy)
                    if d < min_dist:
                        min_dist = d
                        nearest = j
            
            # Binding: is it wall or neighbor?
            if wall[i] <= min_dist / 2.0:
                # Wall-bound: move away from wall
                # Determine which wall
                dists = np.array([best_xy[i, 0], 1.0 - best_xy[i, 0],
                                  best_xy[i, 1], 1.0 - best_xy[i, 1]])
                min_idx = int(np.argmin(dists))
                if min_idx == 0:  # left wall
                    direction = np.array([1.0, 0.0])
                elif min_idx == 1:  # right wall
                    direction = np.array([-1.0, 0.0])
                elif min_idx == 2:  # bottom wall
                    direction = np.array([0.0, 1.0])
                else:  # top wall
                    direction = np.array([0.0, -1.0])
            else:
                # Neighbor-bound: move away from nearest neighbor
                if nearest >= 0:
                    dx = best_xy[i, 0] - best_xy[nearest, 0]
                    dy = best_xy[i, 1] - best_xy[nearest, 1]
                    norm = np.sqrt(dx * dx + dy * dy)
                    if norm > 1e-10:
                        direction = np.array([dx / norm, dy / norm])
                    else:
                        continue
                else:
                    continue
            
            # 1D search along direction
            best_t = 0.0
            best_r_i = r[i]
            for t in np.linspace(0.001, 0.05, 20):
                new_xy = best_xy.copy()
                new_xy[i, 0] = np.clip(new_xy[i, 0] + t * direction[0], 0.01, 0.99)
                new_xy[i, 1] = np.clip(new_xy[i, 1] + t * direction[1], 0.01, 0.99)
                new_r = _compute_radii_from_centers(new_xy)
                new_total = float(np.sum(new_r))
                if new_total > best_total + 1e-12:
                    best_total = new_total
                    best_t = t
                    best_xy = new_xy
                    best_r = new_r
                    improved = True
        
        if not improved:
            break
    
    return best_xy, best_r, best_total


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    best_xy = None
    best_r = None
    best_total = 0.0
    best_valid = False

    def _try_config(xy, label=""):
        nonlocal best_xy, best_r, best_total, best_valid
        r = _compute_radii_from_centers(xy)
        total = float(np.sum(r))
        # Validate
        valid = True
        for i in range(len(xy)):
            x, y, ri = xy[i, 0], xy[i, 1], r[i]
            if ri < 0 or x - ri < -1e-9 or 1.0 - x - ri < -1e-9 or y - ri < -1e-9 or 1.0 - y - ri < -1e-9:
                valid = False
                break
        if valid:
            for i in range(len(xy)):
                for j in range(i + 1, len(xy)):
                    dx = xy[i, 0] - xy[j, 0]
                    dy = xy[i, 1] - xy[j, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    if dist < r[i] + r[j] - 1e-9:
                        valid = False
                        break
                if not valid:
                    break
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_xy = xy.copy()
            best_r = r.copy()
            print(f"  {label}: total={total:.6f}")

    # Phase 1: Multiple initializations + center optimization
    inits = []
    # Grid inits
    for side in [5, 6]:
        inits.append(_grid_init(side))
    # Hex init
    inits.append(_hex_init(N))
    # Random inits
    for seed in range(5):
        inits.append(_random_init(N, seed=seed))

    for idx, xy0 in enumerate(inits):
        if time.time() > deadline - 30:
            break
        label = f"init{idx}"
        
        # Optimize centers
        if time.time() < deadline - 25:
            xy_opt = _optimize_centers(xy0.copy(), deadline - 10, max_iter=200)
            _try_config(xy_opt, label + "_opt")
        else:
            _try_config(xy0, label)
        
        # Local refinement
        if best_valid and best_xy is not None and time.time() < deadline - 15:
            xy_ref, r_ref, total_ref = _refine_with_local_search(best_xy.copy(), deadline - 5)
            _try_config(xy_ref, label + "_ref")
        
        # Dual-gap polish
        if best_valid and best_xy is not None and time.time() < deadline - 10:
            xy_dg, r_dg, total_dg = _dual_gap_polish(best_xy.copy(), deadline - 5)
            _try_config(xy_dg, label + "_dual")

    # Phase 2: More random restarts if time remains
    for seed in range(10, 20):
        if time.time() > deadline - 20:
            break
        xy0 = _random_init(N, seed=seed)
        if time.time() < deadline - 15:
            xy_opt = _optimize_centers(xy0, deadline - 10, max_iter=150)
            _try_config(xy_opt, f"rand{seed}_opt")
        else:
            _try_config(xy0, f"rand{seed}")

    # Fallback
    if best_xy is None or not best_valid:
        side = int(np.ceil(np.sqrt(N)))
        xy = _grid_init(side)
        r = _compute_radii_from_centers(xy)
        best_xy = xy
        best_r = r
        best_total = float(np.sum(r))
        best_valid = True

    # Build result
    n = len(best_xy)
    result = np.zeros((n, 3))
    for i in range(n):
        x, y, ri = best_xy[i, 0], best_xy[i, 1], best_r[i]
        ri = min(ri, x, 1.0 - x, y, 1.0 - y)
        ri = max(ri, 0.001)
        result[i, 2] = ri
        result[i, 0] = np.clip(x, ri, 1.0 - ri)
        result[i, 1] = np.clip(y, ri, 1.0 - ri)

    # Final validation
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
