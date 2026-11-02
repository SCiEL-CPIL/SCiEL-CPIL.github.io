<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 21

def _feasible(circles, perimeter_tol=1e-8, gap_tol=-1e-8):
    """Check if circles are valid: disjoint interiors, bounding rect perimeter <= 4."""
    if circles.shape[0] != N:
        return False
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    if np.any(rs < 0):
        return False
    # bounding rectangle
    min_x = np.min(xs - rs)
    max_x = np.max(xs + rs)
    min_y = np.min(ys - rs)
    max_y = np.max(ys + rs)
    w = max_x - min_x
    h = max_y - min_y
    if 2 * (w + h) > 4 + perimeter_tol:
        return False
    # disjoint interiors: distance between centers >= sum of radii (with small tolerance)
    dists = cdist(circles[:, :2], circles[:, :2])
    min_separation = np.min(dists[np.triu_indices(N, k=1)])
    # Actually need pairwise: for all i<j, dist >= r_i + r_j
    for i in range(N):
        for j in range(i + 1, N):
            d = np.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)
            if d < rs[i] + rs[j] + gap_tol:
                return False
    return True


def _penalty_objective(params, lam=1e4, beta=100):
    """
    Objective: minimize negative sum of radii + penalties for constraint violations.
    params: flattened array of shape (3N,) -> [x0,y0,r0, x1,y1,r1, ...]
    We use a smooth penalty:
      - Perimeter constraint: 2*(w+h) <= 4 -> penalty on max(0, 2*(w+h)-4)
      - Separation: for each pair i<j, dist >= r_i+r_j -> penalty on max(0, r_i+r_j-dist)
      - Radii positive: penalty on max(0, -r_i)
    Use soft versions with log-sum-exp for smoothness where appropriate, but for robustness
    we use a simple squared penalty on violations.
    """
    n = N
    x = params.reshape(n, 3)
    xs, ys, rs = x[:, 0], x[:, 1], x[:, 2]
    
    # Radii positivity
    neg_r = np.maximum(0, -rs)
    penalty_r = np.sum(neg_r ** 2)
    
    # Bounding box
    min_x = np.min(xs - rs)
    max_x = np.max(xs + rs)
    min_y = np.min(ys - rs)
    max_y = np.max(ys + rs)
    w = max_x - min_x
    h = max_y - min_y
    perim = 2 * (w + h)
    perim_viol = max(0, perim - 4)
    penalty_perim = perim_viol ** 2
    
    # Separation constraints
    penalty_sep = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)
            req = rs[i] + rs[j]
            viol = max(0, req - d)
            penalty_sep += viol ** 2
    
    total_penalty = lam * (penalty_r + penalty_perim + penalty_sep)
    return -np.sum(rs) + total_penalty


def _construct_single(x0, y0, r0, max_iter=500, tol=1e-10):
    """Run one optimization from a given initial guess."""
    params0 = np.array([x0, y0, r0])
    
    def obj(p):
        return _penalty_objective(p, lam=1e4, beta=100)
    
    res = minimize(obj, params0, method='BFGS', options={'maxiter': max_iter, 'gtol': tol})
    best_params = res.x
    best_val = -np.sum(best_params.reshape(N, 3)[:, 2])  # we want to maximize sum of radii
    
    # Check feasibility and refine
    circles = best_params.reshape(N, 3)
    if _feasible(circles):
        return circles, np.sum(circles[:, 2])
    
    # Try with tighter constraints via higher lambda
    for lam_try in [1e5, 1e6]:
        def obj2(p):
            return _penalty_objective(p, lam=lam_try, beta=100)
        res2 = minimize(obj2, best_params, method='BFGS', options={'maxiter': max_iter, 'gtol': tol})
        circles2 = res2.x.reshape(N, 3)
        if _feasible(circles2):
            s = np.sum(circles2[:, 2])
            if s > np.sum(circles[:, 2]):
                circles = circles2
                best_val = s
            return circles, best_val
    
    return circles, best_val


def _generate_initial_guesses(n_guesses=12, seed=42):
    """Generate diverse initial guesses."""
    rng = np.random.RandomState(seed)
    guesses = []
    
    # Grid-based: 5x5 grid in a 1x1 square
    side = 5
    for i in range(side):
        for j in range(side):
            if len(guesses) >= n_guesses:
                break
            cx = (i + 0.5) / side
            cy = (j + 0.5) / side
            r = 0.5 / side * 0.95
            guesses.append((cx, cy, r))
        if len(guesses) >= n_guesses:
            break
    
    # Add some random perturbations
    while len(guesses) < n_guesses:
        # Random rectangle aspect ratio
        aspect = rng.uniform(0.5, 2.0)
        w = 2 / (1 + aspect)  # perimeter 4: 2(w+h)=4 -> w+h=2
        h = 2 - w
        # Place circles in a grid within this rectangle
        nx = max(1, int(np.round(np.sqrt(N * w / h))))
        ny = max(1, int(np.ceil(N / nx)))
        for i in range(nx):
            for j in range(ny):
                if len(guesses) >= n_guesses:
                    break
                cx = (i + 0.5) * w / nx
                cy = (j + 0.5) * h / ny
                r = min(w / nx, h / ny) * 0.5 * 0.95
                guesses.append((cx, cy, r))
            if len(guesses) >= n_guesses:
                break
    
    return guesses[:n_guesses]


def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    # Generate initial guesses
    guesses = _generate_initial_guesses(n_guesses=12, seed=42)
    
    best_circles = None
    best_sum = 0.0
    
    for idx, (x0, y0, r0) in enumerate(guesses):
        if time.time() > deadline:
            break
        
        # Build initial params: replicate the guess for all N circles with small offsets
        params0 = np.zeros(3 * N)
        for i in range(N):
            # Slight perturbation to avoid symmetry
            dx = (i % 5 - 2) * 0.01
            dy = (i // 5 - 2) * 0.01
            params0[3*i] = x0 + dx
            params0[3*i+1] = y0 + dy
            params0[3*i+2] = r0 * (1 + 0.05 * np.sin(i))
        
        # Run optimization
        def obj(p):
            return _penalty_objective(p, lam=1e4, beta=100)
        
        try:
            res = minimize(obj, params0, method='BFGS', options={'maxiter': 300, 'gtol': 1e-8})
            circles = res.x.reshape(N, 3)
            
            # Polish with higher lambda if feasible
            if _feasible(circles):
                for lam_try in [1e5, 1e6, 1e7]:
                    if time.time() > deadline:
                        break
                    def obj_polish(p, lam=lam_try):
                        return _penalty_objective(p, lam=lam, beta=100)
                    res_p = minimize(obj_polish, res.x, method='BFGS', options={'maxiter': 200, 'gtol': 1e-10})
                    circles_p = res_p.x.reshape(N, 3)
                    if _feasible(circles_p):
                        s = np.sum(circles_p[:, 2])
                        if s > np.sum(circles[:, 2]):
                            circles = circles_p
                
                s = np.sum(circles[:, 2])
                if s > best_sum:
                    best_sum = s
                    best_circles = circles.copy()
        except Exception:
            continue
    
    if best_circles is not None:
        return best_circles
    
    # Fallback: return the seed solution
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    return np.array(cells[:N])
# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize, linprog
from scipy.spatial.distance import cdist

N = 21

# EVOLVE-BLOCK-START

def _solve_radii_lp(centers, w, h, max_iter=500):
    """
    Given fixed centers in a w x h rectangle (perimeter 2(w+h)=4 so w+h=2),
    solve LP: max sum(r_i) s.t.
      r_i >= 0
      r_i + r_j <= dist_ij for all i<j
      r_i <= min(x_i, w-x_i, y_i, h-y_i)  (circle inside rectangle)
    Returns optimal radii or None.
    """
    n = N
    # Variables: r_0, ..., r_{n-1}
    # Objective: maximize sum(r_i) -> minimize -sum(r_i)
    c_obj = -np.ones(n)
    
    # Inequality constraints: A_ub @ r <= b_ub
    A_rows = []
    b_rows = []
    
    # Pairwise separation: r_i + r_j <= dist_ij
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt((centers[i, 0] - centers[j, 0])**2 + (centers[i, 1] - centers[j, 1])**2)
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_rows.append(row)
            b_rows.append(d)
    
    # Boundary: r_i <= min(x_i, w-x_i, y_i, h-y_i)
    for i in range(n):
        max_r = min(centers[i, 0], w - centers[i, 0], centers[i, 1], h - centers[i, 1])
        max_r = max(max_r, 0.0)  # ensure non-negative bound
        row = np.zeros(n)
        row[i] = 1.0
        A_rows.append(row)
        b_rows.append(max_r)
    
    if len(A_rows) == 0:
        return None
    
    A_ub = np.array(A_rows)
    b_ub = np.array(b_rows)
    
    # Bounds: 0 <= r_i <= inf (the upper bound is handled by A_ub)
    bounds = [(0, None)] * n
    
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.status == 0:
            radii = np.maximum(0, res.x)
            return radii
        else:
            return None
    except Exception:
        return None


def _force_directed_relax(centers, w, h, n_iter=2000, lr=0.005, deadline=None):
    """
    Force-directed relaxation: push centers apart (repulsion) and toward center (attraction),
    keeping them inside the rectangle. This finds good spatial configurations.
    """
    centers = centers.copy()
    n = N
    
    for it in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute pairwise forces
        forces = np.zeros_like(centers)
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                d = np.sqrt(dx*dx + dy*dy)
                if d < 1e-10:
                    # Random nudge
                    dx = (np.random.random() - 0.5) * 0.01
                    dy = (np.random.random() - 0.5) * 0.01
                    d = np.sqrt(dx*dx + dy*dy)
                
                # Repulsion: stronger when closer
                repulse = 1.0 / (d * d + 0.01)
                fx = repulse * dx / d
                fy = repulse * dy / d
                forces[i, 0] += fx
                forces[i, 1] += fy
                forces[j, 0] -= fx
                forces[j, 1] -= fy
        
        # Attraction toward center of rectangle
        cx, cy = w / 2, h / 2
        for i in range(n):
            forces[i, 0] += 0.01 * (cx - centers[i, 0])
            forces[i, 1] += 0.01 * (cy - centers[i, 1])
        
        # Update positions
        for i in range(n):
            centers[i, 0] += lr * forces[i, 0]
            centers[i, 1] += lr * forces[i, 1]
            # Clamp to rectangle (with small margin)
            centers[i, 0] = np.clip(centers[i, 0], 0.01, w - 0.01)
            centers[i, 1] = np.clip(centers[i, 1], 0.01, h - 0.01)
    
    return centers


def _gradient_center_opt(centers, w, h, n_iter=500, lr=0.01, deadline=None):
    """
    Gradient descent on centers: minimize a smooth proxy for the LP radius sum.
    The idea: for each pair, the contribution to radius sum is limited by dist_ij/2 (if both radii equal).
    We maximize sum over pairs of min(dist_ij, some_cap) as a smooth proxy.
    Actually, let's use: for each circle, its max radius is min over all j of dist_ij/2 and boundary.
    Sum of these "max possible radii" is an upper bound. We maximize this.
    """
    centers = centers.copy()
    n = N
    
    def _proxy_objective(c):
        """Negative of sum of max possible radii (to minimize)."""
        total = 0.0
        for i in range(n):
            # Max radius for circle i given centers
            max_r = min(c[i, 0], w - c[i, 0], c[i, 1], h - c[i, 1])
            for j in range(n):
                if i == j:
                    continue
                d = np.sqrt((c[i, 0] - c[j, 0])**2 + (c[i, 1] - c[j, 1])**2)
                max_r = min(max_r, d / 2.0)
            total += max_r
        return -total
    
    # Use simple gradient descent with numerical gradient
    for it in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Numerical gradient
        grad = np.zeros_like(centers)
        eps = 1e-5
        for i in range(n):
            for d in range(2):
                c_plus = centers.copy()
                c_plus[i, d] += eps
                f_plus = _proxy_objective(c_plus)
                c_minus = centers.copy()
                c_minus[i, d] -= eps
                f_minus = _proxy_objective(c_minus)
                grad[i, d] = (f_plus - f_minus) / (2 * eps)
        
        # Update
        centers -= lr * grad
        # Clamp
        for i in range(n):
            centers[i, 0] = np.clip(centers[i, 0], 0.01, w - 0.01)
            centers[i, 1] = np.clip(centers[i, 1], 0.01, h - 0.01)
        
        # Decay learning rate
        lr *= 0.999
    
    return centers


def _feasible(circles, perimeter_tol=1e-8, gap_tol=-1e-8):
    """Check if circles are valid."""
    if circles.shape[0] != N:
        return False
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    if np.any(rs < 0):
        return False
    min_x = np.min(xs - rs)
    max_x = np.max(xs + rs)
    min_y = np.min(ys - rs)
    max_y = np.max(ys + rs)
    w = max_x - min_x
    h = max_y - min_y
    if 2 * (w + h) > 4 + perimeter_tol:
        return False
    for i in range(N):
        for j in range(i + 1, N):
            d = np.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)
            if d < rs[i] + rs[j] + gap_tol:
                return False
    return True


def _make_grid_initial(w, h, nx, ny, rng):
    """Create initial center positions on a grid."""
    centers = []
    for i in range(nx):
        for j in range(ny):
            if len(centers) >= N:
                break
            cx = (i + 0.5) * w / nx + rng.uniform(-0.02, 0.02)
            cy = (j + 0.5) * h / ny + rng.uniform(-0.02, 0.02)
            cx = np.clip(cx, 0.05, w - 0.05)
            cy = np.clip(cy, 0.05, h - 0.05)
            centers.append([cx, cy])
    # Fill remaining if needed
    while len(centers) < N:
        cx = rng.uniform(0.1, w - 0.1)
        cy = rng.uniform(0.1, h - 0.1)
        centers.append([cx, cy])
    return np.array(centers[:N])


def construct():
    start_time = time.time()
    deadline = start_time + 135  # safety margin (150s limit)
    
    best_circles = None
    best_sum = 0.0
    
    rng = np.random.RandomState(12345)
    
    # Try multiple aspect ratios and grid configurations
    aspect_ratios = [1.0, 1.2, 1.5, 2.0, 0.8, 1.3, 1.7, 2.5, 0.7, 1.1]
    grid_configs = [(4, 6), (5, 5), (3, 7), (6, 4), (4, 5), (5, 4), (3, 8), (8, 3), (7, 3), (2, 11)]
    
    attempts = 0
    max_attempts = 20
    
    for ar in aspect_ratios:
        if time.time() > deadline or attempts >= max_attempts:
            break
        
        # Perimeter 4: 2(w+h) = 4 -> w+h = 2
        # w/h = ar -> w = ar*h -> ar*h + h = 2 -> h = 2/(1+ar), w = 2*ar/(1+ar)
        w = 2.0 * ar / (1.0 + ar)
        h = 2.0 / (1.0 + ar)
        
        # Choose grid config
        if attempts < len(grid_configs):
            nx, ny = grid_configs[attempts]
        else:
            nx = max(1, int(np.round(np.sqrt(N * w / h))))
            ny = max(1, int(np.ceil(N / nx)))
        
        # Multiple restarts per configuration
        for restart in range(3):
            if time.time() > deadline or attempts >= max_attempts:
                break
            
            # Initial centers
            centers = _make_grid_initial(w, h, nx, ny, rng)
            
            # Phase 1: Force-directed relaxation
            centers = _force_directed_relax(centers, w, h, n_iter=1500, lr=0.003, deadline=deadline)
            
            # Phase 2: Gradient-based center optimization
            centers = _gradient_center_opt(centers, w, h, n_iter=100, lr=0.005, deadline=deadline)
            
            # Phase 3: Another force-directed pass
            centers = _force_directed_relax(centers, w, h, n_iter=1000, lr=0.002, deadline=deadline)
            
            # Phase 4: Solve LP for radii
            radii = _solve_radii_lp(centers, w, h)
            if radii is None:
                continue
            
            # Build circles array
            circles = np.column_stack([centers, radii])
            
            # Verify feasibility
            if _feasible(circles):
                s = np.sum(radii)
                if s > best_sum:
                    best_sum = s
                    best_circles = circles.copy()
            else:
                # Try to fix: slightly shrink radii
                radii_fixed = radii * 0.99
                circles = np.column_stack([centers, radii_fixed])
                if _feasible(circles):
                    s = np.sum(radii_fixed)
                    if s > best_sum:
                        best_sum = s
                        best_circles = circles.copy()
            
            attempts += 1
    
    # If we found something, also try to improve the best with more iterations
    if best_circles is not None and time.time() < deadline - 10:
        # Extract centers and radii
        best_centers = best_circles[:, :2].copy()
        # Determine w, h from bounding box
        min_x = np.min(best_centers[:, 0])
        max_x = np.max(best_centers[:, 0])
        min_y = np.min(best_centers[:, 1])
        max_y = np.max(best_centers[:, 1])
        # Use the rectangle that contains all centers with some margin
        # Actually, use the aspect ratio that was used
        # For now, just do more force-directed relaxation
        # Estimate w, h from the best configuration
        w_est = max_x - min_x + 0.1
        h_est = max_y - min_y + 0.1
        # Adjust to maintain perimeter constraint
        scale = 2.0 / (w_est + h_est)
        w_est *= scale
        h_est *= scale
        
        # Re-center
        cx = (min_x + max_x) / 2
        cy = (min_y + max_y) / 2
        best_centers -= np.array([cx, cy])
        best_centers += np.array([w_est/2, h_est/2])
        
        # More relaxation
        best_centers = _force_directed_relax(best_centers, w_est, h_est, n_iter=2000, lr=0.001, deadline=deadline)
        radii = _solve_radii_lp(best_centers, w_est, h_est)
        if radii is not None:
            circles = np.column_stack([best_centers, radii])
            if _feasible(circles):
                s = np.sum(radii)
                if s > best_sum:
                    best_sum = s
                    best_circles = circles.copy()
    
    if best_circles is not None:
        return best_circles
    
    # Fallback: simple grid
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= N:
                break
            cells.append([(i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 0.01])
    return np.array(cells[:N])

# EVOLVE-BLOCK-END
>>>>>>> REPLACE
