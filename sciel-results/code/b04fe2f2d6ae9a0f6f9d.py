"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _eval_packing(xy, r):
    """
    Evaluate a packing: returns (total_radius, is_valid).
    Valid means all circles inside [0,1]^2 and disjoint interiors.
    """
    n = len(xy)
    # Check walls
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        if ri < -1e-9:
            return 0.0, False
        if x - ri < -1e-9 or 1.0 - x - ri < -1e-9 or y - ri < -1e-9 or 1.0 - y - ri < -1e-9:
            return 0.0, False
    # Check pairwise distances
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-9:
                return 0.0, False
    return float(np.sum(r)), True


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


def _random_init(n, seed=0):
    """Random initialization with small radii."""
    rng = np.random.RandomState(seed)
    xy = rng.uniform(0.1, 0.9, size=(n, 2))
    r = rng.uniform(0.02, 0.05, size=n)
    return xy, r


def _best_fit_local_search(xy, r, deadline, grid_size=80):
    """
    Iteratively update each circle to the best position (maximizing its radius)
    given all other circles fixed. Vectorized over a grid of candidate positions.
    """
    n = len(xy)
    best_total, best_valid = _eval_packing(xy, r)
    best_xy = xy.copy()
    best_r = r.copy()
    if best_valid:
        best_total = float(np.sum(best_r))
    else:
        best_total = 0.0

    # Candidate grid
    gx = np.linspace(0.0, 1.0, grid_size)
    gy = np.linspace(0.0, 1.0, grid_size)
    GX, GY = np.meshgrid(gx, gy, indexing='ij')
    candidates = np.column_stack([GX.ravel(), GY.ravel()])
    M = len(candidates)

    # Precompute candidate wall clearances
    wall_clear = np.minimum(np.minimum(candidates[:, 0], 1.0 - candidates[:, 0]),
                            np.minimum(candidates[:, 1], 1.0 - candidates[:, 1]))

    max_sweeps = 30
    for sweep in range(max_sweeps):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            # Other circles
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            others_xy = xy[mask]
            others_r = r[mask]

            # Compute clearance to others for all candidates
            if len(others_xy) > 0:
                dx = candidates[:, 0:1] - others_xy[:, 0][np.newaxis, :]
                dy = candidates[:, 1:2] - others_xy[:, 1][np.newaxis, :]
                dists = np.sqrt(dx * dx + dy * dy)  # (M, n-1)
                circle_clear = np.min(dists - others_r[np.newaxis, :], axis=1)  # (M,)
                r_candidates = np.minimum(wall_clear, circle_clear)
            else:
                r_candidates = wall_clear.copy()

            # Find best candidate
            idx = int(np.argmax(r_candidates))
            new_x, new_y = candidates[idx, 0], candidates[idx, 1]
            new_r = r_candidates[idx]

            # Only accept if it improves total radius
            old_r = r[i]
            if new_r > old_r + 1e-10:
                xy[i, 0] = new_x
                xy[i, 1] = new_y
                r[i] = new_r
                improved = True

        # After full sweep, evaluate and update best
        total, valid = _eval_packing(xy, r)
        if valid and total > best_total + 1e-10:
            best_total = total
            best_valid = True
            best_xy = xy.copy()
            best_r = r.copy()

        if not improved:
            break

    return best_xy, best_r, best_total, best_valid


def _smooth_objective(v, n):
    """
    Smooth objective: minimize -sum(r) + penalty for violations.
    v is a vector of length 3*n: [x_0..x_{n-1}, y_0..y_{n-1}, r_0..r_{n-1}]
    Uses smooth max via softplus for wall and pairwise violations.
    """
    x = v[0:n]
    y = v[n:2*n]
    r = v[2*n:3*n]

    # Wall violations: r > min(x, 1-x, y, 1-y)
    wall_min = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    wall_viol = r - wall_min  # can be negative
    # Smooth max(0, viol) via softplus: log(1 + exp(beta*viol)) / beta
    beta = 50.0
    wall_pen = np.sum(np.logaddexp(0.0, beta * wall_viol) / beta)

    # Pairwise violations: r_i + r_j > dist_ij
    total_viol = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            viol = r[i] + r[j] - dist
            total_viol += np.logaddexp(0.0, beta * viol) / beta

    wall_pen_sum = np.sum(np.logaddexp(0.0, beta * wall_viol) / beta)

    return -np.sum(r) + 100.0 * (wall_pen_sum + total_viol)


def _smooth_grad(v, n):
    """
    Gradient of the smooth objective using softplus derivatives.
    """
    x = v[0:n].copy()
    y = v[n:2*n].copy()
    r = v[2*n:3*n].copy()

    grad = np.zeros(3 * n)
    beta = 50.0

    # Wall terms: softplus(beta * (r - wall_min)) / beta
    # d/d(r) = 100 * sigmoid(beta * (r - wall_min))
    # d/d(x) = 100 * sigmoid(...) * (-d(wall_min)/dx)
    wall_min = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    wall_viol = r - wall_min
    sig_wall = 1.0 / (1.0 + np.exp(-beta * wall_viol))  # sigmoid(beta * wall_viol)

    for i in range(n):
        s = sig_wall[i]
        if s < 1e-10:
            continue
        # Determine which wall is binding (closest)
        dists = np.array([x[i], 1.0 - x[i], y[i], 1.0 - y[i]])
        min_idx = int(np.argmin(dists))
        # d(wall_viol)/dr_i = 1, d(wall_viol)/d(x_i) = -d(wall_min)/dx_i
        # wall_min = min of 4 terms; derivative w.r.t. the binding term
        if min_idx == 0:
            # wall_min = x[i], d(wall_min)/dx = 1
            grad[2*n + i] += 100.0 * s
            grad[i] -= 100.0 * s
        elif min_idx == 1:
            # wall_min = 1-x[i], d(wall_min)/dx = -1
            grad[2*n + i] += 100.0 * s
            grad[i] += 100.0 * s
        elif min_idx == 2:
            # wall_min = y[i], d(wall_min)/dy = 1
            grad[2*n + i] += 100.0 * s
            grad[n + i] -= 100.0 * s
        else:
            # wall_min = 1-y[i], d(wall_min)/dy = -1
            grad[2*n + i] += 100.0 * s
            grad[n + i] += 100.0 * s

    # Pairwise terms: softplus(beta * (r_i + r_j - dist)) / beta
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx * dx + dy * dy
            if dist_sq < 1e-20:
                continue
            dist = np.sqrt(dist_sq)
            viol = r[i] + r[j] - dist
            s = 1.0 / (1.0 + np.exp(-beta * viol))  # sigmoid(beta * viol)
            if s < 1e-10:
                continue
            factor = 100.0 * s
            # d(viol)/dx_i = dx/dist, d(viol)/dx_j = -dx/dist
            grad[i] += factor * dx / dist
            grad[n + i] += factor * dy / dist
            grad[j] -= factor * dx / dist
            grad[n + j] -= factor * dy / dist
            grad[2*n + i] += factor
            grad[2*n + j] += factor

    # Objective: -sum(r)
    grad[2*n:3*n] -= 1.0

    return grad


def _simulated_annealing(xy, r, deadline, rng):
    """
    Simulated Annealing for circle packing.
    State: [x_0..x_{n-1}, y_0..y_{n-1}, r_0..r_{n-1}]
    Objective: Maximize sum(r) subject to constraints.
    Energy: sum(r) - lambda * violation.
    """
    n = len(xy)
    state = np.concatenate([xy.flatten(), r])
    
    # Initial temperature
    T = 0.01
    T_min = 1e-6
    cooling = 0.995
    # We want to run as many steps as possible within deadline
    # Estimate steps: roughly 10000 steps should be safe if vectorized eval is fast
    # But _eval_packing is python loops. Let's use a faster eval for SA.
    
    def fast_eval(state_vec):
        x = state_vec[0:n]
        y = state_vec[n:2*n]
        r = state_vec[2*n:3*n]
        
        # Wall check
        wall_min = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
        wall_viol = np.sum(np.maximum(0, r - wall_min))
        
        # Pairwise check
        # Vectorized distance matrix is O(N^2) memory and time, acceptable for N=26
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dist_mat = np.sqrt(dx * dx + dy * dy)
        r_sum_mat = r[:, np.newaxis] + r[np.newaxis, :]
        
        # Only upper triangle
        iu = np.triu_indices(n, k=1)
        viol_dist = np.maximum(0, r_sum_mat[iu] - dist_mat[iu])
        pair_viol = np.sum(viol_dist)
        
        total_viol = wall_viol + pair_viol
        energy = np.sum(r) - 10.0 * total_viol
        return energy, total_viol

    current_energy, _ = fast_eval(state)
    best_state = state.copy()
    best_energy = current_energy
    best_valid = _eval_packing(state[0:n].reshape(-1,2), state[2*n:3*n])[1]
    if best_valid:
         best_val = np.sum(state[2*n:3*n])
    else:
         best_val = -np.inf

    # SA Loop
    while time.time() < deadline and T > T_min:
        # Generate neighbor
        # Move a random circle: x, y, r
        i = rng.randint(n)
        # Gaussian perturbation
        dx = rng.normal(0, 0.01)
        dy = rng.normal(0, 0.01)
        dr = rng.normal(0, 0.005)
        
        new_state = state.copy()
        new_state[i] = np.clip(new_state[i] + dx, 0.01, 0.99)
        new_state[n+i] = np.clip(new_state[n+i] + dy, 0.01, 0.99)
        new_state[2*n+i] = np.clip(new_state[2*n+i] + dr, 0.001, 0.4)
        
        new_energy, new_viol = fast_eval(new_state)
        delta_e = new_energy - current_energy
        
        # Acceptance probability
        if delta_e > 0 or rng.rand() < np.exp(delta_e / T):
            state = new_state
            current_energy = new_energy
            
            # Check if valid and better
            if new_viol == 0:
                val = np.sum(state[2*n:3*n])
                if val > best_val:
                    best_val = val
                    best_state = state.copy()
                    best_valid = True
        
        T *= cooling

    # Return best found
    best_xy = best_state[0:n].reshape(-1, 2)
    best_r = best_state[2*n:3*n]
    return best_xy, best_r


def construct():
    start_time = time.time()
    # Safety margin: 150s limit. Leave 10s for cleanup/fallback.
    deadline = start_time + 135.0 

    best_xy = None
    best_r = None
    best_total = 0.0
    best_valid = False

    rng = np.random.RandomState(42)

    # Phase 1: Grid initialization + local search
    # This provides a strong starting point
    for side in [5, 6, 7]:
        if time.time() > deadline - 60: # Leave time for SA
            break
        xy = _grid_init(side)
        r = xy[:, 2].copy()
        xy = xy[:, :2].copy()

        out_xy, out_r, total, valid = _best_fit_local_search(xy, r, deadline - 60)
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_xy = out_xy
            best_r = out_r

    # Phase 2: Simulated Annealing
    # Use the best result from Phase 1 as the starting point for SA
    if best_valid and best_xy is not None:
        if time.time() < deadline - 10:
            # Run SA for the remaining time
            sa_xy, sa_r = _simulated_annealing(best_xy.copy(), best_r.copy(), deadline - 5, rng)
            
            # Evaluate SA result
            total, valid = _eval_packing(sa_xy, sa_r)
            if valid and total > best_total:
                best_total = total
                best_xy = sa_xy
                best_r = sa_r

    # Fallback if SA failed or didn't improve
    if best_xy is None or not best_valid:
        side = int(np.ceil(np.sqrt(N)))
        xy = _grid_init(side)
        r = xy[:, 2].copy()
        xy = xy[:, :2].copy()
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
        # Fallback to a simple valid grid if everything failed
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
