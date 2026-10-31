"""Circle packing in perimeter-4 rectangle: Simulated annealing + Nelder-Mead simplex optimization."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


def _sa_objective(z, penalty_weight=10000.0):
    """Objective for simulated annealing: negative sum of radii plus penalties."""
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w

    if w < 0.2 or h < 0.2 or np.any(r < 0):
        return 1e6

    obj = -np.sum(r)

    # Vectorized pairwise overlap penalty
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]

    iu = np.triu_indices(N, k=1)
    d_sq_pairs = d_sq[iu]
    r_sum_pairs = r_sum[iu]

    d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
    overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
    overlap_penalty = penalty_weight * np.sum(overlap * overlap)

    obj += overlap_penalty

    # Vectorized boundary penalty
    left_viol = np.maximum(r - x, 0.0)
    right_viol = np.maximum(x + r - w, 0.0)
    bottom_viol = np.maximum(r - y, 0.0)
    top_viol = np.maximum(y + r - h, 0.0)

    boundary_penalty = penalty_weight * (
        np.sum(left_viol * left_viol) +
        np.sum(right_viol * right_viol) +
        np.sum(bottom_viol * bottom_viol) +
        np.sum(top_viol * top_viol)
    )

    obj += boundary_penalty

    # Keep aspect ratio in reasonable range
    if w < 0.4:
        obj += penalty_weight * (0.4 - w) ** 2
    elif w > 1.6:
        obj += penalty_weight * (w - 1.6) ** 2

    return obj


def _is_valid(cx, cy, r, W, H, tol=1e-8):
    """Check if packing is valid."""
    if W <= 0.05 or H <= 0.05:
        return False
    if np.any(cx - r < -tol) or np.any(cy - r < -tol):
        return False
    if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - tol:
                return False
    return True


def _nelder_mead_objective(z, penalty_weight=50000.0):
    """Objective for Nelder-Mead: sum of radii minus penalties (maximize)."""
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w

    if w < 0.2 or h < 0.2 or np.any(r < 0):
        return -1e6

    obj = np.sum(r)

    # Pairwise overlap penalty
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]

    iu = np.triu_indices(N, k=1)
    d_sq_pairs = d_sq[iu]
    r_sum_pairs = r_sum[iu]

    d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
    overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
    overlap_penalty = penalty_weight * np.sum(overlap * overlap)

    obj -= overlap_penalty

    # Boundary penalty
    left_viol = np.maximum(r - x, 0.0)
    right_viol = np.maximum(x + r - w, 0.0)
    bottom_viol = np.maximum(r - y, 0.0)
    top_viol = np.maximum(y + r - h, 0.0)

    boundary_penalty = penalty_weight * (
        np.sum(left_viol * left_viol) +
        np.sum(right_viol * right_viol) +
        np.sum(bottom_viol * bottom_viol) +
        np.sum(top_viol * top_viol)
    )

    obj -= boundary_penalty

    # Aspect ratio penalty
    if w < 0.4:
        obj -= penalty_weight * (0.4 - w) ** 2
    elif w > 1.6:
        obj -= penalty_weight * (w - 1.6) ** 2

    return obj


def _simulated_annealing(z0, time_budget, rng, n_iter=5000):
    """Simulated annealing to optimize circle packing."""
    z = z0.copy().astype(float)
    best_z = z.copy()
    best_obj = _sa_objective(z, penalty_weight=10000.0)
    
    T = 0.1  # Initial temperature
    T_min = 1e-6
    cooling_rate = 0.9995
    
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w
    
    for it in range(n_iter):
        if time.time() > time_budget:
            break
        
        # Generate perturbation
        z_new = z.copy()
        
        # Perturb positions
        for i in range(N):
            if rng.random() < 0.3:
                z_new[3*i] += rng.normal(0, 0.02)
                z_new[3*i+1] += rng.normal(0, 0.02)
        
        # Perturb radii
        if rng.random() < 0.2:
            i = rng.integers(N)
            z_new[3*i+2] *= rng.uniform(0.95, 1.05)
        
        # Perturb width
        if rng.random() < 0.1:
            z_new[-1] += rng.normal(0, 0.01)
        
        # Clamp values
        w_new = z_new[-1]
        h_new = 2.0 - w_new
        for i in range(N):
            r_i = max(0.01, z_new[3*i+2])
            z_new[3*i+2] = r_i
            z_new[3*i] = np.clip(z_new[3*i], r_i, w_new - r_i)
            z_new[3*i+1] = np.clip(z_new[3*i+1], r_i, h_new - r_i)
        
        obj_new = _sa_objective(z_new, penalty_weight=10000.0)
        delta = obj_new - best_obj
        
        # Accept if better or with probability exp(-delta/T)
        if delta < 0 or rng.random() < np.exp(-delta / max(T, T_min)):
            z = z_new
            if obj_new < best_obj:
                best_obj = obj_new
                best_z = z_new.copy()
        
        # Cool
        T = max(T * cooling_rate, T_min)
    
    return best_z


def _nelder_mead_polish(z, time_budget):
    """Polish using Nelder-Mead simplex method."""
    z = z.copy().astype(float)
    
    # Create initial simplex
    n_params = 3 * N + 1
    simplex = [z.copy()]
    for i in range(n_params):
        z_perturbed = z.copy()
        z_perturbed[i] += 0.01 if z_perturbed[i] < 1.0 else -0.01
        simplex.append(z_perturbed)
    
    # Evaluate objective
    obj_values = [_nelder_mead_objective(s, penalty_weight=50000.0) for s in simplex]
    
    # Nelder-Mead parameters
    alpha = 1.0  # Reflection
    gamma = 2.0  # Expansion
    rho = 0.5   # Contraction
    sigma = 0.5  # Shrink
    
    max_iter = 1000
    for it in range(max_iter):
        if time.time() > time_budget:
            break
        
        # Sort by objective (we want to maximize, so sort descending)
        order = np.argsort(obj_values)[::-1]
        simplex = [simplex[i] for i in order]
        obj_values = [obj_values[i] for i in order]
        
        # Check convergence
        if abs(obj_values[0] - obj_values[-1]) < 1e-8:
            break
        
        # Centroid of all but worst point
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflect worst point
        worst = simplex[-1]
        x_r = centroid + alpha * (centroid - worst)
        obj_r = _nelder_mead_objective(x_r, penalty_weight=50000.0)
        
        if obj_r > obj_values[0]:
            # Expand
            x_e = centroid + gamma * (x_r - centroid)
            obj_e = _nelder_mead_objective(x_e, penalty_weight=50000.0)
            if obj_e > obj_r:
                simplex[-1] = x_e
                obj_values[-1] = obj_e
            else:
                simplex[-1] = x_r
                obj_values[-1] = obj_r
        elif obj_r > obj_values[-2]:
            simplex[-1] = x_r
            obj_values[-1] = obj_r
        else:
            # Contract
            x_c = centroid + rho * (worst - centroid)
            obj_c = _nelder_mead_objective(x_c, penalty_weight=50000.0)
            if obj_c > obj_values[-1]:
                simplex[-1] = x_c
                obj_values[-1] = obj_c
            else:
                # Shrink
                for i in range(1, len(simplex)):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    obj_values[i] = _nelder_mead_objective(simplex[i], penalty_weight=50000.0)
    
    return simplex[0]


def _get_initial_guesses():
    """Generate diverse initial guesses for simulated annealing."""
    guesses = []
    rng = np.random.default_rng(42)
    
    # Grid layouts
    for w0 in [0.6, 0.8, 1.0, 1.2, 1.4]:
        h0 = 2.0 - w0
        if h0 < 0.3:
            continue
        for cols in [3, 4, 5]:
            rows = int(np.ceil(N / cols))
            if rows * cols < N:
                continue
            z = np.zeros(3 * N + 1)
            for i in range(N):
                row, col = divmod(i, cols)
                if row >= rows:
                    break
                cx = (col + 0.5) / cols * w0
                cy = (row + 0.5) / rows * h0
                r0 = min(w0 / cols, h0 / rows) * 0.40
                z[3 * i] = cx
                z[3 * i + 1] = cy
                z[3 * i + 2] = r0
            z[-1] = w0
            guesses.append(z)
    
    # Random starts
    for trial in range(30):
        w0 = rng.uniform(0.5, 1.5)
        h0 = 2.0 - w0
        z = np.zeros(3 * N + 1)
        base_r = rng.uniform(0.05, 0.09)
        for i in range(N):
            r0 = base_r * rng.uniform(0.7, 1.3)
            cx = rng.uniform(r0, w0 - r0)
            cy = rng.uniform(r0, h0 - r0)
            z[3 * i] = cx
            z[3 * i + 1] = cy
            z[3 * i + 2] = r0
        z[-1] = w0
        guesses.append(z)
    
    return guesses


def _expand_radii(x, y, r, w, h, time_budget):
    """Expand radii to maximum valid values."""
    x = x.copy()
    y = y.copy()
    r = r.copy()
    
    for _ in range(20):
        if time.time() > time_budget:
            break
        improved = False
        for i in range(N):
            r_wall = min(x[i], w - x[i], y[i], h - y[i])
            r_max = r_wall
            for j in range(N):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                r_allowed = d - r[j]
                if r_allowed < r_max:
                    r_max = r_allowed
            r_max = max(0.0, r_max)
            if r_max > r[i] + 1e-8:
                r[i] = r_max
                improved = True
        if not improved:
            break
    
    return x, y, r


def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    rng = np.random.default_rng(123)
    
    best_value = 0.0
    best_result = None
    
    def check_validity(z, tol=1e-9):
        x = z[0::3][:N]
        y = z[1::3][:N]
        r = z[2::3][:N]
        w = z[-1]
        h = 2.0 - w
        
        if w <= 0.05 or h <= 0.05:
            return False, 0.0
        
        for i in range(N):
            if r[i] <= 0:
                return False, 0.0
            if x[i] < r[i] - tol or x[i] > w - r[i] + tol:
                return False, 0.0
            if y[i] < r[i] - tol or y[i] > h - r[i] + tol:
                return False, 0.0
        
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d2 = dx * dx + dy * dy
                r_sum = r[i] + r[j]
                if d2 < r_sum ** 2 - tol:
                    return False, 0.0
        
        val = float(np.sum(r))
        return True, val
    
    guesses = _get_initial_guesses()
    
    for i, z0 in enumerate(guesses):
        if time.time() > deadline - 30:
            break
        
        # Simulated annealing
        z_sa = _simulated_annealing(z0, deadline - 20, rng, n_iter=3000)
        
        # Nelder-Mead polish
        z_nm = _nelder_mead_polish(z_sa, deadline - 10)
        
        # Expand radii
        x = z_nm[0::3][:N]
        y = z_nm[1::3][:N]
        r = z_nm[2::3][:N]
        w = z_nm[-1]
        h = 2.0 - w
        
        x, y, r = _expand_radii(x, y, r, w, h, deadline - 5)
        
        # Reconstruct
        z_new = np.empty(3 * N + 1)
        z_new[0::3][:N] = x
        z_new[1::3][:N] = y
        z_new[2::3][:N] = r
        z_new[-1] = w
        
        valid, val = check_validity(z_new)
        if valid and val > best_value:
            best_value = val
            best_result = z_new.copy()
    
    # Final polish on best result
    if best_result is not None and time.time() < deadline - 10:
        z_nm = _nelder_mead_polish(best_result, deadline - 5)
        x = z_nm[0::3][:N]
        y = z_nm[1::3][:N]
        r = z_nm[2::3][:N]
        w = z_nm[-1]
        h = 2.0 - w
        x, y, r = _expand_radii(x, y, r, w, h, deadline - 2)
        z_new = np.empty(3 * N + 1)
        z_new[0::3][:N] = x
        z_new[1::3][:N] = y
        z_new[2::3][:N] = r
        z_new[-1] = w
        valid, val = check_validity(z_new)
        if valid and val > best_value:
            best_value = val
            best_result = z_new
    
    if best_result is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    # Format output
    x = best_result[0::3][:N]
    y = best_result[1::3][:N]
    r = best_result[2::3][:N]
    w = best_result[-1]
    h = 2.0 - w
    
    # Safety shrink
    for _ in range(20):
        r_trial = r * (1 - 1e-10)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break
    
    return np.column_stack((x, y, r))
