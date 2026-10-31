"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize, fsolve

N = 26

# EVOLVE-BLOCK-START

def _objective(x):
    radii = x[2*N:]
    return -np.sum(radii)

def _boundary_constraints(x):
    pos = x[:2*N].reshape(N, 2)
    radii = x[2*N:]
    cons = np.concatenate([
        pos[:, 0] - radii,
        1.0 - pos[:, 0] - radii,
        pos[:, 1] - radii,
        1.0 - pos[:, 1] - radii
    ])
    return cons

def _overlap_constraints(x):
    pos = x[:2*N].reshape(N, 2)
    radii = x[2*N:]
    cons = []
    for i in range(N):
        for j in range(i + 1, N):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist_sq = dx * dx + dy * dy
            sum_r = radii[i] + radii[j]
            cons.append(dist_sq - sum_r * sum_r)
    return np.array(cons)

def _all_constraints(x):
    b_cons = _boundary_constraints(x)
    o_cons = _overlap_constraints(x)
    return np.concatenate([b_cons, o_cons])

def _is_valid(pos, radii, tol=1e-6):
    n = len(radii)
    if np.any(pos[:, 0] - radii < -tol) or np.any(pos[:, 0] + radii > 1.0 + tol):
        return False
    if np.any(pos[:, 1] - radii < -tol) or np.any(pos[:, 1] + radii > 1.0 + tol):
        return False
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    if np.any(dist_sq[mask] < (min_dist[mask] - tol)**2):
        return False
    return True

def _post_process(pos, radii, tol=1e-7):
    pos = pos.copy()
    radii = radii.copy()
    for _ in range(100):
        valid = True
        for i in range(N):
            r = radii[i]
            x, y = pos[i]
            max_r = min(x, 1.0-x, y, 1.0-y)
            if r > max_r + tol:
                radii[i] = max(max_r * 0.99999, 1e-5)
                valid = False
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r + tol:
                    if sum_r > 1e-9:
                        k = (dist - tol) / (sum_r + 1e-12)
                    else:
                        k = 0.99
                    if k < 1.0:
                        radii[i] = max(radii[i] * k, 1e-5)
                        radii[j] = max(radii[j] * k, 1e-5)
                        valid = False
        if valid:
            break
    return pos, radii

def _smooth_optimize(pos, radii, beta_start=20.0, beta_end=100.0, n_iters=3):
    """
    Smooth penalty optimization with log-sum-exp based soft constraints.
    """
    pos = pos.copy()
    radii = radii.copy()
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
    for i in range(N):
        bounds.append((1e-5, 0.5))
    
    for it in range(n_iters):
        beta = beta_start * (beta_end / beta_start) ** (it / max(n_iters - 1, 1))
        
        def obj(x):
            pos_x = x[:2*N].reshape(N, 2)
            r_x = x[2*N:]
            b_cons = np.concatenate([
                pos_x[:, 0] - r_x,
                1.0 - pos_x[:, 0] - r_x,
                pos_x[:, 1] - r_x,
                1.0 - pos_x[:, 1] - r_x
            ])
            o_cons = []
            for i in range(N):
                for j in range(i + 1, N):
                    dx = pos_x[i, 0] - pos_x[j, 0]
                    dy = pos_x[i, 1] - pos_x[j, 1]
                    dist_sq = dx * dx + dy * dy
                    sum_r = r_x[i] + r_x[j]
                    o_cons.append(dist_sq - sum_r * sum_r)
            all_cons = np.concatenate([b_cons, np.array(o_cons)])
            z = -beta * all_cons
            penalty = np.sum(np.where(z > 0, z + np.log1p(np.exp(-z)), np.log1p(np.exp(z)))) / beta
            return -np.sum(r_x) + penalty
        
        x0 = np.concatenate([pos.flatten(), radii])
        try:
            result = minimize(obj, x0, method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-8})
            pos_new = result.x[:2*N].reshape(N, 2)
            radii_new = result.x[2*N:]
            pos_pp, radii_pp = _post_process(pos_new, radii_new, tol=1e-8)
            if _is_valid(pos_pp, radii_pp, tol=1e-7):
                pos, radii = pos_pp, radii_pp
        except Exception:
            pass
    return pos, radii

def _force_directed_relaxation(pos, radii, n_steps=2000, k_repel=0.5, k_wall=1.0, dt=0.01, rng=None):
    """
    Force-directed relaxation with adaptive parameters.
    """
    if rng is None:
        rng = np.random.RandomState(0)
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    
    n_phases = 3
    steps_per_phase = n_steps // n_phases
    
    for phase in range(n_phases):
        for step in range(steps_per_phase):
            progress = (phase * steps_per_phase + step) / n_steps
            current_dt = dt * (1.0 - 0.7 * progress)
            forces = np.zeros_like(pos)
            
            if phase == 0:
                k_r = k_repel * 2.0
                k_w = k_wall * 2.0
                grow = False
            elif phase == 1:
                k_r = k_repel
                k_w = k_wall
                grow = True
            else:
                k_r = k_repel * 0.5
                k_w = k_wall * 0.5
                grow = True
            
            for i in range(N):
                for j in range(i+1, N):
                    dx = pos[i, 0] - pos[j, 0]
                    dy = pos[i, 1] - pos[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy) + 1e-12
                    sum_r = radii[i] + radii[j]
                    if dist < sum_r:
                        overlap = sum_r - dist
                        f_mag = overlap * k_r * (1.0 + 5.0 * overlap)
                        fx = (dx / dist) * f_mag
                        fy = (dy / dist) * f_mag
                        forces[i, 0] += fx
                        forces[i, 1] += fy
                        forces[j, 0] -= fx
                        forces[j, 1] -= fy
            
            for i in range(N):
                r = radii[i]
                x, y = pos[i]
                if x - r < 0:
                    forces[i, 0] += (0 - (x - r)) * k_w
                if x + r > 1:
                    forces[i, 0] -= (x + r - 1) * k_w
                if y - r < 0:
                    forces[i, 1] += (0 - (y - r)) * k_w
                if y + r > 1:
                    forces[i, 1] -= (y + r - 1) * k_w
            
            pos += forces * current_dt
            pos[:, 0] = np.clip(pos[:, 0], 0, 1)
            pos[:, 1] = np.clip(pos[:, 1], 0, 1)
            
            if grow:
                for i in range(N):
                    max_r_wall = min(pos[i, 0], 1.0-pos[i, 0], pos[i, 1], 1.0-pos[i, 1])
                    max_r_circ = max_r_wall
                    for j in range(N):
                        if i == j:
                            continue
                        dx = pos[i, 0] - pos[j, 0]
                        dy = pos[i, 1] - pos[j, 1]
                        dist = np.sqrt(dx*dx + dy*dy)
                        possible_r = dist - radii[j]
                        if possible_r < max_r_circ:
                            max_r_circ = possible_r
                    if max_r_circ > radii[i] * 1.001:
                        radii[i] = min(radii[i] * 1.003, max_r_circ * 0.9995)
                    elif max_r_circ < radii[i] * 0.999:
                        radii[i] = max_r_circ * 0.999
    
    pos, radii = _post_process(pos, radii, tol=1e-8)
    return pos, radii

def _identify_contacts(pos, radii, tol=1e-4):
    """
    Identify which constraints are active (contacts).
    Returns a list of tuples:
    ('circle', i, j) for circle-circle contact
    ('wall', i, side) for circle-wall contact, side in [0,1,2,3] (left, right, bottom, top)
    """
    contacts = []
    # Circle-Circle
    for i in range(N):
        for j in range(i+1, N):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            sum_r = radii[i] + radii[j]
            if dist < sum_r + tol:
                contacts.append(('circle', i, j))
    
    # Circle-Wall
    for i in range(N):
        x, y = pos[i]
        r = radii[i]
        if x - r < tol: contacts.append(('wall', i, 0))
        if 1.0 - x - r < tol: contacts.append(('wall', i, 1))
        if y - r < tol: contacts.append(('wall', i, 2))
        if 1.0 - y - r < tol: contacts.append(('wall', i, 3))
        
    return contacts

def _solve_rigid_body(pos, radii, contacts):
    """
    Solve the system of non-linear equations defined by the active contacts.
    Variables: x_i, y_i, r_i for i=0..N-1
    Equations:
    - For ('circle', i, j): dist(i,j) - (r_i + r_j) = 0
    - For ('wall', i, side): 
        side 0 (left):   x_i - r_i = 0
        side 1 (right):  1.0 - x_i - r_i = 0
        side 2 (bottom): y_i - r_i = 0
        side 3 (top):    1.0 - y_i - r_i = 0
    
    We need exactly 3*N equations for 3*N unknowns.
    If we have fewer, the system is under-determined. We can fix some variables or add artificial constraints.
    If we have more, the system is over-determined. We should select a subset or use least squares.
    For a stable packing, we expect exactly 3*N contacts (isostatic).
    """
    n_vars = 3 * N
    n_eqs = len(contacts)
    
    # If we don't have enough constraints, we can't solve uniquely.
    # If we have too many, we try to find a consistent subset or use least squares.
    # For now, let's try to solve if n_eqs == n_vars. 
    # If n_eqs < n_vars, we fix some positions to make it solvable (e.g., fix first circle).
    # If n_eqs > n_vars, we drop the weakest contacts (largest residual) until n_eqs == n_vars.
    
    # Calculate residuals to sort contacts by strength
    residuals = []
    for c in contacts:
        if c[0] == 'circle':
            i, j = c[1], c[2]
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            res = dist - (radii[i] + radii[j])
        else:
            i, side = c[1], c[2]
            x, y = pos[i]
            r = radii[i]
            if side == 0: res = x - r
            elif side == 1: res = 1.0 - x - r
            elif side == 2: res = y - r
            else: res = 1.0 - y - r
        residuals.append(abs(res))
        
    # Sort contacts by residual (strongest first)
    sorted_idx = np.argsort(residuals)
    sorted_contacts = [contacts[i] for i in sorted_idx]
    
    # Select exactly n_vars contacts
    selected_contacts = sorted_contacts[:n_vars]
    
    def equations(vars):
        p = vars[:2*N].reshape(N, 2)
        r = vars[2*N:]
        eqs = []
        for c in selected_contacts:
            if c[0] == 'circle':
                i, j = c[1], c[2]
                dx = p[i, 0] - p[j, 0]
                dy = p[i, 1] - p[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                eqs.append(dist - (r[i] + r[j]))
            else:
                i, side = c[1], c[2]
                x, y = p[i]
                r_i = r[i]
                if side == 0: eqs.append(x - r_i)
                elif side == 1: eqs.append(1.0 - x - r_i)
                elif side == 2: eqs.append(y - r_i)
                else: eqs.append(1.0 - y - r_i)
        return eqs
    
    x0 = np.concatenate([pos.flatten(), radii])
    try:
        sol, info, ier, msg = fsolve(equations, x0, full_output=True)
        # ier == 1 indicates success
        if ier == 1:
            pos_sol = sol[:2*N].reshape(N, 2)
            radii_sol = sol[2*N:]
            # Check validity
            if _is_valid(pos_sol, radii_sol, tol=1e-5):
                # Ensure radii are positive
                if np.all(radii_sol > 1e-5):
                    return pos_sol, radii_sol
    except Exception:
        pass
    return None, None

def _high_precision_polish(pos, radii, time_budget=8.0):
    """
    High-precision polish using SLSQP with progressively tighter tolerances
    and a final coordinate-descent pass.
    """
    start_t = time.time()
    pos = pos.copy()
    radii = radii.copy()
    
    def constraints_func(x):
        return _all_constraints(x)
    
    # First pass: SLSQP with moderate tightness
    bounds = []
    for i in range(N):
        bounds.append((max(0.0, pos[i,0] - 0.03), min(1.0, pos[i,0] + 0.03)))
        bounds.append((max(0.0, pos[i,1] - 0.03), min(1.0, pos[i,1] + 0.03)))
    for i in range(N):
        bounds.append((max(1e-5, radii[i] * 0.98), min(0.5, radii[i] * 1.02)))
    
    x0 = np.concatenate([pos.flatten(), radii])
    try:
        result = minimize(_objective, x0, method='SLSQP', bounds=bounds,
                          constraints={'type': 'ineq', 'fun': constraints_func},
                          options={'maxiter': 300, 'ftol': 1e-12})
        if result.success or result.fun < _objective(x0):
            pos_new = result.x[:2*N].reshape(N, 2)
            radii_new = result.x[2*N:]
            pos_pp, radii_pp = _post_process(pos_new, radii_new, tol=1e-9)
            if _is_valid(pos_pp, radii_pp, tol=1e-8):
                pos, radii = pos_pp, radii_pp
    except Exception:
        pass
    
    # Second pass: tighter SLSQP
    if time.time() - start_t < time_budget * 0.4:
        bounds = []
        for i in range(N):
            bounds.append((max(0.0, pos[i,0] - 0.01), min(1.0, pos[i,0] + 0.01)))
            bounds.append((max(0.0, pos[i,1] - 0.01), min(1.0, pos[i,1] + 0.01)))
        for i in range(N):
            bounds.append((max(1e-5, radii[i] * 0.995), min(0.5, radii[i] * 1.005)))
        
        x0 = np.concatenate([pos.flatten(), radii])
        try:
            result = minimize(_objective, x0, method='SLSQP', bounds=bounds,
                              constraints={'type': 'ineq', 'fun': constraints_func},
                              options={'maxiter': 200, 'ftol': 1e-14})
            if result.success or result.fun < _objective(x0):
                pos_new = result.x[:2*N].reshape(N, 2)
                radii_new = result.x[2*N:]
                pos_pp, radii_pp = _post_process(pos_new, radii_new, tol=1e-10)
                if _is_valid(pos_pp, radii_pp, tol=1e-9):
                    pos, radii = pos_pp, radii_pp
        except Exception:
            pass
    
    # Third pass: coordinate descent on radii (positions fixed)
    if time.time() - start_t < time_budget * 0.7:
        for _ in range(5):
            improved = False
            for i in range(N):
                lo = radii[i] * 0.999
                hi = min(0.5, radii[i] * 1.01)
                wall_max = min(pos[i,0], 1.0-pos[i,0], pos[i,1], 1.0-pos[i,1])
                hi = min(hi, wall_max * 0.999999)
                if hi <= lo:
                    continue
                for _ in range(40):
                    mid = (lo + hi) / 2
                    valid = True
                    for j in range(N):
                        if i == j:
                            continue
                        dx = pos[i,0] - pos[j,0]
                        dy = pos[i,1] - pos[j,1]
                        dist = np.sqrt(dx*dx + dy*dy)
                        if dist < mid + radii[j] + 1e-10:
                            valid = False
                            break
                    if valid:
                        lo = mid
                    else:
                        hi = mid
                new_r = lo * 0.999999
                if new_r > radii[i] + 1e-12:
                    radii[i] = new_r
                    improved = True
            if not improved:
                break
    
    # Fourth pass: coordinate descent on positions (radii fixed)
    if time.time() - start_t < time_budget * 0.85:
        for _ in range(3):
            improved = False
            for i in range(N):
                for coord in range(2):
                    old_val = pos[i, coord]
                    lo = max(0.0, old_val - 0.01)
                    hi = min(1.0, old_val + 0.01)
                    for _ in range(30):
                        mid = (lo + hi) / 2
                        pos_test = pos.copy()
                        pos_test[i, coord] = mid
                        valid = True
                        # Check wall
                        if pos_test[i,0] - radii[i] < -1e-10 or pos_test[i,0] + radii[i] > 1.0 + 1e-10:
                            valid = False
                        if pos_test[i,1] - radii[i] < -1e-10 or pos_test[i,1] + radii[i] > 1.0 + 1e-10:
                            valid = False
                        if valid:
                            for j in range(N):
                                if i == j:
                                    continue
                                dx = pos_test[i,0] - pos_test[j,0]
                                dy = pos_test[i,1] - pos_test[j,1]
                                dist = np.sqrt(dx*dx + dy*dy)
                                if dist < radii[i] + radii[j] + 1e-10:
                                    valid = False
                                    break
                        if valid:
                            lo = mid
                        else:
                            hi = mid
                    new_val = (lo + hi) / 2
                    if abs(new_val - old_val) > 1e-10:
                        pos[i, coord] = new_val
                        improved = True
            if not improved:
                break
    
    pos, radii = _post_process(pos, radii, tol=1e-10)
    return pos, radii

def _initial_greedy_best_fit():
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    pos[0] = [0.5, 0.5]
    radii[0] = 0.25
    for i in range(1, N):
        best_x, best_y, best_r = 0.5, 0.5, 0.0
        found = False
        rng = np.random.RandomState(i * 1000 + 42)
        for _ in range(10000):
            x = rng.uniform(0.01, 0.99)
            y = rng.uniform(0.01, 0.99)
            max_r_wall = min(x, 1.0-x, y, 1.0-y)
            max_r_circ = max_r_wall
            for j in range(i):
                dx = x - pos[j, 0]
                dy = y - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                possible_r = dist - radii[j]
                if possible_r < max_r_circ:
                    max_r_circ = possible_r
            if max_r_circ > best_r and max_r_circ > 0.003:
                best_r = max_r_circ
                best_x = x
                best_y = y
                found = True
        if not found or best_r < 0.003:
            best_x, best_y = 0.5, 0.5
            best_r = 0.003
        pos[i] = [best_x, best_y]
        radii[i] = best_r * 0.97
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _initial_hexagonal():
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    r = 0.12
    dx = r * np.sqrt(3)
    dy = r * 1.5
    k = 0
    for row in range(5):
        offset = dx / 2 if row % 2 == 1 else 0
        start_x = r + offset
        for col in range(5):
            if k >= N:
                break
            x = start_x + col * dx
            y = row * dy + r
            x = np.clip(x, r, 1-r)
            y = np.clip(y, r, 1-r)
            pos[k] = [x, y]
            radii[k] = r
            k += 1
    while k < N:
        pos[k] = [0.5, 0.5]
        radii[k] = 0.05
        k += 1
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _initial_random(seed):
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, (N, 2))
    radii = rng.uniform(0.04, 0.12, N)
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _initial_grid():
    """Structured grid initialization with varying radii."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    # 5x5 grid = 25, plus 1 in center
    k = 0
    for i in range(5):
        for j in range(5):
            if k < N-1:
                pos[k] = [(i+0.5)/5.0, (j+0.5)/5.0]
                # Corner circles can be slightly larger
                if i in (0,4) and j in (0,4):
                    radii[k] = 0.11
                elif i in (0,4) or j in (0,4):
                    radii[k] = 0.10
                else:
                    radii[k] = 0.095
                k += 1
    pos[k] = [0.5, 0.5]
    radii[k] = 0.04
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _vectorized_refine(pos, radii, steps=600, step_size=0.04, rng=None):
    if rng is None:
        rng = np.random.RandomState(42)
    pos = pos.copy()
    radii = radii.copy()
    best_score = np.sum(radii)
    best_pos = pos.copy()
    best_radii = radii.copy()
    for step in range(steps):
        current_step = step_size * (0.997 ** step)
        delta_pos = rng.normal(0, current_step, (N, 2))
        delta_radii = rng.normal(0, current_step * 0.3, N)
        new_pos = pos + delta_pos
        new_radii = radii + delta_radii
        new_pos[:, 0] = np.clip(new_pos[:, 0], 0, 1)
        new_pos[:, 1] = np.clip(new_pos[:, 1], 0, 1)
        new_radii = np.maximum(new_radii, 1e-5)
        new_pos, new_radii = _post_process(new_pos, new_radii, tol=1e-8)
        if _is_valid(new_pos, new_radii, tol=1e-7):
            score = np.sum(new_radii)
            if score > best_score:
                best_score = score
                best_pos = new_pos.copy()
                best_radii = new_radii.copy()
                pos = new_pos
                radii = new_radii
    return best_pos, best_radii

def _local_search(pos, radii, time_budget=5.0, rng=None):
    """
    Local search that perturbs individual circles and accepts improvements.
    """
    if rng is None:
        rng = np.random.RandomState(0)
    start_t = time.time()
    pos = pos.copy()
    radii = radii.copy()
    best_score = np.sum(radii)
    
    while time.time() - start_t < time_budget:
        i = rng.randint(N)
        # Try moving circle i
        delta = rng.normal(0, 0.01, 2)
        new_pos = pos.copy()
        new_pos[i] = np.clip(new_pos[i] + delta, 0, 1)
        new_radii = radii.copy()
        
        # Try to grow radius i
        max_r = min(new_pos[i,0], 1-new_pos[i,0], new_pos[i,1], 1-new_pos[i,1])
        for j in range(N):
            if i == j:
                continue
            dx = new_pos[i,0] - new_pos[j,0]
            dy = new_pos[i,1] - new_pos[j,1]
            dist = np.sqrt(dx*dx + dy*dy)
            max_r = min(max_r, dist - radii[j])
        new_radii[i] = min(radii[i] * 1.005, max_r * 0.9999)
        
        if _is_valid(new_pos, new_radii, tol=1e-8):
            score = np.sum(new_radii)
            if score > best_score:
                best_score = score
                pos = new_pos
                radii = new_radii
        else:
            # Try just moving without growing
            new_radii2 = radii.copy()
            if _is_valid(new_pos, new_radii2, tol=1e-8):
                score = np.sum(new_radii2)
                if score > best_score:
                    best_score = score
                    pos = new_pos
    
    return pos, radii

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_pos = None
    best_radii = None
    best_score = -np.inf
    
    initializers = [
        _initial_greedy_best_fit,
        _initial_hexagonal,
        _initial_grid,
        lambda: _initial_random(1),
        lambda: _initial_random(2),
        lambda: _initial_random(3),
        lambda: _initial_random(4),
        lambda: _initial_random(5),
        lambda: _initial_random(6),
        lambda: _initial_random(7),
    ]
    
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
    for i in range(N):
        bounds.append((1e-5, 0.5))
        
    def constraints_func(x):
        return _all_constraints(x)

    for idx, init_func in enumerate(initializers):
        if time.time() > deadline - 25:
            break
        try:
            pos_init, rad_init = init_func()
        except Exception:
            continue
        
        x0 = np.concatenate([pos_init.flatten(), rad_init])
        
        # Smooth penalty optimization
        try:
            pos_smooth, radii_smooth = _smooth_optimize(pos_init, rad_init, beta_start=20.0, beta_end=80.0, n_iters=3)
            score_smooth = np.sum(radii_smooth)
            if score_smooth > best_score:
                best_score = score_smooth
                best_pos = pos_smooth.copy()
                best_radii = radii_smooth.copy()
        except Exception:
            pass
        
        # Force-directed relaxation
        if time.time() < deadline - 15:
            try:
                pos_fd, radii_fd = _force_directed_relaxation(pos_smooth if best_pos is None else best_pos, 
                                                              radii_smooth if best_radii is None else best_radii,
                                                              n_steps=1000, k_repel=0.5, k_wall=1.0, dt=0.01,
                                                              rng=np.random.RandomState(idx * 13 + 7))
                if _is_valid(pos_fd, radii_fd):
                    score_fd = np.sum(radii_fd)
                    if score_fd > best_score:
                        best_score = score_fd
                        best_pos = pos_fd.copy()
                        best_radii = radii_fd.copy()
            except Exception:
                pass
        
        # Rigid Body Solver - The Key Improvement
        if time.time() < deadline - 10 and best_pos is not None:
            # Identify contacts from the best current solution
            contacts = _identify_contacts(best_pos, best_radii, tol=1e-4)
            pos_rig, radii_rig = _solve_rigid_body(best_pos, best_radii, contacts)
            if pos_rig is not None:
                if _is_valid(pos_rig, radii_rig, tol=1e-6):
                    score_rig = np.sum(radii_rig)
                    if score_rig > best_score:
                        best_score = score_rig
                        best_pos = pos_rig.copy()
                        best_radii = radii_rig.copy()
                        
                    # Try to expand radii after rigid solve
                    # This might allow finding a better contact set
                    pos_exp, rad_exp = _post_process(best_pos, best_radii * 1.001, tol=1e-6)
                    if _is_valid(pos_exp, rad_exp, tol=1e-6):
                        score_exp = np.sum(rad_exp)
                        if score_exp > best_score:
                            best_score = score_exp
                            best_pos = pos_exp.copy()
                            best_radii = rad_exp.copy()
                            
                            # Re-solve with new contacts
                            contacts2 = _identify_contacts(best_pos, best_radii, tol=1e-4)
                            pos_rig2, radii_rig2 = _solve_rigid_body(best_pos, best_radii, contacts2)
                            if pos_rig2 is not None and _is_valid(pos_rig2, radii_rig2, tol=1e-6):
                                score_rig2 = np.sum(radii_rig2)
                                if score_rig2 > best_score:
                                    best_score = score_rig2
                                    best_pos = pos_rig2.copy()
                                    best_radii = radii_rig2.copy()

        # SLSQP with hard constraints
        try:
            result = minimize(_objective, x0, method='SLSQP', bounds=bounds,
                              constraints={'type': 'ineq', 'fun': constraints_func},
                              options={'maxiter': 300, 'ftol': 1e-10})
            if result.success or result.fun < _objective(x0):
                pos_opt = result.x[:2*N].reshape(N, 2)
                radii_opt = result.x[2*N:]
                pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
                if _is_valid(pos_pp, radii_pp):
                    score = np.sum(radii_pp)
                    if score > best_score:
                        best_score = score
                        best_pos = pos_pp.copy()
                        best_radii = radii_pp.copy()
        except Exception:
            pass
        
        # Vectorized refinement
        if time.time() < deadline - 20 and best_pos is not None:
            pos_ref, rad_ref = _vectorized_refine(best_pos, best_radii, steps=500, step_size=0.015, rng=np.random.RandomState(idx * 7 + 1))
            if _is_valid(pos_ref, rad_ref):
                score_ref = np.sum(rad_ref)
                if score_ref > best_score:
                    best_score = score_ref
                    best_pos = pos_ref.copy()
                    best_radii = rad_ref.copy()

    # Local search
    if best_pos is not None and time.time() < deadline - 10:
        pos_ls, radii_ls = _local_search(best_pos, best_radii, time_budget=5.0, rng=np.random.RandomState(99))
        if _is_valid(pos_ls, radii_ls):
            score_ls = np.sum(radii_ls)
            if score_ls > best_score:
                best_score = score_ls
                best_pos = pos_ls.copy()
                best_radii = radii_ls.copy()

    # Final high-precision polish
    if best_pos is not None and time.time() < deadline - 8:
        pos_polished, radii_polished = _high_precision_polish(best_pos, best_radii, time_budget=8.0)
        if _is_valid(pos_polished, radii_polished, tol=1e-8):
            score_polished = np.sum(radii_polished)
            if score_polished > best_score:
                best_score = score_polished
                best_pos = pos_polished.copy()
                best_radii = radii_polished.copy()
        
        # Additional Nelder-Mead polish on radii only (positions fixed)
        if time.time() < deadline - 3:
            try:
                pos_fixed = best_pos.copy()
                radii_nm = best_radii.copy()
                
                def nm_obj(r):
                    r = np.maximum(r, 1e-5)
                    for i in range(N):
                        if pos_fixed[i,0] - r[i] < -1e-10 or pos_fixed[i,0] + r[i] > 1.0 + 1e-10:
                            return 1e10
                        if pos_fixed[i,1] - r[i] < -1e-10 or pos_fixed[i,1] + r[i] > 1.0 + 1e-10:
                            return 1e10
                    for i in range(N):
                        for j in range(i+1, N):
                            dx = pos_fixed[i,0] - pos_fixed[j,0]
                            dy = pos_fixed[i,1] - pos_fixed[j,1]
                            dist = np.sqrt(dx*dx + dy*dy)
                            if dist < r[i] + r[j] + 1e-10:
                                return 1e10 + (r[i] + r[j] - dist)
                    return -np.sum(r)
                
                result_nm = minimize(nm_obj, radii_nm, method='Nelder-Mead',
                                     options={'maxiter': 800, 'xatol': 1e-12, 'fatol': 1e-14,
                                              'initial_simplex': 0.001})
                if result_nm.fun < -np.sum(radii_nm):
                    radii_new = np.maximum(result_nm.x, 1e-5)
                    valid = True
                    for i in range(N):
                        if pos_fixed[i,0] - radii_new[i] < -1e-9 or pos_fixed[i,0] + radii_new[i] > 1.0 + 1e-9:
                            valid = False
                            break
                        if pos_fixed[i,1] - radii_new[i] < -1e-9 or pos_fixed[i,1] + radii_new[i] > 1.0 + 1e-9:
                            valid = False
                            break
                    if valid:
                        for i in range(N):
                            for j in range(i+1, N):
                                dx = pos_fixed[i,0] - pos_fixed[j,0]
                                dy = pos_fixed[i,1] - pos_fixed[j,1]
                                dist = np.sqrt(dx*dx + dy*dy)
                                if dist < radii_new[i] + radii_new[j] + 1e-9:
                                    valid = False
                                    break
                            if not valid:
                                break
                    if valid:
                        score_nm = np.sum(radii_new)
                        if score_nm > best_score:
                            best_score = score_nm
                            best_radii = radii_new.copy()
            except Exception:
                pass

    # Fallback
    if best_pos is None or not _is_valid(best_pos, best_radii):
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        k = 0
        for i in range(5):
            for j in range(5):
                if k < N-1:
                    pos[k] = [(i+0.5)/5.0, (j+0.5)/5.0]
                    radii[k] = 0.095
                    k += 1
        pos[k] = [0.5, 0.02]
        radii[k] = 0.01
        best_pos, best_radii = pos, radii
        best_pos, best_radii = _post_process(best_pos, best_radii)

    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
