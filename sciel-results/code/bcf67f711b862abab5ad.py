"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def _is_valid(pos, radii, tol=1e-8):
    """Strict validity check."""
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

def _post_process(pos, radii):
    """Shrink radii to ensure strict validity while preserving as much as possible."""
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    
    for _ in range(300):
        valid = True
        
        for i in range(N):
            r = radii[i]
            x, y = pos[i]
            max_r = min(x, 1.0 - x, y, 1.0 - y)
            if r > max_r:
                radii[i] = max(max_r * 0.99999, 1e-5)
                valid = False
        
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r:
                    # Scale both down proportionally
                    if sum_r > 0:
                        k = (dist / sum_r) * 0.99999
                    else:
                        k = 0.5
                    if k < 1.0:
                        radii[i] = max(radii[i] * k, 1e-5)
                        radii[j] = max(radii[j] * k, 1e-5)
                        valid = False
                        
        if valid:
            break
            
    return pos, radii

def _objective_and_constraints(x):
    """
    Objective: maximize sum of radii => minimize -sum(radii)
    x = [x1, y1, ..., xN, yN, r1, ..., rN]
    Returns (objective_value, list_of_constraint_values)
    """
    pos = x[:2*N].reshape(N, 2)
    radii = x[2*N:]
    
    obj = -np.sum(radii)
    
    constraints = []
    
    # Boundary constraints: r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
    for i in range(N):
        constraints.append(x[2*i] - x[2*N+i])           # x_i - r_i >= 0
        constraints.append((1.0 - x[2*i]) - x[2*N+i])   # 1 - x_i - r_i >= 0
        constraints.append(x[2*i+1] - x[2*N+i])         # y_i - r_i >= 0
        constraints.append((1.0 - x[2*i+1]) - x[2*N+i]) # 1 - y_i - r_i >= 0
    
    # Overlap constraints: ||p_i - p_j||^2 >= (r_i + r_j)^2
    for i in range(N):
        for j in range(i+1, N):
            dx = x[2*i] - x[2*j]
            dy = x[2*i+1] - x[2*j+1]
            constraints.append(dx*dx + dy*dy - (x[2*N+i] + x[2*N+j])**2)
    
    return obj, constraints

def _slsqp_optimize(pos, radii, maxiter=2000, ftol=1e-14):
    """SLSQP optimization with hard constraints."""
    x = np.concatenate([pos.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((1e-4, 1.0 - 1e-4))
        bounds.append((1e-4, 1.0 - 1e-4))
    for i in range(N):
        bounds.append((1e-4, 0.5))
    
    constraints = []
    for i in range(N):
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i] - x[2*N+i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: (1.0 - x[2*i]) - x[2*N+i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i+1] - x[2*N+i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: (1.0 - x[2*i+1]) - x[2*N+i]})
    
    for i in range(N):
        for j in range(i+1, N):
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: (x[2*i]-x[2*j])**2 + (x[2*i+1]-x[2*j+1])**2 - (x[2*N+i]+x[2*N+j])**2
            })
    
    def obj(x):
        return -np.sum(x[2*N:])
    
    try:
        result = minimize(obj, x, method='SLSQP', bounds=bounds,
                          constraints=constraints,
                          options={'maxiter': maxiter, 'ftol': ftol})
        if result.success or result.fun < -np.sum(radii):
            return result.x
    except Exception:
        pass
    
    return x

def _trust_constr_optimize(pos, radii, maxiter=1000):
    """Trust-constr optimization with nonlinear constraints."""
    x = np.concatenate([pos.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((1e-4, 1.0 - 1e-4))
        bounds.append((1e-4, 1.0 - 1e-4))
    for i in range(N):
        bounds.append((1e-4, 0.5))
    
    # Build constraint functions for trust-constr
    # trust-constr uses scipy.optimize.LinearConstraint or NonlinearConstraint
    from scipy.optimize import NonlinearConstraint
    
    # Boundary constraints
    def boundary_constraints(x):
        c = np.zeros(4*N)
        for i in range(N):
            c[4*i] = x[2*i] - x[2*N+i]
            c[4*i+1] = (1.0 - x[2*i]) - x[2*N+i]
            c[4*i+2] = x[2*i+1] - x[2*N+i]
            c[4*i+3] = (1.0 - x[2*i+1]) - x[2*N+i]
        return c
    
    def boundary_jacobian(x):
        J = np.zeros((4*N, 2*N + N))
        for i in range(N):
            J[4*i, 2*i] = 1.0
            J[4*i, 2*N+i] = -1.0
            J[4*i+1, 2*i] = -1.0
            J[4*i+1, 2*N+i] = -1.0
            J[4*i+2, 2*i+1] = 1.0
            J[4*i+2, 2*N+i] = -1.0
            J[4*i+3, 2*i+1] = -1.0
            J[4*i+3, 2*N+i] = -1.0
        return J
    
    # Overlap constraints
    pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    n_pairs = len(pairs)
    
    def overlap_constraints(x):
        c = np.zeros(n_pairs)
        for k, (i, j) in enumerate(pairs):
            dx = x[2*i] - x[2*j]
            dy = x[2*i+1] - x[2*j+1]
            c[k] = dx*dx + dy*dy - (x[2*N+i] + x[2*N+j])**2
        return c
    
    def overlap_jacobian(x):
        J = np.zeros((n_pairs, 2*N + N))
        for k, (i, j) in enumerate(pairs):
            dx = x[2*i] - x[2*j]
            dy = x[2*i+1] - x[2*j+1]
            J[k, 2*i] = 2*dx
            J[k, 2*j] = -2*dx
            J[k, 2*i+1] = 2*dy
            J[k, 2*j+1] = -2*dy
            J[k, 2*N+i] = -2*(x[2*N+i] + x[2*N+j])
            J[k, 2*N+j] = -2*(x[2*N+i] + x[2*N+j])
        return J
    
    def obj(x):
        return -np.sum(x[2*N:])
    
    def obj_grad(x):
        g = np.zeros(2*N + N)
        g[2*N:] = -1.0
        return g
    
    try:
        from scipy.optimize import NonlinearConstraint
        nl_c = NonlinearConstraint(
            boundary_constraints, 0, np.inf,
            jac=boundary_jacobian
        )
        ol_c = NonlinearConstraint(
            overlap_constraints, 0, np.inf,
            jac=overlap_jacobian
        )
        
        result = minimize(obj, x, method='trust-constr',
                          jac=obj_grad,
                          bounds=bounds,
                          constraints=[nl_c, ol_c],
                          options={'maxiter': maxiter, 'gtol': 1e-10, 'xtol': 1e-10})
        if result.fun < -np.sum(radii) + 1e-12:
            return result.x
    except Exception:
        pass
    
    return x

def _initial_grid():
    """Structured grid initialization."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # 5x5 grid with one removed
    k = 0
    for i in range(5):
        for j in range(5):
            if k >= N:
                break
            pos[k] = [(i + 0.5) / 5.0, (j + 0.5) / 5.0]
            radii[k] = 0.095
            k += 1
    
    return pos, radii

def _initial_hexagonal():
    """Hexagonal lattice initialization."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    r = 0.11
    dx = r * np.sqrt(3)
    dy = r * 1.5
    
    k = 0
    for row in range(6):
        offset = dx / 2 if row % 2 == 1 else 0
        start_x = r + offset
        y = row * dy + r
        
        for col in range(5):
            if k >= N:
                break
            x = start_x + col * dx
            
            if x + r <= 1.0 and x - r >= 0.0 and y + r <= 1.0 and y - r >= 0.0:
                pos[k] = [x, y]
                radii[k] = r
                k += 1
        
        if k >= N:
            break
            
    while k < N:
        rng = np.random.RandomState(k * 7 + 13)
        placed = False
        for _ in range(200):
            x = rng.uniform(0.05, 0.95)
            y = rng.uniform(0.05, 0.95)
            min_dist = min(x, 1 - x, y, 1 - y)
            for j in range(k):
                d = np.sqrt((x - pos[j, 0])**2 + (y - pos[j, 1])**2)
                min_dist = min(min_dist, d - radii[j])
            if min_dist > 0.02:
                pos[k] = [x, y]
                radii[k] = min_dist * 0.95
                k += 1
                placed = True
                break
        if not placed:
            pos[k] = [0.5, 0.5]
            radii[k] = 0.01
            k += 1
            
    return pos, radii

def _initial_corner():
    """Corner-biased initialization with larger corner circles."""
    rng = np.random.RandomState(42)
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # 4 corner circles
    corners = [(0.14, 0.14), (0.86, 0.14), (0.14, 0.86), (0.86, 0.86)]
    for i, (cx, cy) in enumerate(corners):
        pos[i] = [cx, cy]
        radii[i] = 0.13
    
    # 4 edge midpoints
    edges = [(0.5, 0.06), (0.5, 0.94), (0.06, 0.5), (0.94, 0.5)]
    for i, (ex, ey) in enumerate(edges):
        idx = i + 4
        pos[idx] = [ex, ey]
        radii[idx] = 0.055
    
    # Fill remaining with random placement
    for i in range(8, N):
        best_x, best_y, best_r = 0.5, 0.5, 0.0
        for _ in range(3000):
            x = rng.uniform(0.03, 0.97)
            y = rng.uniform(0.03, 0.97)
            max_r = min(x, 1.0 - x, y, 1.0 - y)
            for j in range(i):
                dx = x - pos[j, 0]
                dy = y - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                max_r = min(max_r, dist - radii[j])
            if max_r > best_r and max_r > 0.005:
                best_r = max_r
                best_x = x
                best_y = y
        
        pos[i] = [best_x, best_y]
        radii[i] = max(best_r, 0.005) * 0.97
        
    return pos, radii

def _initial_best_fit():
    """Greedy best-fit initialization."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    pos[0] = [0.5, 0.5]
    radii[0] = 0.25
    
    for i in range(1, N):
        best_x, best_y, best_r = 0.5, 0.5, 0.0
        rng = np.random.RandomState(i * 1000 + 42)
        
        for _ in range(8000):
            x = rng.uniform(0.02, 0.98)
            y = rng.uniform(0.02, 0.98)
            
            max_r_wall = min(x, 1.0 - x, y, 1.0 - y)
            max_r_circ = max_r_wall
            for j in range(i):
                dx = x - pos[j, 0]
                dy = y - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                possible_r = dist - radii[j]
                if possible_r < max_r_circ:
                    max_r_circ = possible_r
            
            if max_r_circ > best_r and max_r_circ > 0.003:
                best_r = max_r_circ
                best_x = x
                best_y = y
        
        if best_r < 0.003:
            best_x, best_y = 0.5, 0.5
            best_r = 0.003
            
        pos[i] = [best_x, best_y]
        radii[i] = best_r * 0.97
        
    return pos, radii

def _simulated_annealing(pos, radii, n_steps=5000, t0=0.04, t_end=0.0005, seed=0):
    """
    Simulated annealing with penalty-based energy.
    """
    rng = np.random.RandomState(seed)
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    n = len(radii)
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    def energy(p, r):
        e = 0.0
        # Boundary penalty
        for i in range(n):
            e += max(0, r[i] - p[i,0])**2 + max(0, p[i,0] - (1-r[i]))**2
            e += max(0, r[i] - p[i,1])**2 + max(0, p[i,1] - (1-r[i]))**2
        # Overlap penalty
        for i in range(n):
            for j in range(i+1, n):
                dx = p[i,0] - p[j,0]
                dy = p[i,1] - p[j,1]
                d2 = dx*dx + dy*dy
                sr = r[i] + r[j]
                if d2 < sr*sr:
                    e += (sr*sr - d2)**2
        return e
    
    current_energy = energy(pos, radii) * 100.0
    
    for step in range(n_steps):
        frac = step / max(n_steps - 1, 1)
        temp = t0 * (t_end / t0) ** frac
        
        i = rng.randint(n)
        dx = rng.normal(0, temp * 0.015)
        dy = rng.normal(0, temp * 0.015)
        dr = rng.normal(0, temp * 0.008)
        
        old_x, old_y, old_r = pos[i, 0], pos[i, 1], radii[i]
        new_x = np.clip(pos[i, 0] + dx, 0.01, 0.99)
        new_y = np.clip(pos[i, 1] + dy, 0.01, 0.99)
        new_r = np.clip(radii[i] + dr, 0.01, 0.4)
        
        pos[i, 0] = new_x
        pos[i, 1] = new_y
        radii[i] = new_r
        
        new_energy = energy(pos, radii) * 100.0
        delta_e = new_energy - current_energy
        
        if delta_e < 0 or rng.random() < np.exp(-delta_e / max(temp, 1e-8)):
            current_energy = new_energy
        else:
            pos[i, 0] = old_x
            pos[i, 1] = old_y
            radii[i] = old_r
        
        if step % 200 == 0:
            pos_pp, radii_pp = _post_process(pos, radii)
            if _is_valid(pos_pp, radii_pp):
                score = np.sum(radii_pp)
                if score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
    
    return best_pos, best_radii

def _local_optimize(pos, radii, deadline, use_trust_constr=True):
    """Run local optimization with SLSQP and optionally trust-constr."""
    x = np.concatenate([pos.flatten(), radii])
    
    # SLSQP
    x_slsqp = _slsqp_optimize(pos, radii, maxiter=1500)
    pos_s = x_slsqp[:2*N].reshape(N, 2)
    rad_s = np.maximum(x_slsqp[2*N:], 1e-4)
    pos_s, rad_s = _post_process(pos_s, rad_s)
    
    best_pos = pos_s.copy()
    best_radii = rad_s.copy()
    best_score = np.sum(rad_s) if _is_valid(pos_s, rad_s) else -np.inf
    
    # Trust-constr
    if use_trust_constr and time.time() < deadline - 15:
        x_tc = _trust_constr_optimize(pos, radii, maxiter=800)
        pos_t = x_tc[:2*N].reshape(N, 2)
        rad_t = np.maximum(x_tc[2*N:], 1e-4)
        pos_t, rad_t = _post_process(pos_t, rad_t)
        if _is_valid(pos_t, rad_t):
            score_t = np.sum(rad_t)
            if score_t > best_score:
                best_pos = pos_t.copy()
                best_radii = rad_t.copy()
                best_score = score_t
    
    # Chain: run SLSQP from trust-constr result
    if use_trust_constr and time.time() < deadline - 10 and best_score > -np.inf:
        x_chain = _slsqp_optimize(best_pos, best_radii, maxiter=1000)
        pos_c = x_chain[:2*N].reshape(N, 2)
        rad_c = np.maximum(x_chain[2*N:], 1e-4)
        pos_c, rad_c = _post_process(pos_c, rad_c)
        if _is_valid(pos_c, rad_c):
            score_c = np.sum(rad_c)
            if score_c > best_score:
                best_pos = pos_c.copy()
                best_radii = rad_c.copy()
                best_score = score_c
    
    return best_pos, best_radii, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Generate multiple initializations
    inits = []
    
    pos_g, rad_g = _initial_grid()
    inits.append(('grid', pos_g, rad_g))
    
    pos_h, rad_h = _initial_hexagonal()
    inits.append(('hex', pos_h, rad_h))
    
    pos_c, rad_c = _initial_corner()
    inits.append(('corner', pos_c, rad_c))
    
    pos_b, rad_b = _initial_best_fit()
    inits.append(('bestfit', pos_b, rad_b))
    
    # Random inits
    for s in [1, 2, 3]:
        rng = np.random.RandomState(s * 100)
        pos_r = rng.uniform(0.1, 0.9, (N, 2))
        radii_r = rng.uniform(0.04, 0.12, N)
        pos_r, radii_r = _post_process(pos_r, radii_r)
        inits.append((f'rand{s}', pos_r, radii_r))
    
    best_pos = inits[0][1].copy()
    best_radii = inits[0][2].copy()
    best_score = np.sum(best_radii)
    
    # Process each initialization
    for idx, (name, pos_init, rad_init) in enumerate(inits):
        if time.time() > deadline - 25:
            break
        
        # Simulated annealing first (faster, good for escaping local minima)
        if time.time() < deadline - 30:
            try:
                pos_sa, rad_sa = _simulated_annealing(
                    pos_init, rad_init,
                    n_steps=3000, t0=0.03, t_end=0.0005,
                    seed=idx * 100 + 7
                )
                pos_pp, radii_pp = _post_process(pos_sa, rad_sa)
                if _is_valid(pos_pp, radii_pp):
                    score = np.sum(radii_pp)
                    if score > best_score:
                        best_score = score
                        best_pos = pos_pp.copy()
                        best_radii = radii_pp.copy()
            except Exception:
                pass
        
        # Local optimization
        if time.time() < deadline - 20:
            try:
                pos_opt, rad_opt, score_opt = _local_optimize(
                    pos_init, rad_init, deadline
                )
                if score_opt > best_score:
                    best_score = score_opt
                    best_pos = pos_opt.copy()
                    best_radii = rad_opt.copy()
            except Exception:
                pass
        
        # Also try local opt from SA result
        if time.time() < deadline - 15 and idx < 3:
            try:
                pos_sa2, rad_sa2 = _simulated_annealing(
                    pos_init, rad_init,
                    n_steps=2000, t0=0.02, t_end=0.0003,
                    seed=idx * 200 + 13
                )
                pos_opt2, rad_opt2, score_opt2 = _local_optimize(
                    pos_sa2, rad_sa2, deadline
                )
                if score_opt2 > best_score:
                    best_score = score_opt2
                    best_pos = pos_opt2.copy()
                    best_radii = rad_opt2.copy()
            except Exception:
                pass
    
    # Final polish: run SLSQP from the best solution
    if time.time() < deadline - 8:
        try:
            x_final = _slsqp_optimize(best_pos, best_radii, maxiter=2000, ftol=1e-15)
            pos_f = x_final[:2*N].reshape(N, 2)
            rad_f = np.maximum(x_final[2*N:], 1e-4)
            pos_f, rad_f = _post_process(pos_f, rad_f)
            if _is_valid(pos_f, rad_f):
                score_f = np.sum(rad_f)
                if score_f > best_score:
                    best_score = score_f
                    best_pos = pos_f.copy()
                    best_radii = rad_f.copy()
        except Exception:
            pass
    
    # Final validation
    best_pos, best_radii = _post_process(best_pos, best_radii)
    
    if not _is_valid(best_pos, best_radii):
        # Fallback: simple grid
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        k = 0
        for i in range(5):
            for j in range(5):
                if k < N - 1:
                    pos[k] = [(i + 0.5) / 5.0, (j + 0.5) / 5.0]
                    radii[k] = 0.095
                    k += 1
        pos[k] = [0.5, 0.02]
        radii[k] = 0.01
        best_pos, best_radii = pos, radii
        best_pos, best_radii = _post_process(best_pos, best_radii)
    
    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
