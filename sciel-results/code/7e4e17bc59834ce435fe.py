"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time

N = 26

# EVOLVE-BLOCK-START

def _is_valid(pos, radii, tol=1e-9):
    """Check if circles are valid (inside square, disjoint interiors)."""
    n = len(radii)
    for i in range(n):
        if pos[i, 0] - radii[i] < -tol or pos[i, 0] + radii[i] > 1.0 + tol:
            return False
        if pos[i, 1] - radii[i] < -tol or pos[i, 1] + radii[i] > 1.0 + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < radii[i] + radii[j] - tol:
                return False
    return True


def _smooth_penalty(pos, radii, beta=50.0):
    """Log-sum-exp smooth penalty focusing on worst violations."""
    n = len(radii)
    violations = []
    # Boundary violations
    for i in range(n):
        if pos[i, 0] < radii[i]:
            violations.append(radii[i] - pos[i, 0])
        if pos[i, 0] > 1.0 - radii[i]:
            violations.append(pos[i, 0] - (1.0 - radii[i]))
        if pos[i, 1] < radii[i]:
            violations.append(radii[i] - pos[i, 1])
        if pos[i, 1] > 1.0 - radii[i]:
            violations.append(pos[i, 1] - (1.0 - radii[i]))
    # Overlap violations
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            min_d = radii[i] + radii[j]
            if dist < min_d:
                violations.append(min_d - dist)
    if not violations:
        return 0.0
    v = np.array(violations)
    # Log-sum-exp for smooth max, plus sum of squares
    m = np.max(v)
    if m < 1e-15:
        return 0.0
    lse = m + np.log(np.sum(np.exp(v - m))) / beta
    sq = np.sum(v**2)
    return lse + 0.1 * sq


def _penalty(pos, radii):
    """Total constraint violation (sum of squared violations)."""
    n = len(radii)
    pen = 0.0
    for i in range(n):
        if pos[i, 0] < radii[i]:
            pen += (radii[i] - pos[i, 0])**2
        if pos[i, 0] > 1.0 - radii[i]:
            pen += (pos[i, 0] - (1.0 - radii[i]))**2
        if pos[i, 1] < radii[i]:
            pen += (radii[i] - pos[i, 1])**2
        if pos[i, 1] > 1.0 - radii[i]:
            pen += (pos[i, 1] - (1.0 - radii[i]))**2
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            min_d = radii[i] + radii[j]
            if dist < min_d:
                pen += (min_d - dist)**2
    return pen


def _score(pos, radii):
    """Sum of radii (the objective to maximize)."""
    return float(np.sum(radii))


def _fix_boundaries(pos, radii):
    """Shrink radii to fit within boundaries."""
    radii = radii.copy()
    for i in range(N):
        max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
        if radii[i] > max_r:
            radii[i] = max(0.001, max_r * 0.999)
    return radii


def _resolve_overlaps(pos, radii, max_iter=300):
    """Iteratively shrink radii to resolve overlaps."""
    radii = radii.copy()
    for _ in range(max_iter):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_d = radii[i] + radii[j]
                if dist < min_d:
                    scale = (dist / min_d) * 0.999 if min_d > 0 else 0.999
                    radii[i] *= scale
                    radii[j] *= scale
                    changed = True
        radii = _fix_boundaries(pos, radii)
        if not changed:
            break
    return radii


def _make_grid(seed=42):
    """5x5 grid with one extra circle."""
    rng = np.random.RandomState(seed)
    side = 5
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    for i in range(5):
        for j in range(5):
            idx = i * 5 + j
            if idx < N:
                pos[idx] = [(i + 0.5) / side, (j + 0.5) / side]
                radii[idx] = 0.5 / side - 0.003
    # 26th circle in a gap
    pos[25] = [0.5, 0.02]
    radii[25] = 0.015
    return pos, radii


def _make_hexagonal(seed=42):
    """Hexagonal packing pattern."""
    rng = np.random.RandomState(seed)
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    r = 0.095
    idx = 0
    for row in range(6):
        y = 0.1 + row * r * np.sqrt(3)
        if y > 0.9:
            break
        offset = r if row % 2 == 0 else 2 * r
        for col in range(5):
            x = offset + col * 2 * r
            if x > 0.9:
                break
            if idx < N:
                pos[idx] = [x, y]
                radii[idx] = r
                idx += 1
    # Fill remaining
    while idx < N:
        pos[idx] = [0.5, 0.5]
        radii[idx] = 0.05
        idx += 1
    return pos, radii


def _make_center_ring(seed=42):
    """Center circle with ring around it."""
    rng = np.random.RandomState(seed)
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    # Center circle
    pos[0] = [0.5, 0.5]
    radii[0] = 0.15
    # Ring of circles around center
    ring_r = 0.12
    ring_dist = radii[0] + ring_r + 0.01
    n_ring = 12
    for k in range(n_ring):
        angle = 2 * np.pi * k / n_ring
        x = 0.5 + ring_dist * np.cos(angle)
        y = 0.5 + ring_dist * np.sin(angle)
        idx = 1 + k
        if idx < N:
            # Clip to square
            x = np.clip(x, ring_r, 1.0 - ring_r)
            y = np.clip(y, ring_r, 1.0 - ring_r)
            pos[idx] = [x, y]
            radii[idx] = ring_r
    # Fill remaining in corners
    remaining = N - 1 - n_ring
    corner_pos = [[0.05, 0.05], [0.95, 0.05], [0.05, 0.95], [0.95, 0.95]]
    for k in range(min(remaining, 4)):
        idx = 1 + n_ring + k
        pos[idx] = corner_pos[k]
        radii[idx] = 0.06
    for k in range(4, remaining):
        idx = 1 + n_ring + k
        pos[idx] = rng.uniform(0.1, 0.9, size=2)
        radii[idx] = 0.05
    return pos, radii


def _make_random(seed=42):
    """Random initial configuration."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, size=(N, 2))
    radii = rng.uniform(0.03, 0.08, size=N)
    return pos, radii


def _make_mixed(seed=42):
    """Mixed: some grid, some random."""
    rng = np.random.RandomState(seed)
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    # 16 grid circles
    side = 4
    idx = 0
    for i in range(4):
        for j in range(4):
            pos[idx] = [(i + 0.5) / side, (j + 0.5) / side]
            radii[idx] = 0.5 / side - 0.005
            idx += 1
    # 10 random circles
    pos[idx:idx+10] = rng.uniform(0.05, 0.95, size=(10, 2))
    radii[idx:idx+10] = rng.uniform(0.03, 0.07, size=10)
    return pos, radii


def _force_directed_anneal(pos, radii, time_budget=30.0, seed=42):
    """Simulated annealing with force-directed moves and smooth penalty."""
    rng = np.random.RandomState(seed)
    pos = pos.copy()
    radii = radii.copy()
    radii = _fix_boundaries(pos, radii)
    radii = _resolve_overlaps(pos, radii)
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = _score(pos, radii)
    
    T = 0.02
    T_min = 1e-7
    cooling = 0.997
    max_iter = 80000
    
    start = time.time()
    no_improve_count = 0
    patience = 5000
    
    for it in range(max_iter):
        if time.time() - start > time_budget:
            break
        
        # Random move: perturb positions and/or radii
        move_type = rng.random()
        if move_type < 0.4:
            # Move one circle
            i = rng.randint(N)
            dx = rng.normal(0, 0.025)
            dy = rng.normal(0, 0.025)
            pos[i, 0] = np.clip(pos[i, 0] + dx, 0.01, 0.99)
            pos[i, 1] = np.clip(pos[i, 1] + dy, 0.01, 0.99)
        elif move_type < 0.7:
            # Change radius of one circle
            i = rng.randint(N)
            dr = rng.normal(0, 0.008)
            radii[i] = np.clip(radii[i] + dr, 0.005, 0.35)
        elif move_type < 0.85:
            # Move two circles
            i, j = rng.choice(N, size=2, replace=False)
            pos[i, 0] = np.clip(pos[i, 0] + rng.normal(0, 0.025), 0.01, 0.99)
            pos[i, 1] = np.clip(pos[i, 1] + rng.normal(0, 0.025), 0.01, 0.99)
            pos[j, 0] = np.clip(pos[j, 0] + rng.normal(0, 0.025), 0.01, 0.99)
            pos[j, 1] = np.clip(pos[j, 1] + rng.normal(0, 0.025), 0.01, 0.99)
        else:
            # Simultaneous small move of all circles (global shift)
            dx = rng.normal(0, 0.005)
            dy = rng.normal(0, 0.005)
            pos[:, 0] = np.clip(pos[:, 0] + dx, 0.01, 0.99)
            pos[:, 1] = np.clip(pos[:, 1] + dy, 0.01, 0.99)
        
        # Fix boundaries
        radii = _fix_boundaries(pos, radii)
        
        # Compute new score with smooth penalty
        pen = _smooth_penalty(pos, radii, beta=50.0)
        new_score = _score(pos, radii) - 50.0 * pen
        old_score = _score(best_pos, best_radii) - 50.0 * _smooth_penalty(best_pos, best_radii, beta=50.0)
        
        delta = new_score - old_score
        if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-10)):
            # Accept
            pos = pos.copy()
            radii = radii.copy()
            if pen < 1e-12:
                s = _score(pos, radii)
                if s > best_score:
                    best_score = s
                    best_pos = pos.copy()
                    best_radii = radii.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
        
        T *= cooling
        if T < T_min:
            T = T_min
        
        # Restart if stuck
        if no_improve_count > patience:
            no_improve_count = 0
            T = max(T, 0.01)
    
    return best_pos, best_radii, best_score


def _gradient_polish(pos, radii, time_budget=15.0):
    """Gradient-based polishing with smooth penalty."""
    from scipy.optimize import minimize
    
    pos = pos.copy()
    radii = radii.copy()
    radii = _fix_boundaries(pos, radii)
    radii = _resolve_overlaps(pos, radii)
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = _score(pos, radii)
    
    def objective(x):
        p = x[:2*N].reshape(N, 2)
        r = x[2*N:]
        # Smooth penalty using log-sum-exp
        violations = []
        for i in range(N):
            if p[i, 0] < r[i]:
                violations.append(r[i] - p[i, 0])
            if p[i, 0] > 1.0 - r[i]:
                violations.append(p[i, 0] - (1.0 - r[i]))
            if p[i, 1] < r[i]:
                violations.append(r[i] - p[i, 1])
            if p[i, 1] > 1.0 - r[i]:
                violations.append(p[i, 1] - (1.0 - r[i]))
        for i in range(N):
            for j in range(i + 1, N):
                dx = p[i, 0] - p[j, 0]
                dy = p[i, 1] - p[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_d = r[i] + r[j]
                if dist < min_d:
                    violations.append(min_d - dist)
        if violations:
            v = np.array(violations)
            m = np.max(v)
            if m < 1e-15:
                pen = 0.0
            else:
                lse = m + np.log(np.sum(np.exp(v - m))) / 50.0
                pen = lse + 0.1 * np.sum(v**2)
        else:
            pen = 0.0
        return -np.sum(r) + 80.0 * pen
    
    x0 = np.concatenate([pos.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((0.005, 0.995))
        bounds.append((0.005, 0.995))
    for i in range(N):
        bounds.append((0.005, 0.35))
    
    start = time.time()
    
    try:
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
        if time.time() - start < time_budget:
            x_opt = result.x
            p_opt = x_opt[:2*N].reshape(N, 2)
            r_opt = x_opt[2*N:]
            r_opt = _fix_boundaries(p_opt, r_opt)
            r_opt = _resolve_overlaps(p_opt, r_opt)
            if _is_valid(p_opt, r_opt):
                s = _score(p_opt, r_opt)
                if s > best_score:
                    best_score = s
                    best_pos = p_opt.copy()
                    best_radii = r_opt.copy()
    except Exception:
        pass
    
    return best_pos, best_radii, best_score


def _slsqp_polish(pos, radii, time_budget=10.0):
    """SLSQP polishing with explicit constraints."""
    from scipy.optimize import minimize
    
    pos = pos.copy()
    radii = radii.copy()
    radii = _fix_boundaries(pos, radii)
    radii = _resolve_overlaps(pos, radii)
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = _score(pos, radii)
    
    def objective(x):
        p = x[:2*N].reshape(N, 2)
        r = x[2*N:]
        return -np.sum(r)
    
    def make_constraint(i, j):
        def constraint(x):
            p = x[:2*N].reshape(N, 2)
            r = x[2*N:]
            dx = p[i, 0] - p[j, 0]
            dy = p[i, 1] - p[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            return dist - r[i] - r[j]
        return constraint
    
    # Boundary constraints
    boundary_cons = []
    for i in range(N):
        boundary_cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i] - x[2*N+i]})
        boundary_cons.append({'type': 'ineq', 'fun': lambda x, i=i: (1.0 - x[2*N+i]) - x[2*i]})
        boundary_cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i+1] - x[2*N+i]})
        boundary_cons.append({'type': 'ineq', 'fun': lambda x, i=i: (1.0 - x[2*N+i]) - x[2*i+1]})
    
    # Overlap constraints (limit to nearest pairs for speed)
    overlap_cons = []
    for i in range(N):
        for j in range(i + 1, N):
            overlap_cons.append({'type': 'ineq', 'fun': make_constraint(i, j)})
    
    x0 = np.concatenate([pos.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((0.005, 0.995))
        bounds.append((0.005, 0.995))
    for i in range(N):
        bounds.append((0.005, 0.35))
    
    start = time.time()
    
    try:
        result = minimize(objective, x0, method='SLSQP', bounds=bounds,
                          constraints=boundary_cons + overlap_cons,
                          options={'maxiter': 300, 'ftol': 1e-12})
        if time.time() - start < time_budget:
            x_opt = result.x
            p_opt = x_opt[:2*N].reshape(N, 2)
            r_opt = x_opt[2*N:]
            r_opt = _fix_boundaries(p_opt, r_opt)
            r_opt = _resolve_overlaps(p_opt, r_opt)
            if _is_valid(p_opt, r_opt):
                s = _score(p_opt, r_opt)
                if s > best_score:
                    best_score = s
                    best_pos = p_opt.copy()
                    best_radii = r_opt.copy()
    except Exception:
        pass
    
    return best_pos, best_radii, best_score


def _high_precision_polish(pos, radii, time_budget=15.0):
    """High-precision final polish with very tight tolerances."""
    from scipy.optimize import minimize
    
    pos = pos.copy()
    radii = radii.copy()
    radii = _fix_boundaries(pos, radii)
    radii = _resolve_overlaps(pos, radii)
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = _score(pos, radii)
    
    def objective(x):
        p = x[:2*N].reshape(N, 2)
        r = x[2*N:]
        violations = []
        for i in range(N):
            if p[i, 0] < r[i]:
                violations.append(r[i] - p[i, 0])
            if p[i, 0] > 1.0 - r[i]:
                violations.append(p[i, 0] - (1.0 - r[i]))
            if p[i, 1] < r[i]:
                violations.append(r[i] - p[i, 1])
            if p[i, 1] > 1.0 - r[i]:
                violations.append(p[i, 1] - (1.0 - r[i]))
        for i in range(N):
            for j in range(i + 1, N):
                dx = p[i, 0] - p[j, 0]
                dy = p[i, 1] - p[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_d = r[i] + r[j]
                if dist < min_d:
                    violations.append(min_d - dist)
        if violations:
            v = np.array(violations)
            m = np.max(v)
            if m < 1e-15:
                pen = 0.0
            else:
                lse = m + np.log(np.sum(np.exp(v - m))) / 100.0
                pen = lse + 0.05 * np.sum(v**2)
        else:
            pen = 0.0
        return -np.sum(r) + 200.0 * pen
    
    x0 = np.concatenate([pos.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((0.005, 0.995))
        bounds.append((0.005, 0.995))
    for i in range(N):
        bounds.append((0.005, 0.35))
    
    start = time.time()
    
    try:
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12})
        if time.time() - start < time_budget:
            x_opt = result.x
            p_opt = x_opt[:2*N].reshape(N, 2)
            r_opt = x_opt[2*N:]
            r_opt = _fix_boundaries(p_opt, r_opt)
            r_opt = _resolve_overlaps(p_opt, r_opt)
            if _is_valid(p_opt, r_opt):
                s = _score(p_opt, r_opt)
                if s > best_score:
                    best_score = s
                    best_pos = p_opt.copy()
                    best_radii = r_opt.copy()
    except Exception:
        pass
    
    return best_pos, best_radii, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    best_pos = None
    best_radii = None
    best_score = -np.inf
    
    strategies = [
        ('grid', 42), ('grid', 123),
        ('hexagonal', 42), ('hexagonal', 99),
        ('center_ring', 42), ('center_ring', 77),
        ('random', 42), ('random', 123), ('random', 7),
        ('mixed', 42), ('mixed', 123),
    ]
    
    for strat_name, seed in strategies:
        if time.time() > deadline - 30:
            break
        
        if strat_name == 'grid':
            pos, radii = _make_grid(seed)
        elif strat_name == 'hexagonal':
            pos, radii = _make_hexagonal(seed)
        elif strat_name == 'center_ring':
            pos, radii = _make_center_ring(seed)
        elif strat_name == 'random':
            pos, radii = _make_random(seed)
        elif strat_name == 'mixed':
            pos, radii = _make_mixed(seed)
        else:
            continue
        
        # Force-directed annealing
        remaining = deadline - time.time()
        if remaining < 15:
            break
        anneal_budget = min(25.0, remaining * 0.45)
        pos_a, radii_a, score_a = _force_directed_anneal(pos, radii, time_budget=anneal_budget, seed=seed)
        
        if score_a > best_score:
            best_score = score_a
            best_pos = pos_a.copy()
            best_radii = radii_a.copy()
        
        # Gradient polish on annealed result
        remaining = deadline - time.time()
        if remaining < 20:
            continue
        polish_budget = min(15.0, remaining * 0.35)
        pos_p, radii_p, score_p = _gradient_polish(pos_a, radii_a, time_budget=polish_budget)
        
        if score_p > best_score:
            best_score = score_p
            best_pos = pos_p.copy()
            best_radii = radii_p.copy()
        
        # SLSQP polish
        remaining = deadline - time.time()
        if remaining < 15:
            continue
        slsqp_budget = min(10.0, remaining * 0.3)
        pos_s, radii_s, score_s = _slsqp_polish(pos_p, radii_p, time_budget=slsqp_budget)
        
        if score_s > best_score:
            best_score = score_s
            best_pos = pos_s.copy()
            best_radii = radii_s.copy()
    
    # Final high-precision polish on best result if time allows
    if best_pos is not None:
        remaining = deadline - time.time()
        if remaining > 20:
            pos_f, radii_f, score_f = _high_precision_polish(best_pos, best_radii, time_budget=min(15.0, remaining - 5))
            if score_f > best_score:
                best_score = score_f
                best_pos = pos_f.copy()
                best_radii = radii_f.copy()
        
        # One more gradient polish pass
        remaining = deadline - time.time()
        if remaining > 15:
            pos_g, radii_g, score_g = _gradient_polish(best_pos, best_radii, time_budget=min(10.0, remaining - 5))
            if score_g > best_score:
                best_score = score_g
                best_pos = pos_g.copy()
                best_radii = radii_g.copy()
    
    # Fallback: simple grid
    if best_pos is None:
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 0.005) for i in range(side) for j in range(side)]
        result = np.array(cells[:N])
        return result
    
    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
