# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog, minimize

TOL = 1e-7
DEADLINE_MARGIN = 30.0
N = 10
PAIRS = [(i, j) for i in range(N) for j in range(i + 1, N)]
NPAIRS = len(PAIRS)

def _is_valid(centers, sides, n=N):
    """Check if a packing is valid."""
    if centers.shape != (n, 2) or sides.shape != (n,):
        return False
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    for i in range(n):
        x0 = centers[i, 0] - sides[i] / 2
        x1 = centers[i, 0] + sides[i] / 2
        y0 = centers[i, 1] - sides[i] / 2
        y1 = centers[i, 1] + sides[i] / 2
        if x0 < -TOL - 1e-10 or x1 > 1 + TOL + 1e-10:
            return False
        if y0 < -TOL - 1e-10 or y1 > 1 + TOL + 1e-10:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            t = (sides[i] + sides[j]) / 2 - TOL
            if abs(centers[i, 0] - centers[j, 0]) < t and abs(centers[i, 1] - centers[j, 1]) < t:
                return False
    return True

def _solve_lp(pattern):
    """
    Solve LP for a given separation pattern.
    pattern: array of length NPAIRS, 0 = x-separation, 1 = y-separation
    Variables: s_i, x_i (left), y_i (bottom) for i in range(N)
    Maximize sum(s_i)
    """
    c = np.zeros(3 * N)
    for i in range(N):
        c[i] = -1.0  # maximize sum of s_i

    A_ub_rows = []
    b_ub_vals = []

    # Boundary: x_i + s_i <= 1, y_i + s_i <= 1
    for i in range(N):
        row = np.zeros(3 * N)
        row[i] = 1.0
        row[N + i] = 1.0
        A_ub_rows.append(row)
        b_ub_vals.append(1.0)

        row = np.zeros(3 * N)
        row[i] = 1.0
        row[2 * N + i] = 1.0
        A_ub_rows.append(row)
        b_ub_vals.append(1.0)

    # Separation constraints
    for idx, (i, j) in enumerate(PAIRS):
        if pattern[idx] == 0:
            # i left of j: s_i + x_i - x_j <= 0
            row = np.zeros(3 * N)
            row[i] = 1.0
            row[N + i] = 1.0
            row[N + j] = -1.0
        else:
            # i below j: s_i + y_i - y_j <= 0
            row = np.zeros(3 * N)
            row[i] = 1.0
            row[2 * N + i] = 1.0
            row[2 * N + j] = -1.0
        A_ub_rows.append(row)
        b_ub_vals.append(0.0)

    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_vals)
    bounds = [(0.0, 1.0)] * (3 * N)

    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if result.success:
            s = result.x[:N]
            x = result.x[N:2*N]
            y = result.x[2*N:3*N]
            centers = np.column_stack([x + s / 2, y + s / 2])
            return centers, s, np.sum(s)
    except:
        pass
    return None, None, -np.inf

def _pattern_fitness(pattern):
    """
    Fitness for a pattern: solve LP, return negative of sum (for minimization).
    If LP fails, return a large penalty.
    """
    centers, sides, val = _solve_lp(pattern)
    if centers is None:
        return 100.0  # infeasible
    if not _is_valid(centers, sides):
        return val + 10.0  # add penalty for invalid
    return -val  # we want to maximize val, so minimize -val

def _generate_structured_patterns():
    """Generate a diverse set of structured patterns."""
    patterns = []

    # All x
    patterns.append(np.zeros(NPAIRS, dtype=int))
    # All y
    patterns.append(np.ones(NPAIRS, dtype=int))

    # Grid patterns for various column counts
    for cols in range(1, N + 1):
        pat = np.zeros(NPAIRS, dtype=int)
        for idx, (i, j) in enumerate(PAIRS):
            ri_i, ci_i = divmod(i, cols)
            ri_j, ci_j = divmod(j, cols)
            if ri_i == ri_j:
                pat[idx] = 0  # x-separation (same row)
            else:
                pat[idx] = 1  # y-separation (different rows)
        patterns.append(pat)

    # "Checkerboard" patterns
    for period in [2, 3, 4, 5]:
        pat = np.zeros(NPAIRS, dtype=int)
        for idx, (i, j) in enumerate(PAIRS):
            if (i + j) % period == 0:
                pat[idx] = 1
        patterns.append(pat)

    # Radial patterns: squares separated by angle from center
    for angle_step in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
        pat = np.zeros(NPAIRS, dtype=int)
        angles = [i * angle_step * np.pi / 180.0 for i in range(N)]
        for idx, (i, j) in enumerate(PAIRS):
            # Use the dominant axis of the vector between them
            dx = np.cos(angles[i]) - np.cos(angles[j])
            dy = np.sin(angles[i]) - np.sin(angles[j])
            if abs(dx) >= abs(dy):
                pat[idx] = 0
            else:
                pat[idx] = 1
        patterns.append(pat)

    return patterns

def _ga_search_patterns(deadline, initial_patterns, pop_size=60, generations=50):
    """
    Genetic algorithm to search over separation patterns.
    """
    rng = np.random.RandomState(42)

    # Initialize population
    population = []
    for pat in initial_patterns[:pop_size]:
        population.append(pat.copy())
    while len(population) < pop_size:
        population.append(rng.randint(0, 2, size=NPAIRS).astype(int))

    # Evaluate initial population
    fitnesses = [_pattern_fitness(p) for p in population]
    best_idx = np.argmin(fitnesses)
    best_pattern = population[best_idx].copy()
    best_fitness = fitnesses[best_idx]

    for gen in range(generations):
        if time.time() > deadline:
            break

        # Sort by fitness
        order = np.argsort(fitnesses)
        population = [population[i] for i in order]
        fitnesses = [fitnesses[i] for i in order]

        # Elitism: keep top 10%
        n_elite = max(1, pop_size // 10)
        new_population = [population[i].copy() for i in range(n_elite)]

        # Fill rest with offspring
        while len(new_population) < pop_size:
            if time.time() > deadline:
                break
            # Tournament selection
            t1 = rng.randint(0, min(5, len(population)))
            t2 = rng.randint(0, min(5, len(population)))
            p1 = population[t1] if fitnesses[t1] <= fitnesses[t2] else population[t2]

            t1 = rng.randint(0, min(5, len(population)))
            t2 = rng.randint(0, min(5, len(population)))
            p2 = population[t1] if fitnesses[t1] <= fitnesses[t2] else population[t2]

            # Crossover: uniform
            child = p1.copy()
            for k in range(NPAIRS):
                if rng.random() < 0.5:
                    child[k] = p2[k]

            # Mutation: flip bits
            for k in range(NPAIRS):
                if rng.random() < 0.1:
                    child[k] = 1 - child[k]

            new_population.append(child)

        population = new_population
        fitnesses = [_pattern_fitness(p) for p in population]

        gen_best_idx = np.argmin(fitnesses)
        if fitnesses[gen_best_idx] < best_fitness:
            best_fitness = fitnesses[gen_best_idx]
            best_pattern = population[gen_best_idx].copy()

    return best_pattern, -best_fitness  # return pattern and the actual value

def _slsqp_polish(centers, sides, n=N, deadline=None):
    """
    NLP polish using SLSQP with smooth non-overlap constraints.
    """
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[i] = sides[i]
        x0[n + i] = centers[i, 0] - sides[i] / 2
        x0[2 * n + i] = centers[i, 1] - sides[i] / 2

    def objective(v):
        return -np.sum(v[:n])

    def objective_grad(v):
        g = np.zeros(3 * n)
        g[:n] = -1.0
        return g

    constraints = []

    # Boundary constraints
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[n + i],
                           'jac': lambda v, i=i: _jac_xi(i, n)})
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[2 * n + i],
                           'jac': lambda v, i=i: _jac_yi(i, n)})
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[n + i] - v[i],
                           'jac': lambda v, i=i: _jac_xi_plus_s(i, n)})
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[2 * n + i] - v[i],
                           'jac': lambda v, i=i: _jac_yi_plus_s(i, n)})
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[i],
                           'jac': lambda v, i=i: _jac_si(i, n)})

    # Non-overlap constraints (smooth)
    for i in range(n):
        for j in range(i + 1, n):
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i, j=j: _overlap_fun(v, i, j, n),
                               'jac': lambda v, i=i, j=j: _overlap_jac(v, i, j, n)})

    bounds = [(1e-8, 1.0)] * n + [(0.0, 1.0)] * n + [(0.0, 1.0)] * n

    best_val = -np.inf
    best_centers = centers.copy()
    best_sides = sides.copy()

    try:
        if deadline is not None and time.time() > deadline:
            return best_centers, best_sides, best_val

        result = minimize(objective, x0, method='SLSQP', jac=objective_grad,
                          bounds=bounds, constraints=constraints,
                          options={'maxiter': 300, 'ftol': 1e-12})
        if result.success:
            s = result.x[:n]
            x = result.x[n:2 * n]
            y = result.x[2 * n:3 * n]
            c = np.column_stack([x + s / 2, y + s / 2])
            if _is_valid(c, s, n):
                val = np.sum(s)
                if val > best_val:
                    best_val = val
                    best_centers = c
                    best_sides = s
    except:
        pass

    # Try from perturbed starting points
    if deadline is not None and time.time() > deadline - 10:
        return best_centers, best_sides, best_val

    rng = np.random.RandomState(123)
    for trial in range(3):
        if deadline is not None and time.time() > deadline:
            break
        x0_p = x0.copy()
        x0_p[:n] *= (1.0 + 0.05 * rng.randn(n))
        x0_p[n:2 * n] += 0.02 * rng.randn(n)
        x0_p[2 * n:3 * n] += 0.02 * rng.randn(n)
        x0_p = np.clip(x0_p, 1e-8, 1.0)

        try:
            result = minimize(objective, x0_p, method='SLSQP', jac=objective_grad,
                              bounds=bounds, constraints=constraints,
                              options={'maxiter': 200, 'ftol': 1e-12})
            if result.success:
                s = result.x[:n]
                x = result.x[n:2 * n]
                y = result.x[2 * n:3 * n]
                c = np.column_stack([x + s / 2, y + s / 2])
                if _is_valid(c, s, n):
                    val = np.sum(s)
                    if val > best_val:
                        best_val = val
                        best_centers = c
                        best_sides = s
        except:
            pass

    return best_centers, best_sides, best_val

def _jac_xi(i, n):
    g = np.zeros(3 * n)
    g[n + i] = 1.0
    return g

def _jac_yi(i, n):
    g = np.zeros(3 * n)
    g[2 * n + i] = 1.0
    return g

def _jac_si(i, n):
    g = np.zeros(3 * n)
    g[i] = 1.0
    return g

def _jac_xi_plus_s(i, n):
    g = np.zeros(3 * n)
    g[i] = -1.0
    g[n + i] = -1.0
    return g

def _jac_yi_plus_s(i, n):
    g = np.zeros(3 * n)
    g[i] = -1.0
    g[2 * n + i] = -1.0
    return g

def _overlap_fun(v, i, j, n):
    cx_i = v[n + i] + v[i] / 2.0
    cx_j = v[n + j] + v[j] / 2.0
    cy_i = v[2 * n + i] + v[i] / 2.0
    cy_j = v[2 * n + j] + v[j] / 2.0
    dx = cx_i - cx_j
    dy = cy_i - cy_j
    r = (v[i] + v[j]) / 2.0
    return dx * dx + dy * dy - r * r

def _overlap_jac(v, i, j, n):
    cx_i = v[n + i] + v[i] / 2.0
    cx_j = v[n + j] + v[j] / 2.0
    cy_i = v[2 * n + i] + v[i] / 2.0
    cy_j = v[2 * n + j] + v[j] / 2.0
    dx = cx_i - cx_j
    dy = cy_i - cy_j
    r = (v[i] + v[j]) / 2.0

    grad = np.zeros(3 * n)
    grad[i] = dx * 0.5 + dy * 0.5 - r
    grad[j] = -dx * 0.5 - dy * 0.5 - r
    grad[n + i] = dx
    grad[n + j] = -dx
    grad[2 * n + i] = dy
    grad[2 * n + j] = -dy
    return grad

def _local_polish(centers, sides, n=N, max_iter=100, deadline=None):
    """Local improvement by trying to increase each side."""
    best_val = np.sum(sides)
    centers = centers.copy()
    sides = sides.copy()

    for iteration in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(n):
            for delta in [0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00001, 0.000005]:
                new_sides = sides.copy()
                new_sides[i] += delta
                if new_sides[i] <= 0 or new_sides[i] > 1:
                    continue
                if _is_valid(centers, new_sides, n):
                    sides = new_sides
                    best_val = np.sum(sides)
                    improved = True
                    break
        if not improved:
            break

    return centers, sides, best_val

def squares_in_square10() -> tuple:
    n = N
    start_time = time.time()
    deadline = start_time + 630 - DEADLINE_MARGIN

    # Baseline grid packing
    k = int(np.ceil(np.sqrt(n)))
    side = 1.0 / k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx = (cols + 0.5) * side
    cy = (rows + 0.5) * side
    best_centers = np.stack([cx, cy], axis=1).astype(np.float64)
    best_sides = np.full(n, side, dtype=np.float64)
    best_val = np.sum(best_sides)

    # Phase 1: Evaluate structured patterns
    structured = _generate_structured_patterns()
    for pat in structured:
        if time.time() > deadline:
            break
        centers_p, sides_p, val_p = _solve_lp(pat)
        if centers_p is not None and _is_valid(centers_p, sides_p, n):
            if val_p > best_val:
                best_val = val_p
                best_centers = centers_p.copy()
                best_sides = sides_p.copy()

    # Phase 2: GA search for better patterns
    if time.time() < deadline - 60:
        ga_pat, ga_val = _ga_search_patterns(deadline - 60, structured, pop_size=50, generations=40)
        if ga_val > best_val:
            centers_p, sides_p, val_p = _solve_lp(ga_pat)
            if centers_p is not None and _is_valid(centers_p, sides_p, n):
                best_val = val_p
                best_centers = centers_p.copy()
                best_sides = sides_p.copy()

    # Phase 3: SLSQP polish from best solution
    if time.time() < deadline - 30:
        centers_s, sides_s, val_s = _slsqp_polish(best_centers, best_sides, n, deadline=deadline - 10)
        if val_s > best_val:
            best_val = val_s
            best_centers = centers_s.copy()
            best_sides = sides_s.copy()

    # Phase 4: Local polish
    if time.time() < deadline - 5:
        centers_l, sides_l, val_l = _local_polish(best_centers, best_sides, n, max_iter=100, deadline=deadline)
        if val_l > best_val:
            best_val = val_l
            best_centers = centers_l.copy()
            best_sides = sides_l.copy()

    # Phase 5: Try SLSQP from a few other good LP solutions
    if time.time() < deadline - 15:
        for pat in structured[:15]:
            if time.time() > deadline - 5:
                break
            centers_p, sides_p, val_p = _solve_lp(pat)
            if centers_p is not None and _is_valid(centers_p, sides_p, n):
                if val_p >= best_val - 0.03:
                    centers_s, sides_s, val_s = _slsqp_polish(centers_p, sides_p, n, deadline=deadline - 5)
                    if val_s > best_val:
                        best_val = val_s
                        best_centers = centers_s.copy()
                        best_sides = sides_s.copy()

    # Final validation
    if not _is_valid(best_centers, best_sides, n):
        k = int(np.ceil(np.sqrt(n)))
        side = 1.0 / k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        best_centers = np.stack([cx, cy], axis=1).astype(np.float64)
        best_sides = np.full(n, side, dtype=np.float64)

    return best_centers, best_sides

# EVOLVE-BLOCK-END
