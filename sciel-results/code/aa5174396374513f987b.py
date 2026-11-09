"""Broadband antireflection coating design: global + local optimization.

Combines the fast characteristic-matrix reflectance model with:
  1. Diverse candidate generation (symmetric, graded, random, quarter-wave).
  2. Differential evolution for global search on thicknesses for fixed material sequences.
  3. Nelder-Mead for local polish.
  4. Simulated annealing for material/thickness joint search.
"""
import time
import numpy as np


def _reflectance(materials, thicknesses, problem):
    """Compute mean reflectance (%) over the band using characteristic matrix method."""
    if not materials:
        n0 = problem["ambient_index"]
        ns = problem["substrate_index"]
        return 100.0 * ((ns - n0) / (ns + n0)) ** 2

    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    band = problem["band_nm"]
    n_wl = 401
    wl = np.linspace(band[0], band[1], n_wl)

    n_mat = np.array([problem["materials"][m] for m in materials])
    d = np.array(thicknesses, dtype=float)
    N = len(materials)

    # Phase delta for each layer at each wavelength
    delta = 2.0 * np.pi * n_mat[:, None] * d[:, None] / wl[None, :]

    # Characteristic matrix product
    A = np.ones(n_wl, dtype=complex)
    B = np.zeros(n_wl, dtype=complex)
    C = np.zeros(n_wl, dtype=complex)
    D = np.ones(n_wl, dtype=complex)

    for i in range(N):
        c = np.cos(delta[i])
        s = np.sin(delta[i])
        n_i = n_mat[i]

        A_new = A * c + 1j * B * n_i * s
        B_new = 1j * A * s / n_i + B * c
        C_new = C * c + 1j * D * n_i * s
        D_new = 1j * C * s / n_i + D * c
        A, B, C, D = A_new, B_new, C_new, D_new

    # Reflectance
    num = n0 * (A + B * ns) - (C + D * ns)
    den = n0 * (A + B * ns) + (C + D * ns)
    r = num / den
    R = np.abs(r) ** 2
    return np.mean(R) * 100.0


def _valid(materials, thicknesses, problem):
    """Check if the design is valid."""
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]

    if len(materials) > max_layers:
        return False
    if len(materials) != len(thicknesses):
        return False
    for t in thicknesses:
        if t < 0 or t > max_layer_nm:
            return False
    if sum(thicknesses) > max_total_nm:
        return False
    return True


def _clamp_thicknesses(thicknesses, problem):
    """Clamp thicknesses to valid range and ensure total doesn't exceed max."""
    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem["max_total_nm"]

    clamped = [max(0.0, min(max_layer_nm, t)) for t in thicknesses]

    total = sum(clamped)
    if total > max_total_nm:
        scale = max_total_nm / total
        clamped = [t * scale for t in clamped]

    return clamped


def _random_stack(problem, n_layers=None):
    """Generate a random valid stack."""
    if n_layers is None:
        n_layers = np.random.randint(2, problem["max_layers"] + 1)

    materials = []
    thicknesses = []

    last = None
    for _ in range(n_layers):
        if last is None:
            m = np.random.choice(['L', 'H'])
        else:
            if np.random.random() < 0.7:
                m = 'H' if last == 'L' else 'L'
            else:
                m = np.random.choice(['L', 'H'])
        materials.append(m)
        thicknesses.append(np.random.uniform(10, problem["max_layer_nm"]))
        last = m

    thicknesses = _clamp_thicknesses(thicknesses, problem)
    return materials, thicknesses


def _nelder_mead(objective, x0, bounds, maxiter=200, timeout=None, start_time=None):
    """Nelder-Mead simplex optimization with bounds."""
    if start_time is None:
        start_time = time.time()

    n = len(x0)
    x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])

    # Initial simplex
    simplex = [x0.copy()]
    for i in range(n):
        xi = x0.copy()
        step = 0.05 * (bounds[i][1] - bounds[i][0]) if bounds[i][1] - bounds[i][0] > 1 else 0.05
        xi[i] += step
        xi[i] = np.clip(xi[i], bounds[i][0], bounds[i][1])
        simplex.append(xi)

    fvals = np.array([objective(simplex[i]) for i in range(n + 1)])
    best_idx = np.argmin(fvals)
    fbest = fvals[best_idx]
    xbest = simplex[best_idx].copy()

    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5

    for _ in range(maxiter):
        if timeout and time.time() - start_time > timeout:
            break

        # Sort simplex
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = fvals[order]

        if fvals[0] < fbest:
            fbest = fvals[0]
            xbest = simplex[0].copy()

        # Check convergence
        if np.allclose(fvals, fvals[0], rtol=1e-6, atol=1e-6):
            break

        # Centroid of all points except worst
        centroid = np.mean(simplex[:-1], axis=0)
        worst = simplex[-1]

        # Reflection
        x_r = centroid + alpha * (centroid - worst)
        x_r = np.clip(x_r, [b[0] for b in bounds], [b[1] for b in bounds])
        f_r = objective(x_r)

        if f_r < fvals[0]:
            # Expansion
            x_e = centroid + gamma * (centroid - worst)
            x_e = np.clip(x_e, [b[0] for b in bounds], [b[1] for b in bounds])
            f_e = objective(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                fvals[-1] = f_e
            else:
                simplex[-1] = x_r
                fvals[-1] = f_r
        elif f_r < fvals[-2]:
            simplex[-1] = x_r
            fvals[-1] = f_r
        else:
            # Contraction
            x_c = centroid + rho * (worst - centroid)
            x_c = np.clip(x_c, [b[0] for b in bounds], [b[1] for b in bounds])
            f_c = objective(x_c)
            if f_c < fvals[-1]:
                simplex[-1] = x_c
                fvals[-1] = f_c
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    simplex[i] = np.clip(simplex[i], [b[0] for b in bounds], [b[1] for b in bounds])
                    fvals[i] = objective(simplex[i])

    return xbest, fbest


def _differential_evolution(objective, bounds, pop_size=20, max_iters=50, F=0.8, CR=0.7, deadline=None):
    """Differential evolution for global search."""
    n = len(bounds)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])

    # Initialize population
    pop = np.random.uniform(lo, hi, size=(pop_size, n))
    fitness = np.array([objective(pop[i]) for i in range(pop_size)])
    best_idx = np.argmin(fitness)
    xbest = pop[best_idx].copy()
    fbest = fitness[best_idx]

    for gen in range(max_iters):
        if deadline and time.time() > deadline:
            break

        for i in range(pop_size):
            if deadline and time.time() > deadline:
                break

            # Select three distinct random individuals
            candidates = [j for j in range(pop_size) if j != i]
            a, b, c = np.random.choice(candidates, size=3, replace=False)

            # Mutation
            v = pop[a] + F * (pop[b] - pop[c])
            v = np.clip(v, lo, hi)

            # Crossover
            trial = pop[i].copy()
            for j in range(n):
                if np.random.random() < CR:
                    trial[j] = v[j]
                else:
                    trial[j] = pop[i][j]
            trial = np.clip(trial, lo, hi)

            # Selection
            ft = objective(trial)
            if ft < fitness[i]:
                pop[i] = trial
                fitness[i] = ft
                if ft < fbest:
                    fbest = ft
                    xbest = trial.copy()

    return xbest, fbest


def _optimize_local(materials, thicknesses, problem, deadline):
    """Run local optimization on a given stack."""
    max_layer_nm = problem["max_layer_nm"]

    start_time = time.time()
    timeout = min(10.0, max(0.1, deadline - start_time))

    def objective(x):
        if time.time() > deadline:
            return 100.0
        t = _clamp_thicknesses(x, problem)
        if not _valid(materials, t, problem):
            return 100.0
        return _reflectance(materials, t, problem)

    x0 = np.array(thicknesses, dtype=float)
    bounds = [(0.0, max_layer_nm)] * len(x0)

    try:
        x_opt, f_opt = _nelder_mead(objective, x0, bounds, maxiter=100, timeout=timeout, start_time=start_time)
        t_opt = _clamp_thicknesses(x_opt, problem)
        if _valid(materials, t_opt, problem):
            f_opt_check = _reflectance(materials, t_opt, problem)
            if f_opt_check < objective(x0):
                return materials, t_opt, f_opt_check
    except Exception:
        pass

    return materials, thicknesses, objective(x0)


def _optimize_de(materials, thicknesses, problem, deadline, pop_size=15, max_iters=30):
    """Run differential evolution on thicknesses for a given material sequence."""
    max_layer_nm = problem["max_layer_nm"]

    start_time = time.time()
    timeout = min(15.0, max(0.1, deadline - start_time))

    def objective(x):
        if time.time() > deadline:
            return 100.0
        t = _clamp_thicknesses(x, problem)
        if not _valid(materials, t, problem):
            return 100.0
        return _reflectance(materials, t, problem)

    x0 = np.array(thicknesses, dtype=float)
    bounds = [(0.0, max_layer_nm)] * len(x0)

    try:
        x_opt, f_opt = _differential_evolution(objective, bounds, pop_size=pop_size, max_iters=max_iters, deadline=deadline)
        t_opt = _clamp_thicknesses(x_opt, problem)
        if _valid(materials, t_opt, problem):
            f_opt_check = _reflectance(materials, t_opt, problem)
            return materials, t_opt, f_opt_check
    except Exception:
        pass

    return materials, thicknesses, objective(x0)


def _simulated_annealing(problem, deadline, initial_materials=None, initial_thicknesses=None):
    """Simulated annealing to find good thicknesses and materials."""
    max_layers = problem["max_layers"]
    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem["max_total_nm"]

    if initial_materials is None or initial_thicknesses is None:
        materials, thicknesses = _random_stack(problem)
    else:
        materials = list(initial_materials)
        thicknesses = list(initial_thicknesses)

    current_score = _reflectance(materials, thicknesses, problem)
    best_materials = list(materials)
    best_thicknesses = list(thicknesses)
    best_score = current_score

    T = 5.0
    cooling = 0.997

    while time.time() < deadline:
        T *= cooling
        if T < 0.005:
            break

        new_materials = list(materials)
        new_thicknesses = list(thicknesses)

        action = np.random.random()
        if action < 0.55:
            idx = np.random.randint(len(new_thicknesses))
            delta = np.random.normal(0, 25)
            new_thicknesses[idx] = max(0, min(max_layer_nm, new_thicknesses[idx] + delta))
        elif action < 0.75:
            idx = np.random.randint(len(new_materials))
            new_materials[idx] = 'H' if new_materials[idx] == 'L' else 'L'
        elif action < 0.88:
            if len(new_materials) < max_layers:
                pos = np.random.randint(len(new_materials) + 1)
                m = 'L' if np.random.random() < 0.5 else 'H'
                new_materials.insert(pos, m)
                new_thicknesses.insert(pos, np.random.uniform(20, 100))
        else:
            if len(new_materials) > 2:
                pos = np.random.randint(len(new_materials))
                new_materials.pop(pos)
                new_thicknesses.pop(pos)

        new_thicknesses = _clamp_thicknesses(new_thicknesses, problem)
        if not _valid(new_materials, new_thicknesses, problem):
            continue

        new_score = _reflectance(new_materials, new_thicknesses, problem)

        if new_score < current_score or np.random.random() < np.exp(-(new_score - current_score) / T):
            materials = new_materials
            thicknesses = new_thicknesses
            current_score = new_score

            if new_score < best_score:
                best_materials = list(materials)
                best_thicknesses = list(thicknesses)
                best_score = new_score

    return best_materials, best_thicknesses, best_score


def design(problem):
    """Design a broadband antireflection coating stack."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    band = problem["band_nm"]
    centre = 0.5 * (band[0] + band[1])
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]

    # Quarter-wave thicknesses at center wavelength
    qwL = centre / (4.0 * nL)
    qwH = centre / (4.0 * nH)

    # Generate diverse candidate patterns
    candidates = []

    # Quarter-wave alternating patterns
    patterns = [
        (['L', 'H', 'L', 'H'], [qwL, qwH, qwL, qwH]),
        (['H', 'L', 'H', 'L'], [qwH, qwL, qwH, qwL]),
        (['L', 'H', 'L', 'H', 'L'], [qwL, qwH, qwL, qwH, qwL]),
        (['H', 'L', 'H', 'L', 'H'], [qwH, qwL, qwH, qwL, qwH]),
        (['L', 'H', 'L', 'H', 'L', 'H'], [qwL, qwH, qwL, qwH, qwL, qwH]),
        (['H', 'L', 'H', 'L', 'H', 'L'], [qwH, qwL, qwH, qwL, qwH, qwL]),
        (['L', 'H', 'L', 'H', 'L', 'H', 'L'], [qwL, qwH, qwL, qwH, qwL, qwH, qwL]),
        (['H', 'L', 'H', 'L', 'H', 'L', 'H'], [qwH, qwL, qwH, qwL, qwH, qwL, qwH]),
    ]

    for mats, thicks in patterns:
        thicks = _clamp_thicknesses(thicks, problem)
        if _valid(mats, thicks, problem):
            score = _reflectance(mats, thicks, problem)
            candidates.append((score, mats, thicks))

    # Symmetric patterns
    for length in range(3, min(max_layers, 11) + 1, 2):
        half = (length + 1) // 2
        for _ in range(10):
            half_pattern = tuple(np.random.choice(['L', 'H']) for _ in range(half))
            pattern = list(half_pattern) + list(half_pattern[-2::-1])
            thicks = [np.random.uniform(30, 150) for _ in range(length)]
            thicks = _clamp_thicknesses(thicks, problem)
            if _valid(pattern, thicks, problem):
                score = _reflectance(pattern, thicks, problem)
                candidates.append((score, pattern, thicks))

    # Random initializations
    for _ in range(10):
        if time.time() > deadline:
            break
        mats, thicks = _random_stack(problem)
        score = _reflectance(mats, thicks, problem)
        candidates.append((score, mats, thicks))

    # Sort by initial score
    candidates.sort(key=lambda x: x[0])

    best_materials = candidates[0][1] if candidates else ['L']
    best_thicknesses = candidates[0][2] if candidates else [qwL]
    best_score = candidates[0][0] if candidates else 100.0

    # Global optimization (DE) on top candidates
    # We prioritize candidates with fewer layers first to save time, but also check diverse ones
    # Let's take top 10 candidates
    for score, mats, thicks in candidates[:10]:
        if time.time() > deadline - 30:
            break
        opt_mats, opt_thicks, opt_score = _optimize_de(mats, thicks, problem, deadline, pop_size=15, max_iters=30)
        if opt_score < best_score:
            best_materials = opt_mats
            best_thicknesses = opt_thicks
            best_score = opt_score

    # Local optimization (Nelder-Mead) on top candidates
    for score, mats, thicks in candidates[:15]:
        if time.time() > deadline - 20:
            break
        opt_mats, opt_thicks, opt_score = _optimize_local(mats, thicks, problem, deadline)
        if opt_score < best_score:
            best_materials = opt_mats
            best_thicknesses = opt_thicks
            best_score = opt_score

    # Simulated annealing with remaining time
    if time.time() < deadline - 15:
        sa_mats, sa_thicks, sa_score = _simulated_annealing(
            problem, deadline, best_materials, best_thicknesses)
        if sa_score < best_score:
            best_materials = sa_mats
            best_thicknesses = sa_thicks
            best_score = sa_score

        # Local polish after SA
        if time.time() < deadline - 5:
            opt_mats, opt_thicks, opt_score = _optimize_local(
                best_materials, best_thicknesses, problem, deadline)
            if opt_score < best_score:
                best_materials = opt_mats
                best_thicknesses = opt_thicks
                best_score = opt_score

    # Final validation
    best_thicknesses = _clamp_thicknesses(best_thicknesses, problem)
    if not _valid(best_materials, best_thicknesses, problem):
        best_materials = ['L']
        best_thicknesses = [qwL]

    return best_materials, best_thicknesses
