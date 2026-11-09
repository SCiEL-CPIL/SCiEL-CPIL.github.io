"""Broadband antireflection coating design: quarter-wave init + fast DE + Nelder-Mead."""
# EVOLVE-BLOCK-START
import time
import numpy as np

# Precompute evaluation wavelengths
_WL = np.linspace(400.0, 800.0, 401)
_K = 2.0 * np.pi / _WL  # 1/nm, shape (401,)

N_AMB = 1.0
N_SUB = 1.52
N_L = 1.38
N_H = 2.35
MAX_LAYERS = 20
MAX_TOTAL_NM = 1000.0
MAX_LAYER_NM = 400.0


def _reflectance(labels, thicknesses):
    """Vectorized mean reflectance (%) over 401 wavelengths using characteristic matrix."""
    n = len(labels)
    if n == 0:
        r0 = (N_SUB - N_AMB) / (N_SUB + N_AMB)
        return 100.0 * (r0 * r0)

    N = len(_WL)
    M00 = np.ones(N, dtype=np.complex128)
    M01 = np.zeros(N, dtype=np.complex128)
    M10 = np.zeros(N, dtype=np.complex128)
    M11 = np.ones(N, dtype=np.complex128)

    for j in range(n):
        n_j = N_L if labels[j] == 'L' else N_H
        d_j = thicknesses[j]
        delta = _K * n_j * d_j
        cos_d = np.cos(delta)
        sin_d = np.sin(delta)

        L00 = cos_d
        L01 = 1j * sin_d / n_j
        L10 = 1j * n_j * sin_d
        L11 = cos_d

        new_M00 = M00 * L00 + M01 * L10
        new_M01 = M00 * L01 + M01 * L11
        new_M10 = M10 * L00 + M11 * L10
        new_M11 = M10 * L01 + M11 * L11
        M00, M01, M10, M11 = new_M00, new_M01, new_M10, new_M11

    Y = M01 / M00
    R = np.abs((N_AMB - Y) / (N_AMB + Y)) ** 2
    return float(np.mean(R) * 100.0)


def _valid(labels, thicknesses):
    if len(labels) != len(thicknesses):
        return False
    if len(labels) > MAX_LAYERS or len(labels) == 0:
        return False
    total = 0.0
    for t in thicknesses:
        if t < -1e-9 or t > MAX_LAYER_NM + 1e-9:
            return False
        total += t
    if total > MAX_TOTAL_NM + 1e-9:
        return False
    for m in labels:
        if m not in ('L', 'H'):
            return False
    return True


def _clamp_thicknesses(thicknesses):
    return [min(max(t, 0.0), MAX_LAYER_NM) for t in thicknesses]


def _quarter_wave_init(labels, center_wl=600.0):
    """Initialize thicknesses as quarter-wave at center wavelength."""
    n = len(labels)
    thicks = np.zeros(n)
    for j in range(n):
        n_j = N_L if labels[j] == 'L' else N_H
        thicks[j] = center_wl / (4.0 * n_j)
    # Scale to fit within total constraint if needed
    total = np.sum(thicks)
    if total > MAX_TOTAL_NM:
        thicks *= MAX_TOTAL_NM / total
    return np.clip(thicks, 0, MAX_LAYER_NM)


def _fast_de(objective, x0, bounds, pop_size=20, max_iter=30, F=0.8, CR=0.9, seed=42):
    """Fast DE with vectorized crossover and fewer function evals."""
    rng = np.random.RandomState(seed)
    n = len(bounds)
    lb = np.array(bounds[:, 0], dtype=float)
    ub = np.array(bounds[:, 1], dtype=float)

    # Initialize population with x0 as best initial guess
    pop = np.zeros((pop_size, n))
    pop[0] = x0
    for i in range(1, pop_size):
        pop[i] = rng.uniform(lb, ub)

    fitness = np.array([objective(p) for p in pop])
    best_idx = np.argmin(fitness)
    best_x = pop[best_idx].copy()
    best_f = fitness[best_idx]

    for it in range(max_iter):
        order = np.argsort(fitness)
        pop = pop[order]
        fitness = fitness[order]
        best_x = pop[0].copy()
        best_f = fitness[0]

        # Vectorized mutation and crossover
        new_pop = pop.copy()
        new_fitness = fitness.copy()

        for i in range(1, pop_size):
            # Select three distinct random individuals (vectorized selection)
            candidates = np.arange(1, pop_size)
            a, b, c = rng.choice(candidates, size=3, replace=False)
            
            # Mutation
            v = pop[a] + F * (pop[b] - pop[c])
            v = np.clip(v, lb, ub)

            # Vectorized crossover
            mask = rng.random(n) < CR
            mask[rng.randint(n)] = True  # Ensure at least one gene from v
            trial = np.where(mask, v, pop[i])

            # Selection
            f_trial = objective(trial)
            if f_trial < fitness[i]:
                new_pop[i] = trial
                new_fitness[i] = f_trial

        pop = new_pop
        fitness = new_fitness

    return best_x, best_f


def _nelder_mead(objective, x0, max_iter=300, tol=1e-7, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5):
    """Nelder-Mead with tighter tolerance for better convergence."""
    n = len(x0)
    x = np.array(x0, dtype=float)
    simplex = [x.copy()]
    for i in range(n):
        xi = x.copy()
        step = 0.1 * abs(xi[i]) if abs(xi[i]) > 1e-6 else 0.1
        xi[i] += step
        simplex.append(xi)
    fvals = [objective(s) for s in simplex]

    for it in range(max_iter):
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]

        if abs(fvals[-1] - fvals[0]) < tol:
            break

        centroid = np.mean(simplex[:-1], axis=0)

        xr = centroid + alpha * (centroid - simplex[-1])
        fr = objective(xr)
        if fr < fvals[0]:
            xe = centroid + gamma * (centroid - simplex[-1])
            fe = objective(xe)
            if fe < fr:
                simplex[-1] = xe
                fvals[-1] = fe
            else:
                simplex[-1] = xr
                fvals[-1] = fr
        elif fr < fvals[-2]:
            simplex[-1] = xr
            fvals[-1] = fr
        else:
            xc = centroid + rho * (simplex[-1] - centroid)
            fc = objective(xc)
            if fc < fvals[-1]:
                simplex[-1] = xc
                fvals[-1] = fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    fvals[i] = objective(simplex[i])
    return simplex[0], fvals[0]


def _make_objective(labels):
    """Create objective function for given layer labels."""
    def objective(x):
        thicks = list(x)
        if any(t < -1e-9 or t > MAX_LAYER_NM + 1e-9 for t in thicks):
            return 100.0
        if sum(thicks) > MAX_TOTAL_NM + 1e-9:
            return 100.0
        return _reflectance(labels, thicks)
    return objective


def _optimize_stack(labels, time_budget, seed=42):
    """Optimize thicknesses for given labels using quarter-wave init + fast DE + Nelder-Mead."""
    deadline = time.time() + time_budget
    n = len(labels)
    if n > 16:
        return 100.0, None

    objective = _make_objective(labels)

    # Bounds: each thickness in [0, MAX_LAYER_NM]
    bounds = np.zeros((n, 2))
    bounds[:, 0] = 0.0
    bounds[:, 1] = MAX_LAYER_NM

    # Start with quarter-wave initialization
    x0 = _quarter_wave_init(labels)
    f0 = objective(x0)

    # Phase 1: Fast DE from quarter-wave init
    remaining = deadline - time.time()
    if remaining > 5.0:
        pop_size = min(20, max(8, int(remaining / 10.0)))
        max_iter_de = min(25, max(5, int(remaining / 20.0)))
        
        x_de, f_de = _fast_de(objective, x0, bounds, pop_size=pop_size, max_iter=max_iter_de, seed=seed)
        
        if f_de < f0:
            x_best = x_de
            f_best = f_de
        else:
            x_best = x0
            f_best = f0
    else:
        x_best = x0
        f_best = f0

    # Phase 2: Nelder-Mead refinement (most important phase)
    remaining = deadline - time.time()
    if remaining > 2.0:
        max_iter_nm = min(500, max(50, int(remaining / 0.05)))
        x_nm, f_nm = _nelder_mead(objective, x_best, max_iter=max_iter_nm)
        
        if f_nm < f_best:
            x_best = x_nm
            f_best = f_nm

    thicks = _clamp_thicknesses(list(x_best))
    total = sum(thicks)
    if total > MAX_TOTAL_NM:
        scale = MAX_TOTAL_NM / total
        thicks = [t * scale for t in thicks]

    if _valid(labels, thicks):
        return f_best, thicks
    return 100.0, None


def design(problem):
    """Design a broadband antireflection coating."""
    deadline = time.time() + 145.0  # Safety margin under 150s wall-clock

    best_score = 100.0
    best_labels = ['L']
    best_thicks = [600.0 / (4.0 * N_L)]

    # Prioritize: 3, 5, 7 layers first (most likely to be optimal for broadband AR)
    # Then try 9, 11, 13, 15 layers
    n_layers_priority = [3, 5, 7, 9, 11, 13, 15]
    seed_counter = 0

    for n_layers in n_layers_priority:
        if time.time() > deadline - 15.0:
            break

        # Two alternating patterns: start with L or start with H
        for start_with_L in [True, False]:
            if time.time() > deadline - 10.0:
                break

            labels = []
            for i in range(n_layers):
                labels.append('L' if (i % 2 == 0) == start_with_L else 'H')

            # Allocate more time to smaller stacks (more likely optimal)
            if n_layers <= 7:
                time_budget = min(25.0, max(5.0, (deadline - time.time()) - 15.0))
            else:
                time_budget = min(15.0, max(3.0, (deadline - time.time()) - 15.0))
            
            score, thicks = _optimize_stack(labels, time_budget, seed=seed_counter)
            seed_counter += 1

            if score < best_score:
                best_score = score
                best_labels = labels
                best_thicks = thicks

    # If we have significant time left, try a few non-alternating patterns
    if time.time() < deadline - 25.0:
        rng = np.random.RandomState(42)
        for trial in range(4):
            if time.time() > deadline - 12.0:
                break
            n_layers = rng.randint(4, 12)
            labels = []
            for i in range(n_layers):
                if i == 0:
                    labels.append('L' if rng.random() < 0.6 else 'H')  # Bias toward L start
                else:
                    if rng.random() < 0.8:
                        labels.append('H' if labels[-1] == 'L' else 'L')
                    else:
                        labels.append(labels[-1])

            time_budget = min(12.0, max(3.0, (deadline - time.time()) - 12.0))
            score, thicks = _optimize_stack(labels, time_budget, seed=seed_counter)
            seed_counter += 1

            if score < best_score:
                best_score = score
                best_labels = labels
                best_thicks = thicks

    # Final validity check
    if not _valid(best_labels, best_thicks):
        centre = 600.0
        best_labels = ['L']
        best_thicks = [centre / (4.0 * N_L)]

    return best_labels, best_thicks
# EVOLVE-BLOCK-END
