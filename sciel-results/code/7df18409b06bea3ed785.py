"""Broadband antireflection coating design: symmetric templates + simulated annealing + Nelder-Mead."""
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


def _quarter_wave(labels, center_wl=600.0):
    """Quarter-wave thicknesses at center wavelength."""
    thicks = np.zeros(len(labels))
    for j in range(len(labels)):
        n_j = N_L if labels[j] == 'L' else N_H
        thicks[j] = center_wl / (4.0 * n_j)
    total = np.sum(thicks)
    if total > MAX_TOTAL_NM:
        thicks *= MAX_TOTAL_NM / total
    return np.clip(thicks, 0, MAX_LAYER_NM)


def _symmetric_qw(labels, center_wl=600.0):
    """Symmetric quarter-wave: mirror the second half to first."""
    n = len(labels)
    thicks = np.zeros(n)
    half = (n + 1) // 2
    for j in range(half):
        n_j = N_L if labels[j] == 'L' else N_H
        thicks[j] = center_wl / (4.0 * n_j)
    for j in range(n):
        mirror = n - 1 - j
        if j < half:
            thicks[mirror] = thicks[j]
    total = np.sum(thicks)
    if total > MAX_TOTAL_NM:
        thicks *= MAX_TOTAL_NM / total
    return np.clip(thicks, 0, MAX_LAYER_NM)


def _maier_7layer(center_wl=600.0):
    """Maier's 7-layer symmetric AR coating: L H L H L H L with optimized thicknesses."""
    labels = ['L', 'H', 'L', 'H', 'L', 'H', 'L']
    # Known good starting point for Maier 7-layer (approximate)
    # Quarter-wave at center, then we'll optimize
    thicks = _symmetric_qw(labels, center_wl)
    return labels, thicks


def _maier_5layer(center_wl=600.0):
    """5-layer symmetric: L H L H L"""
    labels = ['L', 'H', 'L', 'H', 'L']
    thicks = _symmetric_qw(labels, center_wl)
    return labels, thicks


def _maier_9layer(center_wl=600.0):
    """9-layer symmetric: L H L H L H L H L"""
    labels = ['L', 'H', 'L', 'H', 'L', 'H', 'L', 'H', 'L']
    thicks = _symmetric_qw(labels, center_wl)
    return labels, thicks


def _alternating_qw(n_layers, start_L=True, center_wl=600.0):
    """Alternating L/H pattern with quarter-wave init."""
    labels = []
    for i in range(n_layers):
        labels.append('L' if (i % 2 == 0) == start_L else 'H')
    thicks = _quarter_wave(labels, center_wl)
    return labels, thicks


def _graded_index(n_layers, center_wl=600.0):
    """Graded index: start with L, end with H (or vice versa), no strict alternation."""
    labels = []
    for i in range(n_layers):
        if i < n_layers // 2:
            labels.append('L')
        else:
            labels.append('H')
    thicks = _quarter_wave(labels, center_wl)
    return labels, thicks


def _simulated_annealing(objective, x0, bounds, time_budget, seed=42, T0=50.0, T_end=0.01, cooling=0.995):
    """Simulated annealing with vectorized evaluation."""
    rng = np.random.RandomState(seed)
    n = len(x0)
    lb = np.array(bounds[:, 0], dtype=float)
    ub = np.array(bounds[:, 1], dtype=float)

    x = np.array(x0, dtype=float)
    f = objective(x)
    best_x = x.copy()
    best_f = f

    deadline = time.time() + time_budget
    T = T0
    step_size = 10.0  # nm initial step

    while time.time() < deadline:
        # Generate neighbor
        idx = rng.randint(n)
        delta = rng.uniform(-step_size, step_size)
        x_new = x.copy()
        x_new[idx] = np.clip(x_new[idx] + delta, lb[idx], ub[idx])

        # Check total constraint
        if np.sum(x_new) > MAX_TOTAL_NM + 1e-9:
            # Scale back
            x_new *= MAX_TOTAL_NM / np.sum(x_new)
            x_new = np.clip(x_new, lb, ub)

        f_new = objective(x_new)

        # Acceptance criterion
        if f_new < f or rng.random() < np.exp(-(f_new - f) / T):
            x = x_new
            f = f_new
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Cool down
        T *= cooling
        if T < T_end:
            T = T_end
            step_size *= 0.9

    return best_x, best_f


def _nelder_mead(objective, x0, max_iter=200, tol=1e-6, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5):
    """Nelder-Mead refinement."""
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
    """Optimize thicknesses for given labels using SA + Nelder-Mead."""
    deadline = time.time() + time_budget
    n = len(labels)
    if n > 16:
        return 100.0, None

    objective = _make_objective(labels)

    # Bounds
    bounds = np.zeros((n, 2))
    bounds[:, 0] = 0.0
    bounds[:, 1] = MAX_LAYER_NM

    # Start with symmetric quarter-wave initialization
    x0 = _symmetric_qw(labels)
    f0 = objective(x0)

    # Phase 1: Simulated annealing (good for escaping local minima)
    remaining = deadline - time.time()
    if remaining > 3.0:
        sa_budget = min(remaining - 2.0, 10.0)
        x_sa, f_sa = _simulated_annealing(objective, x0, bounds, time_budget=sa_budget, seed=seed)
        if f_sa < f0:
            x_best = x_sa
            f_best = f_sa
        else:
            x_best = x0
            f_best = f0
    else:
        x_best = x0
        f_best = f0

    # Phase 2: Nelder-Mead refinement
    remaining = deadline - time.time()
    if remaining > 1.0:
        max_iter_nm = min(300, max(30, int(remaining / 0.02)))
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

    # Generate candidate templates
    templates = []

    # Maier symmetric designs (known good starting points)
    templates.append(('Maier 5L', _maier_5layer()))
    templates.append(('Maier 7L', _maier_7layer()))
    templates.append(('Maier 9L', _maier_9layer()))

    # Alternating patterns with various layer counts
    for n in [3, 5, 7, 9, 11]:
        templates.append((f'Alt {n}L L-start', _alternating_qw(n, start_L=True)))
        templates.append((f'Alt {n}L H-start', _alternating_qw(n, start_L=False)))

    # Graded index
    for n in [5, 7, 9]:
        templates.append((f'Graded {n}L', _graded_index(n)))

    # Prioritize: Maier 7L first (known best), then others
    # Sort by expected quality: Maier 7L, 9L, 5L, then alternating
    priority_order = [
        'Maier 7L', 'Maier 9L', 'Maier 5L',
        'Alt 7L L-start', 'Alt 7L H-start',
        'Alt 5L L-start', 'Alt 5L H-start',
        'Alt 9L L-start', 'Alt 9L H-start',
        'Alt 3L L-start', 'Alt 3L H-start',
        'Alt 11L L-start', 'Alt 11L H-start',
        'Graded 5L', 'Graded 7L', 'Graded 9L',
    ]

    template_map = {name: data for name, data in templates}

    seed_counter = 0
    for name in priority_order:
        if time.time() > deadline - 10.0:
            break
        if name not in template_map:
            continue

        labels, thicks_init = template_map[name]
        if not _valid(labels, thicks_init):
            continue

        # Allocate time based on priority
        if name.startswith('Maier'):
            time_budget = min(20.0, max(5.0, (deadline - time.time()) - 10.0))
        else:
            time_budget = min(12.0, max(3.0, (deadline - time.time()) - 10.0))

        score, thicks = _optimize_stack(labels, time_budget, seed=seed_counter)
        seed_counter += 1

        if score < best_score:
            best_score = score
            best_labels = labels
            best_thicks = thicks

    # If time remains, try a few random symmetric patterns
    if time.time() < deadline - 15.0:
        rng = np.random.RandomState(123)
        for trial in range(3):
            if time.time() > deadline - 10.0:
                break
            n_layers = rng.randint(5, 13)
            # Generate symmetric pattern
            half = (n_layers + 1) // 2
            first_half = []
            for i in range(half):
                first_half.append('L' if rng.random() < 0.5 else 'H')
            labels = first_half + first_half[-1::-1] if n_layers % 2 == 1 else first_half + first_half[::-1]

            # Ensure validity
            if len(labels) > MAX_LAYERS:
                continue
            thicks_init = _symmetric_qw(labels)
            if not _valid(labels, thicks_init):
                continue

            time_budget = min(10.0, max(3.0, (deadline - time.time()) - 10.0))
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
