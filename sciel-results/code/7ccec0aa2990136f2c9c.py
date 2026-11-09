"""Broadband antireflection coating design: fixed graded-index profile + L-BFGS-B."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

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


def _clamp_and_project(x):
    """Clip to [0, MAX_LAYER_NM] and scale down so total <= MAX_TOTAL_NM."""
    x = np.clip(np.asarray(x, dtype=float), 0.0, MAX_LAYER_NM)
    total = x.sum()
    if total > MAX_TOTAL_NM:
        x = x * (MAX_TOTAL_NM / total)
    return x


def _objective(x):
    """Objective for a fixed graded-index label sequence (20 alternating layers)."""
    x = _clamp_and_project(x)
    if x.sum() == 0:
        return 100.0
    # Build labels: start with H (higher index near air helps match 1.0 -> 1.52? 
    # Actually graded from air(1.0) to glass(1.52): we want index to increase.
    # L=1.38, H=2.35. Neither is between 1.0 and 1.52 perfectly, but H then L alternating
    # creates a graded effect. Let's use H L H L ... starting with H.
    labels = []
    for i in range(len(x)):
        labels.append('H' if i % 2 == 0 else 'L')
    return _reflectance(labels, list(x))


def _objective_alt(x):
    """Alternative: start with L."""
    x = _clamp_and_project(x)
    if x.sum() == 0:
        return 100.0
    labels = []
    for i in range(len(x)):
        labels.append('L' if i % 2 == 0 else 'H')
    return _reflectance(labels, list(x))


def _initial_quadratic(n_layers, start_with_H=True):
    """Generate initial thicknesses from a quadratic index-match profile.
    Target: match ambient (1.0) to substrate (1.52) via a smooth profile.
    We assign each layer a target index between 1.0 and 1.52, then pick L or H
    and set thickness to approximate quarter-wave at center wavelength.
    """
    # Actually, for a fixed alternating pattern, we just need good initial thicknesses.
    # Use a profile where thickness varies to create a graded effect.
    # Center wavelength for quarter-wave
    wl_c = 600.0
    d_L = wl_c / (4.0 * N_L)  # ~108.7
    d_H = wl_c / (4.0 * N_H)  # ~63.8
    
    x = np.zeros(n_layers)
    for i in range(n_layers):
        if start_with_H:
            is_H = (i % 2 == 0)
        else:
            is_H = (i % 2 == 1)
        if is_H:
            x[i] = d_H
        else:
            x[i] = d_L
    # Add a slight taper to break symmetry
    t = np.linspace(0.8, 1.2, n_layers)
    x = x * t
    return _clamp_and_project(x)


def _lbfgs_optimize(obj_func, x0, time_budget):
    """Run L-BFGS-B with a time budget."""
    deadline = time.time() + time_budget
    n = len(x0)
    bounds = [(0.0, MAX_LAYER_NM)] * n
    
    # Custom callback to check time
    def callback(xk):
        return time.time() > deadline
    
    try:
        res = minimize(
            obj_func,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={
                'maxiter': 200,
                'maxfun': 500,
                'ftol': 1e-10,
                'gtol': 1e-8,
                'callback': callback
            }
        )
        x_opt = _clamp_and_project(res.x)
        f_opt = obj_func(x_opt)
        return f_opt, x_opt
    except Exception:
        return 100.0, x0


def _nelder_mead(objective, x0, max_iter=200, tol=1e-8, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5):
    """Fallback Nelder-Mead for small problems."""
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


def design(problem):
    """Design a broadband antireflection coating using fixed graded-index + L-BFGS-B."""
    t_start = time.time()
    deadline = t_start + 140.0  # Safety margin under 150s

    best_score = 100.0
    best_labels = None
    best_thicks = None

    n_layers = 20  # Use max layers for best broadband performance

    # Try both starting patterns: H-first and L-first
    patterns = [
        ('H', _objective),
        ('L', _objective_alt),
    ]

    for start_char, obj_func in patterns:
        if time.time() > deadline - 20.0:
            break
        
        # Generate multiple initial points
        rng = np.random.RandomState(42)
        initial_points = []
        
        # Quadratic profile
        x0_q = _initial_quadratic(n_layers, start_with_H=(start_char == 'H'))
        initial_points.append(x0_q)
        
        # Quarter-wave uniform
        x0_qw = np.zeros(n_layers)
        for i in range(n_layers):
            is_H = (i % 2 == 0) if start_char == 'H' else (i % 2 == 1)
            x0_qw[i] = 600.0 / (4.0 * (N_H if is_H else N_L))
        initial_points.append(x0_qw)
        
        # Random starts
        for _ in range(5):
            x0_r = rng.uniform(20.0, 150.0, size=n_layers)
            initial_points.append(_clamp_and_project(x0_r))
        
        # Tapered starts
        for base in [50.0, 80.0, 100.0]:
            for slope in [-0.3, 0.0, 0.3]:
                x0_t = base * (1.0 + slope * np.linspace(0, 1, n_layers))
                initial_points.append(_clamp_and_project(x0_t))
        
        for x0 in initial_points:
            if time.time() > deadline - 15.0:
                break
            
            time_budget = min(10.0, max(2.0, deadline - time.time() - 10.0))
            f_opt, x_opt = _lbfgs_optimize(obj_func, x0, time_budget)
            
            if f_opt < best_score:
                best_score = f_opt
                # Reconstruct labels
                labels = []
                for i in range(n_layers):
                    if start_char == 'H':
                        labels.append('H' if i % 2 == 0 else 'L')
                    else:
                        labels.append('L' if i % 2 == 0 else 'H')
                best_labels = labels
                best_thicks = list(x_opt)
    
    # If L-BFGS-B didn't find anything good, try Nelder-Mead on a reduced set
    if best_score > 5.0 and time.time() < deadline - 30.0:
        # Use the best starting point from above
        if best_thicks is None:
            x0 = _initial_quadratic(n_layers, start_with_H=True)
            labels = ['H' if i % 2 == 0 else 'L' for i in range(n_layers)]
        else:
            x0 = np.array(best_thicks)
            labels = best_labels
        
        time_budget = min(30.0, max(5.0, deadline - time.time() - 10.0))
        x_nm, f_nm = _nelder_mead(_objective, x0, max_iter=100)
        x_nm = _clamp_and_project(x_nm)
        f_nm = _objective(x_nm)
        
        if f_nm < best_score:
            best_score = f_nm
            best_thicks = list(x_nm)
            # Labels stay the same (H-first pattern)
            if best_labels is None:
                best_labels = ['H' if i % 2 == 0 else 'L' for i in range(n_layers)]
    
    # Final validity check
    if best_labels is None or best_thicks is None or not _valid(best_labels, best_thicks):
        # Fallback: single quarter-wave layer
        best_labels = ['L']
        best_thicks = [600.0 / (4.0 * N_L)]
    
    return best_labels, best_thicks
# EVOLVE-BLOCK-END
