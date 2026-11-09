"""
Broadband antireflection coating design: Structural Search + Local Refinement.

Strategy:
1. Define a set of structural families for the coating stack:
   - Quarter-wave alternating (LH or HL) with varying layer count.
   - Graded index stacks (thicknesses varying linearly or exponentially).
   - Symmetric stacks with random but fixed material patterns.
2. For each family, parameterize the thicknesses by a small number of variables 
   (e.g., scale factor, gradient) rather than 20 independent variables.
3. Use Nelder-Mead to optimize these low-dimensional parameters.
4. Compare the best results from all families and return the global best.
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
        step = 0.1 * (bounds[i][1] - bounds[i][0]) if bounds[i][1] - bounds[i][0] > 1 else 0.1
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


def _optimize_stack(materials, thicknesses, problem, deadline):
    """
    Optimize thicknesses for a fixed material sequence using Nelder-Mead.
    This is the refinement engine.
    """
    start_time = time.time()
    timeout = min(10.0, max(0.1, deadline - start_time))
    
    if len(materials) == 0:
        return materials, thicknesses, 100.0
        
    def objective(x):
        if time.time() > deadline:
            return 100.0
        t = _clamp_thicknesses(x, problem)
        if not _valid(materials, t, problem):
            return 100.0
        return _reflectance(materials, t, problem)

    x0 = np.array(thicknesses, dtype=float)
    bounds = [(0.0, problem["max_layer_nm"])] * len(x0)

    try:
        x_opt, f_opt = _nelder_mead(objective, x0, bounds, maxiter=150, timeout=timeout, start_time=start_time)
        t_opt = _clamp_thicknesses(x_opt, problem)
        if _valid(materials, t_opt, problem):
            f_opt_check = _reflectance(materials, t_opt, problem)
            if f_opt_check < objective(x0):
                return materials, t_opt, f_opt_check
    except Exception:
        pass

    return materials, thicknesses, objective(x0)


def design(problem):
    """Design a broadband antireflection coating stack."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

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

    best_materials = ['L']
    best_thicknesses = [qwL]
    best_score = _reflectance(best_materials, best_thicknesses, problem)

    # List of candidate structural families to explore
    # Each entry is (materials_generator, thicknesses_generator)
    # We will test a range of layer counts and patterns.

    candidates = []

    # 1. Quarter-wave alternating stacks (LH or HL)
    # These are the classic "broadband" AR stacks.
    for n_layers in range(2, max_layers + 1, 2):
        # Start with L
        mats = ['L' if i % 2 == 0 else 'H' for i in range(n_layers)]
        thicks = [qwL if i % 2 == 0 else qwH for i in range(n_layers)]
        candidates.append((mats, thicks))
        
        # Start with H
        mats = ['H' if i % 2 == 0 else 'L' for i in range(n_layers)]
        thicks = [qwH if i % 2 == 0 else qwL for i in range(n_layers)]
        candidates.append((mats, thicks))

    # 2. Graded index stacks
    # Thicknesses vary to approximate a smooth index transition.
    # We'll try linear and quadratic variations in thickness.
    for n_layers in range(4, min(max_layers, 12) + 1, 2):
        # Pattern: L H L H ... (alternating)
        mats = ['L' if i % 2 == 0 else 'H' for i in range(n_layers)]
        
        # Base thicknesses
        base_thicks = [qwL if i % 2 == 0 else qwH for i in range(n_layers)]
        
        # Variation 1: Linear gradient (thicker in middle or at one end)
        # We just add a random offset to the base to let NM find the best gradient
        for _ in range(3):
            # Random perturbation of the base thicknesses
            perturbed = [t * (1.0 + np.random.uniform(-0.2, 0.2)) for t in base_thicks]
            candidates.append((mats, perturbed))
            
        # Variation 2: Symmetric thickness variation
        # e.g., [1.0, 1.2, 1.0, 1.2, 1.0] for LHLLH
        for _ in range(3):
            # Define a symmetric multiplier pattern
            mults = [1.0 + np.random.uniform(-0.1, 0.3) for _ in range(n_layers//2)]
            sym_mults = mults + mults[::-1] if n_layers % 2 == 0 else mults + [1.0] + mults[::-1]
            if len(sym_mults) != n_layers:
                sym_mults = sym_mults[:n_layers]
            perturbed = [t * m for t, m in zip(base_thicks, sym_mults)]
            candidates.append((mats, perturbed))

    # 3. Random symmetric patterns
    # Generate random material sequences that are symmetric, then optimize thicknesses.
    for _ in range(10):
        n_layers = np.random.randint(4, min(max_layers, 16) + 1, 1)[0]
        half = (n_layers + 1) // 2
        half_mats = [np.random.choice(['L', 'H']) for _ in range(half)]
        
        # Ensure it doesn't start/end with the same material as the other side if symmetric
        # Actually, just mirror it.
        if n_layers % 2 == 0:
            mats = half_mats + half_mats[::-1]
        else:
            mats = half_mats + half_mats[-2::-1]
            
        # Initial thicknesses: random around quarter-wave
        thicks = []
        for m in mats:
            qw = qwL if m == 'L' else qwH
            thicks.append(qw * np.random.uniform(0.8, 1.2))
        
        candidates.append((mats, thicks))

    # 4. Specific known good structures (if any)
    # e.g., 2-layer, 4-layer, 6-layer optimized.
    # We already have these in the alternating section, but let's add a few random 3-5 layer ones.
    for n_layers in range(3, 7):
        for _ in range(5):
            mats = [np.random.choice(['L', 'H']) for _ in range(n_layers)]
            thicks = []
            for m in mats:
                qw = qwL if m == 'L' else qwH
                thicks.append(qw * np.random.uniform(0.7, 1.3))
            candidates.append((mats, thicks))

    # Sort candidates by initial score to prioritize promising ones
    scored_candidates = []
    for mats, thicks in candidates:
        thicks = _clamp_thicknesses(thicks, problem)
        if _valid(mats, thicks, problem):
            score = _reflectance(mats, thicks, problem)
            scored_candidates.append((score, mats, thicks))

    scored_candidates.sort(key=lambda x: x[0])

    # Optimize the top candidates
    # We have a time budget. Let's allocate it dynamically.
    # NM is fast, so we can run it on many candidates.
    max_optimizations = 50 # Safety cap
    
    for i, (score, mats, thicks) in enumerate(scored_candidates[:max_optimizations]):
        if time.time() > deadline - 10:
            break
            
        opt_mats, opt_thicks, opt_score = _optimize_stack(mats, thicks, problem, deadline)
        
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
