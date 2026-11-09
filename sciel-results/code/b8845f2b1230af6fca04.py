"""Rewritten program: simulated annealing + vectorized characteristic-matrix AR design."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def design(problem):
    """Design a broadband antireflection coating via simulated annealing with
    a vectorized exact characteristic-matrix reflectance objective."""

    t_start = time.time()
    DEADLINE = 150.0 - 8.0  # safety margin

    ambient = problem.get("ambient_index", 1.0)
    substrate = problem.get("substrate_index", 1.52)
    materials = problem.get("materials", {"L": 1.38, "H": 2.35})
    n_L = materials["L"]
    n_H = materials["H"]
    band = problem.get("band_nm", [400, 800])
    lam_min, lam_max = band[0], band[1]
    max_layers = problem.get("max_layers", 20)
    max_total = problem.get("max_total_nm", 1000)
    max_layer = problem.get("max_layer_nm", 400)

    n_wl = 401
    wavelengths = np.linspace(lam_min, lam_max, n_wl)

    # Precompute 1/lambda for speed
    inv_lam = 1.0 / wavelengths

    def compute_reflectance(labels, thicknesses):
        """Exact characteristic-matrix method, vectorized over wavelengths.
        thicknesses: numpy array of physical thicknesses in nm.
        Returns mean reflectance in percent."""
        n_layers = len(labels)
        if n_layers == 0:
            # Bare substrate
            r = (substrate - ambient) / (substrate + ambient)
            return float(np.abs(r)**2 * 100.0)

        # Accumulate the 2x2 characteristic matrix M across layers.
        # M starts as identity; M = M @ M_j for each layer j.
        M00 = np.ones(n_wl, dtype=complex)
        M01 = np.zeros(n_wl, dtype=complex)
        M10 = np.zeros(n_wl, dtype=complex)
        M11 = np.ones(n_wl, dtype=complex)

        for i in range(n_layers):
            n_j = materials[labels[i]]
            d = thicknesses[i]
            delta = 2.0 * np.pi * n_j * d * inv_lam
            c = np.cos(delta)
            s = np.sin(delta)
            # Avoid division by zero: where s==0, use limit (M01 -> 0, M10 -> 0 effectively)
            # Actually at s=0 the matrix is [[1,0],[0,1]] for that layer (delta=0 or pi)
            # But delta=pi gives c=-1, s=0 -> M = [[-1, 0],[0, -1]]
            # Handle: where |s| < eps, set M01=0, M10=0 (since i/s and i*s both -> 0 in limit)
            eps = 1e-12
            safe_s = np.where(np.abs(s) < eps, 1.0, s)
            M01_j = 1j / (n_j * safe_s)
            M10_j = 1j * n_j * s
            # Zero out where s was ~0
            M01_j = np.where(np.abs(s) < eps, 0.0, M01_j)
            M10_j = np.where(np.abs(s) < eps, 0.0, M10_j)

            # M_new = M @ M_j
            new00 = M00 * c + M01 * M10_j
            new01 = M00 * M01_j + M01 * c
            new10 = M10 * c + M11 * M10_j
            new11 = M10 * M01_j + M11 * c
            M00, M01, M10, M11 = new00, new01, new10, new11

        Y_sub = substrate
        Y = (M00 + M01 * Y_sub) / (M10 + M11 * Y_sub)
        r = (Y - ambient) / (Y + ambient)
        R = np.abs(r)**2
        return float(np.mean(R) * 100.0)

    def make_symmetric_labels(n_layers):
        """Build a symmetric label sequence starting with L on air side."""
        if n_layers % 2 != 0:
            n_layers -= 1
        if n_layers < 2:
            return ["L"]
        half = n_layers // 2
        half_labels = []
        for i in range(half):
            half_labels.append("L" if i % 2 == 0 else "H")
        full = half_labels + list(reversed(half_labels))
        return full

    def make_alternating_labels(n_layers):
        """Simple alternating L,H starting with L."""
        return ["L" if i % 2 == 0 else "H" for i in range(n_layers)]

    def project(x):
        """Clip to [0, max_layer] and enforce total <= max_total."""
        x = np.clip(x, 0.0, max_layer)
        s = np.sum(x)
        if s > max_total:
            x = x * (max_total / s)
            x = np.clip(x, 0.0, max_layer)
        return x

    def objective(x):
        x = project(x)
        return compute_reflectance(labels, x)

    def simulated_annealing(labels, x0, max_iter=30000, t0=100.0, cooling=0.9995,
                            step_scale=50.0):
        """Simulated annealing on the continuous thickness vector."""
        n_var = len(x0)
        x = x0.copy()
        f = objective(x)
        best_x = x.copy()
        best_f = f
        rng = np.random.default_rng(42)

        for it in range(max_iter):
            if time.time() > DEADLINE:
                break
            # Random perturbation
            step = step_scale * (1.0 + 0.5 * np.random.randn(n_var))
            x_new = x + step
            x_new = project(x_new)
            f_new = objective(x_new)
            if f_new < f:
                x, f = x_new, f_new
                if f_new < best_f:
                    best_x, best_f = x_new.copy(), f_new
            else:
                # Accept with probability exp(-(f_new - f)/T)
                T = t0 * (cooling ** it)
                if T > 1e-10:
                    if np.log(np.random.rand()) < -(f_new - f) / T:
                        x, f = x_new, f_new
        return best_x, best_f

    def local_refine(labels, x0, max_iter=500):
        """L-BFGS-B local refinement."""
        n_var = len(x0)
        bounds = [(0.0, max_layer)] * n_var
        cons = [{'type': 'ineq', 'fun': lambda x: max_total - np.sum(x)}]

        def obj(x):
            x = np.clip(x, 0.0, max_layer)
            s = np.sum(x)
            if s > max_total:
                x = x * (max_total / s)
            return compute_reflectance(labels, x)

        try:
            res = minimize(obj, x0, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 30})
            x_best = np.clip(res.x, 0.0, max_layer)
            s = np.sum(x_best)
            if s > max_total:
                x_best = x_best * (max_total / s)
            f_best = obj(x_best)
            return x_best, f_best
        except Exception:
            return x0, objective(x0)

    # Strategy: try several label patterns, run SA + local refine on each, keep best.
    best_labels = None
    best_thick = None
    best_score = float('inf')

    # Candidate label patterns
    candidate_patterns = []
    for nl in [12, 10, 14, 8, 16]:
        candidate_patterns.append(make_symmetric_labels(nl))
    # Also try non-symmetric alternating
    for nl in [12, 10, 14]:
        candidate_patterns.append(make_alternating_labels(nl))
    # A few more diverse patterns
    # L H H L L H H L ... (double-H blocks)
    def make_double_h(n_layers):
        if n_layers % 2 != 0:
            n_layers -= 1
        half = n_layers // 2
        half_labels = []
        for i in range(half):
            if i % 3 == 0:
                half_labels.append("L")
            else:
                half_labels.append("H")
        full = half_labels + list(reversed(half_labels))
        return full
    for nl in [12, 10]:
        candidate_patterns.append(make_double_h(nl))

    # Deduplicate
    seen = set()
    unique_patterns = []
    for p in candidate_patterns:
        key = tuple(p)
        if key not in seen:
            seen.add(key)
            unique_patterns.append(p)

    center = 0.5 * (lam_min + lam_max)

    for labels in unique_patterns:
        if time.time() > DEADLINE - 5:
            break

        n_var = len(labels)
        if n_var > max_layers:
            continue

        # Initial guess: quarter-wave at band center
        x0 = np.zeros(n_var)
        for i, lab in enumerate(labels):
            n_i = materials[lab]
            x0[i] = center / (4.0 * n_i)
        x0 = project(x0)

        # Run SA
        sa_iters = 20000
        x_sa, f_sa = simulated_annealing(labels, x0, max_iter=sa_iters)

        # Local refine from SA result
        x_ref, f_ref = local_refine(labels, x_sa)

        # Also try local refine from x0 directly
        x_ref0, f_ref0 = local_refine(labels, x0)

        # Pick best of the three
        candidates = [(x_sa, f_sa), (x_ref, f_ref), (x_ref0, f_ref0)]
        x_best, f_best = min(candidates, key=lambda t: t[1])

        # Validate
        x_best = project(x_best)
        f_best = compute_reflectance(labels, x_best)

        if f_best < best_score:
            best_score = f_best
            best_labels = labels
            best_thick = list(x_best)

    # If we have spare time, do a second SA pass with different seed on the best pattern
    if best_labels is not None and time.time() < DEADLINE - 10:
        labels = best_labels
        n_var = len(labels)
        x0 = np.array(best_thick, dtype=float)
        # Perturb and re-run SA
        rng = np.random.default_rng(123)
        x_pert = x0 + 0.1 * rng.randn(n_var)
        x_pert = project(x_pert)
        x_sa2, f_sa2 = simulated_annealing(labels, x_pert, max_iter=15000, t0=50.0, cooling=0.9997)
        x_ref2, f_ref2 = local_refine(labels, x_sa2)
        if f_ref2 < best_score:
            best_score = f_ref2
            best_thick = list(project(x_ref2))

    # Fallback
    if best_labels is None:
        best_labels = ["L"]
        best_thick = [center / (4.0 * n_L)]

    # Final validation
    best_thick = [max(0.0, min(max_layer, t)) for t in best_thick]
    s = sum(best_thick)
    if s > max_total:
        scale = max_total / s
        best_thick = [t * scale for t in best_thick]

    return best_labels, best_thick
# EVOLVE-BLOCK-END
