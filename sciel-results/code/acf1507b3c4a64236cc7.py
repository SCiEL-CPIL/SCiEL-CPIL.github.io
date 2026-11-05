"""Combined spectral quadrature design for luminosity distance.

Strategy:
1. Use a robust parameterization (nodes via gaps, weights via softmax) to ensure
   strictly increasing nodes in (0,1) and positive weights summing to 1.
2. Minimize the worst-case relative error over a diverse set of CPL cosmologies
   and redshifts.
3. Use Differential Evolution for global search, followed by L-BFGS-B and
   Nelder-Mead for local refinement.
4. Include a moment-matching initialization based on Chebyshev polynomials to
   provide a strong starting point that captures the general shape of the
   integrand.
"""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import roots_legendre, roots_cheby1


def _E_of_a(a, Om, w0, wa):
    """CPL Hubble expansion E(a)=H(a)/H0 for flat universe."""
    a = np.asarray(a, dtype=float)
    matter = Om * a ** (-3.0)
    if abs(w0 + 1.0) < 1e-10 and abs(wa) < 1e-10:
        de = (1.0 - Om)
    elif abs(w0 + 1.0) < 1e-10:
        # w0 = -1, wa != 0: E^2 = Om a^-3 + (1-Om) exp(3 wa (1-a))
        exp_arg = 3.0 * wa * (1.0 - a)
        exp_arg = np.clip(exp_arg, -500.0, 500.0)
        de = (1.0 - Om) * np.exp(exp_arg)
    else:
        exponent = -3.0 * (1.0 + w0 + wa)
        exp_arg = 3.0 * wa * (1.0 - a) / (1.0 + w0)
        exp_arg = np.clip(exp_arg, -500.0, 500.0)
        de = (1.0 - Om) * np.power(a, exponent) * np.exp(exp_arg)
    return np.sqrt(matter + de)


def _integrand_for_nodes(nodes, Om, w0, wa):
    """Compute 1/E(a_i) for the given nodes."""
    a_vals = 1.0 / (1.0 + nodes)
    return 1.0 / _E_of_a(a_vals, Om, w0, wa)


def _ref_integral(z_max, Om, w0, wa):
    """High-accuracy reference integral ∫_0^{z_max} dz/E(a(z)) via Gauss-Legendre."""
    n_gl = 256
    gz, gw = roots_legendre(n_gl)
    z_pts = 0.5 * z_max * (gz + 1.0)
    w_pts = 0.5 * z_max * gw
    a_pts = 1.0 / (1.0 + z_pts)
    e_vals = _E_of_a(a_pts, Om, w0, wa)
    return np.sum(w_pts / e_vals)


def construct():
    t0 = time.time()
    DEADLINE = 85.0  # safety margin under 90s kill

    # --- Training cosmologies: diverse CPL models ---
    models = []
    Om_list = [0.2, 0.3, 0.35, 0.4]
    w0_list = [-1.2, -1.0, -0.8, -0.5, 0.0, 0.3]
    wa_list = [-0.5, -0.2, 0.0, 0.2, 0.5]
    for Om in Om_list:
        for w0 in w0_list:
            for wa in wa_list:
                models.append((Om, w0, wa))

    # Redshifts (z_max) to cover the integration domain
    z_list = [0.5, 1.0, 2.0, 3.0, 5.0]

    # Pre-compute reference integrals
    n_models = len(models)
    n_z = len(z_list)
    ref = np.zeros((n_models, n_z))
    for i, (Om, w0, wa) in enumerate(models):
        for j, zm in enumerate(z_list):
            ref[i, j] = _ref_integral(zm, Om, w0, wa)

    # --- Parameterization ---
    # Nodes: 7 gap parameters (positive) + 1 offset to ensure nodes in (0,1)
    # Let gaps = exp(g_i) for i=0..6. Total gap = sum(gaps).
    # Nodes are cumulative sums of normalized gaps, scaled to fit in (0,1).
    # Specifically:
    #   raw_nodes = cumsum(gaps) / sum(gaps)  -> in (0,1]
    #   nodes = raw_nodes * (1 - 2*eps) + eps  -> in (eps, 1-eps)
    # Weights: 8 logits -> softmax -> positive, sum=1

    def decode(params):
        g = params[:7]
        v = params[7:]
        # Nodes from gaps
        gaps = np.exp(g)
        total_gaps = np.sum(gaps)
        raw_nodes = np.cumsum(gaps) / total_gaps
        # raw_nodes is in (0,1], with last element = 1
        # Map to (eps, 1-eps)
        eps = 1e-7
        nodes = raw_nodes * (1.0 - 2.0 * eps) + eps
        # Ensure strictly increasing
        nodes = np.clip(nodes, eps, 1.0 - eps)
        # Weights via softmax
        v_shifted = v - np.max(v)
        w_exp = np.exp(v_shifted)
        weights = w_exp / np.sum(w_exp)
        return nodes, weights

    def objective(params):
        if time.time() > DEADLINE:
            return 1e10
        nodes, weights = decode(params)
        # Check strict increase
        if np.any(np.diff(nodes) < 1e-14):
            return 1e8
        max_err = 0.0
        for i, (Om, w0, wa) in enumerate(models):
            f_vals = _integrand_for_nodes(nodes, Om, w0, wa)
            q = np.dot(weights, f_vals)
            for j, zm in enumerate(z_list):
                rv = ref[i, j]
                if abs(rv) < 1e-15:
                    continue
                rel_err = abs(q - rv) / abs(rv)
                if rel_err > max_err:
                    max_err = rel_err
        return max_err

    # --- Initial guess: Moment-matching with Chebyshev polynomials ---
    # Use Chebyshev nodes and weights as a starting point
    n = 8
    cheb_nodes, cheb_weights = roots_cheby1(n)
    # Map Chebyshev nodes from [-1,1] to [0,1]
    cheb_nodes_mapped = 0.5 * (cheb_nodes + 1.0)
    cheb_weights_mapped = cheb_weights / 2.0

    # Encode Chebyshev nodes to gap parameterization
    # We need to find g such that decode(g, v) gives nodes close to cheb_nodes_mapped
    # This is an inverse problem. Instead, we'll use a simple heuristic:
    # Assume uniform gaps initially, then optimize.
    g_init = np.zeros(7)
    v_init = np.log(cheb_weights_mapped)
    x0 = np.concatenate([g_init, v_init])

    bounds = [(-8.0, 8.0)] * 7 + [(-8.0, 8.0)] * 8

    best_x = x0.copy()
    best_val = objective(best_x)

    # --- Phase 1: Differential Evolution for global search ---
    if time.time() < DEADLINE - 30.0:
        try:
            de_res = differential_evolution(
                objective, bounds, maxiter=50, popsize=8,
                tol=1e-6, seed=42, polish=False,
                callback=lambda xk, conv: (time.time() > DEADLINE - 30.0)
            )
            if de_res.fun < best_val:
                best_val = de_res.fun
                best_x = de_res.x.copy()
        except Exception:
            pass

    # --- Phase 2: L-BFGS-B local refinement ---
    if time.time() < DEADLINE - 10.0:
        try:
            res = minimize(objective, best_x, method='L-BFGS-B',
                           bounds=bounds,
                           options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-14})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    # --- Phase 3: Nelder-Mead final polish ---
    if time.time() < DEADLINE - 3.0:
        try:
            res = minimize(objective, best_x, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    # --- Decode and finalize ---
    nodes, weights = decode(best_x)

    # Enforce strict increase
    nodes = np.clip(nodes, 1e-9, 1.0 - 1e-9)
    for i in range(1, len(nodes)):
        if nodes[i] - nodes[i - 1] < 1e-12:
            nodes[i] = nodes[i - 1] + 1e-12

    # Enforce positivity and normalize
    weights = np.maximum(weights, 1e-15)
    weights = weights / np.sum(weights)

    return {"nodes": nodes, "weights": weights}
