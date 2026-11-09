"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

# ---------- Cosmology: H(a)/H0 for flat CPL dark-energy models ----------
# E(a) = sqrt( Omega_m a^-3 + Omega_r a^-4 + Omega_k a^-2 + Omega_L a^(-3(1+w0+w1*(1-a))) )
# With Omega_k = 0, Omega_L = 1 - Omega_m - Omega_r.
# Luminosity distance (c/H0 units):
#   dL = (1+z) * int_0^z  dz' / E(1/(1+z'))
#      = (1+z) * z * int_0^1 du / E(a(u))  where a(u)=1/(1+z*u), u in [0,1]
# Because du = dz'/(1+z), and 1/(1+z u) = a.
# Actually: dL = (1+z) * int_0^z dz'/E(a). Let u = z'/z => dz' = z du, a = 1/(1+z u).
#   dL = (1+z)*z * int_0^1 du / E(1/(1+z u)).  Good.

def E_of_a(a, om_m, om_r, w0, w1):
    a = np.asarray(a, dtype=float)
    om_l = 1.0 - om_m - om_r
    # dark energy density term
    # rho_L(a) ~ a^{-3(1+w0+w1*(1-a))}
    exp_l = -3.0 * (1.0 + w0 + w1 * (1.0 - a))
    term_l = om_l * np.power(a, exp_l)
    term_m = om_m * np.power(a, -3.0)
    term_r = om_r * np.power(a, -4.0)
    return np.sqrt(np.maximum(term_m + term_r + term_l, 1e-300))


def ref_lumdist(z, om_m, om_r, w0, w1, n=20000):
    """High-accuracy reference: composite Simpson on u in [0,1]."""
    u = np.linspace(0.0, 1.0, n + 1)
    a = 1.0 / (1.0 + z * u)
    f = 1.0 / E_of_a(a, om_m, om_r, w0, w1)
    # Simpson's rule (n must be even)
    if n % 2 != 0:
        n += 1
        u = np.linspace(0.0, 1.0, n + 1)
        a = 1.0 / (1.0 + z * u)
        f = 1.0 / E_of_a(a, om_m, om_r, w0, w1)
    h = 1.0 / n
    s = f[0] + f[-1] + 4.0 * np.sum(f[1:-1:2]) + 2.0 * np.sum(f[2:-1:2])
    integral = h * s / 3.0
    return (1.0 + z) * z * integral


def quad_lumdist(z, nodes, weights, om_m, om_r, w0, w1):
    a = 1.0 / (1.0 + z * nodes)
    f = 1.0 / E_of_a(a, om_m, om_r, w0, w1)
    integral = np.dot(weights, f)
    return (1.0 + z) * z * integral


# ---------- Representative model set (design set, disjoint from validation) ----------
def design_models():
    models = []
    # LambdaCDM
    models.append((0.3, 0.0, 0.0, 0.0))
    # Matter only
    models.append((1.0, 0.0, -1.0, 0.0))
    # Constant w family
    for w in (-0.5, -0.8, -1.0, -1.2, -1.5):
        models.append((0.3, 0.0, w, 0.0))
    # CPL grid
    for w0 in (-0.5, -0.8, -1.0, -1.2, -1.5):
        for w1 in (-0.5, 0.0, 0.5, 1.0):
            om_l = 0.7
            om_m = 1.0 - om_l
            # ensure positivity of om_m
            if om_m <= 0:
                continue
            models.append((om_m, 0.0, w0, w1))
    # Some with radiation
    models.append((0.27, 0.005, -1.0, 0.0))
    models.append((0.3, 0.005, -0.5, 0.5))
    return models


def design_zs():
    zs = []
    for z in np.logspace(-2, 1.3, 40):
        zs.append(z)
    return zs


def worst_rel_error(nodes, weights, models, zs):
    worst = 0.0
    for (om_m, om_r, w0, w1) in models:
        for z in zs:
            ref = ref_lumdist(z, om_m, om_r, w0, w1)
            q = quad_lumdist(z, nodes, weights, om_m, om_r, w0, w1)
            err = abs(q - ref) / ref
            if err > worst:
                worst = err
    return worst


# ---------- Parameterization ensuring positivity and ordering ----------
def params_to_nodes_weights(p):
    # p[0:8] -> log-gaps for nodes (strictly increasing in (0,1))
    # p[8:16] -> log-weights (positive), normalized to sum 1
    log_gaps = p[:8]
    log_ws = p[8:16]
    # node_0 from log_gaps[0] mapped to (0, 0.5)
    # gaps mapped via softplus to positive
    gaps = np.exp(log_gaps)
    # scale so total span < 1
    # node_i = base + cumsum of scaled gaps
    # Use a soft mapping: nodes = 1/(1+exp(-x)) style cumulative
    # Simpler: nodes strictly increasing via cumulative of exp gaps, then rescale to (eps, 1-eps)
    cum = np.cumsum(gaps)
    span = cum[-1]
    nodes = 0.01 + 0.98 * (cum / span)
    # ensure strictly increasing (exp gaps > 0 so cum strictly increasing)
    # weights
    ws = np.exp(log_ws)
    ws = ws / np.sum(ws)
    return nodes, ws


def objective(p, models, zs):
    nodes, weights = params_to_nodes_weights(p)
    return worst_rel_error(nodes, weights, models, zs)


def construct():
    t0 = time.time()
    deadline = t0 + 75.0  # safety margin before 90s kill

    models = design_models()
    zs = design_zs()

    # Precompute reference values for speed (cache)
    ref_cache = []
    for (om_m, om_r, w0, w1) in models:
        for z in zs:
            ref_cache.append(ref_lumdist(z, om_m, om_r, w0, w1))
    ref_cache = np.array(ref_cache)
    model_keys = [(om_m, om_r, w0, w1) for (om_m, om_r, w0, w1) in models]

    def fast_worst(nodes, weights):
        # vectorize over models and zs
        # nodes shape (8,), weights (8,)
        # For each model, for each z: a = 1/(1+z*nodes)
        worst = 0.0
        idx = 0
        for (om_m, om_r, w0, w1) in model_keys:
            for z in zs:
                a = 1.0 / (1.0 + z * nodes)
                f = 1.0 / E_of_a(a, om_m, om_r, w0, w1)
                q = (1.0 + z) * z * np.dot(weights, f)
                ref = ref_cache[idx]
                err = abs(q - ref) / ref
                if err > worst:
                    worst = err
                idx += 1
        return worst

    def obj(p):
        nodes, weights = params_to_nodes_weights(p)
        return fast_worst(nodes, weights)

    # Initial: moment-matching style. Start from a spread and refine.
    # Use several restarts.
    rng = np.random.default_rng(12345)

    best_p = None
    best_val = np.inf

    # A good initial guess: nodes roughly geometric spread, weights ~ 1/8
    # log_gaps such that nodes spread out; log_ws all equal
    init_log_gaps = np.log(np.linspace(0.05, 1.0, 8))
    init_log_ws = np.zeros(8)
    p0 = np.concatenate([init_log_gaps, init_log_ws])

    n_restarts = 6
    for r in range(n_restarts):
        if time.time() > deadline - 15.0:
            break
        if r == 0:
            p = p0.copy()
        else:
            p = p0 + rng.normal(0.0, 0.5, size=16)
        try:
            from scipy.optimize import minimize
            res = minimize(obj, p, method='L-BFGS-B',
                           options={'maxiter': 60, 'ftol': 1e-10, 'gtol': 1e-8,
                                    'maxfun': 200})
            val = res.fun
            if val < best_val:
                best_val = val
                best_p = res.x
        except Exception:
            # fallback: simple gradient-free local search
            pass

    if best_p is None:
        # Fallback: composite midpoint
        n = 8
        return {"nodes": (np.arange(n) + 0.5) / n, "weights": np.full(n, 1.0 / n)}

    nodes, weights = params_to_nodes_weights(best_p)

    # Enforce strictly increasing and positivity, sum weights to 1
    nodes = np.sort(nodes)
    # ensure in (0,1) and strictly increasing
    eps = 1e-6
    nodes = np.clip(nodes, eps, 1.0 - eps)
    # enforce strict increase
    for i in range(1, len(nodes)):
        if nodes[i] <= nodes[i-1]:
            nodes[i] = nodes[i-1] + 1e-5
    nodes[-1] = min(nodes[-1], 1.0 - eps)
    weights = np.maximum(weights, 1e-12)
    weights = weights / np.sum(weights)

    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
