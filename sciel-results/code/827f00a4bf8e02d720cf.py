"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution

_DEADLINE = time.time() + 85.0  # safety margin before 90s wall-clock kill

# Precompute a fast 64-point Gauss-Legendre reference rule on [0,1]
_REF_NODES, _REF_WEIGHTS = np.polynomial.legendre.leggauss(64)
_REF_NODES = 0.5 * (_REF_NODES + 1.0)
_REF_WEIGHTS = 0.5 * _REF_WEIGHTS


def _E_cpl(a, w0, wa, om_m, om_r, om_k=0.0):
    """E(a) = H(a)/H0 for a CPL dark-energy model (flat universe)."""
    h2 = (om_m * a**(-3.0) + om_r * a**(-4.0) + om_k
          + (1.0 - om_m - om_r - om_k) * a**(-3.0 * (1.0 + w0 + wa)))
    return np.sqrt(np.maximum(h2, 1e-300))


def _true_dL(z, w0, wa, om_m, om_r, om_k=0.0):
    """Reference luminosity distance via 64-point GL quadrature."""
    a = 1.0 / (1.0 + z * _REF_NODES)
    integrand = 1.0 / _E_cpl(a, w0, wa, om_m, om_r, om_k)
    return (1.0 + z) * z * np.dot(_REF_WEIGHTS, integrand)


def _quad_dL(nodes, weights, z, w0, wa, om_m, om_r, om_k=0.0):
    """Candidate 8-node quadrature luminosity distance."""
    a = 1.0 / (1.0 + z * nodes)
    integrand = 1.0 / _E_cpl(a, w0, wa, om_m, om_r, om_k)
    return (1.0 + z) * z * np.dot(weights, integrand)


# Diverse design cosmologies (flat, time-varying DE).
_COSMOS = [
    (0.0, 0.0, 0.3, 0.05),
    (-0.5, 0.5, 0.3, 0.05),
    (-1.0, 0.0, 0.3, 0.05),
    (-1.5, 1.0, 0.3, 0.05),
    (-0.5, 1.0, 0.2, 0.03),
    (-0.5, 0.5, 0.27, 0.04),
    (-0.5, 0.5, 0.35, 0.06),
    (-0.5, 0.5, 0.3, 0.02),
    (-0.5, 0.5, 0.3, 0.08),
    (-0.5, 0.5, 0.4, 0.05),
    (-0.5, 0.5, 0.2, 0.05),
    (-0.5, 0.5, 0.3, 0.0),
]
_ZVALS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]


def _objective(x):
    """Worst relative error (%) over the design set. x = nodes(8) + log_weights(8)."""
    nodes = x[:8]
    weights = np.exp(x[8:])
    weights = weights / weights.sum()
    worst = 0.0
    for (w0, wa, om_m, om_r) in _COSMOS:
        for z in _ZVALS:
            ref = _true_dL(z, w0, wa, om_m, om_r)
            qd = _quad_dL(nodes, weights, z, w0, wa, om_m, om_r)
            err = abs(qd - ref) / abs(ref) * 100.0
            if err > worst:
                worst = err
    return worst


def _enforce(x):
    """Project a raw candidate to strictly-increasing nodes in (0,1) and positive weights."""
    nodes = x[:8].copy()
    # Strictly increasing in (0,1): enforce via cumulative positive gaps.
    # Map raw node values to a sorted strictly-increasing sequence in (0,1).
    nodes = np.sort(nodes)
    # Add small strictly-increasing offsets to guarantee strict increase & interior.
    eps = 1e-3
    nodes = nodes - nodes[0] + eps
    nodes = nodes / (nodes[-1] - nodes[0] + eps)  # normalize to [0,1]
    nodes = nodes * (1.0 - 2 * eps) + eps  # pull into (eps, 1-eps)
    # Ensure strict increase (already true after sort + positive span)
    return nodes, np.exp(x[8:])


def construct():
    start = time.time()

    # Initial reference: composite midpoint (valid, positive, sum=1).
    n = 8
    best_nodes = (np.arange(n) + 0.5) / n
    best_weights = np.full(n, 1.0 / n)
    best_score = None

    def _score_of(nodes, weights):
        worst = 0.0
        for (w0, wa, om_m, om_r) in _COSMOS:
            for z in _ZVALS:
                ref = _true_dL(z, w0, wa, om_m, om_r)
                qd = _quad_dL(nodes, weights, z, w0, wa, om_m, om_r)
                err = abs(qd - ref) / abs(ref) * 100.0
                if err > worst:
                    worst = err
        return worst

    best_score = _score_of(best_nodes, best_weights)

    # Try differential evolution with a deadline guard.
    try:
        def obj_wrapped(x):
            if time.time() > _DEADLINE:
                return 1e12
            nodes, weights = _enforce(x)
            return _objective(np.concatenate([nodes, np.log(weights)]))

        bounds = [(0.0, 1.0)] * 8 + [(-5.0, 5.0)] * 8
        res = differential_evolution(
            obj_wrapped, bounds,
            seed=0, maxiter=200, popsize=12,
            tol=1e-8, mutation=(0.3, 1.0), recombination=0.8,
            polish=False, updating='deferred',
        )
        if res.x is not None and time.time() < _DEADLINE:
            cand_nodes, cand_weights = _enforce(res.x)
            cand_weights = cand_weights / cand_weights.sum()
            cand_score = _score_of(cand_nodes, cand_weights)
            if cand_score < best_score:
                best_nodes, best_weights, best_score = cand_nodes, cand_weights, cand_score
    except Exception:
        pass

    # Final validity enforcement.
    best_nodes = np.sort(best_nodes)
    best_nodes = best_nodes - best_nodes[0] + 1e-3
    best_nodes = best_nodes / (best_nodes[-1] - best_nodes[0] + 1e-3)
    best_nodes = best_nodes * (1.0 - 2e-3) + 1e-3
    best_weights = np.maximum(best_weights, 1e-300)
    best_weights = best_weights / best_weights.sum()

    return {"nodes": best_nodes, "weights": best_weights}
# EVOLVE-BLOCK-END
