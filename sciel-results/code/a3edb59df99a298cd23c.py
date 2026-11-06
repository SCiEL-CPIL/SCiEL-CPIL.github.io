"""Universal 8-node positive quadrature for dimensionless luminosity distance."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import eval_chebyt
from scipy.optimize import minimize


def _cpl_g(x, w0, wa):
    """Model-dependent factor g(x)=1/E(1-x) for CPL. x = z/(1+z), a=1-x."""
    a = np.clip(1.0 - x, 1e-12, 1.0)
    exponent = 3.0 * (1.0 + w0 + wa / a)
    log_g = exponent * np.log(a)
    g = np.exp(log_g)
    return np.where(np.isfinite(g) & (g > 0), g, 1.0)


def _project_positive(w, tol=1e-12):
    """Project weights onto the cone w_i>0, sum w = 1 via bisection on Lagrange multiplier."""
    w = np.asarray(w, dtype=float).copy()
    n = w.size

    def _sum_after(lambda_):
        return np.maximum(w - lambda_, tol).sum()

    lo, hi = -1e6, 1e6
    target = 1.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        s = _sum_after(mid)
        if s > target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-15:
            break
    lam = (lo + hi) / 2.0
    w_new = np.maximum(w - lam, tol)
    w_new = w_new / w_new.sum()
    return w_new


def _chebyshev_basis(x, n_terms):
    """Return Chebyshev basis T_0..T_{n_terms-1} evaluated at x (mapped to [-1,1])."""
    t = 2.0 * x - 1.0
    B = np.empty((len(x), n_terms))
    for k in range(n_terms):
        B[:, k] = eval_chebyt(k, t)
    return B


def _solve(w0_grid, wa_grid):
    n_models = len(w0_grid) * len(wa_grid)
    M = 2000
    x = np.linspace(1e-4, 1.0 - 1e-4, M)

    # Build true integrand library and true integrals
    f_true = np.empty(n_models)
    lib = np.empty((n_models, M))
    k = 0
    for w0 in w0_grid:
        for wa in wa_grid:
            g = _cpl_g(x, w0, wa)
            f = x * g
            lib[k] = f
            f_true[k] = np.trapz(f, x)
            k += 1
    f_true = np.maximum(f_true, 1e-30)

    # Relative-error weighting: emphasize models with small integral
    wgt = 1.0 / np.sqrt(np.maximum(f_true, 1e-3))
    wgt = wgt / wgt.max()

    # --- Step 1: Choose nodes via weighted Chebyshev-Lobatto-like distribution
    # We want nodes that capture the variation of the integrand across the CPL space.
    # Use a density proportional to the average |df/dx| across models, then place
    # nodes as quantiles of that density (clenshaw-curtis style on the warped domain).
    # Simpler robust choice: Gauss-Legendre nodes on [0,1] as initial, then refine.
    nodes_init, weights_init = np.polynomial.legendre.leggauss(8)
    nodes_init = 0.5 * nodes_init + 0.5
    weights_init = 0.5 * weights_init
    weights_init = weights_init / weights_init.sum()

    # --- Step 2: Alternating optimization
    # For fixed nodes, solve for weights via weighted least squares (with positivity).
    # For fixed weights, optimize nodes to minimize worst relative error.

    nodes = nodes_init.copy()
    weights = weights_init.copy()

    n_terms_cheb = 10  # basis dimension for the model-dependent factor

    def _compute_samples(nd):
        """Compute S[m, i] = nd[i] * g(nd[i], w0_m, wa_m) for all models."""
        S = np.empty((n_models, 8))
        idx = 0
        for w0 in w0_grid:
            for wa in wa_grid:
                S[idx, :] = nd * _cpl_g(nd, w0, wa)
                idx += 1
        return S

    def _worst_rel_err(nd, wt):
        S = _compute_samples(nd)
        Q = S @ wt
        err = np.abs(Q - f_true) / f_true
        return err.max()

    for it in range(300):
        # --- Update weights for fixed nodes: weighted least squares with positivity
        S = _compute_samples(nodes)  # (n_models, 8)
        # Weighted LS: minimize sum_m wgt_m * (S_m @ w - f_true_m)^2
        W = np.diag(wgt)
        # Normal equations: (S^T W S) w = S^T W f_true
        SW = S.T * wgt[:, None]  # (8, n_models)
        A_ww = SW @ S           # (8, 8)
        b_w = SW @ f_true       # (8,)
        try:
            w_ls = np.linalg.solve(A_ww, b_w)
        except np.linalg.LinAlgError:
            w_ls, *_ = np.linalg.lstsq(A_ww, b_w, rcond=None)
        weights = _project_positive(w_ls)

        # --- Update nodes for fixed weights: minimize worst relative error
        def obj_nodes(v):
            nd = np.sort(np.clip(v, 1e-4, 1.0 - 1e-4))
            # enforce strict increase
            for i in range(1, 8):
                if nd[i] <= nd[i - 1] + 1e-7:
                    nd[i] = nd[i - 1] + 1e-7
            nd[-1] = min(nd[-1], 1.0 - 1e-4)
            return _worst_rel_err(nd, weights)

        res = minimize(obj_nodes, nodes, method="L-BFGS-B",
                       bounds=[(1e-4, 1.0 - 1e-4)] * 8,
                       options={"maxiter": 100, "ftol": 1e-14, "gtol": 1e-10})
        if res.success and np.all(np.isfinite(res.x)):
            nodes = np.sort(res.x)
            nodes = np.clip(nodes, 1e-4, 1.0 - 1e-4)
            for i in range(1, 8):
                if nodes[i] <= nodes[i - 1] + 1e-7:
                    nodes[i] = nodes[i - 1] + 1e-7

    # --- Step 3: Final polish with Nelder-Mead on nodes
    def obj_nodes_nm(v):
        nd = np.sort(np.clip(v, 1e-4, 1.0 - 1e-4))
        for i in range(1, 8):
            if nd[i] <= nd[i - 1] + 1e-8:
                nd[i] = nd[i - 1] + 1e-8
        nd[-1] = min(nd[-1], 1.0 - 1e-4)
        return _worst_rel_err(nd, weights)

    res2 = minimize(obj_nodes_nm, nodes, method="Nelder-Mead",
                    options={"maxiter": 2000, "xatol": 1e-12, "fatol": 1e-14})
    if res2.success and np.all(np.isfinite(res2.x)):
        nodes = np.sort(res2.x)
        nodes = np.clip(nodes, 1e-4, 1.0 - 1e-4)
        for i in range(1, 8):
            if nodes[i] <= nodes[i - 1] + 1e-8:
                nodes[i] = nodes[i - 1] + 1e-8

    # Final weight solve for the polished nodes
    S = _compute_samples(nodes)
    SW = S.T * wgt[:, None]
    A_ww = SW @ S
    b_w = SW @ f_true
    try:
        w_ls = np.linalg.solve(A_ww, b_w)
    except np.linalg.LinAlgError:
        w_ls, *_ = np.linalg.lstsq(A_ww, b_w, rcond=None)
    weights = _project_positive(w_ls)

    # Final sanity
    weights = np.maximum(weights, 1e-12)
    weights = weights / weights.sum()
    nodes = np.clip(np.sort(nodes), 1e-4, 1.0 - 1e-4)

    return {"nodes": nodes, "weights": weights}


def construct():
    w0_grid = np.array([-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    wa_grid = np.array([-1.0, -0.6, -0.3, 0.0, 0.3, 0.6, 1.0])
    try:
        rule = _solve(w0_grid, wa_grid)
    except Exception:
        n = 8
        rule = {"nodes": (np.arange(n) + 0.5) / n, "weights": np.full(n, 1.0 / n)}

    nodes = np.asarray(rule["nodes"], dtype=float)
    weights = np.asarray(rule["weights"], dtype=float)
    nodes = np.clip(np.sort(nodes), 1e-9, 1 - 1e-9)
    for i in range(1, 8):
        if nodes[i] <= nodes[i - 1] + 1e-9:
            nodes[i] = nodes[i - 1] + 1e-9
    weights = np.maximum(weights, 1e-12)
    weights = weights / weights.sum()
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
