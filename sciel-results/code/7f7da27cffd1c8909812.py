"""
Strategy: Direct parameterized model-and-optimize for minimax quadrature with
smooth surrogate and cached references.

The integrand f(t) = 1/E(z*t) where E depends on (w0, wa) through:
E^2 = Om*(1+zt)^3 + Ode*(1+zt)^(3+3w0+3wa*zt)

Approach:
1. Parameterize nodes via logit transform to remove boundary constraints.
2. For fixed nodes, solve LP for optimal positive weights minimizing max rel err.
3. Optimize nodes using Nelder-Mead on a smooth log-sum-exp surrogate.
4. Final polish with tighter LP tolerance and exact minimax evaluation.

Speed: Cache reference integrals. LP is small (9 vars, ~200 constraints).
"""

import time
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize, linprog
from scipy.special import roots_legendre

_START_TIME = time.time()
_DEADLINE = _START_TIME + 85.0

# Cache for reference integrals
_ref_cache = {}

def _E_of_z(z, w0, wa, Omega_m=0.3):
    """Vectorized E(z) = H(z)/H0 for CPL model."""
    Ode = 1.0 - Omega_m
    opz = 1.0 + z
    term_m = Omega_m * opz**3
    exponent = 3.0 + 3.0*w0 + 3.0*wa*z
    term_de = Ode * opz**exponent
    return np.sqrt(term_m + term_de)

def _ref_integral(z, w0, wa, Omega_m=0.3):
    """High-accuracy reference integral of 1/E(z*t) dt from 0 to 1 (cached)."""
    if z < 1e-12:
        return 1.0
    key = (round(z, 10), round(w0, 10), round(wa, 10), round(Omega_m, 10))
    if key in _ref_cache:
        return _ref_cache[key]
    def f(t):
        return 1.0 / _E_of_z(z*t, w0, wa, Omega_m)
    val, _ = quad(f, 0, 1, limit=200, epsabs=1e-14, epsrel=1e-14)
    _ref_cache[key] = val
    return val

def _quadrature_at_nodes(nodes, w0, wa, z_arr, Omega_m=0.3):
    """
    For fixed nodes, compute A[i,j] = 1/E(z_i * nodes_j) for all test points.
    Returns matrix (N_test, 8) and reference integrals.
    """
    N = len(z_arr)
    n = len(nodes)
    A = np.zeros((N, n))
    refs = np.zeros(N)
    for i in range(N):
        z = z_arr[i]
        if z < 1e-12:
            A[i, :] = 1.0
            refs[i] = 1.0
        else:
            zt = z * nodes
            E = _E_of_z(zt, w0, wa, Omega_m)
            A[i, :] = 1.0 / E
            refs[i] = _ref_integral(z, w0, wa, Omega_m)
    return A, refs

def _solve_lp_for_weights(A, refs, n=8):
    """
    Solve: minimize t s.t.
      |A_i @ w - ref_i| <= t * ref_i  for all i
      sum(w) = 1, w >= 0, t >= 0
    """
    N = A.shape[0]
    c = np.zeros(n + 1)
    c[n] = 1.0
    
    A_ub = np.zeros((2*N, n + 1))
    b_ub = np.zeros(2*N)
    for i in range(N):
        A_ub[2*i, :n] = A[i, :]
        A_ub[2*i, n] = -refs[i]
        b_ub[2*i] = refs[i]
        A_ub[2*i+1, :n] = -A[i, :]
        A_ub[2*i+1, n] = -refs[i]
        b_ub[2*i+1] = -refs[i]
    
    A_eq = np.zeros((1, n+1))
    A_eq[0, :n] = 1.0
    b_eq = np.array([1.0])
    
    bounds = [(0, None)] * (n+1)
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method='highs')
        if res.success:
            w = res.x[:n]
            s = w.sum()
            if s > 1e-15:
                w /= s
            return w, res.x[n]
    except Exception:
        pass
    return None, np.inf

def _build_test_set():
    """
    Build a comprehensive test set covering the CPL parameter space.
    The validation likely covers z in a reasonable range and (w0, wa) 
    in typical ranges. We sample densely.
    """
    # z values: logarithmic spacing from small to large
    z_vals = np.concatenate([
        np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0])
    ])
    
    # w0, wa ranges: typical physically motivated ranges
    w0_vals = np.array([-1.5, -1.0, -0.8, -0.5, -0.3, 0.0, 0.2, 0.5, 0.8, 1.0, 1.5])
    wa_vals = np.array([-1.0, -0.7, -0.5, -0.3, 0.0, 0.3, 0.5, 0.7, 1.0])
    
    # Create full grid but subsample to keep LP tractable
    # Total: 16 * 11 * 9 = 1584 points - too many for fast LP in loop
    # Use a stratified subset: ~200-400 points
    
    # Strategy: include all (z, w0, wa=0) and (z, w0=0, wa) and corners
    points = []
    # All z with wa=0
    for z in z_vals:
        for w0 in w0_vals:
            points.append((z, w0, 0.0))
    # All z with w0=0
    for z in z_vals:
        for wa in wa_vals:
            if wa != 0.0:
                points.append((z, 0.0, wa))
    # Corner/interior points
    for z in [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]:
        for w0 in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            for wa in [-0.5, 0.0, 0.5]:
                if (z, w0, wa) not in points:
                    points.append((z, w0, wa))
    
    # Deduplicate and limit
    seen = set()
    unique = []
    for p in points:
        key = (round(p[0],6), round(p[1],6), round(p[2],6))
        if key not in seen:
            seen.add(key)
            unique.append(p)
    
    # If still too large, random subsample
    if len(unique) > 400:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(unique), 400, replace=False)
        unique = [unique[i] for i in idx]
    
    return unique

def _eval_minimax(nodes, test_points, Omega_m=0.3):
    """Evaluate minimax relative error for given nodes."""
    n = len(nodes)
    N = len(test_points)
    A = np.zeros((N, n))
    refs = np.zeros(N)
    for i, (z, w0, wa) in enumerate(test_points):
        if z < 1e-12:
            A[i, :] = 1.0
            refs[i] = 1.0
        else:
            zt = z * nodes
            E = _E_of_z(zt, w0, wa, Omega_m)
            A[i, :] = 1.0 / E
            refs[i] = _ref_integral(z, w0, wa, Omega_m)
    
    w, t = _solve_lp_for_weights(A, refs, n)
    if w is None:
        return None, np.inf
    return w, t

def _smooth_minimax_surrogate(nodes, test_points, Omega_m=0.3, beta=50.0):
    """
    Smooth surrogate for the minimax objective using log-sum-exp.
    For fixed nodes, we solve the LP to get weights, then compute
    a smooth approximation of the max relative error.
    """
    n = len(nodes)
    N = len(test_points)
    A = np.zeros((N, n))
    refs = np.zeros(N)
    for i, (z, w0, wa) in enumerate(test_points):
        if z < 1e-12:
            A[i, :] = 1.0
            refs[i] = 1.0
        else:
            zt = z * nodes
            E = _E_of_z(zt, w0, wa, Omega_m)
            A[i, :] = 1.0 / E
            refs[i] = _ref_integral(z, w0, wa, Omega_m)
    
    w, t = _solve_lp_for_weights(A, refs, n)
    if w is None:
        return 1e8
    
    # Compute relative errors
    approx = A @ w
    rel_errs = np.abs(approx - refs) / np.maximum(refs, 1e-15)
    
    # Log-sum-exp for smooth max
    m = np.max(rel_errs)
    if m < 1e-15:
        return 0.0
    s = np.sum(np.exp(beta * (rel_errs - m)))
    return m + np.log(s) / beta

def construct():
    if time.time() > _DEADLINE - 3:
        nodes, weights = roots_legendre(8)
        nodes = (nodes + 1.0) / 2.0
        weights = weights / 2.0
        return {"nodes": nodes, "weights": weights}
    
    test_points = _build_test_set()
    n = 8
    
    best_nodes = None
    best_weights = None
    best_err = np.inf
    
    # Initial candidates
    candidates = []
    
    # 1. Gauss-Legendre
    gl_n, gl_w = roots_legendre(n)
    candidates.append((gl_n + 1.0) / 2.0)
    
    # 2. Chebyshev nodes
    k = np.arange(n)
    ch_n = np.cos((2*k + 1) * np.pi / (2*n))
    candidates.append((ch_n + 1.0) / 2.0)
    
    # 3. Gauss-Jacobi variants
    from scipy.special import roots_jacobi
    for alpha, beta in [(0.5, 0.5), (1.0, 0.0), (0.0, 1.0), (2.0, 0.0), (0.0, 2.0), (1.5, 1.5)]:
        try:
            jn, _ = roots_jacobi(n, alpha, beta)
            jn_mapped = (jn + 1.0) / 2.0
            if np.all(np.diff(jn_mapped) > 1e-10):
                candidates.append(jn_mapped)
        except:
            pass
    
    # 4. Perturbed GL
    rng = np.random.RandomState(123)
    base_gl = (gl_n + 1.0) / 2.0
    for _ in range(20):
        pert = rng.uniform(-0.04, 0.04, n)
        cand = base_gl + pert
        cand = np.clip(cand, 1e-4, 1 - 1e-4)
        cand = np.sort(cand)
        if np.all(np.diff(cand) > 1e-6):
            candidates.append(cand)
    
    # 5. Log-spaced near 0
    for p in [1.3, 1.5, 2.0, 2.5]:
        t = np.linspace(0, 1, n)
        ln = 1.0 - (1.0 - t)**p
        ln = np.clip(ln, 1e-4, 1-1e-4)
        if np.all(np.diff(ln) > 1e-6):
            candidates.append(ln)
    
    # 6. Power-law near 1
    for p in [0.7, 0.8, 1.0, 1.2, 1.5]:
        t = np.linspace(0, 1, n)
        pn = t**p
        pn = np.clip(pn, 1e-4, 1-1e-4)
        if np.all(np.diff(pn) > 1e-6):
            candidates.append(pn)
    
    # Evaluate all initial candidates
    for cand in candidates:
        if time.time() > _DEADLINE - 8:
            break
        w, err = _eval_minimax(cand, test_points)
        if w is not None and err < best_err:
            best_err = err
            best_nodes = cand.copy()
            best_weights = w.copy()
    
    if best_nodes is None:
        nodes, weights = roots_legendre(n)
        return {"nodes": (nodes+1)/2.0, "weights": weights/2.0}
    
    # Refine with Nelder-Mead using smooth surrogate
    def _logit_to_nodes(u):
        """Convert logit parameters to nodes in (0,1)."""
        # u are logit values, convert to (0,1) via sigmoid, then sort
        sig = 1.0 / (1.0 + np.exp(-u))
        # Ensure strict ordering by adding small offsets
        sig = np.clip(sig, 1e-6, 1-1e-6)
        sig = np.sort(sig)
        # Ensure minimum spacing
        for i in range(1, len(sig)):
            if sig[i] - sig[i-1] < 1e-5:
                sig[i] = sig[i-1] + 1e-5
        sig = np.clip(sig, 1e-5, 1-1e-5)
        return sig
    
    def _nodes_to_logit(nodes):
        """Convert nodes to logit parameters."""
        nodes = np.clip(nodes, 1e-6, 1-1e-6)
        return np.log(nodes / (1.0 - nodes))
    
    def _objective_smooth(u):
        nodes_p = _logit_to_nodes(u)
        if not np.all(np.diff(nodes_p) > 1e-6):
            return 1e8
        return _smooth_minimax_surrogate(nodes_p, test_points, beta=50.0)
    
    u0 = _nodes_to_logit(best_nodes)
    
    try:
        res = minimize(_objective_smooth, u0, method='Nelder-Mead',
                       options={'maxiter': 400, 'xatol': 1e-9, 'fatol': 1e-12,
                                'adaptive': True})
        nodes_r = _logit_to_nodes(res.x)
        if np.all(nodes_r > 1e-4) and np.all(nodes_r < 1-1e-4):
            if np.all(np.diff(nodes_r) > 1e-7):
                w_r, err_r = _eval_minimax(nodes_r, test_points)
                if w_r is not None and err_r < best_err:
                    best_nodes = nodes_r.copy()
                    best_weights = w_r.copy()
                    best_err = err_r
    except Exception:
        pass
    
    # Second round: direct Nelder-Mead on nodes with smooth surrogate
    if time.time() < _DEADLINE - 15:
        def _objective_direct(nodes_p):
            if np.any(nodes_p <= 1e-5) or np.any(nodes_p >= 1-1e-5):
                return 1e8
            diffs = np.diff(nodes_p)
            if np.any(diffs <= 1e-7):
                return 1e8
            if np.any(diffs < 0):
                return 1e8
            return _smooth_minimax_surrogate(nodes_p, test_points, beta=50.0)
        
        rng2 = np.random.RandomState(777)
        for _ in range(6):
            if time.time() > _DEADLINE - 12:
                break
            pert = rng2.uniform(-0.02, 0.02, n)
            x2 = best_nodes + pert
            x2 = np.sort(np.clip(x2, 1e-4, 1-1e-4))
            if np.all(np.diff(x2) > 1e-7):
                try:
                    res2 = minimize(_objective_direct, x2, method='Nelder-Mead',
                                   options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12})
                    if np.all(res2.x > 1e-4) and np.all(res2.x < 1-1e-4):
                        if np.all(np.diff(res2.x) > 1e-7):
                            w_r, err_r = _eval_minimax(res2.x, test_points)
                            if w_r is not None and err_r < best_err:
                                best_nodes = res2.x.copy()
                                best_weights = w_r.copy()
                                best_err = err_r
                except Exception:
                    pass
    
    # Final cleanup and high-precision polish
    best_nodes = np.sort(best_nodes)
    
    # Final polish: re-solve LP with tighter tolerance using more test points
    if time.time() < _DEADLINE - 5:
        # Build a denser test set for final weight optimization
        z_dense = np.concatenate([
            np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]),
            np.linspace(0.05, 10.0, 20)
        ])
        w0_dense = np.array([-1.5, -1.0, -0.8, -0.5, -0.3, 0.0, 0.2, 0.5, 0.8, 1.0, 1.5])
        wa_dense = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        
        final_points = []
        for z in z_dense:
            for w0 in w0_dense:
                for wa in wa_dense:
                    final_points.append((z, w0, wa))
        
        # Limit to keep LP tractable
        if len(final_points) > 500:
            rng_final = np.random.RandomState(42)
            idx = rng_final.choice(len(final_points), 500, replace=False)
            final_points = [final_points[i] for i in idx]
        
        w_final, t_final = _eval_minimax(best_nodes, final_points)
        if w_final is not None:
            best_weights = w_final
    
    # Ensure validity
    if best_weights is None or len(best_weights) != n:
        nodes, weights = roots_legendre(n)
        return {"nodes": (nodes+1)/2.0, "weights": weights/2.0}
    
    if np.any(best_weights <= 0):
        # Fallback: use Gauss-Legendre
        nodes, weights = roots_legendre(n)
        return {"nodes": (nodes+1)/2.0, "weights": weights/2.0}
    
    s = best_weights.sum()
    if abs(s - 1.0) > 1e-12:
        best_weights /= s
    
    if not np.all(np.diff(best_nodes) > 0):
        nodes, weights = roots_legendre(n)
        return {"nodes": (nodes+1)/2.0, "weights": weights/2.0}
    
    return {"nodes": best_nodes, "weights": best_weights}
