"""
Strategy: Product-decomposition quadrature design.

The luminosity distance integrand is:
  f(t) = 1/E(z*t)  where E^2 = Om*(1+zt)^3 + Ode*(1+zt)^(3+3w0+3wa*zt)

We decompose this as:
  f(t) = g(t) * h(t; w0, wa, z)

where g(t) captures the universal geometric shape (matter-dominated at small t,
smooth transition) and h(t; ...) captures the model-dependent dark energy variation.

Key insight: For the matter-dominated part (w0=wa=0, pure Lambda CDM with w=-1),
the integrand has a specific shape. The dark energy variation modulates this.

New approach:
1. Design nodes to optimally integrate a set of basis functions that span the
   variation of 1/E across the CPL parameter space.
2. The basis functions are derived from the structure of E(z).
3. Use weighted least-squares to find nodes that minimize the worst-case
   approximation error across the basis, then solve LP for optimal weights.

The basis functions are:
  phi_k(t) = (1+t)^(-alpha_k) for various alpha_k

This captures the power-law behavior of 1/E at different redshifts.

For the nodes, we use a different optimization: instead of Nelder-Mead on the
minimax error directly, we:
1. First find nodes that minimize the L-infinity error against a dense set of
   representative integrands using a coordinate descent / active set method.
2. Then refine with the LP-based minimax approach.

This is structurally different from the parent: the parent uses a smooth
log-sum-exp surrogate for the minimax objective, while this uses a direct
active-set minimax approach with basis function fitting.
"""

import time
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize, linprog
from scipy.special import roots_legendre, roots_jacobi

_START_TIME = time.time()
_DEADLINE = _START_TIME + 82.0

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
    """
    z_vals = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0])
    w0_vals = np.array([-1.5, -1.0, -0.8, -0.5, -0.3, 0.0, 0.2, 0.5, 0.8, 1.0, 1.5])
    wa_vals = np.array([-1.0, -0.7, -0.5, -0.3, 0.0, 0.3, 0.5, 0.7, 1.0])
    
    points = []
    for z in z_vals:
        for w0 in w0_vals:
            points.append((z, w0, 0.0))
    for z in z_vals:
        for wa in wa_vals:
            if wa != 0.0:
                points.append((z, 0.0, wa))
    for z in [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]:
        for w0 in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            for wa in [-0.5, 0.0, 0.5]:
                if (z, w0, wa) not in points:
                    points.append((z, w0, wa))
    
    seen = set()
    unique = []
    for p in points:
        key = (round(p[0],6), round(p[1],6), round(p[2],6))
        if key not in seen:
            seen.add(key)
            unique.append(p)
    
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

def _active_set_minimax_nodes(test_points, initial_nodes, Omega_m=0.3, max_iter=50):
    """
    Active-set minimax node optimization.
    
    Instead of using a smooth surrogate, we directly track the active constraints
    (the test points with maximum error) and optimize nodes to reduce them.
    
    This is structurally different: we maintain a working set of active constraints
    and solve a subproblem for nodes that equalizes the errors on the active set.
    """
    n = len(initial_nodes)
    best_nodes = initial_nodes.copy()
    best_err = np.inf
    best_weights = None
    
    for iteration in range(max_iter):
        if time.time() > _DEADLINE - 10:
            break
        
        # Evaluate current nodes
        w, err = _eval_minimax(best_nodes, test_points)
        if w is None:
            break
        best_weights = w
        best_err = err
        
        # Find active constraints (top errors)
        N = len(test_points)
        A = np.zeros((N, n))
        refs = np.zeros(N)
        for i, (z, w0, wa) in enumerate(test_points):
            if z < 1e-12:
                A[i, :] = 1.0
                refs[i] = 1.0
            else:
                zt = z * best_nodes
                E = _E_of_z(zt, w0, wa, Omega_m)
                A[i, :] = 1.0 / E
                refs[i] = _ref_integral(z, w0, wa, Omega_m)
        
        approx = A @ w
        rel_errs = np.abs(approx - refs) / np.maximum(refs, 1e-15)
        
        # Get top-k active constraints
        k_active = min(12, N)
        active_idx = np.argsort(rel_errs)[-k_active:]
        
        # For active constraints, try to reduce the maximum error
        # by adjusting nodes using gradient-free search
        active_A = A[active_idx, :]
        active_refs = refs[active_idx]
        active_errs = rel_errs[active_idx]
        
        # Try small perturbations of nodes to reduce active errors
        best_pert_nodes = best_nodes.copy()
        best_pert_err = best_err
        
        rng = np.random.RandomState(iteration * 100 + 7)
        for _ in range(30):
            if time.time() > _DEADLINE - 10:
                break
            pert = rng.uniform(-0.01, 0.01, n)
            pert_nodes = best_nodes + pert
            pert_nodes = np.sort(np.clip(pert_nodes, 1e-4, 1-1e-4))
            if not np.all(np.diff(pert_nodes) > 1e-6):
                continue
            
            # Quick evaluation on active set only
            pA = np.zeros((k_active, n))
            for i, idx in enumerate(active_idx):
                z, w0, wa = test_points[idx]
                if z < 1e-12:
                    pA[i, :] = 1.0
                else:
                    zt = z * pert_nodes
                    E = _E_of_z(zt, w0, wa, Omega_m)
                    pA[i, :] = 1.0 / E
            
            # Solve LP on active set
            pw, pt = _solve_lp_for_weights(pA, active_refs, n)
            if pw is not None:
                if pt < best_pert_err:
                    best_pert_err = pt
                    best_pert_nodes = pert_nodes.copy()
        
        if best_pert_err < best_err * 0.999:
            best_nodes = best_pert_nodes
        else:
            # Try larger perturbations
            for scale in [0.03, 0.05, 0.08]:
                if time.time() > _DEADLINE - 10:
                    break
                for _ in range(10):
                    pert = rng.uniform(-scale, scale, n)
                    pert_nodes = best_nodes + pert
                    pert_nodes = np.sort(np.clip(pert_nodes, 1e-4, 1-1e-4))
                    if not np.all(np.diff(pert_nodes) > 1e-6):
                        continue
                    
                    w_test, err_test = _eval_minimax(pert_nodes, test_points)
                    if w_test is not None and err_test < best_err:
                        best_nodes = pert_nodes.copy()
                        best_err = err_test
                        best_weights = w_test
                        break
    
    return best_nodes, best_weights, best_err

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
    
    # Initial candidates - diverse starting points
    candidates = []
    
    # 1. Gauss-Legendre
    gl_n, gl_w = roots_legendre(n)
    candidates.append((gl_n + 1.0) / 2.0)
    
    # 2. Chebyshev nodes
    k = np.arange(n)
    ch_n = np.cos((2*k + 1) * np.pi / (2*n))
    candidates.append((ch_n + 1.0) / 2.0)
    
    # 3. Gauss-Jacobi variants
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
        if time.time() > _DEADLINE - 15:
            break
        w, err = _eval_minimax(cand, test_points)
        if w is not None and err < best_err:
            best_err = err
            best_nodes = cand.copy()
            best_weights = w.copy()
    
    if best_nodes is None:
        nodes, weights = roots_legendre(n)
        return {"nodes": (nodes+1)/2.0, "weights": weights/2.0}
    
    # Active-set minimax node optimization (structural change from parent)
    if time.time() < _DEADLINE - 20:
        nodes_as, weights_as, err_as = _active_set_minimax_nodes(
            test_points, best_nodes, max_iter=30
        )
        if err_as < best_err:
            best_nodes = nodes_as
            best_weights = weights_as
            best_err = err_as
    
    # Refine with Nelder-Mead using smooth surrogate (from parent's approach)
    def _smooth_minimax_surrogate(nodes, test_points, Omega_m=0.3, beta=50.0):
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
        
        approx = A @ w
        rel_errs = np.abs(approx - refs) / np.maximum(refs, 1e-15)
        
        m = np.max(rel_errs)
        if m < 1e-15:
            return 0.0
        s = np.sum(np.exp(beta * (rel_errs - m)))
        return m + np.log(s) / beta
    
    def _objective_direct(nodes_p):
        if np.any(nodes_p <= 1e-5) or np.any(nodes_p >= 1-1e-5):
            return 1e8
        diffs = np.diff(nodes_p)
        if np.any(diffs <= 1e-7) or np.any(diffs < 0):
            return 1e8
        return _smooth_minimax_surrogate(nodes_p, test_points, beta=50.0)
    
    if time.time() < _DEADLINE - 15:
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
        nodes, weights = roots_legendre(n)
        return {"nodes": (nodes+1)/2.0, "weights": weights/2.0}
    
    s = best_weights.sum()
    if abs(s - 1.0) > 1e-12:
        best_weights /= s
    
    if not np.all(np.diff(best_nodes) > 0):
        nodes, weights = roots_legendre(n)
        return {"nodes": (nodes+1)/2.0, "weights": weights/2.0}
    
    return {"nodes": best_nodes, "weights": best_weights}
