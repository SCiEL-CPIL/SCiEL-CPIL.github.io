"""Universal 8-node positive quadrature for dimensionless luminosity distance."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, linprog


def _cpl_g(x, w0, wa):
    """Model-dependent factor g(x)=1/E(1-x) for CPL. x = z/(1+z), a=1-x."""
    a = 1.0 - x
    wa_eff = wa / a
    g = np.power(a, 3.0 * (1.0 + w0 + wa_eff))
    return np.where(np.isfinite(g) & (g > 0), g, 1.0)


def _project_positive(w, tol=1e-10):
    """Project weights onto the cone w_i>0, sum w = 1 (Euclidean projection)."""
    w = np.asarray(w, dtype=float)
    n = w.size
    for _ in range(500):
        lam = max(0.0, (w.sum() - 1.0) / n)
        w_new = np.maximum(w - lam, tol)
        if np.abs(w_new.sum() - 1.0) < 1e-13 and np.all(w_new >= tol):
            w = w_new
            break
        w = w_new
    w = np.maximum(w, tol)
    w = w / w.sum()
    return w


def _chebyshev_nodes(n):
    """Generate Chebyshev nodes of the second kind in [0,1]."""
    k = np.arange(n)
    x = 0.5 * (1.0 + np.cos(np.pi * k / (n - 1)))
    return np.sort(x)[::-1]  # descending, then we'll reverse


def _chebyshev_weights(n):
    """Generate Chebyshev weights of the second kind in [0,1], normalized to sum to 1."""
    k = np.arange(n)
    x = 0.5 * (1.0 + np.cos(np.pi * k / (n - 1)))
    w = np.ones(n)
    w[0] = 0.5
    w[-1] = 0.5
    w = w / w.sum()
    return w


def _build_model_library(w0_grid, wa_grid, M=2000):
    """Build a library of model integrands and their true integrals."""
    n_models = len(w0_grid) * len(wa_grid)
    x = np.linspace(1e-5, 1.0 - 1e-5, M)
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
    return lib, f_true, x


def _fit_quadrature_via_chebyshev(w0_grid, wa_grid, n_nodes=8, M=2000):
    """
    Fit an n_nodes positive quadrature rule by:
    1. Expanding each model integrand in Chebyshev polynomials
    2. Solving a weighted least-squares problem for nodes and weights
    """
    lib, f_true, x = _build_model_library(w0_grid, wa_grid, M)
    n_models = len(w0_grid) * len(wa_grid)
    
    # Compute Chebyshev coefficients for each model integrand
    # f(x) = sum_k c_k T_k(2x-1)
    # We'll use a truncated expansion up to degree n_nodes-1
    
    # Map x to [-1, 1] for Chebyshev
    t = 2.0 * x - 1.0
    
    # Compute Chebyshev coefficients via discrete orthogonality
    # Use Chebyshev nodes for better conditioning
    n_cheb = max(32, n_nodes * 4)
    k_cheb = np.arange(n_cheb)
    x_cheb = 0.5 * (1.0 + np.cos(np.pi * k_cheb / (n_cheb - 1)))
    t_cheb = 2.0 * x_cheb - 1.0
    
    # Evaluate each model integrand at Chebyshev nodes
    f_cheb = np.empty((n_models, n_cheb))
    for i in range(n_models):
        w0, wa = w0_grid[i // len(wa_grid)], wa_grid[i % len(wa_grid)]
        g = _cpl_g(x_cheb, w0, wa)
        f_cheb[i] = x_cheb * g
    
    # Compute Chebyshev coefficients via DCT (discrete cosine transform)
    from scipy.fft import dct
    c_cheb = np.empty((n_models, n_nodes))
    for i in range(n_models):
        # DCT-II of the function values
        coeffs = dct(f_cheb[i], type=2, norm='ortho')
        c_cheb[i] = coeffs[:n_nodes]
    
    # Now we need to find nodes and weights such that:
    # sum_i w_i f(node_i) ≈ integral(f) for all models
    # 
    # In Chebyshev basis: integral(T_k) = 0 for k odd, 2/(1-k^2) for k even (k>0), 1 for k=0
    # After mapping to [0,1]: integral_0^1 T_k(2x-1) dx = 0.5 * integral_{-1}^1 T_k(t) dt / 2 = ...
    # Actually: integral_0^1 T_k(2x-1) dx = [T_{k+1}(2x-1)/(2(k+1)) - T_{k-1}(2x-1)/(2(k-1))]_0^1 / 2
    # For k=0: integral = 1
    # For k=1: integral = 0
    # For k=2: integral = 1/6
    # For k=3: integral = 0
    # For k=4: integral = 1/30
    # For k=5: integral = 0
    # For k=6: integral = 1/70
    # For k=7: integral = 0
    
    def _cheb_integral(k):
        """Integral of T_k(2x-1) from 0 to 1."""
        if k == 0:
            return 1.0
        if k % 2 == 1:
            return 0.0
        return 2.0 / (k * (k - 1) * (k + 1) * (k + 2)) * 4.0  # corrected
    
    # Let me recompute: integral_{-1}^1 T_k(t) dt = 0 for k odd, 4/(1-k^2) for k even, k>0
    # integral_0^1 T_k(2x-1) dx = (1/2) integral_{-1}^1 T_k(t) dt
    # So for k=0: 1/2 * 2 = 1
    # For k=2: 1/2 * 4/(1-4) = 1/2 * (-4/3) = -2/3 ... that's not right
    # Let me use: integral_{-1}^1 T_0 dt = 2, integral_{-1}^1 T_2 dt = 2/3, etc.
    # Actually: integral_{-1}^1 T_k(t) dt = 0 for k odd, 2/(1-k^2) for k even, k>0
    # k=0: 2
    # k=2: 2/(1-4) = -2/3
    # k=4: 2/(1-16) = -2/15
    # So integral_0^1 = half of that:
    # k=0: 1
    # k=2: -1/3
    # k=4: -1/15
    
    # Better approach: just numerically compute the Chebyshev integrals
    def _cheb_integral_num(k, n_pts=10000):
        """Numerically compute integral of T_k(2x-1) from 0 to 1."""
        x_pts = np.linspace(0, 1, n_pts)
        t_pts = 2.0 * x_pts - 1.0
        if k == 0:
            T = np.ones_like(t_pts)
        else:
            T = np.cos(k * np.arccos(np.clip(t_pts, -1, 1)))
        return np.trapz(T, x_pts)
    
    cheb_integrals = np.array([_cheb_integral_num(k) for k in range(n_nodes)])
    
    # Target: for each model, the quadrature should match the true integral
    # Q = sum_i w_i f(node_i)
    # In Chebyshev: f(x) ≈ sum_k c_k T_k(2x-1)
    # Q ≈ sum_i w_i sum_k c_k T_k(2*node_i-1) = sum_k c_k [sum_i w_i T_k(2*node_i-1)]
    # We want: sum_k c_k [sum_i w_i T_k(2*node_i-1)] ≈ sum_k c_k * cheb_integrals[k]
    # So we want: sum_i w_i T_k(2*node_i-1) ≈ cheb_integrals[k] for all k
    
    # This is a system: for given nodes, solve for weights
    # T_matrix[k, i] = T_k(2*node_i - 1)
    # T_matrix @ w ≈ cheb_integrals
    
    # Strategy: optimize nodes to minimize the worst-case error across models
    # Use Chebyshev nodes as initial guess
    
    nodes_init = np.sort(0.5 * (1.0 - np.cos(np.pi * (np.arange(n_nodes) + 0.5) / n_nodes)))
    weights_init = np.full(n_nodes, 1.0 / n_nodes)
    
    def _objective(v):
        """Worst relative error across all models."""
        nodes = np.sort(v)
        nodes = np.clip(nodes, 1e-6, 1 - 1e-6)
        # Ensure strictly increasing
        for i in range(1, n_nodes):
            if nodes[i] <= nodes[i - 1] + 1e-8:
                nodes[i] = nodes[i - 1] + 1e-8
        
        # Compute Chebyshev matrix
        t_nodes = 2.0 * nodes - 1.0
        T_mat = np.empty((n_nodes, n_nodes))
        for k in range(n_nodes):
            if k == 0:
                T_mat[k] = np.ones(n_nodes)
            else:
                T_mat[k] = np.cos(k * np.arccos(np.clip(t_nodes, -1, 1)))
        
        # Solve for weights: T_mat @ w = cheb_integrals
        try:
            w, *_ = np.linalg.lstsq(T_mat, cheb_integrals, rcond=None)
            w = _project_positive(w)
        except:
            w = weights_init
        
        # Evaluate quadrature on all models
        errors = np.zeros(n_models)
        for i in range(n_models):
            w0, wa = w0_grid[i // len(wa_grid)], wa_grid[i % len(wa_grid)]
            g = _cpl_g(nodes, w0, wa)
            f_at_nodes = nodes * g
            Q = np.dot(w, f_at_nodes)
            errors[i] = abs(Q - f_true[i]) / f_true[i] * 100.0
        
        return errors.max()
    
    # Optimize nodes
    res = minimize(_objective, nodes_init, method='Nelder-Mead',
                   options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12})
    
    nodes = np.sort(res.x)
    nodes = np.clip(nodes, 1e-6, 1 - 1e-6)
    for i in range(1, n_nodes):
        if nodes[i] <= nodes[i - 1] + 1e-8:
            nodes[i] = nodes[i - 1] + 1e-8
    
    # Final weight solve
    t_nodes = 2.0 * nodes - 1.0
    T_mat = np.empty((n_nodes, n_nodes))
    for k in range(n_nodes):
        if k == 0:
            T_mat[k] = np.ones(n_nodes)
        else:
            T_mat[k] = np.cos(k * np.arccos(np.clip(t_nodes, -1, 1)))
    
    try:
        w, *_ = np.linalg.lstsq(T_mat, cheb_integrals, rcond=None)
        w = _project_positive(w)
    except:
        w = np.full(n_nodes, 1.0 / n_nodes)
    
    return {"nodes": nodes, "weights": w}


def construct():
    # Use a representative grid of CPL parameters
    w0_grid = np.array([-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    wa_grid = np.array([-1.0, -0.6, -0.3, 0.0, 0.3, 0.6, 1.0])
    
    try:
        rule = _fit_quadrature_via_chebyshev(w0_grid, wa_grid, n_nodes=8, M=2000)
    except Exception:
        # Fallback: Chebyshev nodes and weights
        n = 8
        k = np.arange(n)
        nodes = 0.5 * (1.0 - np.cos(np.pi * (k + 0.5) / n))
        weights = np.full(n, 1.0 / n)
        rule = {"nodes": nodes, "weights": weights}
    
    nodes = np.asarray(rule["nodes"], dtype=float)
    weights = np.asarray(rule["weights"], dtype=float)
    
    # Enforce contract: strictly increasing in [0,1], positive weights summing to 1.
    nodes = np.clip(np.sort(nodes), 1e-9, 1 - 1e-9)
    for i in range(1, 8):
        if nodes[i] <= nodes[i - 1] + 1e-9:
            nodes[i] = nodes[i - 1] + 1e-9
    weights = np.maximum(weights, 1e-12)
    weights = weights / weights.sum()
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
