"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def construct():
    """
    Discover an eight-node positive quadrature rule for dimensionless luminosity distance.
    
    Uses a two-phase optimization: first optimize nodes with uniform weights,
    then jointly optimize nodes and weights using a better parameterization.
    """

    def E_of_a(a, w0, wa):
        """Hubble parameter normalized by H0 for CPL model."""
        a = np.maximum(a, 1e-15)
        if abs(w0 + 1.0) < 1e-10:
            return np.sqrt(a**(-3) + (1.0 + wa) * np.exp(-3.0 * wa * (1.0 - a)))
        else:
            exponent = -3.0 * (1.0 + w0)
            prefactor = (1.0 + w0 + wa)
            exp_term = np.exp(-3.0 * wa * (1.0 - a) / (1.0 + w0))
            return np.sqrt(a**(-3) + prefactor * a**exponent * exp_term)

    def exact_luminosity_distance(z, w0, wa):
        """Compute exact dL/(c/H0) via adaptive quadrature."""
        from scipy.integrate import quad
        
        def integrand(zp):
            a = 1.0 / (1.0 + zp)
            return (1.0 + zp) * zp / E_of_a(a, w0, wa)
        
        result, _ = quad(integrand, 0.0, z, limit=200)
        return result

    def quadrature_luminosity_distance(z, nodes, weights, w0, wa):
        """Compute quadrature approximation of dL/(c/H0)."""
        a_vals = 1.0 / (1.0 + z * nodes)
        E_vals = E_of_a(a_vals, w0, wa)
        return (1.0 + z) * z * np.sum(weights / E_vals)

    # Training cosmologies - broader coverage of parameter space
    training_cosmos = [
        (0.0, 0.0),      # Lambda CDM
        (-0.5, 0.5),     # quintessence-like
        (-1.5, 0.5),     # phantom-like
        (-1.0, 0.5),     # w0 = -1
        (-1.0, -0.5),    # w0 = -1, negative wa
        (-0.5, -0.5),    # another
        (-1.5, -0.5),    # another
        (0.0, 0.5),      # another
        (-1.0, 0.0),     # pure cosmological constant with w0=-1
        (-0.8, 0.3),     # intermediate
        (-1.2, -0.3),    # intermediate
        (0.2, -0.2),     # matter-dominated-ish
    ]

    # Test redshifts - include a wider range
    test_zs = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0])

    # Precompute exact values
    exact_values = {}
    for (w0, wa) in training_cosmos:
        exact_values[(w0, wa)] = np.array([exact_luminosity_distance(z, w0, wa) for z in test_zs])

    n = 8
    
    def compute_worst_error(nodes, weights):
        """Compute worst-case relative error across all training cosmologies."""
        max_rel_err = 0.0
        for (w0, wa) in training_cosmos:
            exact = exact_values[(w0, wa)]
            for i_z, z in enumerate(test_zs):
                approx = quadrature_luminosity_distance(z, nodes, weights, w0, wa)
                rel_err = abs(approx - exact[i_z]) / max(exact[i_z], 1e-15)
                max_rel_err = max(max_rel_err, rel_err)
        return max_rel_err

    # Phase 1: Optimize nodes with uniform weights
    # Use Chebyshev nodes as initial guess (good for polynomial approximation)
    def sigmoid(t):
        return 1.0 / (1.0 + np.exp(-t))
    
    # Initial: Chebyshev-like nodes mapped to (0,1)
    # Chebyshev nodes on [0,1]: 0.5*(1+cos((2k+1)*pi/(2n)))
    cheb_nodes = 0.5 * (1.0 + np.cos((2*np.arange(n)+1) * np.pi / (2*n)))
    cheb_nodes = np.sort(cheb_nodes)  # sort to be increasing
    # Clip to avoid endpoints
    cheb_nodes = np.clip(cheb_nodes, 0.01, 0.99)
    
    t_init = np.log(cheb_nodes / (1.0 - cheb_nodes))  # logit
    uniform_weights = np.ones(n) / n
    
    def objective_nodes(params):
        t = params
        nodes = sigmoid(t)
        # Enforce strictly increasing
        diffs = np.diff(nodes)
        if np.any(diffs <= 1e-10):
            return 1e10
        return compute_worst_error(nodes, uniform_weights)
    
    result1 = minimize(objective_nodes, t_init, method='Nelder-Mead',
                       options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10})
    
    nodes_phase1 = sigmoid(result1.x)
    nodes_phase1 = np.clip(nodes_phase1, 1e-4, 1.0 - 1e-4)
    
    # Ensure strict increase
    for i in range(1, n):
        if nodes_phase1[i] <= nodes_phase1[i-1] + 1e-8:
            nodes_phase1[i] = nodes_phase1[i-1] + 1e-6
    
    # Phase 2: Jointly optimize nodes and weights
    # Parameterize weights as exp(log_w) normalized
    def objective_joint(params):
        t = params[:n]
        log_w = params[n:]
        
        nodes = sigmoid(t)
        w = np.exp(log_w)
        w = w / np.sum(w)
        
        # Enforce strictly increasing nodes
        diffs = np.diff(nodes)
        if np.any(diffs <= 1e-10):
            return 1e10 + np.sum(np.minimum(diffs, 0)**2)
        
        return compute_worst_error(nodes, w)
    
    t_init2 = np.log(nodes_phase1 / (1.0 - nodes_phase1))
    log_w_init2 = np.zeros(n)
    x0 = np.concatenate([t_init2, log_w_init2])
    
    # Try multiple optimization methods and keep the best
    best_result = None
    best_score = float('inf')
    
    # Method 1: Nelder-Mead
    res1 = minimize(objective_joint, x0, method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-9, 'fatol': 1e-11})
    if res1.fun < best_score:
        best_score = res1.fun
        best_result = res1
    
    # Method 2: COBYLA
    try:
        res2 = minimize(objective_joint, x0, method='COBYLA',
                       options={'maxiter': 3000, 'rhobeg': 1e-3})
        if res2.fun < best_score:
            best_score = res2.fun
            best_result = res2
    except:
        pass
    
    # Method 3: Powell
    try:
        res3 = minimize(objective_joint, x0, method='Powell',
                       options={'maxiter': 3000, 'xtol': 1e-9, 'ftol': 1e-11})
        if res3.fun < best_score:
            best_score = res3.fun
            best_result = res3
    except:
        pass
    
    if best_result is None:
        best_result = res1
    
    t_opt = best_result.x[:n]
    log_w_opt = best_result.x[n:]
    
    nodes = sigmoid(t_opt)
    weights = np.exp(log_w_opt)
    weights = weights / np.sum(weights)
    
    # Ensure nodes are strictly increasing and in (0,1)
    nodes = np.clip(nodes, 1e-4, 1.0 - 1e-4)
    
    # Ensure strict increase with minimum gap
    min_gap = 1e-6
    for i in range(1, n):
        if nodes[i] <= nodes[i-1] + min_gap:
            nodes[i] = nodes[i-1] + min_gap
    
    # Final normalization and validation
    weights = weights / np.sum(weights)
    
    # Verify constraints
    assert len(nodes) == 8
    assert len(weights) == 8
    assert np.all(np.diff(nodes) > 0)
    assert np.all(nodes > 0) and np.all(nodes < 1)
    assert np.all(weights > 0)
    assert abs(np.sum(weights) - 1.0) < 1e-12
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
