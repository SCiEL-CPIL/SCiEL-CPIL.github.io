"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, linprog
from scipy.integrate import quad

def construct():
    """
    Discover an eight-node positive quadrature rule for dimensionless luminosity distance
    in spatially flat, time-varying dark-energy models.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    # Define E(z) for CPL models
    def E_cpl(z, w0, wa, Omega_m=0.3):
        """Compute E(z) = H(z)/H0 for CPL model."""
        Omega_DE = 1.0 - Omega_m
        # Standard CPL: rho_DE ~ (1+z)^(-3(1+w0)) * exp(-3*wa*z/(1+z))
        # More accurately: E(z)^2 = Om*(1+z)^3 + (1-Om)*(1+z)^(3(1+w0+wa))
        # Using the power-law approximation which is standard for CPL
        term_m = Omega_m * (1.0 + z)**3
        term_de = Omega_DE * (1.0 + z)**(3.0 * (1.0 + w0 + wa))
        return np.sqrt(term_m + term_de)
    
    def E_lcdm(z, Omega_m=0.3):
        Omega_L = 1.0 - Omega_m
        return np.sqrt(Omega_m * (1+z)**3 + Omega_L)

    # Set of test models - comprehensive coverage of parameter space
    test_models = []
    # Omega_m values
    for Om in [0.25, 0.30, 0.35, 0.40]:
        # w0, wa pairs covering the plausible range
        for w0, wa in [
            (-1.0, 0.0),    # LCDM
            (-1.2, 0.2),
            (-1.0, 0.4),
            (-0.8, 0.3),
            (-0.5, 0.5),
            (-0.3, 0.2),
            (-1.5, 0.1),
            (-1.0, -0.3),
            (-0.8, -0.5),
            (-0.5, -0.2),
            (-1.2, -0.4),
            (0.0, 0.0),
            (-2.0, 0.0),
            (-1.0, 0.8),
            (-0.5, 1.0),
            (-1.5, 0.5),
        ]:
            test_models.append((Om, w0, wa))
    
    # Test redshifts - cover the range where quadrature matters
    z_values = np.array([0.1, 0.2, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0])
    
    # Build test cases
    test_cases = []
    for Om, w0, wa in test_models:
        if w0 == -1.0 and wa == 0.0:
            E_func = lambda z, Om=Om: E_lcdm(z, Om)
        else:
            E_func = lambda z, w0=w0, wa=wa, Om=Om: E_cpl(z, w0, wa, Om)
        
        for z in z_values:
            test_cases.append((E_func, z))
    
    n_cases = len(test_cases)
    n_nodes = 8
    
    # Precompute true integrals using adaptive quadrature
    true_integrals = np.zeros(n_cases)
    for k, (E_func, z) in enumerate(test_cases):
        integrand = lambda t: 1.0 / E_func(z * t)
        val, _ = quad(integrand, 0, 1, limit=100)
        true_integrals[k] = val
    
    def solve_weights_lp(nodes):
        """
        Solve LP to find weights w that minimize max relative error.
        The quadrature approximates: ∫_0^1 f(t) dt ≈ sum_i w_i f(x_i)
        where f(t) = 1/E(z*t)
        Relative error for case k: |sum_i w_i f_k(x_i) - true_k| / true_k
        We want to minimize max_k relative error subject to sum w_i = 1, w_i >= 0
        """
        # A[k, i] = f_k(x_i) / true_k = 1/(E_k(z_k * x_i) * true_k)
        A = np.zeros((n_cases, n_nodes))
        for k, (E_func, z) in enumerate(test_cases):
            for i in range(n_nodes):
                A[k, i] = 1.0 / (E_func(z * nodes[i]) * true_integrals[k])
        
        # Minimize t subject to:
        # A[k,:] @ w - t <= 1 for all k
        # -A[k,:] @ w - t <= -1 for all k
        # sum w = 1
        # w >= 0, t >= 0
        
        c = np.zeros(n_nodes + 1)
        c[-1] = 1.0
        
        A_ub = np.zeros((2 * n_cases, n_nodes + 1))
        b_ub = np.zeros(2 * n_cases)
        
        for k in range(n_cases):
            A_ub[2*k, :n_nodes] = A[k, :]
            A_ub[2*k, -1] = -1.0
            b_ub[2*k] = 1.0
            
            A_ub[2*k+1, :n_nodes] = -A[k, :]
            A_ub[2*k+1, -1] = -1.0
            b_ub[2*k+1] = -1.0
            
        A_eq = np.zeros((1, n_nodes + 1))
        A_eq[0, :n_nodes] = 1.0
        b_eq = np.array([1.0])
        
        bounds = [(0, None)] * n_nodes + [(0, None)]
        
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        
        if res.success:
            return res.x[:n_nodes], res.x[-1]
        else:
            return np.ones(n_nodes) / n_nodes, 1.0

    def objective_nodes(params):
        """
        Optimize nodes. params are log-spacings that define nodes via cumulative sum.
        """
        exp_nodes = np.exp(params)
        nodes = np.cumsum(exp_nodes)
        nodes = nodes / nodes[-1]
        nodes = np.maximum.accumulate(nodes)
        nodes = np.clip(nodes, 1e-15, 1 - 1e-15)
        
        for k in range(1, n_nodes):
            if nodes[k] <= nodes[k-1]:
                nodes[k] = nodes[k-1] + 1e-12
        
        weights, max_err = solve_weights_lp(nodes)
        
        if not np.all(weights >= 0) or abs(np.sum(weights) - 1.0) > 1e-6:
            return 1e6
            
        return max_err

    # Initial guesses for node optimization
    gl_nodes, _ = np.polynomial.legendre.leggauss(n_nodes)
    init_nodes = (gl_nodes + 1.0) / 2.0
    init_params = np.full(n_nodes, np.log(1.0/n_nodes))
    
    best_params = init_params.copy()
    best_obj = objective_nodes(init_params)
    
    # Multiple restarts with different initializations
    restarts = [
        init_params,
        np.log(np.linspace(0.05, 0.95, n_nodes)),
        np.log(1.0 - np.linspace(0.05, 0.95, n_nodes)),
        np.log(np.sort(np.random.uniform(0.05, 0.95, n_nodes))),
        np.log(np.sort(np.random.uniform(0.02, 0.98, n_nodes))),
        # Chebyshev-like nodes
        np.log(0.5 * (1 + np.cos(np.pi * (np.arange(n_nodes) + 0.5) / n_nodes)))[::-1],
    ]
    
    for i, init_p in enumerate(restarts):
        if time.time() > deadline - 25.0:
            break
        try:
            res = minimize(objective_nodes, init_p, method='Nelder-Mead',
                           options={'maxiter': 150, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True})
            if res.fun < best_obj:
                best_obj = res.fun
                best_params = res.x
        except Exception:
            pass
            
    # Convert best params to nodes
    exp_nodes = np.exp(best_params)
    nodes = np.cumsum(exp_nodes)
    nodes = nodes / nodes[-1]
    nodes = np.maximum.accumulate(nodes)
    nodes = np.clip(nodes, 1e-15, 1 - 1e-15)
    
    for k in range(1, n_nodes):
        if nodes[k] <= nodes[k-1]:
            nodes[k] = nodes[k-1] + 1e-12
            
    weights, _ = solve_weights_lp(nodes)
    weights = np.maximum(weights, 1e-15)
    weights = weights / np.sum(weights)
    
    # Final validation
    if not np.all(np.diff(nodes) > 0) or not np.all(nodes > 0) or not np.all(nodes < 1) or not np.all(weights > 0):
        nodes = (np.arange(n_nodes) + 0.5) / n_nodes
        weights = np.full(n_nodes, 1.0 / n_nodes)
    
    # Ensure weights sum to 1 within 1e-12
    weights = weights / np.sum(weights)
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
