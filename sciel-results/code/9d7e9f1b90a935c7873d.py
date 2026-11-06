"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog
from scipy.integrate import quad

def _E_sq(a, Om, w0, wa):
    """Compute E^2(a) for flat CPL dark energy model."""
    w = w0 + wa * (1.0 - a)
    return Om * a**(-3) + (1.0 - Om) * a**(-3.0 * (1.0 + w))

def _E(a, Om, w0, wa):
    return np.sqrt(_E_sq(a, Om, w0, wa))

def _exact_integral(z, Om, w0, wa):
    """Compute the exact integral using adaptive quadrature (high accuracy)."""
    def integrand(x):
        a = 1.0 / (1.0 + z * x)
        return 1.0 / _E(a, Om, w0, wa)
    val, err = quad(integrand, 0, 1, limit=200, epsabs=1e-12, epsrel=1e-12)
    return (1.0 + z) * z * val

def _quadrature_value(nodes, weights, z, Om, w0, wa):
    """Compute quadrature approximation."""
    a_vals = 1.0 / (1.0 + z * nodes)
    E_vals = _E(a_vals, Om, w0, wa)
    return (1.0 + z) * z * np.sum(weights / E_vals)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Diverse cosmological models for training - much larger set
    cosmo_params = []
    for Om in [0.25, 0.28, 0.30, 0.32, 0.35]:
        for w0 in [-1.5, -1.2, -1.0, -0.8, -0.5]:
            for wa in [-0.5, -0.3, 0.0, 0.3, 0.5, 1.0]:
                cosmo_params.append((Om, w0, wa))
    
    z_list = [0.2, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    
    # Create a diverse test set (not all combinations, but a good sample)
    test_cases = []
    rng = np.random.RandomState(42)
    n_cosmo = len(cosmo_params)
    n_z = len(z_list)
    # Sample up to 500 test cases
    max_cases = 500
    indices = rng.choice(n_cosmo * n_z, size=min(max_cases, n_cosmo * n_z), replace=False)
    for idx in indices:
        ci = idx % n_cosmo
        zi = idx // n_cosmo
        test_cases.append((z_list[zi], cosmo_params[ci]))
    
    # Pre-compute exact values and integrand values at candidate nodes
    # We'll try several node configurations and for each, solve an LP for optimal weights
    
    def _solve_lp_for_nodes(nodes, test_cases, exact_vals, deadline):
        """Solve LP to find weights minimizing worst-case relative error."""
        n_test = len(test_cases)
        n_nodes = len(nodes)
        
        # For each test case j, the quadrature is:
        # Q_j = (1+z_j)*z_j * sum_i w_i / E_j(a_i)
        # We want |Q_j - I_j| / I_j <= t for all j
        # => -t <= (Q_j - I_j)/I_j <= t
        # => -t * I_j <= (1+z_j)*z_j * sum_i w_i/E_j(a_i) - I_j <= t * I_j
        # => I_j - t*I_j <= (1+z_j)*z_j * sum_i w_i/E_j(a_i) <= I_j + t*I_j
        
        # Variables: w_0, ..., w_7, t (9 variables)
        # Objective: minimize t (index 8)
        # Constraints:
        #   sum(w_i) = 1
        #   w_i >= 0
        #   For each test j:
        #     (1+z_j)*z_j * sum_i w_i/E_j(a_i) - I_j <= t * I_j
        #     -( (1+z_j)*z_j * sum_i w_i/E_j(a_i) - I_j ) <= t * I_j
        
        c = np.zeros(n_nodes + 1)
        c[n_nodes] = 1.0  # minimize t
        
        # Inequality constraints: A_ub @ x <= b_ub
        A_rows = []
        b_rows = []
        
        for j, (z, (Om, w0, wa)) in enumerate(test_cases):
            exact = exact_vals[j]
            if exact < 1e-10:
                continue
            a_vals = 1.0 / (1.0 + z * nodes)
            E_vals = _E(a_vals, Om, w0, wa)
            coeff = (1.0 + z) * z / E_vals  # length n_nodes
            
            # Constraint 1: (1+z)*z * sum_i w_i/E_j(a_i) - I_j <= t * I_j
            # => sum_i coeff[i]*w_i - I_j - t*I_j <= 0
            row1 = np.zeros(n_nodes + 1)
            row1[:n_nodes] = coeff
            row1[n_nodes] = -exact
            A_rows.append(row1)
            b_rows.append(0.0)
            
            # Constraint 2: -( (1+z)*z * sum_i w_i/E_j(a_i) - I_j ) <= t * I_j
            # => -sum_i coeff[i]*w_i + I_j - t*I_j <= 0
            row2 = np.zeros(n_nodes + 1)
            row2[:n_nodes] = -coeff
            row2[n_nodes] = -exact
            A_rows.append(row2)
            b_rows.append(0.0)
        
        if not A_rows:
            return None, np.inf
        
        A_ub = np.array(A_rows)
        b_ub = np.array(b_rows)
        
        # Equality constraint: sum(w_i) = 1
        A_eq = np.zeros((1, n_nodes + 1))
        A_eq[0, :n_nodes] = 1.0
        b_eq = np.array([1.0])
        
        # Bounds: w_i >= 0, t >= 0
        bounds = [(0, None)] * n_nodes + [(0, None)]
        
        try:
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                           bounds=bounds, method='highs')
            if result.success:
                weights = result.x[:n_nodes]
                t_val = result.x[n_nodes]
                # Ensure weights sum to 1
                weights = np.maximum(weights, 0)
                weights = weights / weights.sum()
                return weights, t_val
            else:
                return None, np.inf
        except Exception:
            return None, np.inf
    
    # Candidate node configurations
    node_configs = []
    
    # 1. Gauss-Legendre nodes mapped to [0,1]
    gl_nodes, _ = np.polynomial.legendre.leggauss(8)
    node_configs.append((gl_nodes + 1.0) / 2.0)
    
    # 2. Midpoint rule
    node_configs.append((np.arange(8) + 0.5) / 8.0)
    
    # 3. Chebyshev nodes mapped to [0,1]
    k = np.arange(8)
    cheb_nodes = (1.0 + np.cos(np.pi * (2*k + 1) / 16.0)) / 2.0
    node_configs.append(cheb_nodes)
    
    # 4. Slightly shifted Gauss-Legendre (more weight toward x=0)
    for shift in [0.05, 0.1, 0.15, 0.2]:
        shifted = gl_nodes.copy()
        # Apply a power transform to shift nodes toward 0
        t = (shifted + 1.0) / 2.0  # map to [0,1]
        t_shifted = t ** (1.0 + shift)  # compress toward 0
        node_configs.append(t_shifted)
    
    # 5. Random configurations
    rng2 = np.random.RandomState(123)
    for _ in range(10):
        rand_nodes = np.sort(rng2.uniform(0.02, 0.98, 8))
        node_configs.append(rand_nodes)
    
    # 6. Spaced configurations with varying density
    for alpha in [0.5, 0.7, 1.0, 1.3, 1.5, 2.0]:
        t = np.linspace(0, 1, 9)[1:-1]  # 7 interior points... need 8
        t = np.linspace(0, 1, 9)[1:-1]
        # Actually let's do: nodes[i] = ((i+1)/(9))^alpha for i=0..7
        i = np.arange(1, 9)
        nodes_alpha = (i / 9.0) ** alpha
        node_configs.append(nodes_alpha)
    
    # Pre-compute exact values for test cases
    exact_vals = []
    for z, (Om, w0, wa) in test_cases:
        exact_vals.append(_exact_integral(z, Om, w0, wa))
    
    best_weights = None
    best_nodes = None
    best_err = np.inf
    
    for cfg_idx, nodes in enumerate(node_configs):
        if time.time() > deadline - 10.0:
            break
        
        nodes = np.sort(nodes)
        nodes = np.clip(nodes, 1e-6, 1.0 - 1e-6)
        for i in range(1, 8):
            if nodes[i] <= nodes[i-1]:
                nodes[i] = nodes[i-1] + 1e-6
        
        weights, err = _solve_lp_for_nodes(nodes, test_cases, exact_vals, deadline)
        
        if weights is not None and err < best_err:
            best_err = err
            best_weights = weights
            best_nodes = nodes.copy()
    
    # If LP didn't find good solution, fall back to Gauss-Legendre weights
    if best_weights is None:
        gl_nodes, gl_weights = np.polynomial.legendre.leggauss(8)
        best_nodes = (gl_nodes + 1.0) / 2.0
        best_weights = gl_weights / 2.0
    else:
        best_weights = np.maximum(best_weights, 1e-10)
        best_weights = best_weights / best_weights.sum()
    
    # Final validation and cleanup
    nodes = np.sort(best_nodes)
    nodes = np.clip(nodes, 1e-6, 1.0 - 1e-6)
    for i in range(1, 8):
        if nodes[i] <= nodes[i-1]:
            nodes[i] = nodes[i-1] + 1e-6
    
    weights = np.maximum(best_weights, 1e-10)
    weights = weights / weights.sum()
    
    if not np.all(np.diff(nodes) > 0) or not np.all(weights > 0) or abs(weights.sum() - 1.0) > 1e-12:
        return {"nodes": (np.arange(8) + 0.5) / 8.0, "weights": np.full(8, 1.0 / 8.0)}
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
