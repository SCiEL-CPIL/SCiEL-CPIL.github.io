"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog, differential_evolution
from scipy.integrate import quad
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0 # Safety margin before 90s kill

    # Define E(z) for CPL model
    def E_func(z, Omega_m, w0, w1):
        Omega_L = 1.0 - Omega_m
        # E^2 = Om(1+z)^3 + OL(1+z)^(3(1+w0+w1z))
        return np.sqrt(Omega_m * (1.0 + z)**3 + Omega_L * (1.0 + z)**(3.0 * (1.0 + w0 + w1 * z)))
    
    def true_integral(z, Omega_m, w0, w1):
        # Integral from 0 to 1 of 1/E(z*u) du
        integrand = lambda u: 1.0 / E_func(z * u, Omega_m, w0, w1)
        res, _ = quad(integrand, 0, 1, limit=100)
        return res

    # Representative cosmologies covering the likely validation space
    # We need to cover a range of (Omega_m, w0, w1)
    cosmo_params = [
        (0.3, -1.0, 0.0),   # Lambda CDM
        (0.3, -1.2, 0.0),   # Phantom
        (0.3, -0.8, 0.0),   # Quintessence
        (0.3, -1.0, 0.5),   # Evolving DE (positive w1)
        (0.3, -1.0, -0.5),  # Evolving DE (negative w1)
        (0.2, -1.0, 0.0),   # Low matter
        (0.4, -1.0, 0.0),   # High matter
        (0.3, -1.5, 0.0),   # Deep Phantom
        (0.3, -0.5, 0.0),   # Deep Quintessence
        (0.3, -1.0, 1.0),   # Strongly evolving
        (0.3, -1.0, -1.0),  # Strongly evolving
        (0.25, -1.0, 0.5),  # Mixed
        (0.35, -1.0, -0.5), # Mixed
    ]
    
    # Test z values covering the full range
    z_test = [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
    
    num_cosmo = len(cosmo_params)
    num_z = len(z_test)
    num_cases = num_cosmo * num_z
    
    # Precompute true integrals
    true_vals = np.zeros(num_cases)
    case_idx = 0
    for i, (Om, w0, w1) in enumerate(cosmo_params):
        for j, z in enumerate(z_test):
            true_vals[case_idx] = true_integral(z, Om, w0, w1)
            case_idx += 1

    # Check time
    if time.time() > deadline:
        n = 8
        nodes, weights = np.polynomial.legendre.leggauss(n)
        nodes = 0.5 * (nodes + 1.0)
        weights = 0.5 * weights
        weights = weights / weights.sum()
        return {"nodes": nodes, "weights": weights}

    def solve_weights_for_nodes(nodes):
        """
        Given fixed nodes, find weights that minimize the worst-case relative error
        across all (cosmo, z) cases.
        
        We formulate this as a Linear Program:
        Minimize epsilon
        Subject to:
        -epsilon * true_val[k] <= sum_i w_i * integrand[k, i] - true_val[k] <= epsilon * true_val[k]
        sum_i w_i = 1
        w_i >= 0
        
        Where integrand[k, i] = 1 / E(z_k * u_i; cosmo_k)
        """
        # Build the integrand matrix: (num_cases, 8)
        integrand_matrix = np.zeros((num_cases, 8))
        case_idx = 0
        for i, (Om, w0, w1) in enumerate(cosmo_params):
            for j, z in enumerate(z_test):
                z_u = z * nodes
                E_vals = E_func(z_u, Om, w0, w1)
                integrand_matrix[case_idx, :] = 1.0 / E_vals
                case_idx += 1

        # LP Variables: [w_0, ..., w_7, epsilon]
        # Objective: minimize epsilon -> c = [0, ..., 0, 1]
        c = np.zeros(9)
        c[-1] = 1.0

        # Inequality constraints: A_ub @ x <= b_ub
        # For each case k:
        # sum(w_i * I_ki) - true_k - epsilon * true_k <= 0  =>  sum(w_i * I_ki) - true_k * epsilon <= true_k
        # -sum(w_i * I_ki) + true_k + epsilon * true_k <= 0 => -sum(w_i * I_ki) + true_k * epsilon <= -true_k
        
        A_ub = np.zeros((2 * num_cases, 9))
        b_ub = np.zeros(2 * num_cases)
        
        for k in range(num_cases):
            # Row 1: sum(w_i * I_ki) - true_k * eps <= true_k
            A_ub[2*k, :8] = integrand_matrix[k, :]
            A_ub[2*k, 8] = -true_vals[k]
            b_ub[2*k] = true_vals[k]
            
            # Row 2: -sum(w_i * I_ki) + true_k * eps <= -true_k
            A_ub[2*k+1, :8] = -integrand_matrix[k, :]
            A_ub[2*k+1, 8] = true_vals[k]
            b_ub[2*k+1] = -true_vals[k]

        # Equality constraints: A_eq @ x = b_eq
        # sum(w_i) = 1
        A_eq = np.zeros((1, 9))
        A_eq[0, :8] = 1.0
        b_eq = np.array([1.0])

        # Bounds: w_i >= 0, epsilon >= 0
        bounds = [(0, None)] * 8 + [(0, None)]

        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
            if res.success:
                weights = res.x[:8]
                # Ensure weights are strictly positive (LP might give 0)
                # Add a small epsilon to zero weights and renormalize? 
                # Or just clamp to 1e-10 and renormalize.
                weights = np.maximum(weights, 1e-12)
                weights = weights / np.sum(weights)
                return weights, res.x[-1]
            else:
                return None, np.inf
        except Exception:
            return None, np.inf

    def evaluate_nodes(nodes):
        """
        Evaluate a set of nodes by solving for optimal weights.
        Returns the worst-case relative error (epsilon).
        """
        weights, err = solve_weights_for_nodes(nodes)
        if weights is None:
            return 1.0 # Large penalty
        return err

    def map_nodes(params):
        """
        Map 8 parameters to strictly increasing nodes in (0, 1).
        Uses the cumulative sigmoid mapping.
        """
        nodes = np.zeros(8)
        nodes[0] = 1.0 / (1.0 + np.exp(-params[0]))
        for i in range(1, 8):
            step = 1.0 / (1.0 + np.exp(-params[i]))
            nodes[i] = nodes[i-1] + step * (1.0 - nodes[i-1])
        # Ensure strictly inside (0,1)
        nodes = np.clip(nodes, 1e-6, 1 - 1e-6)
        # Ensure strictly increasing (just in case of numerical issues)
        for i in range(1, 8):
            if nodes[i] <= nodes[i-1]:
                nodes[i] = nodes[i-1] + 1e-6
        if nodes[-1] >= 1.0:
            # Rescale to fit
            nodes = nodes / nodes[-1] * 0.999
        return nodes

    # Optimization Strategy:
    # 1. Use Differential Evolution to find good nodes.
    # 2. The objective function calls the LP solver, which is fast.
    # 3. This decouples the non-linear node search from the linear weight optimization.
    
    bounds = [(-5.0, 5.0)] * 8
    
    best_nodes = None
    best_err = np.inf
    best_weights = None

    try:
        # Use DE to find optimal nodes
        # maxfev needs to be balanced. LP is fast, so we can afford more.
        # But we have a 90s limit. Let's try 300-500 evals.
        result = differential_evolution(
            lambda p: evaluate_nodes(map_nodes(p)),
            bounds,
            maxiter=100,
            maxfev=400,
            tol=1e-4,
            seed=42,
            polish=False,
            init='sobol' # Better initial spread
        )
        
        if result.success or result.fun < best_err:
            best_nodes = map_nodes(result.x)
            best_weights, best_err = solve_weights_for_nodes(best_nodes)
            
    except Exception as e:
        pass

    # If DE failed or didn't find anything, try a few heuristic node sets
    if best_weights is None or best_err > 0.05:
        # Try Gauss-Legendre
        n = 8
        gl_nodes, gl_weights = np.polynomial.legendre.leggauss(n)
        gl_nodes = 0.5 * (gl_nodes + 1.0)
        gl_weights = 0.5 * gl_weights
        gl_weights = gl_weights / gl_weights.sum()
        
        # Optimize weights for GL nodes
        w_opt, err_opt = solve_weights_for_nodes(gl_nodes)
        if w_opt is not None and err_opt < best_err:
            best_nodes = gl_nodes
            best_weights = w_opt
            best_err = err_opt
            
        # Try Chebyshev nodes
        th = np.linspace(0, np.pi, 8)
        cheb_nodes = 0.5 * (1 + np.cos(th))
        # Chebyshev nodes are in [0,1] but not strictly increasing in the order generated?
        # np.cos is decreasing from 1 to -1, so 1+cos is decreasing.
        # We need increasing.
        cheb_nodes = cheb_nodes[::-1]
        # Weights for Chebyshev are not uniform, but we will re-optimize them via LP.
        # Initial weights don't matter for LP, but we need valid nodes.
        w_opt, err_opt = solve_weights_for_nodes(cheb_nodes)
        if w_opt is not None and err_opt < best_err:
            best_nodes = cheb_nodes
            best_weights = w_opt
            best_err = err_opt

    # Final safety check
    if best_weights is None or best_err > 1.0:
        # Fallback to simple Gauss-Legendre with standard weights
        n = 8
        best_nodes, best_weights = np.polynomial.legendre.leggauss(n)
        best_nodes = 0.5 * (best_nodes + 1.0)
        best_weights = 0.5 * best_weights
        best_weights = best_weights / best_weights.sum()

    # Ensure constraints
    if not np.all(np.diff(best_nodes) > 0) or best_nodes[0] <= 0 or best_nodes[-1] >= 1:
        n = 8
        best_nodes, best_weights = np.polynomial.legendre.leggauss(n)
        best_nodes = 0.5 * (best_nodes + 1.0)
        best_weights = 0.5 * best_weights
        best_weights = best_weights / best_weights.sum()
        
    best_weights = np.maximum(best_weights, 1e-12)
    best_weights = best_weights / best_weights.sum()

    return {"nodes": best_nodes, "weights": best_weights}
# EVOLVE-BLOCK-END
