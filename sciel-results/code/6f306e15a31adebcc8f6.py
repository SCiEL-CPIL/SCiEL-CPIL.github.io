import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import quad

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill

    # Define E(a) for CPL dark energy in flat universe
    def _E(a, w0, w1, Omega_m=0.3):
        Omega_de = 1.0 - Omega_m
        a_safe = np.maximum(a, 1e-10)
        return np.sqrt(Omega_m * a_safe**(-3) + Omega_de * a_safe**(-3*(1+w0)) * np.exp(3*w1*(1-a_safe)/a_safe))

    def _true_distance(z_max, w0, w1):
        def f(z):
            a = 1.0 / (1.0 + z)
            return (1.0 + z) / _E(a, w0, w1)
        val, _ = quad(f, 0, z_max, limit=200, epsabs=1e-12, epsrel=1e-12)
        return val

    def _quad_distance(z_max, nodes, weights, w0, w1):
        z_nodes = z_max * nodes
        a_nodes = 1.0 / (1.0 + z_nodes)
        vals = (1.0 + z_nodes) / _E(a_nodes, w0, w1)
        return z_max * np.sum(weights * vals)

    # Test cosmologies: diverse CPL models
    test_cosmos = [
        (-1.0, 0.0),      # LambdaCDM
        (-1.0, 0.5),      # CPL w1>0
        (-1.0, -0.5),     # CPL w1<0
        (-0.9, 0.3),      # Quintessence-like
        (-0.95, -0.3),    # Another variant
        (-1.2, 0.0),      # Phantom
        (-0.8, 0.5),      # Strong evolution
        (-1.0, 1.0),      # Strong CPL
        (-1.1, 0.4),      # Additional
        (-0.85, -0.4),    # Additional
        (-0.7, 0.8),      # Very strong evolution
        (-1.3, -0.2),     # Phantom with slight evolution
    ]

    # Redshifts to test
    test_zs = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]

    # Pre-compute true distances for speed
    true_vals = np.zeros((len(test_cosmos), len(test_zs)))
    for i, (w0, w1) in enumerate(test_cosmos):
        for j, z in enumerate(test_zs):
            true_vals[i, j] = _true_distance(z, w0, w1)

    def _worst_error(nodes, weights):
        """Compute worst relative error over all test cosmologies and redshifts."""
        max_err = 0.0
        for i, (w0, w1) in enumerate(test_cosmos):
            for j, z in enumerate(test_zs):
                approx = _quad_distance(z, nodes, weights, w0, w1)
                true_val = true_vals[i, j]
                err = abs(approx - true_val) / true_val * 100.0
                if err > max_err:
                    max_err = err
        return max_err

    # NEW APPROACH: Weighted least-squares basis fit
    # Model the integrand (1+z)/E(a) as a linear combination of basis functions
    # that span the variation across CPL parameters.
    
    # Basis functions: use monomials in a (scale factor) up to degree 7
    # Since we have 8 nodes, we can integrate 8 basis functions exactly.
    
    def _basis_matrix(a_vals, num_basis=8):
        """Compute basis function values at given scale factors."""
        # Use Chebyshev polynomials for better numerical stability
        # Map a in (0,1] to x in [-1,1]
        x = 2.0 * a_vals - 1.0
        x = np.clip(x, -1.0 + 1e-10, 1.0 - 1e-10)
        basis = np.zeros((len(a_vals), num_basis))
        for k in range(num_basis):
            basis[:, k] = np.polynomial.chebyshev.chebval(x, np.zeros(num_basis, dtype=float))
            # Actually, let's use Chebyshev polynomials properly
        # Recompute using proper Chebyshev evaluation
        for k in range(num_basis):
            coeffs = np.zeros(num_basis)
            coeffs[k] = 1.0
            basis[:, k] = np.polynomial.chebyshev.chebval(x, coeffs)
        return basis

    # Step 1: For each test cosmology and redshift, compute the integrand values
    # at a fine grid, and fit to basis functions.
    
    # Actually, a better approach: 
    # We want nodes/weights such that for each basis function phi_k,
    # sum_i w_i * phi_k(a_i) ≈ integral_0^1 phi_k(a) * (1+z(a)) / E(a) da
    # But this depends on the cosmology.
    
    # Alternative: Use the fact that the integrand can be written as
    # (1+z)/E(a) = (1+z) * sqrt(1/(Omega_m*a^-3 + Omega_de*a^-3(1+w0)*exp(...)))
    # 
    # Let's define a "geometric" factor g(z) = (1+z) and a "model" factor m(a; w0,w1) = 1/E(a)
    # Then the integral is ∫ g(z) * m(a) dz from 0 to z_max
    # 
    # Since z and a are related by a = 1/(1+z), we can change variables:
    # ∫_0^{z_max} (1+z)/E(a) dz = ∫_{a_min}^{1} 1/(a*E(a)) da
    # where a_min = 1/(1+z_max)
    #
    # So the integral is ∫_{a_min}^{1} 1/(a*E(a)) da
    #
    # This is independent of z_max in the sense that the integrand is the same,
    # just the limits change. So we can design a quadrature rule for the unit interval
    # [0,1] in a-space, and then for a given z_max, we integrate from a_min to 1.
    #
    # But our quadrature rule has fixed nodes in [0,1] (in z-space). Let's re-examine.
    #
    # The task says: dL/(c/H0) = (1+z)*z*sum_i weights[i]/E(a_i), where a_i=1/(1+z*nodes[i])
    # So the nodes are in z-space (dimensionless redshift fraction), and we evaluate
    # at z_i = z_max * nodes[i].
    #
    # So the quadrature is: ∫_0^{z_max} f(z) dz ≈ z_max * sum_i w_i * f(z_max * x_i)
    # where x_i are the nodes in [0,1].
    #
    # This is a standard quadrature rule on [0,1] applied to the scaled integral.
    #
    # For a fixed cosmology, f(z) = (1+z)/E(1/(1+z))
    #
    # The key insight: f(z) has a specific structure. Let's expand it in terms of
    # a basis that captures the variation across CPL parameters.
    #
    # Define: f(z; w0, w1) = (1+z) / sqrt(Omega_m*(1+z)^3 + Omega_de*(1+z)^{3(1+w0)} * exp(3*w1*z/(1+z)))
    #
    # = (1+z) / sqrt(Omega_m*(1+z)^3 + Omega_de*(1+z)^{3+3w0} * exp(3*w1*z/(1+z)))
    #
    # For small z, this is approximately 1/sqrt(Omega_m) * (1 - 0.5*Omega_de/Omega_m + ...)
    #
    # Let's use a different basis: expand in powers of z/(1+z) or similar.
    #
    # Actually, let's use a pragmatic approach:
    # 1. For each test cosmology, compute the integrand on a fine grid.
    # 2. Fit to a set of basis functions that are common across all cosmologies.
    # 3. Choose nodes/weights that integrate these basis functions well.
    
    # Basis functions: use {1, z, z^2, ..., z^7} in z-space, but scaled to [0,1]
    # Actually, since we're integrating on [0, z_max] with scaled nodes, let's use
    # basis in x = z/z_max ∈ [0,1].
    
    # For a given cosmology, f(z_max * x) = (1 + z_max*x) / E(1/(1+z_max*x))
    #
    # Let's define basis functions phi_k(x) that are independent of z_max and cosmology.
    # Use Chebyshev polynomials on [0,1].
    
    def _cheb_basis(x, n=8):
        """Evaluate first n Chebyshev polynomials on [0,1] mapped to [-1,1]."""
        t = 2.0 * x - 1.0
        basis = np.zeros((len(x), n))
        for k in range(n):
            # T_k(t)
            if k == 0:
                basis[:, k] = 1.0
            elif k == 1:
                basis[:, k] = t
            else:
                # Recurrence: T_k(t) = 2*t*T_{k-1}(t) - T_{k-2}(t)
                basis[:, k] = 2.0 * t * basis[:, k-1] - basis[:, k-2]
        return basis

    # Step 1: For each test cosmology and redshift, compute the integrand values
    # on a fine grid in x ∈ [0,1].
    
    fine_grid = np.linspace(0.01, 0.99, 100)  # Avoid endpoints for numerical stability
    basis_fine = _cheb_basis(fine_grid, 8)
    
    # For each cosmology and z_max, compute f(z_max * x) and fit to basis.
    # The integral is z_max * ∫_0^1 f(z_max * x) dx
    #
    # We want nodes/weights such that:
    # sum_i w_i * phi_k(x_i) ≈ ∫_0^1 phi_k(x) * f(z_max * x) / f_avg dx
    #
    # This is getting complicated. Let's try a different approach:
    #
    # Direct optimization with a better initialization:
    # 1. Start from Gauss-Legendre nodes/weights.
    # 2. Use the basis fit to guide the optimization.
    
    # Actually, let's implement the basis fit approach properly:
    #
    # For each test case (cosmology, z_max), we have:
    # I = z_max * ∫_0^1 f(x) dx, where f(x) = (1 + z_max*x) / E(1/(1+z_max*x))
    #
    # We want to find nodes x_i and weights w_i such that:
    # z_max * sum_i w_i * f(x_i) ≈ I
    #
    # Equivalently: sum_i w_i * f(x_i) ≈ ∫_0^1 f(x) dx
    #
    # Now, expand f(x) in the Chebyshev basis:
    # f(x) ≈ sum_k c_k * phi_k(x)
    #
    # Then ∫_0^1 f(x) dx ≈ sum_k c_k * ∫_0^1 phi_k(x) dx
    #
    # And sum_i w_i * f(x_i) ≈ sum_k c_k * sum_i w_i * phi_k(x_i)
    #
    # So we want: sum_i w_i * phi_k(x_i) ≈ ∫_0^1 phi_k(x) dx for all k
    #
    # This is a system of equations that defines the quadrature rule.
    # If we can find nodes/weights that satisfy this for all 8 basis functions,
    # then the quadrature will be exact for any function in the span of the basis.
    
    # Compute the integrals of the basis functions over [0,1]
    def _cheb_integral(k, n=8):
        """Compute ∫_0^1 T_k(2x-1) dx."""
        # Change variable: t = 2x-1, dx = dt/2, x:0->1 => t:-1->1
        # ∫_0^1 T_k(2x-1) dx = (1/2) ∫_{-1}^{1} T_k(t) dt
        # ∫_{-1}^{1} T_k(t) dt = 0 for k odd, 2/(1-k^2) for k even (k>0), 2 for k=0
        if k == 0:
            return 1.0  # (1/2) * 2 = 1
        elif k % 2 == 1:
            return 0.0
        else:
            return (1.0 / (1 - k*k))  # (1/2) * 2/(1-k^2)
    
    cheb_integrals = np.array([_cheb_integral(k) for k in range(8)])
    
    # Now, we need to find nodes x_i and weights w_i such that:
    # sum_i w_i * phi_k(x_i) = cheb_integrals[k] for k=0,...,7
    #
    # This is a system of 8 equations with 16 unknowns (8 nodes, 8 weights).
    # We can use the nodes as parameters and solve for the weights.
    
    def _solve_weights(nodes):
        """Given nodes, solve for weights that integrate the basis exactly."""
        B = _cheb_basis(nodes, 8)  # 8x8 matrix
        try:
            weights = np.linalg.solve(B, cheb_integrals)
        except np.linalg.LinAlgError:
            return None, 1e10
        # Check positivity
        if np.any(weights <= 0):
            return None, 1e10
        return weights, 0.0
    
    # Step 2: Optimize the nodes to find a set where the weights are positive
    # and the quadrature performs well on the test cases.
    
    def _node_objective(node_params):
        if time.time() > deadline - 2:
            return 1e10
        
        # Transform to ensure strictly increasing nodes in (0,1)
        # Use cumulative sum of exponentials
        exp_u = np.exp(node_params)
        cumsum_u = np.cumsum(exp_u)
        nodes = cumsum_u / cumsum_u[-1]
        
        weights, penalty = _solve_weights(nodes)
        if weights is None:
            return penalty
        
        # Evaluate the worst error on test cases
        max_err = 0.0
        for i, (w0, w1) in enumerate(test_cosmos):
            for j, z in enumerate(test_zs):
                approx = _quad_distance(z, nodes, weights, w0, w1)
                true_val = true_vals[i, j]
                err = abs(approx - true_val) / true_val * 100.0
                if err > max_err:
                    max_err = err
        
        return max_err
    
    # Start with nodes that are roughly evenly spaced
    # Inverse transform: if nodes = cumsum(exp(u))/total, then for evenly spaced nodes,
    # u_i should be roughly log(i/(i-1)) for i=1,...,8 (with u_0=0)
    # Actually, let's just use random initialization and let the optimizer find a good set.
    
    np.random.seed(42)
    best_nodes = None
    best_weights = None
    best_err = 1e10
    
    # Try multiple random initializations
    for trial in range(20):
        if time.time() > deadline - 10:
            break
        
        u0 = np.random.uniform(-2, 2, 8)
        
        result = minimize(_node_objective, u0, method='Nelder-Mead',
                         options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
        
        if result.fun < best_err:
            exp_u = np.exp(result.x)
            cumsum_u = np.cumsum(exp_u)
            best_nodes = cumsum_u / cumsum_u[-1]
            best_weights, _ = _solve_weights(best_nodes)
            best_err = result.fun
    
    # If the basis fit approach didn't work well, fall back to the original approach
    if best_nodes is None or best_err > 1.0:
        # Use the original Gauss-Legendre initialization and optimization
        t, w = np.polynomial.legendre.leggauss(8)
        nodes_init = 0.5 * (t + 1.0)
        weights_init = 0.5 * w
        weights_init = weights_init / np.sum(weights_init)
        
        best_nodes = nodes_init.copy()
        best_weights = weights_init.copy()
        best_err = _worst_error(best_nodes, best_weights)
        
        def _objective(params):
            if time.time() > deadline - 2:
                return 1e10
            nodes = params[:8]
            weights = params[8:]
            
            if np.any(nodes[1:] <= nodes[:-1]):
                return 1e10
            if np.any(nodes <= 0) or np.any(nodes >= 1):
                return 1e10
            if np.any(weights <= 0):
                return 1e10
            
            weights = weights / np.sum(weights)
            return _worst_error(nodes, weights)
        
        x0 = np.concatenate([best_nodes, best_weights])
        try:
            result_nm = minimize(_objective, x0, method='Nelder-Mead',
                                options={'maxiter': 300, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True})
            if result_nm.fun < best_err:
                best_nodes = result_nm.x[:8]
                best_weights = result_nm.x[8:] / np.sum(result_nm.x[8:])
                best_err = result_nm.fun
        except Exception:
            pass
    
    # Final constraint enforcement
    best_nodes = np.sort(best_nodes)
    best_nodes = np.clip(best_nodes, 1e-8, 1.0 - 1e-8)
    for j in range(1, 8):
        if best_nodes[j] <= best_nodes[j-1]:
            best_nodes[j] = best_nodes[j-1] + 1e-8
    best_weights = np.maximum(best_weights, 1e-12)
    best_weights = best_weights / np.sum(best_weights)

    return {"nodes": best_nodes, "weights": best_weights}
