"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.integrate import quad


def construct():
    """
    Discover an eight-node positive quadrature rule for dimensionless luminosity distance.
    Uses a structured search over node positions with softmax-weighted positive weights,
    optimized against a set of representative CPL dark-energy models.
    """
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill

    n = 8

    # --- Define a set of representative CPL models for optimization ---
    # w0, wa pairs covering the parameter space of interest
    models = [
        (0.0, 0.0),    # LCDM
        (-1.0, 0.0),   # Cosmological constant
        (-0.5, 0.5),   # Quintessence-like
        (-1.5, 1.0),   # Phantom-like
        (0.0, -1.0),   # Varying w
        (-0.8, 0.3),   # Moderate CPL
        (-1.2, -0.5),  # Negative wa
        (-0.3, 0.8),   # Another corner
        (0.5, -0.5),   # Positive w0
        (-2.0, 2.0),   # Extreme
        (-0.7, 0.0),   # Near LCDM
        (-1.0, 0.5),   # CC with positive wa
        (0.0, 0.5),    # Matter-dominated with positive wa
        (-0.4, -0.3),  # Another point
        (1.0, 0.0),    # High w0
    ]

    # Also include some z values to test at various redshifts
    z_test_values = [0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]

    def E_of_a(a, w0, wa):
        """H(a)/H0 for CPL model: E^2 = Omega_m * a^-3 + (1-Omega_m) * a^(-3*(1+w0+wa*(1-a)))"""
        Omega_m = 0.3
        if a <= 0:
            return 1e30
        # CPL: w(a) = w0 + wa*(1-a), so integral of (1+w) da gives the exponent
        # E^2 = Omega_m * a^-3 + (1-Omega_m) * exp(3*wa*(1-a)) * a^(-3*(1+w0))
        # More precisely: E^2 = Omega_m a^-3 + (1-Omega_m) a^(-3(1+w0)) * exp(3*wa*(1-a))
        # Let's use the standard form:
        # E(z)^2 = (1+z)^3 * Omega_m + (1-Omega_m) * (1+z)^(3*(1+w0+wa)) * exp(-3*wa*z/(1+z))
        # In terms of a: z = 1/a - 1
        # E(a)^2 = a^-3 * Omega_m + (1-Omega_m) * a^(-3*(1+w0+wa)) * exp(3*wa*(1-a))
        # Wait, let me be more careful.
        # H^2/H0^2 = Omega_m a^-3 + Omega_de * exp(3 * integral_0^z (1+w(z')) dz')
        # For CPL: w(z) = w0 + wa * z/(1+z) = w0 + wa*(1-a)
        # integral_0^z (1+w(z')) dz' = integral_0^z (1+w0+wa*z'/(1+z')) dz'
        # = (1+w0)*z + wa*ln(1+z)
        # So E(a)^2 = Omega_m * a^-3 + (1-Omega_m) * exp(3*((1+w0)*(1/a-1) + wa*ln(1/a)))
        # = Omega_m * a^-3 + (1-Omega_m) * exp(3*(1+w0)*(1/a-1)) * (1/a)^(3*wa)
        # = a^-3 * [Omega_m + (1-Omega_m) * exp(3*(1+w0)*(1/a-1)) * a^(-3*wa)]
        
        # Simpler numerical approach:
        z = 1.0/a - 1.0
        # integral of (1+w) dz from 0 to z
        # = (1+w0)*z + wa*ln(1+z)
        integral_val = (1.0 + w0) * z + wa * np.log1p(z)
        E2 = Omega_m * a**(-3) + (1.0 - Omega_m) * np.exp(3.0 * integral_val)
        return np.sqrt(E2)

    def luminosity_distance_integral(z, w0, wa):
        """
        dL/(c/H0) = (1+z) * integral_0^z dz' / E(a')
        where a' = 1/(1+z')
        Change variable: let u = z', integral_0^z du/E(1/(1+u))
        """
        def integrand(u):
            a = 1.0 / (1.0 + u)
            return 1.0 / E_of_a(a, w0, wa)
        
        result, _ = quad(integrand, 0, z, limit=200, epsabs=1e-12, epsrel=1e-12)
        return (1.0 + z) * result

    def quadrature_estimate(z, nodes, weights, w0, wa):
        """
        Quadrature approximation: (1+z) * sum_i weights[i] * z * f(node_i)
        where f(u) = 1/E(a) and a = 1/(1+u)
        Actually the formula is: dL/(c/H0) = (1+z)*z*sum_i weights[i]/E(a_i)
        with a_i = 1/(1+z*nodes[i])
        So nodes are in [0,1] and they scale the redshift.
        """
        total = 0.0
        for i in range(n):
            a_i = 1.0 / (1.0 + z * nodes[i])
            total += weights[i] / E_of_a(a_i, w0, wa)
        return (1.0 + z) * z * total

    def relative_error(z, nodes, weights, w0, wa):
        ref = luminosity_distance_integral(z, w0, wa)
        approx = quadrature_estimate(z, nodes, weights, w0, wa)
        if abs(ref) < 1e-30:
            return 0.0
        return abs(approx - ref) / abs(ref) * 100.0  # percent

    def objective(params):
        """
        params: 16 values = 8 node logits + 8 weight logits
        Nodes are in (0,1) strictly increasing via sigmoid transform of sorted logits.
        Weights are positive via softmax, normalized to sum to 1.
        """
        if time.time() > deadline:
            return 1e6
        
        node_logits = params[:n]
        weight_logits = params[n:2*n]
        
        # Transform node logits to strictly increasing nodes in (0,1)
        # Use cumulative sigmoid or sorted sigmoids
        # Simple approach: sort the sigmoids
        raw_nodes = 1.0 / (1.0 + np.exp(-node_logits))
        nodes = np.sort(raw_nodes)
        # Ensure strict increase with small separation
        for i in range(1, n):
            if nodes[i] - nodes[i-1] < 1e-8:
                nodes[i] = nodes[i-1] + 1e-8
        # Clamp to [1e-6, 1-1e-6]
        nodes = np.clip(nodes, 1e-6, 1.0 - 1e-6)
        
        # Softmax for weights
        exp_w = np.exp(weight_logits - np.max(weight_logits))
        weights = exp_w / np.sum(exp_w)
        
        # Compute worst-case relative error over training models and z values
        max_err = 0.0
        for w0, wa in models:
            for z in z_test_values:
                err = relative_error(z, nodes, weights, w0, wa)
                if err > max_err:
                    max_err = err
        
        return max_err

    # Initial guess: midpoint rule
    initial_nodes = (np.arange(n) + 0.5) / n
    initial_weights = np.ones(n) / n
    
    # Inverse transform for initial params
    initial_node_logits = np.log(initial_nodes / (1 - initial_nodes))
    initial_weight_logits = np.log(initial_weights)
    
    initial_params = np.concatenate([initial_node_logits, initial_weight_logits])
    
    best_params = initial_params.copy()
    best_obj = objective(initial_params)
    
    # Try L-BFGS-B
    try:
        result = minimize(
            objective,
            initial_params,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
        )
        if result.fun < best_obj:
            best_params = result.x
            best_obj = result.fun
    except Exception:
        pass
    
    if time.time() > deadline - 5.0:
        # Not enough time, return best so far
        pass
    else:
        # Try a few more restarts with different initial points
        np.random.seed(42)
        for restart in range(3):
            if time.time() > deadline - 5.0:
                break
            # Perturb the best solution
            perturbed = best_params + np.random.randn(2*n) * 0.5
            try:
                result = minimize(
                    objective,
                    perturbed,
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
                )
                if result.fun < best_obj:
                    best_params = result.x
                    best_obj = result.fun
            except Exception:
                pass
    
    # Extract final nodes and weights
    node_logits = best_params[:n]
    weight_logits = best_params[n:2*n]
    
    raw_nodes = 1.0 / (1.0 + np.exp(-node_logits))
    nodes = np.sort(raw_nodes)
    for i in range(1, n):
        if nodes[i] - nodes[i-1] < 1e-8:
            nodes[i] = nodes[i-1] + 1e-8
    nodes = np.clip(nodes, 1e-6, 1.0 - 1e-6)
    
    exp_w = np.exp(weight_logits - np.max(weight_logits))
    weights = exp_w / np.sum(exp_w)
    
    # Final validation: check constraints
    assert len(nodes) == n
    assert len(weights) == n
    assert np.all(np.diff(nodes) > 0), "Nodes not strictly increasing"
    assert np.all(nodes > 0) and np.all(nodes < 1), "Nodes out of [0,1]"
    assert np.all(weights > 0), "Weights not positive"
    assert abs(np.sum(weights) - 1.0) < 1e-12, "Weights don't sum to 1"
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
