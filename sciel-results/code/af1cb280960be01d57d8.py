"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def construct():
    """
    Construct an 8-node positive quadrature rule for the dimensionless
    luminosity-distance integral in flat CPL dark-energy models.
    
    The integrand is f(z) = z / E(a) where a = 1/(1+z) and E(a) = H(a)/H0.
    We want to approximate I = integral_0^z f(z') dz' for all z in [0, z_max].
    
    The quadrature: I(z) ≈ z * sum_i w_i * f(z * x_i)
    where x_i are the nodes in [0,1] and w_i are weights summing to 1.
    
    Strategy: Use Gauss-Legendre quadrature on [0,1] which is optimal for
    smooth functions. The integrand z/E(a) is smooth for reasonable CPL parameters.
    """
    # Deadline guard
    t_start = time.time()
    deadline = t_start + 85.0  # 5s safety margin before 90s kill
    
    # Try to optimize the nodes and weights for a family of CPL models
    # E(a) = a^(-3(1+w0+wa)) * (1 + wa*(1-a))
    # where w0 and wa are the CPL parameters
    
    # The integral is: dL/(c/H0) = (1+z) * integral_0^z z'/E(a(z')) dz'
    # Let u = z'/z, so z' = z*u, dz' = z*du
    # integral_0^z z'/E(a(z')) dz' = z * integral_0^1 u * E(a(z*u))^{-1} du
    
    # So we need to approximate: integral_0^1 g(u) du where g(u) = u / E(a(z*u))
    # The quadrature: integral_0^1 g(u) du ≈ sum_i w_i * g(x_i)
    
    # For Gauss-Legendre with 8 points on [0,1]:
    # Nodes are the roots of P_8(x) mapped to [0,1]
    
    # Standard Gauss-Legendre 8-point on [-1,1]:
    gl_nodes = np.array([
        -0.9602898564975362,
        -0.7966664774136267,
        -0.5255803466885128,
        -0.1834346424956500,
         0.1834346424956500,
         0.5255803466885128,
         0.7966664774136267,
         0.9602898564975362
    ])
    gl_weights = np.array([
        0.1012285362903762,
        0.2223810344533744,
        0.3137066458778872,
        0.3626837833783619,
        0.3626837833783619,
        0.3137066458778872,
        0.2223810344533744,
        0.1012285362903762
    ])
    
    # Map from [-1,1] to [0,1]
    nodes = (gl_nodes + 1.0) / 2.0
    weights = gl_weights / 2.0
    
    # Sort nodes to be strictly increasing
    sort_idx = np.argsort(nodes)
    nodes = nodes[sort_idx]
    weights = weights[sort_idx]
    
    # Verify weights sum to 1
    w_sum = np.sum(weights)
    weights = weights / w_sum
    
    # Verify nodes are in [0,1] and strictly increasing
    assert np.all(nodes > 0) and np.all(nodes < 1), "Nodes must be in (0,1)"
    assert np.all(np.diff(nodes) > 0), "Nodes must be strictly increasing"
    assert np.all(weights > 0), "Weights must be positive"
    assert abs(np.sum(weights) - 1.0) < 1e-12, "Weights must sum to 1"
    
    # Check deadline
    if time.time() > deadline - 5.0:
        return {"nodes": nodes, "weights": weights}
    
    # Try to improve by optimizing for a family of CPL models
    # Define a set of test cosmologies (w0, wa) pairs
    test_models = [
        (0.0, 0.0),    # LCDM
        (-0.5, 0.0),   # w0 = -0.5
        (0.5, 0.0),    # w0 = 0.5
        (-1.0, 0.0),   # cosmological constant
        (-1.0, -0.5),  # phantom
        (-1.0, 0.5),   # quintessence-like
        (-0.5, -0.5),  # another phantom
        (-0.5, 0.5),   # another quintessence
        (0.0, -0.5),   # time-varying
        (0.0, 0.5),    # time-varying
        (-1.5, 0.0),   # strong phantom
        (-0.8, -0.3),  # mixed
        (-0.8, 0.3),   # mixed
        (-1.2, -0.2),  # mixed
        (-1.2, 0.2),   # mixed
    ]
    
    def E_of_a(a, w0, wa):
        """E(a) = H(a)/H0 for CPL model in flat universe."""
        # E^2 = Omega_m * a^{-3} + Omega_r * a^{-4} + Omega_L * a^{-3(1+w0+wa)} * (1+wa*(1-a))^2
        # For simplicity, assume Omega_m = 0.3, Omega_r = 0, Omega_L = 0.7
        Omega_m = 0.3
        Omega_L = 0.7
        term_m = Omega_m * a**(-3)
        term_L = Omega_L * a**(-3*(1+w0+wa)) * (1 + wa*(1-a))**2
        return np.sqrt(term_m + term_L)
    
    def integrand(u, z, w0, wa):
        """g(u) = u / E(a(z*u)) where a = 1/(1+z*u)"""
        a = 1.0 / (1.0 + z * u)
        return u / E_of_a(a, w0, wa)
    
    def reference_integral(z, w0, wa, n_points=10000):
        """High-accuracy reference integral using many points."""
        u = np.linspace(0, 1, n_points)
        vals = integrand(u, z, w0, wa)
        return np.trapz(vals, u)
    
    def quadrature_integral(nodes, weights, z, w0, wa):
        """Quadrature approximation of the integral."""
        u = nodes
        vals = integrand(u, z, w0, wa)
        return np.sum(weights * vals)
    
    def objective(nodes, weights):
        """Worst-case relative error across test models and redshifts."""
        max_rel_err = 0.0
        test_zs = np.array([0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0])
        
        for w0, wa in test_models:
            for z in test_zs:
                ref = reference_integral(z, w0, wa, n_points=5000)
                if abs(ref) < 1e-15:
                    continue
                quad = quadrature_integral(nodes, weights, z, w0, wa)
                rel_err = abs(quad - ref) / abs(ref)
                max_rel_err = max(max_rel_err, rel_err)
        
        return max_rel_err
    
    # Try a few optimization approaches within time budget
    best_nodes = nodes.copy()
    best_weights = weights.copy()
    best_err = objective(best_nodes, best_weights)
    
    # Simple local optimization: perturb nodes and weights
    # Use coordinate descent with small steps
    np.random.seed(42)
    
    for iteration in range(50):
        if time.time() > deadline - 10.0:
            break
        
        # Perturb one node at a time
        for i in range(8):
            # Try small perturbations to node i
            for delta in [-0.01, 0.01, -0.005, 0.005]:
                new_nodes = best_nodes.copy()
                new_nodes[i] += delta
                # Ensure strictly increasing and in (0,1)
                if not (np.all(new_nodes > 0) and np.all(new_nodes < 1)):
                    continue
                if not np.all(np.diff(new_nodes) > 0):
                    continue
                
                # Renormalize weights
                new_weights = best_weights / np.sum(best_weights)
                
                err = objective(new_nodes, new_weights)
                if err < best_err:
                    best_nodes = new_nodes
                    best_weights = new_weights
                    best_err = err
        
        # Perturb one weight at a time
        for i in range(8):
            for delta in [-0.01, 0.01, -0.005, 0.005]:
                new_weights = best_weights.copy()
                new_weights[i] += delta
                # Ensure positive and sum to 1
                if not np.all(new_weights > 0):
                    continue
                new_weights = new_weights / np.sum(new_weights)
                
                err = objective(best_nodes, new_weights)
                if err < best_err:
                    best_weights = new_weights
                    best_err = err
    
    # Final check
    if time.time() > deadline - 5.0:
        pass  # Just return what we have
    
    # Ensure final validity
    sort_idx = np.argsort(best_nodes)
    best_nodes = best_nodes[sort_idx]
    best_weights = best_weights[sort_idx]
    best_weights = best_weights / np.sum(best_weights)
    
    assert np.all(best_nodes > 0) and np.all(best_nodes < 1)
    assert np.all(np.diff(best_nodes) > 0)
    assert np.all(best_weights > 0)
    assert abs(np.sum(best_weights) - 1.0) < 1e-12
    
    return {"nodes": best_nodes, "weights": best_weights}
# EVOLVE-BLOCK-END
