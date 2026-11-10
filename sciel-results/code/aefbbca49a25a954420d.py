"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def construct():
    # We design an 8-node positive quadrature rule for the integral
    # I(z) = (1+z) * integral_0^z u / E(1/(1+u)) du  (in dimensionless form with a=1/(1+z))
    # Actually the task says: dL/(c/H0) = (1+z)*z*sum_i weights[i]/E(a_i) where a_i=1/(1+z*nodes[i])
    # Wait, let's re-read carefully:
    # dL/(c/H0) = (1+z)*z*sum_i weights[i]/E(a_i)
    # This is NOT a standard quadrature of an integral. It is a specific formula.
    # Let's check the standard formula for luminosity distance in flat universe:
    # d_L = (c/H0) * (1+z) * integral_0^z dz' / E(z')
    # The task says: dL/(c/H0) = (1+z)*z*sum_i weights[i]/E(a_i)
    # This looks like a quadrature approximation of the integral, but with an extra factor of z?
    # Let's check: if we approximate integral_0^z dz'/E(z') by z * sum w_i / E(z_i), 
    # then dL/(c/H0) = (1+z) * z * sum w_i / E(z_i).
    # This is a quadrature rule for the integral from 0 to z of f(z') dz', where f(z') = 1/E(z').
    # The nodes are z_i = z * nodes[i].
    # So the quadrature is: integral_0^z f(z') dz' ≈ z * sum_i w_i f(z * nodes[i])
    # This is a standard quadrature rule on [0,1] mapped to [0,z].
    # The nodes are in [0,1], weights sum to 1.
    # The integrand is f(z') = 1/E(z').
    # But E(z) depends on the cosmology. The task says "Minimize the worst relative error against independent adaptive integration on disjoint public validation cosmologies."
    # So we need to find nodes and weights that work well for a range of cosmologies.
    
    # The current program uses Gauss-Legendre. We can try to optimize the nodes and weights
    # to minimize the worst-case error over a set of representative cosmologies.
    
    # Since we don't have the specific validation cosmologies, we can use a set of representative
    # CPL (Chevallier-Polarski-Linder) cosmologies.
    # E^2(z) = (1+z)^2 + Omega_m (1+z)^3 + Omega_Lambda (1+z)^{3(1+w0+wa*z)}
    # with Omega_m + Omega_Lambda = 1 (flat universe).
    
    # Let's define a set of cosmologies to optimize for.
    cosmologies = [
        {'Om': 0.3, 'OL': 0.7, 'w0': -1.0, 'wa': 0.0},   # Lambda CDM
        {'Om': 0.3, 'OL': 0.7, 'w0': -0.5, 'wa': 0.5},   # Quintessence
        {'Om': 0.3, 'OL': 0.7, 'w0': -1.5, 'wa': 0.5},   # Phantom
        {'Om': 0.2, 'OL': 0.8, 'w0': -1.0, 'wa': 0.0},   # Different matter density
        {'Om': 0.4, 'OL': 0.6, 'w0': -1.0, 'wa': 0.0},   # Different matter density
        {'Om': 0.3, 'OL': 0.7, 'w0': -1.0, 'wa': 1.0},   # Strong evolution
    ]
    
    # We'll optimize the nodes and weights to minimize the worst-case relative error
    # over these cosmologies and a range of redshifts.
    
    # Initial guess: Gauss-Legendre
    x, w = np.polynomial.legendre.leggauss(8)
    nodes0 = (x + 1.0) / 2.0
    weights0 = w / 2.0
    
    # We'll optimize in terms of log(nodes) and log(weights) to ensure positivity.
    # But nodes must be in [0,1], so we can use a transformation.
    # Let's use nodes = sigmoid(t) / (1 + sum sigmoid(t)) to ensure they are in (0,1) and sum to 1?
    # No, nodes don't need to sum to 1. Weights sum to 1.
    # Let's use nodes = 1 / (1 + exp(-t)) to ensure nodes in (0,1).
    # And weights = exp(s) / sum(exp(s)) to ensure positive and sum to 1.
    
    def compute_error(params):
        t_nodes, t_weights = params[:8], params[8:16]
        nodes = 1.0 / (1.0 + np.exp(-t_nodes))
        # Ensure nodes are strictly increasing
        nodes = np.sort(nodes)
        # Add small epsilon to avoid 0 or 1
        nodes = np.clip(nodes, 1e-12, 1.0 - 1e-12)
        weights = np.exp(t_weights)
        weights = weights / np.sum(weights)
        
        max_err = 0.0
        for cosmo in cosmologies:
            Om = cosmo['Om']
            OL = cosmo['OL']
            w0 = cosmo['w0']
            wa = cosmo['wa']
            
            def E(z):
                return np.sqrt(Om * (1+z)**3 + OL * (1+z)**(3*(1+w0+wa*z)))
            
            # Test at several redshifts
            for z in [0.1, 0.5, 1.0, 2.0, 3.0]:
                # True integral: integral_0^z dz'/E(z')
                # Use adaptive quadrature for reference
                from scipy.integrate import quad
                true_val, _ = quad(lambda zp: 1.0/E(zp), 0, z, limit=200)
                true_dL = (1+z) * true_val
                
                # Quadrature approximation
                z_nodes = z * nodes
                approx_integral = z * np.sum(weights / E(z_nodes))
                approx_dL = (1+z) * approx_integral
                
                if true_dL > 1e-12:
                    rel_err = abs(approx_dL - true_dL) / true_dL
                    max_err = max(max_err, rel_err)
        
        return max_err
    
    # Optimize
    x0 = np.concatenate([np.log(nodes0 / (1 - nodes0)), np.log(weights0)])
    result = minimize(compute_error, x0, method='Nelder-Mead', 
                      options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
    
    t_nodes, t_weights = result.x[:8], result.x[8:16]
    nodes = 1.0 / (1.0 + np.exp(-t_nodes))
    nodes = np.sort(nodes)
    nodes = np.clip(nodes, 1e-12, 1.0 - 1e-12)
    weights = np.exp(t_weights)
    weights = weights / np.sum(weights)
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
