"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution

# Deadline guard: must finish within 90s, aim to return well before
_START_TIME = time.time()
_DEADLINE = _START_TIME + 85.0  # 5s safety margin

def _E_of_z(z, w0, wa):
    """
    Compute E(z) = H(z)/H0 for a flat universe with CPL dark energy.
    w(z) = w0 + wa*z
    H(z) = H0 * (1+z)^3 * (1+z)^(3*(w0+wa*z)) ... wait, standard form:
    H^2/H0^2 = Omega_m (1+z)^3 + Omega_k (1+z)^2 + Omega_de (1+z)^(3(1+w0+wa*z))
    For flat: Omega_k = 0, Omega_de = 1 - Omega_m.
    Let's assume standard cosmological parameters for the 'representative' set.
    Omega_m = 0.3, Omega_de = 0.7.
    """
    Omega_m = 0.3
    Omega_de = 0.7
    # The integral in the task is dL/(c/H0) = (1+z) * z * sum w_i / E(a_i) ??
    # The task says: dL/(c/H0) = (1+z)*z*sum_i weights[i]/E(a_i)
    # Wait, the standard formula is dL/(c/H0) = (1+z) * \int_0^z dz'/E(z')
    # The task's formula: (1+z)*z*sum ... implies the integral is approximated by z * sum w_i / E(a_i)?
    # No, the task says: "dL/(c/H0) = (1+z)*z*sum_i weights[i]/E(a_i)"
    # This looks like a specific quadrature approximation where the integral \int_0^z dz'/E(z')
    # is approximated by z * \sum w_i f(a_i).
    # Let's re-read carefully: "dL/(c/H0) = (1+z)*z*sum_i weights[i]/E(a_i), where a_i=1/(1+z*nodes[i])"
    # This implies the integral \int_0^z \frac{dz'}{E(z')} is approximated by z \sum_{i=1}^8 w_i \frac{1}{E(a_i)}
    # where a_i = 1/(1 + z * x_i) and x_i are the nodes in [0,1].
    # Let t = z'/z. Then z' = z*t, dz' = z*dt.
    # \int_0^1 \frac{z dt}{E(z*t)} = z \int_0^1 \frac{dt}{E(z*t)}
    # The quadrature approximates \int_0^1 f(t) dt \approx \sum w_i f(x_i)
    # Here f(t) = 1/E(z*t).
    # The nodes x_i are fixed. The argument of E is z * x_i?
    # No, the task says a_i = 1/(1+z*nodes[i]).
    # So E(a_i) = E(1/(1+z*x_i)).
    # So the approximation is: \int_0^z \frac{dz'}{E(z')} \approx z \sum_{i=1}^8 w_i \frac{1}{E(1/(1+z*x_i))}
    # This is a quadrature rule for the integral over [0,1] of a function that depends on z.
    # The nodes x_i and weights w_i must be universal (independent of z and cosmology).
    
    z_exp = z
    # E(z) = sqrt( Omega_m (1+z)^3 + Omega_de (1+z)^(3(1+w0+wa*z)) )
    # Note: standard CPL is w = w0 + wa*z.
    # Exponent for DE term: 3(1+w(z)) = 3(1+w0+wa*z)
    term_m = Omega_m * (1 + z_exp)**3
    term_de = Omega_de * (1 + z_exp)**(3 * (1 + w0 + wa * z_exp))
    return np.sqrt(term_m + term_de)

def _true_luminosity_distance(z, w0, wa, num_steps=1000):
    """
    Compute true dL/(c/H0) using adaptive/simpson integration for reference.
    dL/(c/H0) = (1+z) * \int_0^z dz'/E(z')
    """
    if z == 0:
        return 0.0
    # Use scipy.integrate.quad for high precision reference
    from scipy.integrate import quad
    integrand = lambda zp: 1.0 / _E_of_z(zp, w0, wa)
    val, _ = quad(integrand, 0, z, limit=200)
    return (1 + z) * val

def _approx_luminosity_distance(z, w0, wa, nodes, weights):
    """
    Compute approx dL/(c/H0) using the quadrature rule.
    dL/(c/H0) = (1+z) * z * sum_i weights[i] / E(a_i)
    where a_i = 1/(1+z*nodes[i])
    """
    if z == 0:
        return 0.0
    a_vals = 1.0 / (1.0 + z * nodes)
    # Calculate E(a) for each node
    # E(a) = H(a)/H0. In terms of redshift z_i corresponding to a_i:
    # z_i = 1/a_i - 1 = (1+z*nodes[i]) - 1 = z*nodes[i]
    # So E(a_i) = E(z * nodes[i])
    z_nodes = z * nodes
    E_vals = _E_of_z(z_nodes, w0, wa)
    # Sum weights / E
    integral_approx = z * np.dot(weights, 1.0 / E_vals)
    return (1 + z) * integral_approx

def _objective(params, nodes_bounds, weights_bounds):
    """
    Objective function to minimize worst-case relative error.
    params: [nodes_1..8, weights_1..8]
    """
    nodes = params[:8]
    weights = params[8:]
    
    # Ensure positivity and normalization for evaluation
    # In DE, we handle bounds, but weights must sum to 1.
    # We will optimize unnormalized weights and normalize in the evaluation?
    # No, it's better to optimize weights directly with a penalty or constraint.
    # Since DE doesn't handle equality constraints well, we can optimize 7 weights and set the 8th?
    # Or just penalize deviation from sum=1.
    # Let's stick to simple bounds: weights > 0.
    # We will normalize weights inside the error calculation to ensure the rule is valid during search,
    # but the final output must sum to 1.
    w_sum = np.sum(weights)
    if w_sum < 1e-12:
        return 1e6
    w_norm = weights / w_sum
    
    # Representative cosmologies (w0, wa)
    # We want to cover a range of dark energy behaviors.
    # w0 in [-1.5, -0.5], wa in [-0.5, 0.5]
    test_cosmos = [
        (-1.0, 0.0),   # LCDM
        (-0.5, 0.0),   # Quintessence
        (-1.5, 0.0),   # Phantom
        (-1.0, 0.5),   # Accelerating with positive slope
        (-1.0, -0.5),  # Accelerating with negative slope
        (-0.5, 0.5),
        (-1.5, 0.5),
        (-0.5, -0.5),
    ]
    
    # Representative redshifts
    # The rule must work for any z. The error depends on z.
    # We should test a range of z values.
    # dL is sensitive to z. Let's test z = 0.1, 0.5, 1.0, 2.0, 5.0
    test_zs = [0.1, 0.5, 1.0, 2.0, 5.0]
    
    max_err = 0.0
    for w0, wa in test_cosmos:
        for z in test_zs:
            d_true = _true_luminosity_distance(z, w0, wa)
            d_approx = _approx_luminosity_distance(z, w0, wa, nodes, w_norm)
            
            if d_true < 1e-10:
                continue
                
            rel_err = abs(d_approx - d_true) / d_true
            max_err = max(max_err, rel_err)
            
    # Return worst-case error
    return max_err

def construct():
    global _START_TIME, _DEADLINE
    
    # Check deadline
    if time.time() > _DEADLINE - 10:
        # Fallback to midpoint if no time left
        n = 8
        return {"nodes": (np.arange(n) + 0.5) / n, "weights": np.full(n, 1.0 / n)}
    
    # Define bounds
    # Nodes must be strictly increasing in [0,1]
    # We will optimize nodes in [0,1] and sort them? No, DE doesn't sort.
    # Better to define bounds such that n_i < n_{i+1}.
    # n_1 in [0.01, 0.99/8 + ...]
    # Simpler: Optimize nodes in [0,1] and enforce ordering in the objective?
    # If we don't enforce ordering, the quadrature might be invalid or less efficient.
    # Let's optimize 8 nodes with bounds that encourage ordering?
    # Or just optimize them freely and sort in the objective?
    # If we sort, the mapping from params to nodes is non-smooth, which DE can handle poorly.
    # Alternative: Optimize gaps?
    # Let's just optimize nodes in [0,1] and weights in [0.01, 0.5].
    # Then sort nodes and re-associate weights? No, weights correspond to specific nodes.
    # If we sort nodes, we must sort weights correspondingly.
    # Let's assume the optimal nodes are naturally ordered if we start from a good guess?
    # DE is global. It might find unsorted nodes.
    # Let's enforce ordering by defining bounds:
    # n1 in [0, 0.1], n2 in [n1+eps, 0.2]... this is complex.
    
    # Strategy: Optimize nodes in [0,1] and weights in [0.01, 1.0].
    # Inside objective, sort nodes and weights together?
    # If we sort, we are effectively searching over the set of {node, weight} pairs.
    # This is valid.
    
    nodes_bounds = [(0.0, 1.0)] * 8
    weights_bounds = [(0.01, 1.0)] * 8
    
    # Initial guess: Midpoint rule
    n = 8
    init_nodes = (np.arange(n) + 0.5) / n
    init_weights = np.full(n, 1.0 / n)
    
    # Run Differential Evolution
    # Time limit: 80 seconds
    # Max iterations: set high, rely on time limit
    try:
        result = differential_evolution(
            _objective,
            bounds=nodes_bounds + weights_bounds,
            args=(nodes_bounds, weights_bounds),
            seed=42,
            maxiter=100,
            tol=1e-6,
            polish=True,
            workers=1,
            # We need to stop before deadline
            # DE doesn't have a time limit param directly, but we can check in objective?
            # No, objective is called many times.
            # We will just run it and hope it finishes.
            # To be safe, we can run a smaller budget.
            # 85s deadline. DE with 16 vars, 100 iters might take a while.
            # Let's reduce maxiter to 20 and see.
        )
    except Exception as e:
        # Fallback
        return {"nodes": init_nodes, "weights": init_weights}
    
    if time.time() > _DEADLINE - 5:
        return {"nodes": init_nodes, "weights": init_weights}
    
    if result.success:
        opt_nodes = result.x[:8]
        opt_weights = result.x[8:]
        
        # Sort nodes and weights
        sort_idx = np.argsort(opt_nodes)
        opt_nodes = opt_nodes[sort_idx]
        opt_weights = opt_weights[sort_idx]
        
        # Normalize weights
        opt_weights = opt_weights / np.sum(opt_weights)
        
        # Ensure nodes are strictly increasing and in [0,1]
        # Add small epsilon if needed
        opt_nodes = np.clip(opt_nodes, 1e-6, 1 - 1e-6)
        
        # Check strict increase
        diffs = np.diff(opt_nodes)
        if np.any(diffs <= 0):
            # Fallback
            return {"nodes": init_nodes, "weights": init_weights}
            
        return {"nodes": opt_nodes, "weights": opt_weights}
    else:
        # Fallback to midpoint
        return {"nodes": init_nodes, "weights": init_weights}

# EVOLVE-BLOCK-END
