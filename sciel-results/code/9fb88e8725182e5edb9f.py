"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import quad


def construct():
    """
    Find an 8-node positive quadrature rule for dimensionless luminosity distance.
    
    The integral is: dL/(c/H0) = (1+z) * z * ∫_0^1 da / [a * E(a)]
    where E(a) = H(a)/H0 depends on the cosmological model.
    
    We use a weighted Gauss-Legendre approach: we want to find nodes x_i and weights w_i
    such that sum_i w_i / E(a_i) approximates ∫_0^1 da/(a*E(a)) for a variety of E(a).
    
    Key insight: For CPL (Chevallier-Polarski-Linder) models:
    E(a) = a^(-3(1+w0+wa)) * (1 + wa*(1-a))^(-wa)  ... let me be more careful.
    
    Actually, E(a) = (1+z)^(3(1+w0+wa)) * (1 - wa*z/(1+z))^(-wa)
    In terms of a: z = 1/a - 1, so 1+z = 1/a
    E(a) = (1/a)^(3(1+w0+wa)) * (1 - wa*(1/a-1)/(1/a))^(-wa)
         = a^(-3(1+w0+wa)) * (1 - wa*(1-a)/a)^(-wa)
         = a^(-3(1+w0+wa)) * (a - wa*(1-a))^(-wa) * a^wa
         = a^(-3(1+w0+wa)+wa) * (a - wa + wa*a)^(-wa)
         = a^(-3-3w0-3wa+wa) * (a(1+wa) - wa)^(-wa)
         = a^(-3-3w0-2wa) * (a(1+wa) - wa)^(-wa)
    
    So 1/(a*E(a)) = a^(3+3w0+2wa) / [a * (a(1+wa) - wa)^(-wa)]
    Wait, let me redo this.
    
    dL/(c/H0) = (1+z) * ∫_0^z dz'/E(z')
    
    Change variable: a = 1/(1+z), so z = 1/a - 1, dz = -da/a^2
    When z=0, a=1; when z=z, a=1/(1+z)
    
    dL/(c/H0) = (1+z) * ∫_{1/(1+z)}^{1} da/a^2 * 1/E(a)
    
    Hmm, but the task says: dL/(c/H0) = (1+z)*z*sum_i weights[i]/E(a_i)
    where a_i = 1/(1+z*nodes[i])
    
    Let me re-derive. The standard formula is:
    dL = (c/H0) * (1+z) * ∫_0^z dz'/E(z')
    
    Let u = z'/z, so z' = z*u, dz' = z*du
    dL = (c/H0) * (1+z) * z * ∫_0^1 du / E(z*u)
    
    So the integral is ∫_0^1 du / E(z*u), and we approximate this with sum_i w_i / E(z*x_i)
    where x_i are the nodes and w_i are the weights.
    
    In terms of a: a = 1/(1+z'), so a_i = 1/(1+z*x_i).
    
    E(a) = H(a)/H0. For CPL:
    E(z) = (1+z)^(3(1+w0+wa)) * (1 - wa*z/(1+z))^(-wa)
    
    So 1/E(z*x_i) = 1/[(1+z*x_i)^(3(1+w0+wa)) * (1 - wa*z*x_i/(1+z*x_i))^(-wa)]
    
    This is complex. Let me just use the fact that for different cosmologies, E(a) varies,
    and we need a universal quadrature.
    
    Strategy: Use Gauss-Legendre as a starting point, then optimize to minimize worst-case
    error over a set of representative cosmologies.
    """
    start_time = time.time()
    deadline = start_time + 80  # 80 second deadline with 10s margin
    
    # Representative cosmologies for optimization
    # (w0, wa) pairs covering a range of dark energy models
    # LambdaCDM: w0=-1, wa=0
    # CPL variations
    cosmologies = [
        (-1.0, 0.0),     # LambdaCDM
        (-1.0, 0.1),
        (-1.0, -0.1),
        (-0.9, 0.0),
        (-0.9, 0.1),
        (-0.9, -0.1),
        (-0.8, 0.0),
        (-0.8, 0.1),
        (-0.8, -0.1),
        (-0.7, 0.0),
        (-0.7, 0.1),
        (-0.7, -0.1),
        (-1.1, 0.0),
        (-1.1, 0.1),
        (-1.1, -0.1),
    ]
    
    # Redshifts to test at
    z_values = np.array([0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0])
    
    def E_of_a(a, w0, wa):
        """Compute E(a) = H(a)/H0 for CPL model."""
        # E(a) = a^(-3(1+w0+wa)) * (1 + wa*(1-a))^(-wa)
        # Wait, let me be more careful.
        # H(z)/H0 = (1+z)^(3(1+w0+wa)) * (1 - wa*z/(1+z))^(-wa)
        # In terms of a: 1+z = 1/a, z = (1-a)/a
        # H(a)/H0 = (1/a)^(3(1+w0+wa)) * (1 - wa*(1-a)/a / (1/a))^(-wa)
        #          = a^(-3(1+w0+wa)) * (1 - wa*(1-a))^(-wa)
        #          = a^(-3-3w0-3wa) * (1 - wa + wa*a)^(-wa)
        #          = a^(-3-3w0-3wa) * (wa*a + 1 - wa)^(-wa)
        
        # Handle edge cases
        a_safe = np.maximum(a, 1e-15)
        base1 = a_safe ** (-3.0 * (1.0 + w0 + wa))
        base2_arg = wa * a_safe + (1.0 - wa)
        base2_arg_safe = np.maximum(np.abs(base2_arg), 1e-15)
        # Need to handle sign of base2_arg for non-integer powers
        base2 = base2_arg_safe ** (-wa)
        return base1 * base2
    
    def integrand_u(u, z, w0, wa):
        """Integrand for the integral ∫_0^1 du / E(z*u)."""
        z_u = z * u
        # E(z_u) = (1+z_u)^(3(1+w0+wa)) * (1 - wa*z_u/(1+z_u))^(-wa)
        one_plus_z = 1.0 + z_u
        arg = 1.0 - wa * z_u / one_plus_z
        # Handle potential negative arg for some parameters
        if arg <= 0:
            return 1e10  # Large penalty
        E = one_plus_z ** (3.0 * (1.0 + w0 + wa)) * arg ** (-wa)
        return 1.0 / E
    
    def exact_integral(z, w0, wa):
        """Compute the exact integral ∫_0^1 du / E(z*u) using adaptive quadrature."""
        result, _ = quad(lambda u: integrand_u(u, z, w0, wa), 0, 1, limit=200)
        return result
    
    def quadrature_value(nodes, weights, z, w0, wa):
        """Compute quadrature approximation sum_i w_i / E(z*x_i)."""
        z_nodes = z * nodes
        one_plus_z = 1.0 + z_nodes
        arg = 1.0 - wa * z_nodes / one_plus_z
        arg_safe = np.maximum(arg, 1e-15)
        E = one_plus_z ** (3.0 * (1.0 + w0 + wa)) * arg_safe ** (-wa)
        return np.sum(weights / E)
    
    def worst_case_error(nodes, weights):
        """Compute worst-case relative error over all test cosmologies and redshifts."""
        max_err = 0.0
        for w0, wa in cosmologies:
            for z in z_values:
                exact = exact_integral(z, w0, wa)
                approx = quadrature_value(nodes, weights, z, w0, wa)
                if exact != 0:
                    rel_err = abs(approx - exact) / abs(exact)
                    max_err = max(max_err, rel_err)
        return max_err
    
    # Start with Gauss-Legendre nodes and weights
    from numpy.polynomial.legendre import leggauss
    x_gl, w_gl = leggauss(8)
    x0 = (x_gl + 1.0) / 2.0
    w0_init = w_gl / 2.0
    
    # Ensure strictly increasing and in (0,1)
    x0 = np.sort(x0)
    # Make sure they're strictly in (0,1)
    x0 = np.clip(x0, 1e-6, 1 - 1e-6)
    # Ensure strict increase
    for i in range(1, 8):
        if x0[i] <= x0[i-1]:
            x0[i] = x0[i-1] + 1e-6
    
    # Optimize using differential evolution first for global exploration
    def objective(params):
        if time.time() > deadline - 5:
            return 1e10
        nodes = np.sort(params[:8])
        weights = params[8:]
        # Check constraints
        if nodes[0] <= 0 or nodes[-1] >= 1:
            return 1e10
        if np.any(np.diff(nodes) <= 0):
            return 1e10
        if np.any(weights <= 0):
            return 1e10
        if abs(np.sum(weights) - 1.0) > 1e-8:
            return 1e10
        return worst_case_error(nodes, weights)
    
    # Use a simpler approach: optimize nodes with weights derived from a basis
    # Actually, let's try a direct optimization with penalty for constraint violations
    
    def objective_constrained(params):
        if time.time() > deadline - 5:
            return 1e10
        nodes = np.sort(params[:8])
        weights = params[8:]
        # Soft constraints
        penalty = 0.0
        if nodes[0] <= 0:
            penalty += 1e6 * abs(nodes[0])
        if nodes[-1] >= 1:
            penalty += 1e6 * abs(nodes[-1] - 1)
        diffs = np.diff(nodes)
        if np.any(diffs <= 0):
            penalty += 1e6 * np.sum(np.maximum(0, -diffs))
        if np.any(weights <= 0):
            penalty += 1e6 * np.sum(np.maximum(0, -weights))
        weight_sum_dev = abs(np.sum(weights) - 1.0)
        penalty += 1e4 * weight_sum_dev ** 2
        
        if penalty > 1e-3:
            return penalty
        
        return worst_case_error(nodes, weights) + penalty
    
    # Try differential evolution for global optimization
    bounds = [(1e-6, 1-1e-6)] * 8 + [(1e-4, 0.5)] * 8
    best_result = None
    best_score = float('inf')
    
    try:
        de_result = differential_evolution(
            objective_constrained,
            bounds=bounds,
            seed=42,
            maxiter=50,
            popsize=10,
            tol=1e-6,
            polish=False,
            init='sobol'
        )
        if de_result.fun < best_score:
            best_score = de_result.fun
            best_result = de_result.x
    except Exception as e:
        pass
    
    if time.time() > deadline - 20:
        # Time to refine with local optimization
        if best_result is not None:
            nodes = np.sort(best_result[:8])
            weights = best_result[8:]
        else:
            nodes = x0.copy()
            weights = w0_init.copy()
        
        # Local refinement
        def local_objective(params):
            if time.time() > deadline - 2:
                return 1e10
            nodes = np.sort(params[:8])
            weights = params[8:]
            penalty = 0.0
            if nodes[0] <= 0:
                penalty += 1e6
            if nodes[-1] >= 1:
                penalty += 1e6
            diffs = np.diff(nodes)
            if np.any(diffs <= 0):
                penalty += 1e6
            if np.any(weights <= 0):
                penalty += 1e6
            weight_sum_dev = abs(np.sum(weights) - 1.0)
            penalty += 1e4 * weight_sum_dev ** 2
            if penalty > 1e-3:
                return penalty
            return worst_case_error(nodes, weights) + penalty
        
        try:
            res = minimize(
                local_objective,
                x0=np.concatenate([nodes, weights]),
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-12}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_result = res.x
        except Exception:
            pass
        
        if best_result is not None:
            nodes = np.sort(best_result[:8])
            weights = best_result[8:]
        else:
            nodes = x0
            weights = w0_init
        
        # Final constraint enforcement
        nodes = np.clip(nodes, 1e-8, 1 - 1e-8)
        for i in range(1, 8):
            if nodes[i] <= nodes[i-1]:
                nodes[i] = nodes[i-1] + 1e-8
        weights = np.maximum(weights, 1e-10)
        weights = weights / np.sum(weights)
        
        return {"nodes": nodes, "weights": weights}
    
    # Refine with local optimization
    if best_result is not None:
        nodes = np.sort(best_result[:8])
        weights = best_result[8:]
    else:
        nodes = x0.copy()
        weights = w0_init.copy()
    
    def local_objective(params):
        if time.time() > deadline - 2:
            return 1e10
        nodes = np.sort(params[:8])
        weights = params[8:]
        penalty = 0.0
        if nodes[0] <= 0:
            penalty += 1e6
        if nodes[-1] >= 1:
            penalty += 1e6
        diffs = np.diff(nodes)
        if np.any(diffs <= 0):
            penalty += 1e6
        if np.any(weights <= 0):
            penalty += 1e6
        weight_sum_dev = abs(np.sum(weights) - 1.0)
        penalty += 1e4 * weight_sum_dev ** 2
        if penalty > 1e-3:
            return penalty
        return worst_case_error(nodes, weights) + penalty
    
    try:
        res = minimize(
            local_objective,
            x0=np.concatenate([nodes, weights]),
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-12}
        )
        if res.fun < best_score:
            best_score = res.fun
            best_result = res.x
    except Exception:
        pass
    
    if best_result is not None:
        nodes = np.sort(best_result[:8])
        weights = best_result[8:]
    else:
        nodes = x0
        weights = w0_init
    
    # Final constraint enforcement
    nodes = np.clip(nodes, 1e-8, 1 - 1e-8)
    for i in range(1, 8):
        if nodes[i] <= nodes[i-1]:
            nodes[i] = nodes[i-1] + 1e-8
    weights = np.maximum(weights, 1e-10)
    weights = weights / np.sum(weights)
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
