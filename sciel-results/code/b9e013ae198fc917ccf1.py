"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as so
import scipy.integrate as si

def _E_of_a(a, w0, wa):
    """Compute E(a) = H(a)/H0 for CPL model with parameters w0, wa.
    a is scale factor.
    E^2 = (1-a)^(-3*(1+w0+wa)) * (1-a)^(3*wa) * (1 + Omega_m*(1-a)^-3) is not quite right.
    Standard CPL: w(a) = w0 + wa*(1-a)
    H^2/H0^2 = Omega_m a^-3 + Omega_L exp(3 * int_0^a (1+w(a'))/a' da')
    int (1+w)/a da = int (1+w0+wa(1-a))/a da = (1+w0+wa)ln(a) - wa*a
    So E^2 = Omega_m a^-3 + Omega_L * a^(3*(1+w0+wa)) * exp(-3*wa*a)
    """
    Omega_m = 0.3
    Omega_L = 0.7
    # Ensure numerical stability
    a = np.clip(a, 1e-6, 1.0)
    term_m = Omega_m * a**(-3)
    # Handle the exponent carefully
    # log_term = 3*(1+w0+wa)*np.log(a) - 3*wa*a
    # term_L = Omega_L * np.exp(log_term)
    # For small a, a^-3 dominates.
    # Let's use the standard formula.
    w_a = w0 + wa * (1 - a)
    # Integral of (1+w)/a from 1 to a (since a goes 1->0 or 0->1?)
    # dL = c/H0 * (1+z) int_0^z dz'/E(z')
    # Change variable to a = 1/(1+z). z=0 -> a=1, z=z_max -> a_min.
    # dz = -da/a^2.
    # Integrand becomes (1/a) * 1/E(a) * da/a^2 = 1/(a^3 E(a)) da.
    # So dL/(c/H0) = int_{a_min}^{1} 1/(a^3 E(a)) da.
    
    # Wait, the task says:
    # dL/(c/H0) = (1+z)*z*sum_i weights[i]/E(a_i)
    # This formula in the prompt seems to imply the nodes are in z-space?
    # "Nodes must be strictly increasing in [0,1]"
    # "a_i = 1/(1+z*nodes[i])"
    # If nodes are in [0,1], let u be the node. z_eff = z * u? No.
    # "a_i=1/(1+z*nodes[i])" implies the quadrature is over the variable u in [0,1]
    # such that the integration variable is mapped from z to u.
    # Let's re-read carefully:
    # dL/(c/H0) = (1+z)*z*sum_i weights[i]/E(a_i)
    # This looks like a specific quadrature formula for the integral int_0^z (1+z')/E(z') dz'?
    # No, standard luminosity distance is dL = (1+z) int_0^z dz'/E(z').
    # The prompt formula: (1+z)*z*sum... this is strange.
    # Let's look at the seed: nodes are (i+0.5)/8.
    # If nodes are u_i in [0,1], and the formula is (1+z)*z * sum w_i / E(a_i)
    # This implies the integral is approximated by z * sum w_i / E(a_i) ?
    # Let's check dimensions.
    # If z is large, dL ~ z^2.
    # Let's assume the evaluator uses this EXACT formula.
    # I must optimize nodes u_i and weights w_i to minimize error against "independent adaptive integration".
    # What is the exact integral?
    # The prompt says: "Minimize the worst relative error against independent adaptive integration"
    # It does NOT give the exact integral formula explicitly in the text, but implies it's the standard cosmological integral.
    # Standard: dL = (c/H0) (1+z) \int_0^z \frac{dz'}{E(z')}
    # Let's verify if the prompt formula is a quadrature rule for this.
    # If I use midpoint rule on u in [0,1] with a_i = 1/(1+z*u_i),
    # then dz' = -da'/a'^2.
    # Integral = \int_{a(z)}^1 \frac{1}{E(a')} \frac{da'}{a'^2}
    # Let u = (1-a)/(1-a_min)? No.
    # The prompt defines a_i = 1/(1+z*nodes[i]).
    # This suggests the variable of integration is t in [0,1] where z' = z*t.
    # Then dz' = z dt.
    # Integral = (1+z) \int_0^1 \frac{z dt}{E(1/(1+z t))}
    # = (1+z) z \int_0^1 \frac{dt}{E(a(t))}
    # So the quadrature is sum_i w_i / E(a_i) approximating \int_0^1 \frac{dt}{E(a(t))}
    # And the total dL is (1+z) * z * (quadrature sum).
    # This matches the prompt formula exactly.
    
    # So the task is:
    # Find u_i, w_i (i=1..8) such that \sum w_i / E(a_i) approximates \int_0^1 \frac{dt}{E(a(t))}
    # for a given z, and this must work for ALL z and ALL cosmologies.
    # Wait, "The same nodes and weights must work across all models."
    # And "Minimize the worst relative error against independent adaptive integration on disjoint public validation cosmologies."
    # The error is in dL.
    # dL_quad = (1+z) z \sum w_i / E(a_i)
    # dL_exact = (1+z) \int_0^z dz'/E(z') = (1+z) z \int_0^1 \frac{dt}{E(1/(1+zt))}
    # Relative error = |dL_quad - dL_exact| / dL_exact
    # = | \sum w_i / E(a_i) - \int_0^1 \frac{dt}{E(1/(1+zt))} | / \int_0^1 \frac{dt}{E(1/(1+zt))}
    
    # Since (1+z)z cancels out in the relative error (assuming z>0),
    # we are minimizing the relative error of the quadrature approximation of I(z, w0, wa) = \int_0^1 \frac{dt}{E(1/(1+zt))}
    # using \sum w_i / E(1/(1+z u_i)).
    
    # The integrand f(t) = 1/E(1/(1+zt)) depends on z.
    # For small z, f(t) is nearly constant.
    # For large z, f(t) varies significantly.
    # We need a universal rule.
    
    # Strategy:
    # 1. Define a set of "training" cosmologies (w0, wa) and redshifts z.
    # 2. Define the objective: max relative error over the training set.
    # 3. Optimize u_i, w_i using SLSQP or similar, enforcing w_i > 0, sum(w)=1, 0 < u_1 < ... < u_8 < 1.
    # 4. Return the best found.

def _exact_integral(z, w0, wa):
    """Compute exact integral \int_0^1 \frac{dt}{E(1/(1+zt))} using adaptive quadrature."""
    Omega_m = 0.3
    Omega_L = 0.7
    
    def integrand(t):
        a = 1.0 / (1.0 + z * t)
        if a <= 0:
            return 0.0
        # E(a) calculation
        # E^2 = Omega_m a^-3 + Omega_L a^(3*(1+w0+wa)) * exp(-3*wa*a)
        # Note: the exponent in CPL is usually derived from w(a)=w0+wa(1-a)
        # \int_1^a (1+w(a'))/a' da' = (1+w0+wa) \ln a - wa (a-1)
        # So E^2 = Omega_m a^-3 + Omega_L a^{3(1+w0+wa)} e^{-3wa(a-1)}
        # Let's use this form.
        log_term = 3.0 * (1.0 + w0 + wa) * np.log(a) - 3.0 * wa * (a - 1.0)
        # To avoid overflow/underflow, compute log(E^2) if possible?
        # E^2 can be very small or large.
        # For a near 1, log_term ~ 0.
        # For a near 0, log(a) -> -inf.
        
        # Let's calculate E^2 directly with care.
        term_m = Omega_m * (a ** -3)
        # term_L = Omega_L * exp(log_term)
        # If log_term is very negative, term_L is 0.
        if log_term < -700:
            term_L = 0.0
        else:
            term_L = Omega_L * np.exp(log_term)
        
        E2 = term_m + term_L
        if E2 <= 0:
            return 0.0
        E = np.sqrt(E2)
        return 1.0 / E

    # Integrate from 0 to 1
    res, err = si.quad(integrand, 0.0, 1.0, limit=100, epsabs=1e-12, epsrel=1e-12)
    return res

def _quadrature_value(nodes, weights, z, w0, wa):
    """Compute \sum w_i / E(a_i)"""
    Omega_m = 0.3
    Omega_L = 0.7
    total = 0.0
    for u, w in zip(nodes, weights):
        a = 1.0 / (1.0 + z * u)
        if a <= 0:
            continue
        log_term = 3.0 * (1.0 + w0 + wa) * np.log(a) - 3.0 * wa * (a - 1.0)
        if log_term < -700:
            term_L = 0.0
        else:
            term_L = Omega_L * np.exp(log_term)
        term_m = Omega_m * (a ** -3)
        E2 = term_m + term_L
        if E2 <= 0:
            continue
        E = np.sqrt(E2)
        total += w / E
    return total

def _objective(params, training_data):
    """
    params: 16 vector [u1..u8, w1..w8]
    training_data: list of (z, w0, wa, exact_val)
    Returns worst-case relative error.
    """
    nodes = params[:8]
    weights = params[8:]
    
    # Check constraints (soft penalty or hard via bounds)
    # SLSQP handles bounds and equality constraints.
    # Inequality: nodes strictly increasing.
    # We will use bounds for nodes in [0,1] and weights in [0,1].
    # We need to enforce u_i < u_{i+1}.
    # SLSQP can take inequality constraints.
    
    max_err = 0.0
    for z, w0, wa, exact_val in training_data:
        if exact_val == 0:
            continue
        quad_val = _quadrature_value(nodes, weights, z, w0, wa)
        rel_err = abs(quad_val - exact_val) / exact_val
        if rel_err > max_err:
            max_err = rel_err
    return max_err

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s margin
    
    # Training data: sample of cosmologies and redshifts
    # w0 in [-1.5, -0.5], wa in [-1.0, 1.0]
    # z in [0.1, 2.0]
    # We need to be efficient.
    
    # Initial guess: Gauss-Legendre nodes and weights
    nodes_init, weights_init = np.polynomial.legendre.leggauss(8)
    # leggauss gives nodes in [-1, 1]. Map to [0, 1]: u = (x+1)/2
    nodes_init = (nodes_init + 1.0) / 2.0
    weights_init = weights_init / 2.0
    
    # Ensure strict ordering (leggauss is strictly increasing)
    # Ensure weights positive (they are)
    # Ensure sum weights = 1 (they are)
    
    params_init = np.concatenate([nodes_init, weights_init])
    
    # Define training set
    # To keep it fast, use a small grid.
    w0_vals = [-1.2, -1.0, -0.9, -0.8, -0.7, -0.5]
    wa_vals = [-0.5, 0.0, 0.5]
    z_vals = [0.1, 0.3, 0.5, 1.0, 1.5, 2.0]
    
    training_data = []
    for w0 in w0_vals:
        for wa in wa_vals:
            for z in z_vals:
                try:
                    exact_val = _exact_integral(z, w0, wa)
                    if exact_val > 0:
                        training_data.append((z, w0, wa, exact_val))
                except Exception:
                    pass
                
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
        
    if not training_data:
        # Fallback to initial guess
        return {"nodes": nodes_init, "weights": weights_init}
    
    # Optimization
    # Bounds
    # Nodes: 0 < u_i < 1, and u_i < u_{i+1}
    # Weights: w_i > 0, sum w_i = 1
    
    # To handle u_i < u_{i+1}, we can parameterize nodes as cumulative sums of positive deltas?
    # Or just use SLSQP with inequality constraints.
    # Inequality: u_{i+1} - u_i - eps >= 0
    
    def fun(x):
        if time.time() > deadline:
            return 1e10
        return _objective(x, training_data)
    
    # Constraints
    # Sum weights = 1
    cons_eq = [
        {'type': 'eq', 'fun': lambda x: np.sum(x[8:]) - 1.0}
    ]
    
    # Inequalities: u_{i+1} - u_i > 0
    cons_ineq = []
    for i in range(7):
        cons_ineq.append({
            'type': 'ineq', 
            'fun': lambda x, i=i: x[i+1] - x[i] - 1e-6
        })
    
    # Bounds
    # Nodes in [1e-6, 1-1e-6]
    # Weights in [1e-6, 1]
    bounds = []
    for i in range(8):
        bounds.append((1e-6, 1.0 - 1e-6)) # Nodes
    for i in range(8):
        bounds.append((1e-6, 1.0)) # Weights
    
    best_params = params_init
    best_score = fun(params_init)
    
    # Try SLSQP
    try:
        res = so.minimize(fun, params_init, method='SLSQP', bounds=bounds, 
                          constraints=cons_eq + cons_ineq, options={'maxiter': 100, 'ftol': 1e-12})
        if res.success and res.fun < best_score:
            best_params = res.x
            best_score = res.fun
    except Exception:
        pass
    
    # Try a few random restarts if time permits
    if time.time() < deadline - 5.0:
        for _ in range(5):
            # Perturb initial guess
            nodes_pert = nodes_init + np.random.normal(0, 0.05, 8)
            nodes_pert = np.sort(np.clip(nodes_pert, 1e-5, 1-1e-5))
            # Ensure strict increase
            for i in range(1, 8):
                if nodes_pert[i] <= nodes_pert[i-1]:
                    nodes_pert[i] = nodes_pert[i-1] + 1e-5
            
            weights_pert = weights_init + np.random.normal(0, 0.02, 8)
            weights_pert = np.clip(weights_pert, 1e-6, None)
            weights_pert = weights_pert / np.sum(weights_pert)
            
            params_pert = np.concatenate([nodes_pert, weights_pert])
            
            try:
                res = so.minimize(fun, params_pert, method='SLSQP', bounds=bounds, 
                                  constraints=cons_eq + cons_ineq, options={'maxiter': 50, 'ftol': 1e-12})
                if res.success and res.fun < best_score:
                    best_params = res.x
                    best_score = res.fun
            except Exception:
                pass
            
            if time.time() > deadline - 2.0:
                break
    
    # Final check and cleanup
    final_nodes = best_params[:8]
    final_weights = best_params[8:]
    
    # Enforce strict increasing
    for i in range(1, 8):
        if final_nodes[i] <= final_nodes[i-1]:
            final_nodes[i] = final_nodes[i-1] + 1e-12
    
    # Enforce positive weights and sum to 1
    final_weights = np.clip(final_weights, 1e-12, None)
    final_weights = final_weights / np.sum(final_weights)
    
    # Check sum to 1 within 1e-12
    # The division ensures sum is 1.
    
    return {"nodes": final_nodes, "weights": final_weights}
# EVOLVE-BLOCK-END
