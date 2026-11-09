"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog, differential_evolution
from scipy.integrate import quad
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0 # Safety margin (limit is 90s)
    
    # 1. Define Cosmology Models and Validation Set
    # Representative CPL models covering the expected parameter space
    # (Omega_m, w0, w1)
    cosmo_params = [
        (0.3, -1.0, 0.0),   # Lambda CDM
        (0.3, -1.2, 0.0),   # Phantom
        (0.3, -0.8, 0.0),   # Quintessence
        (0.3, -1.0, 0.5),   # Evolving DE (w1 > 0)
        (0.3, -1.0, -0.5),  # Evolving DE (w1 < 0)
        (0.2, -1.0, 0.0),   # Low matter
        (0.4, -1.0, 0.0),   # High matter
        (0.3, -1.5, 0.5),   # Strongly evolving phantom
        (0.3, -0.5, -0.5),  # Strongly evolving quintessence
    ]
    
    # Test z values: spread across the domain [0, 1]
    z_test = np.linspace(0.05, 1.0, 20)
    
    def E_func(z, Omega_m, w0, w1):
        Omega_L = 1.0 - Omega_m
        # E^2 = Om(1+z)^3 + OL(1+z)^(3(1+w0+w1z))
        # Handle z=0 case safely (1+0=1, so 1^x = 1)
        return np.sqrt(Omega_m * (1.0 + z)**3 + Omega_L * (1.0 + z)**(3.0 * (1.0 + w0 + w1 * z)))
    
    def true_integral(z, Omega_m, w0, w1):
        # Integral from 0 to 1 of 1/E(z*u) du
        integrand = lambda u: 1.0 / E_func(z * u, Omega_m, w0, w1)
        res, _ = quad(integrand, 0, 1, limit=100, epsabs=1e-12, epsrel=1e-12)
        return res

    # Precompute true integrals
    num_cosmo = len(cosmo_params)
    num_z = len(z_test)
    true_vals = np.zeros((num_cosmo, num_z))
    for i, (Om, w0, w1) in enumerate(cosmo_params):
        for j, z in enumerate(z_test):
            true_vals[i, j] = true_integral(z, Om, w0, w1)
    
    if time.time() > deadline:
        # Fallback to Gauss-Legendre
        n = 8
        nodes, weights = np.polynomial.legendre.leggauss(n)
        nodes = 0.5 * (nodes + 1.0)
        weights = 0.5 * weights
        weights = weights / weights.sum()
        return {"nodes": nodes, "weights": weights}

    # 2. Strategy: Minimax Linear Programming for Weights given Nodes
    # For fixed nodes u_i, the quadrature estimate for a specific (cosmo, z) is:
    # Q = sum_i w_i * f_i
    # We want to minimize max_{k} |Q_k - T_k| / T_k
    # Let e_k be the error. We want to minimize E such that:
    # -E * T_k <= Q_k - T_k <= E * T_k
    # Q_k - T_k = sum_i w_i * f_{k,i} - T_k
    # So:
    # sum_i w_i * f_{k,i} - E * T_k <= T_k
    # -sum_i w_i * f_{k,i} - E * T_k <= -T_k
    
    # Variables: w_1..w_8, E
    # Constraints:
    # 1. w_i >= 0
    # 2. sum w_i = 1
    # 3. The error bounds for all k
    
    def solve_weights_lp(nodes):
        """
        Solve for optimal weights for given nodes using LP.
        Returns weights or None if infeasible/timeout.
        """
        n_nodes = len(nodes)
        n_constraints = num_cosmo * num_z
        
        # Matrix A_ub for inequality constraints: A_ub @ x <= b_ub
        # x = [w_1, ..., w_8, E]
        # Constraints for each k:
        # 1) sum_i w_i f_{k,i} - E T_k <= T_k
        #    -> [f_{k,1}, ..., f_{k,8}, -T_k] @ x <= T_k
        # 2) -sum_i w_i f_{k,i} - E T_k <= -T_k
        #    -> [-f_{k,1}, ..., -f_{k,8}, -T_k] @ x <= -T_k
        
        A_ub = np.zeros((2 * n_constraints, n_nodes + 1))
        b_ub = np.zeros(2 * n_constraints)
        
        idx = 0
        for i_c in range(num_cosmo):
            Om, w0, w1 = cosmo_params[i_c]
            for i_z in range(num_z):
                z = z_test[i_z]
                T_k = true_vals[i_c, i_z]
                
                # Compute f_{k,i} = 1/E(z * u_i)
                z_u = z * nodes
                E_vals = E_func(z_u, Om, w0, w1)
                f_vals = 1.0 / E_vals
                
                # Constraint 1
                A_ub[idx, :n_nodes] = f_vals
                A_ub[idx, n_nodes] = -T_k
                b_ub[idx] = T_k
                idx += 1
                
                # Constraint 2
                A_ub[idx, :n_nodes] = -f_vals
                A_ub[idx, n_nodes] = -T_k
                b_ub[idx] = -T_k
                idx += 1
        
        # Objective: Minimize E -> c = [0, 0, ..., 0, 1]
        c = np.zeros(n_nodes + 1)
        c[-1] = 1.0
        
        # Bounds: w_i >= 0, E >= 0
        bounds = [(0, None)] * n_nodes + [(0, None)]
        
        # Equality constraint: sum w_i = 1
        A_eq = np.zeros((1, n_nodes + 1))
        A_eq[0, :n_nodes] = 1.0
        b_eq = np.array([1.0])
        
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
            if res.success:
                return res.x[:n_nodes]
            else:
                return None
        except Exception:
            return None

    # 3. Node Search Strategy
    # Instead of DE over 16 vars, we use a two-stage approach:
    # Stage 1: Generate a set of candidate node sets.
    # Stage 2: For each candidate, solve LP for weights.
    # Stage 3: Pick the candidate with the best worst-case error.
    # Stage 4: Local refinement of the best node set.
    
    def evaluate_node_set(nodes):
        """
        Given nodes, find optimal weights via LP and return worst-case relative error.
        """
        weights = solve_weights_lp(nodes)
        if weights is None:
            return np.inf, None
            
        # Check if weights are valid (positive, sum 1)
        if np.any(weights < -1e-12) or abs(np.sum(weights) - 1.0) > 1e-10:
             # Clip small negatives
             weights = np.maximum(weights, 0)
             weights = weights / np.sum(weights)
        
        # Calculate worst-case error
        max_err = 0.0
        for i_c in range(num_cosmo):
            Om, w0, w1 = cosmo_params[i_c]
            for i_z in range(num_z):
                z = z_test[i_z]
                T_k = true_vals[i_c, i_z]
                z_u = z * nodes
                E_vals = E_func(z_u, Om, w0, w1)
                f_vals = 1.0 / E_vals
                Q_k = np.dot(weights, f_vals)
                err = abs(Q_k - T_k) / T_k
                if err > max_err:
                    max_err = err
        return max_err, weights

    # Generate initial candidate node sets
    # 1. Gauss-Legendre (8 points)
    n_gl = 8
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(n_gl)
    gl_nodes = 0.5 * (gl_nodes + 1.0)
    
    # 2. Gauss-Jacobi with alpha, beta adjusted to weight endpoints more?
    # The integrand 1/E(z*u) is often smooth but can have varying curvature.
    # Let's try a few perturbations of GL nodes.
    
    candidates = [gl_nodes]
    
    # Add some structured perturbations
    # Shift nodes towards center or edges
    # Simple scaling: nodes' = (nodes - 0.5) * scale + 0.5
    for scale in [0.8, 0.9, 1.0, 1.1, 1.2]:
        new_nodes = (gl_nodes - 0.5) * scale + 0.5
        if np.all(new_nodes > 0) and np.all(new_nodes < 1) and np.all(np.diff(new_nodes) > 0):
            candidates.append(new_nodes)
            
    # Add random perturbations (deterministic seed for reproducibility)
    rng = np.random.RandomState(42)
    for _ in range(20):
        perturb = rng.uniform(-0.05, 0.05, 8)
        new_nodes = gl_nodes + perturb
        # Enforce ordering and bounds
        new_nodes = np.sort(new_nodes)
        if np.all(new_nodes > 0.01) and np.all(new_nodes < 0.99):
            candidates.append(new_nodes)

    best_err = np.inf
    best_nodes = gl_nodes
    best_weights = gl_weights
    
    # Evaluate candidates
    for cand_nodes in candidates:
        if time.time() > deadline - 10: # Leave time for refinement
            break
        err, wts = evaluate_node_set(cand_nodes)
        if err < best_err:
            best_err = err
            best_nodes = cand_nodes.copy()
            best_weights = wts.copy()

    # 4. Local Refinement of Nodes
    # Use Nelder-Mead to refine the nodes, using the LP-solved weights as the objective function.
    # Objective: f(nodes) = worst_case_error(nodes, optimal_weights(nodes))
    
    def objective_for_nodes(nodes):
        if time.time() > deadline:
            return np.inf
        # Ensure nodes are valid
        if np.any(nodes <= 0) or np.any(nodes >= 1) or not np.all(np.diff(nodes) > 0):
            return np.inf
        err, _ = evaluate_node_set(nodes)
        return err

    # Initial simplex
    x0 = best_nodes
    # Perturbations for simplex
    step = 0.01
    simplex = [x0]
    for i in range(8):
        perturbed = x0.copy()
        perturbed[i] += step
        if np.all(perturbed > 0) and np.all(perturbed < 1) and np.all(np.diff(perturbed) > 0):
            simplex.append(perturbed)
        else:
            perturbed = x0.copy()
            perturbed[i] -= step
            if np.all(perturbed > 0) and np.all(perturbed < 1) and np.all(np.diff(perturbed) > 0):
                simplex.append(perturbed)
            else:
                simplex.append(x0) # Fallback

    # Run Nelder-Mead
    try:
        from scipy.optimize import minimize
        res = minimize(objective_for_nodes, x0, method='Nelder-Mead', 
                       options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
        if res.fun < best_err:
            best_nodes = res.x
            # Re-solve LP for final weights
            _, best_weights = evaluate_node_set(best_nodes)
            best_err = res.fun
    except Exception:
        pass

    # Final Check
    if best_weights is None:
        # Fallback
        n = 8
        nodes, weights = np.polynomial.legendre.leggauss(n)
        nodes = 0.5 * (nodes + 1.0)
        weights = 0.5 * weights
        weights = weights / weights.sum()
        return {"nodes": nodes, "weights": weights}

    # Ensure strict positivity and normalization
    best_weights = np.maximum(best_weights, 1e-15)
    best_weights = best_weights / np.sum(best_weights)
    
    # Ensure nodes are strictly increasing and in (0,1)
    if not (np.all(best_nodes > 0) and np.all(best_nodes < 1) and np.all(np.diff(best_nodes) > 1e-12)):
        # Fallback
        n = 8
        nodes, weights = np.polynomial.legendre.leggauss(n)
        nodes = 0.5 * (nodes + 1.0)
        weights = 0.5 * weights
        weights = weights / weights.sum()
        return {"nodes": nodes, "weights": weights}

    return {"nodes": best_nodes, "weights": best_weights}

# EVOLVE-BLOCK-END
