"""Basis-function projection approach: fit integrand shape to physical basis, derive optimal quadrature."""
import numpy as np
import time
import scipy.optimize as so

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    Omega_m = 0.3
    Omega_L = 0.7

    # High-accuracy reference using 128-point Gauss-Legendre
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(128)
    gl_nodes = (gl_nodes + 1.0) / 2.0
    gl_weights = gl_weights / 2.0

    def _reference_integral(z, w0, wa):
        """Compute integral of 1/E(a) da from a=1 to a=1/(1+z) using 128-point GL."""
        a_vals = 1.0 / (1.0 + z * gl_nodes)
        a_vals = np.clip(a_vals, 1e-15, 1.0)
        log_term = 3.0 * (1.0 + w0 + wa) * np.log(a_vals) - 3.0 * wa * (a_vals - 1.0)
        log_term = np.clip(log_term, -700, 700)
        term_L = Omega_L * np.exp(log_term)
        term_m = Omega_m * (a_vals ** -3)
        E2 = np.maximum(term_m + term_L, 1e-300)
        E = np.sqrt(E2)
        inv_E = 1.0 / E
        return np.dot(gl_weights, inv_E)

    # Diverse test cases for evaluation
    z_vals = np.logspace(-1, np.log10(8.0), 10)
    w0_vals = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0]
    wa_vals = [-1.5, -0.5, 0.0, 0.5, 1.0, 1.5]

    test_cases = []
    for z in z_vals:
        for w0 in w0_vals:
            for wa in wa_vals:
                test_cases.append((z, w0, wa))

    # Precompute reference values
    ref_vals = []
    for z, w0, wa in test_cases:
        if time.time() > deadline:
            ref_vals.append(1.0)
            continue
        try:
            val = _reference_integral(z, w0, wa)
            ref_vals.append(val)
        except Exception:
            ref_vals.append(1.0)

    def _quadrature_eval(nodes, weights, z, w0, wa):
        """Evaluate the 8-point quadrature."""
        a_vals = 1.0 / (1.0 + z * nodes)
        a_vals = np.clip(a_vals, 1e-12, 1.0)
        log_term = 3.0 * (1.0 + w0 + wa) * np.log(a_vals) - 3.0 * wa * (a_vals - 1.0)
        log_term = np.clip(log_term, -700, 700)
        term_L = Omega_L * np.exp(log_term)
        term_m = Omega_m * (a_vals ** -3)
        E2 = np.maximum(term_m + term_L, 1e-300)
        E = np.sqrt(E2)
        inv_E = 1.0 / E
        return np.dot(weights, inv_E)

    def _worst_rel_error(nodes, weights):
        """Compute worst-case relative error in percent."""
        max_err = 0.0
        for i, (z, w0, wa) in enumerate(test_cases):
            ref = ref_vals[i]
            if ref <= 0:
                continue
            qval = _quadrature_eval(nodes, weights, z, w0, wa)
            rel_err = abs(qval - ref) / ref * 100.0
            if rel_err > max_err:
                max_err = rel_err
        return max_err

    # ===== BASIS-FUNCTION PROJECTION APPROACH =====
    # The integrand 1/E(a) has a known structure in CPL models:
    # E(a)^2 = Omega_m * a^-3 + Omega_L * a^(-3*(1+w0+wa)) * exp(3*wa*(a-1))
    # 
    # We represent the integrand as a linear combination of basis functions
    # that capture the dominant physical behavior, then find an 8-node rule
    # that exactly integrates this fitted form.
    
    # Basis functions in terms of u = 1-a (redshift-like variable):
    # f(u) = sum_j c_j * phi_j(u)
    # where phi_j are chosen to capture the integrand shape across cosmologies
    
    def _compute_integrand_values(z, w0, wa, a_grid):
        """Compute 1/E(a) on a grid of a values."""
        a_vals = np.clip(a_grid, 1e-15, 1.0)
        log_term = 3.0 * (1.0 + w0 + wa) * np.log(a_vals) - 3.0 * wa * (a_vals - 1.0)
        log_term = np.clip(log_term, -700, 700)
        term_L = Omega_L * np.exp(log_term)
        term_m = Omega_m * (a_vals ** -3)
        E2 = np.maximum(term_m + term_L, 1e-300)
        E = np.sqrt(E2)
        return 1.0 / E

    # Create a fine grid in a-space for fitting
    a_grid = np.linspace(0.05, 1.0, 200)
    
    # Build basis functions: powers of (1-a) and exponential terms
    # u = 1 - a
    u_grid = 1.0 - a_grid
    
    # Basis set: u^0, u^1, u^2, u^3, u^4, exp(-k*u) for several k
    # This captures both polynomial and exponential behavior
    k_vals = [0.5, 1.0, 2.0, 4.0]
    
    def _build_basis(u):
        """Build basis function matrix for given u values."""
        n_basis = 5 + len(k_vals)  # 5 power terms + 4 exponential terms
        B = np.zeros((len(u), n_basis))
        # Power terms: u^0, u^1, u^2, u^3, u^4
        for p in range(5):
            B[:, p] = u ** p
        # Exponential terms: exp(-k*u)
        for i, k in enumerate(k_vals):
            B[:, 5 + i] = np.exp(-k * u)
        return B
    
    B_grid = _build_basis(u_grid)
    
    # Fit coefficients for each test case and collect them
    # We want a single set of basis that works well across all cosmologies
    # Strategy: fit each cosmology, then find a "universal" basis representation
    
    # For the quadrature design, we need to find nodes x_i and weights w_i
    # such that sum_i w_i * f(x_i) ≈ integral_0^1 f(u) du
    # for f in our basis space.
    
    # The optimal rule exactly integrates each basis function:
    # sum_i w_i * phi_j(x_i) = integral_0^1 phi_j(u) du  for j=1..n_basis
    # with n_basis = 8 (we have 8 nodes and 8 weights)
    
    # Select 8 basis functions from our set
    # Use: u^0, u^1, u^2, u^3, u^4, u^5, exp(-1*u), exp(-2*u)
    k_sel = [1.0, 2.0]
    
    def _basis_phi(j, u):
        """Evaluate basis function j at u."""
        if j < 6:
            return u ** j
        elif j == 6:
            return np.exp(-1.0 * u)
        else:
            return np.exp(-2.0 * u)
    
    # Compute exact integrals of each basis function over [0,1]
    def _integral_basis(j):
        """Compute integral of basis function j over [0,1]."""
        if j < 6:
            return 1.0 / (j + 1.0)
        elif j == 6:
            return 1.0 - np.exp(-1.0)
        else:
            return 1.0 - np.exp(-2.0)
    
    target_integrals = np.array([_integral_basis(j) for j in range(8)])
    
    # Now find 8 nodes x_1 < x_2 < ... < x_8 in (0,1) and 8 positive weights w_i
    # such that sum_i w_i * phi_j(x_i) = target_integrals[j] for j=0..7
    # and sum_i w_i = 1 (which is target_integrals[0])
    
    # This is a system of 8 equations in 16 unknowns (8 nodes + 8 weights)
    # We need additional constraints. We'll use a parameterized approach:
    # - Nodes are determined by a smooth distribution parameterized by 7 free params
    # - Weights are solved from the linear system
    
    def _solve_weights(nodes):
        """Given 8 nodes, solve for weights that exactly integrate the 8 basis functions."""
        A = np.zeros((8, 8))
        for j in range(8):
            for i in range(8):
                A[j, i] = _basis_phi(j, nodes[i])
        try:
            w = np.linalg.solve(A, target_integrals)
            return w
        except np.linalg.LinAlgError:
            return None
    
    # Parameterize nodes using log-spaced or smooth distribution
    # x_i = 1 - exp(-c * t_i) where t_i are increasing
    # Or use a beta-like distribution
    
    def _param_to_nodes(params):
        """Convert 7 parameters to 8 strictly increasing nodes in (0,1)."""
        # Use a smooth monotone mapping
        # params define the shape of the node distribution
        t = np.linspace(0, 1, 8)
        # Apply a smooth deformation
        # x_i = t_i + sum_j a_j * sin(pi * t_i) * cos(j * pi * t_i)
        # Simpler: use a power-law spacing
        # Let gaps be proportional to exp(params[i])
        gaps = np.exp(params)
        gaps = gaps / gaps.sum()
        nodes = np.cumsum(gaps)
        # Map to (0, 1) with some margin
        nodes = 0.01 + 0.98 * nodes
        return nodes
    
    def _nodes_to_param(nodes):
        """Convert 8 nodes to 7 parameters (log of gaps)."""
        gaps = np.diff(nodes)
        gaps = np.maximum(gaps, 1e-10)
        return np.log(gaps)
    
    # Objective: minimize worst-case error, but first ensure the rule
    # is well-conditioned (positive weights)
    
    def _objective_nodes(params):
        """Objective for node optimization."""
        nodes = _param_to_nodes(params)
        w = _solve_weights(nodes)
        if w is None:
            return 1e6
        # Check if weights are positive
        if np.any(w <= 0):
            # Penalize negative weights
            return 1e6 + np.sum(np.maximum(0, -w))
        # Normalize weights to sum to 1
        w = w / np.sum(w)
        return _worst_rel_error(nodes, w)
    
    # Start from Gauss-Legendre nodes
    gl8_nodes, gl8_weights = np.polynomial.legendre.leggauss(8)
    gl8_nodes = (gl8_nodes + 1.0) / 2.0
    start_params = _nodes_to_param(gl8_nodes)
    
    best_worst_err = float('inf')
    best_nodes = gl8_nodes.copy()
    best_weights = gl8_weights.copy()
    
    # Try to solve the linear system for GL nodes
    w_gl = _solve_weights(gl8_nodes)
    if w_gl is not None and np.all(w_gl > 0):
        w_gl = w_gl / np.sum(w_gl)
        err = _worst_rel_error(gl8_nodes, w_gl)
        if err < best_worst_err:
            best_worst_err = err
            best_weights = w_gl.copy()
    
    # Optimize nodes using differential evolution (global search)
    if time.time() < deadline - 20.0:
        try:
            def _de_objective(params):
                nodes = _param_to_nodes(params)
                w = _solve_weights(nodes)
                if w is None:
                    return 1e6
                if np.any(w <= 0):
                    return 1e6 + np.sum(np.maximum(0, -w))
                w = w / np.sum(w)
                return _worst_rel_error(nodes, w)
            
            bounds = [(np.log(0.1), np.log(10.0))] * 7
            de_result = so.differential_evolution(
                _de_objective, bounds,
                seed=42, maxiter=100, tol=1e-8,
                popsize=15, mutation=(0.3, 1.0), recombination=0.7
            )
            if de_result.fun < best_worst_err:
                best_params = de_result.x
                best_nodes = _param_to_nodes(best_params)
                w_de = _solve_weights(best_nodes)
                if w_de is not None and np.all(w_de > 0):
                    best_weights = w_de / np.sum(w_de)
                    best_worst_err = de_result.fun
        except Exception:
            pass
    
    # Refine with Nelder-Mead
    if time.time() < deadline - 10.0:
        try:
            def _nm_objective(params):
                nodes = _param_to_nodes(params)
                w = _solve_weights(nodes)
                if w is None:
                    return 1e6
                if np.any(w <= 0):
                    return 1e6 + np.sum(np.maximum(0, -w))
                w = w / np.sum(w)
                return _worst_rel_error(nodes, w)
            
            start_p = _nodes_to_param(best_nodes)
            nm_result = so.minimize(
                _nm_objective, start_p,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            if nm_result.fun < best_worst_err:
                best_nodes = _param_to_nodes(nm_result.x)
                w_nm = _solve_weights(best_nodes)
                if w_nm is not None and np.all(w_nm > 0):
                    best_weights = w_nm / np.sum(w_nm)
                    best_worst_err = nm_result.fun
        except Exception:
            pass
    
    # Also try direct joint optimization of nodes and weights
    # with a penalty for not exactly integrating the basis
    if time.time() < deadline - 5.0:
        def _joint_param_to_nw(params):
            """Convert 16 params to nodes and weights."""
            x = params[:8]
            logw = params[8:16]
            
            # Nodes via monotone mapping
            nodes = np.zeros(8)
            nodes[0] = 1.0 / (1.0 + np.exp(-x[0]))
            for i in range(1, 8):
                gap = np.exp(x[i] - x[i-1])
                nodes[i] = nodes[i-1] + gap
            nodes = np.clip(nodes, 1e-12, 1.0 - 1e-12)
            
            w = np.exp(logw)
            w = w / np.sum(w)
            return nodes, w
        
        def _joint_objective(params):
            nodes, weights = _joint_param_to_nw(params)
            # Penalty for not integrating basis functions
            basis_err = 0.0
            for j in range(8):
                quad_val = sum(weights[i] * _basis_phi(j, nodes[i]) for i in range(8))
                basis_err += (quad_val - target_integrals[j]) ** 2
            # Plus worst-case error
            wc_err = _worst_rel_error(nodes, weights)
            return wc_err + 0.1 * basis_err
        
        # Start from current best
        x_start = np.zeros(8)
        x_start[0] = np.log(best_nodes[0] / (1.0 - best_nodes[0]))
        for i in range(1, 8):
            x_start[i] = x_start[i-1] + np.log(max(best_nodes[i] - best_nodes[i-1], 1e-10))
        logw_start = np.log(np.maximum(best_weights, 1e-12))
        start_joint = np.concatenate([x_start, logw_start])
        
        try:
            res = so.minimize(
                _joint_objective, start_joint,
                method='Nelder-Mead',
                options={'maxiter': 1500, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            if res.fun < best_worst_err:
                n, w = _joint_param_to_nw(res.x)
                if np.all(np.diff(n) > 0) and np.all(w > 0):
                    w = w / np.sum(w)
                    err = _worst_rel_error(n, w)
                    if err < best_worst_err:
                        best_nodes = n
                        best_weights = w
                        best_worst_err = err
        except Exception:
            pass
    
    # Final cleanup: ensure nodes strictly increasing, weights positive and sum to 1
    best_nodes = np.clip(best_nodes, 1e-12, 1.0 - 1e-12)
    for i in range(1, 8):
        if best_nodes[i] <= best_nodes[i-1]:
            best_nodes[i] = best_nodes[i-1] + 1e-12
    best_weights = np.clip(best_weights, 1e-12, None)
    best_weights = best_weights / np.sum(best_weights)
    
    return {"nodes": best_nodes, "weights": best_weights}
