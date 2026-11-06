"""
Strategy: Decoupled Deterministic Node Search + Constrained Weight Optimization.

The task requires finding an 8-node quadrature rule that minimizes the worst-case
relative error for the luminosity distance integral across various CPL cosmologies.

Current approach (score 76.9) uses L-BFGS-B on 16 parameters, which is slow and
stuck in local minima.

New approach:
1. Recognize that for a fixed set of nodes $t_i$, the approximation error is linear
   in the weights $w_i$.
2. We can minimize the Mean Squared Error (MSE) of the approximation over a large
   set of training cosmologies and redshifts.
   Let $A_{k,i} = 1/E(z_k t_i; w0_k, wa_k)$ and $b_k = \int_0^1 \frac{dt}{E(z_k t; w0_k, wa_k)}$.
   We want to find $w$ such that $\| A w - b \|^2$ is minimized, subject to $w_i > 0$ and $\sum w_i = 1$.
3. This is a convex quadratic programming problem. We can solve it efficiently using
   `scipy.optimize.minimize` with SLSQP or, better yet, since it's a least squares
   problem with linear constraints, we can use `scipy.optimize.lsq_linear` or a
   custom projected gradient method. However, `lsq_linear` doesn't handle the
   equality constraint $\sum w_i = 1$ directly in all versions/configurations easily
   without transformation.
   
   Actually, a simpler and very robust approach for weights given nodes:
   Use `scipy.optimize.minimize` with method='SLSQP' on the weights only (8 vars).
   This is much faster than optimizing 16 vars because the Hessian is positive definite
   (or close to it) and the landscape is convex.
   
   Even better: Since the weights must sum to 1 and be positive, we can parameterize
   the weights using softmax or just enforce constraints.
   
   Let's refine the node search:
   Instead of random nodes, we can use a structured grid or a few families of nodes:
   - Gauss-Legendre
   - Gauss-Jacobi (with various alpha, beta)
   - Chebyshev nodes
   - Equidistant nodes
   - And a coarse random search.
   
   For each candidate node set:
   1. Solve for optimal weights (minimizing MSE on training set).
   2. Evaluate worst-case error on a separate validation set (or hold-out training set).
   3. Keep the best node/weight pair.
   
   To maximize coverage, we will:
   1. Generate a large, diverse training set of (z, w0, wa).
   2. Pre-compute the integrals $b_k$ and matrix $A$ for a few representative node sets?
      No, $A$ depends on nodes.
      
   Time budget: 90s.
   Let's use a two-stage approach:
   Stage 1: Coarse search over node configurations.
   - Try GL nodes.
   - Try GL nodes with small perturbations.
   - Try random nodes (seeded).
   For each, solve for weights using a fast method.
   
   How to solve for weights fast?
   The problem is: min_w || A w - b ||^2 s.t. w >= 0, sum(w) = 1.
   This is a standard quadratic program.
   We can use `scipy.optimize.minimize` with method='SLSQP' or 'trust-constr'.
   Given only 8 variables, SLSQP should be very fast (milliseconds per node set).
   
   Let's implement this.
"""

import time
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
from scipy.special import roots_legendre, roots_jacobi, roots_chebyshev

_START_TIME = time.time()
_DEADLINE = _START_TIME + 85.0  # Safety margin

def _E_of_z(z, w0, wa, Omega_m=0.3):
    """Compute E(z) = H(z)/H0 for CPL model."""
    Omega_de = 1.0 - Omega_m
    # Avoid numerical issues for very large z
    # (1+z) can be large.
    term_m = Omega_m * (1.0 + z)**3
    # Exponent: 3 * (1 + w0 + wa * z)
    exp_term = 3.0 * (1.0 + w0 + wa * z)
    term_de = Omega_de * (1.0 + z)**exp_term
    return np.sqrt(term_m + term_de)

def _true_integral(z, w0, wa, Omega_m=0.3):
    """Compute the true integral \int_0^1 dt / E(z*t)."""
    if z == 0:
        return 1.0
    def integrand(t):
        # t in [0,1]
        # z*t is the effective redshift
        return 1.0 / _E_of_z(z * t, w0, wa, Omega_m)
    # Use high precision quad
    val, _ = quad(integrand, 0, 1, limit=200, epsabs=1e-12, epsrel=1e-12)
    return val

def _generate_training_set(num_z=10, num_w0=5, num_wa=5):
    """Generate a diverse set of (z, w0, wa) for training."""
    # z: log-spaced from 0.1 to 5.0
    z_vals = np.logspace(-1, np.log10(5.0), num_z)
    # w0: typical range -1.5 to -0.5
    w0_vals = np.linspace(-1.5, -0.5, num_w0)
    # wa: typical range -0.5 to 0.5
    wa_vals = np.linspace(-0.5, 0.5, num_wa)
    
    samples = []
    for z in z_vals:
        for w0 in w0_vals:
            for wa in wa_vals:
                samples.append((z, w0, wa))
    return samples

def _solve_weights(nodes, training_data, Omega_m=0.3):
    """
    Solve for weights given fixed nodes to minimize MSE on training data.
    min_w || A w - b ||^2 s.t. w >= 0, sum(w) = 1
    """
    n = len(nodes)
    num_samples = len(training_data)
    
    # Pre-compute A and b
    A = np.zeros((num_samples, n))
    b = np.zeros(num_samples)
    
    for k, (z, w0, wa) in enumerate(training_data):
        b[k] = _true_integral(z, w0, wa, Omega_m)
        for i in range(n):
            A[k, i] = 1.0 / _E_of_z(z * nodes[i], w0, wa, Omega_m)
    
    # Objective: || A w - b ||^2
    def objective(w):
        res = A.dot(w) - b
        return np.dot(res, res)
    
    def gradient(w):
        res = A.dot(w) - b
        return 2.0 * A.T.dot(res)
    
    # Constraints
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    bounds = [(1e-8, 1.0)] * n
    
    # Initial guess: uniform weights
    w0 = np.ones(n) / n
    
    try:
        res = minimize(objective, w0, method='SLSQP', jac=gradient, 
                       bounds=bounds, constraints=constraints,
                       options={'maxiter': 100, 'ftol': 1e-12})
        if res.success:
            w_opt = res.x
            # Ensure strict positivity and normalization
            w_opt = np.maximum(w_opt, 1e-10)
            w_opt = w_opt / np.sum(w_opt)
            return w_opt
        else:
            # Fallback to uniform if optimization fails
            return w0
    except Exception:
        return w0

def _evaluate_worst_case_error(nodes, weights, test_data, Omega_m=0.3):
    """Calculate the worst-case relative error percentage."""
    max_err = 0.0
    for z, w0, wa in test_data:
        true_val = _true_integral(z, w0, wa, Omega_m)
        approx_val = np.dot(weights, 1.0 / _E_of_z(z * nodes, w0, wa, Omega_m))
        if true_val > 1e-10:
            rel_err = abs(approx_val - true_val) / true_val
            if rel_err > max_err:
                max_err = rel_err
    return max_err * 100.0

def construct():
    if time.time() > _DEADLINE - 5:
        # Fallback: Gauss-Legendre
        n = 8
        nodes, weights = roots_legendre(n)
        nodes = (nodes + 1) / 2
        weights = weights / 2
        return {"nodes": nodes, "weights": weights}

    # 1. Generate data
    # Use a subset for training to speed up weight solving,
    # and a separate set for validation to avoid overfitting.
    # Since we only have 90s, we need to be efficient.
    # Let's use ~100 points for training.
    train_data = _generate_training_set(num_z=8, num_w0=4, num_wa=3) # 96 points
    # Validation set: different points
    val_z = np.logspace(-0.5, np.log10(3.0), 5)
    val_w0 = np.linspace(-1.2, -0.8, 3)
    val_wa = np.linspace(-0.3, 0.3, 3)
    val_data = [(z, w0, wa) for z in val_z for w0 in val_w0 for wa in val_wa]
    
    best_score = float('inf')
    best_nodes = None
    best_weights = None
    
    # 2. Define candidate node sets
    candidate_nodes = []
    
    # A. Gauss-Legendre
    n = 8
    gl_nodes, gl_weights = roots_legendre(n)
    gl_nodes = (gl_nodes + 1) / 2
    candidate_nodes.append(gl_nodes)
    
    # B. Gauss-Jacobi (alpha, beta) variations
    # Jacobi polynomials P_n^(alpha, beta)
    # alpha, beta > -1
    for alpha in [0.5, 1.0, 2.0]:
        for beta in [0.5, 1.0, 2.0]:
            try:
                j_nodes, j_weights = roots_jacobi(n, alpha, beta)
                j_nodes = (j_nodes + 1) / 2
                candidate_nodes.append(j_nodes)
            except Exception:
                pass
                
    # C. Chebyshev nodes
    try:
        c_nodes, _ = roots_chebyshev(n, 1)
        c_nodes = (c_nodes + 1) / 2
        candidate_nodes.append(c_nodes)
    except Exception:
        pass

    # D. Random perturbations of GL nodes
    rng = np.random.default_rng(42)
    for _ in range(20):
        perturbed = gl_nodes.copy()
        # Add small noise, ensure they stay in [0,1] and sorted
        noise = rng.uniform(-0.05, 0.05, size=n)
        perturbed += noise
        perturbed = np.clip(perturbed, 1e-6, 1 - 1e-6)
        perturbed = np.sort(perturbed)
        # Ensure strict increase (clip handles mostly, but check diffs)
        if np.all(np.diff(perturbed) > 1e-6):
            candidate_nodes.append(perturbed)
            
    # E. Equidistant
    eq_nodes = np.linspace(0.05, 0.95, n)
    candidate_nodes.append(eq_nodes)

    # 3. Evaluate each candidate
    # We need to be careful with time.
    # Solving weights takes ~10-50ms per node set (rough estimate).
    # Evaluating error takes ~10-20ms per node set.
    # Total per candidate: ~50-100ms.
    # With ~30 candidates, this is ~3s. Very safe.
    
    for nodes in candidate_nodes:
        if time.time() > _DEADLINE - 10:
            break
            
        # Ensure nodes are strictly increasing in [0,1]
        if not np.all(np.diff(nodes) > 1e-6):
            continue
        if nodes[0] < 1e-6 or nodes[-1] > 1 - 1e-6:
            continue
            
        # Solve for optimal weights
        weights = _solve_weights(nodes, train_data)
        
        # Evaluate worst-case error on validation set
        # Note: Using validation set is better for generalization estimate
        # But the task says "Minimize worst relative error against ... disjoint public validation cosmologies".
        # The evaluator uses its own set. We should try to minimize error on a set
        # that is representative of the evaluator's set.
        # Since we don't know the exact set, minimizing on a diverse training set
        # and checking on a diverse validation set is a good proxy.
        
        err_val = _evaluate_worst_case_error(nodes, weights, val_data)
        err_train = _evaluate_worst_case_error(nodes, weights, train_data)
        
        # Use a combination or just validation?
        # Usually, minimizing training error can lead to overfitting the specific
        # quadrature nodes to the training integrals.
        # However, the "universal" rule implies we want good performance everywhere.
        # Let's use the validation error as the primary metric for selection,
        # but if validation is empty or too small, use training.
        
        if err_val < best_score:
            best_score = err_val
            best_nodes = nodes
            best_weights = weights
            
    # 4. Refine the best candidate with a local search on nodes?
    # If we have time, we can do a few steps of local optimization on the nodes
    # using the previously found weights as a starting point, or re-solving weights.
    
    if best_nodes is not None and time.time() < _DEADLINE - 20:
        # Local refinement of nodes using Nelder-Mead or similar
        # Objective: Worst-case error on val_data
        # Variables: nodes (8)
        
        def node_objective(nodes_flat):
            nodes = nodes_flat
            if not np.all(np.diff(nodes) > 1e-6):
                return 1e10
            if nodes[0] < 1e-6 or nodes[-1] > 1 - 1e-6:
                return 1e10
            w = _solve_weights(nodes, train_data)
            return _evaluate_worst_case_error(nodes, w, val_data)
        
        # Start from best_nodes
        x0 = best_nodes
        try:
            res = minimize(node_objective, x0, method='Nelder-Mead',
                           options={'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-6})
            if res.fun < best_score:
                best_nodes = res.x
                best_weights = _solve_weights(best_nodes, train_data)
                best_score = res.fun
        except Exception:
            pass

    # 5. Final sanity check and formatting
    if best_nodes is None:
        # Fallback
        best_nodes, best_weights = roots_legendre(8)
        best_nodes = (best_nodes + 1) / 2
        best_weights = best_weights / 2

    # Ensure weights sum to 1
    best_weights = best_weights / np.sum(best_weights)
    
    # Ensure nodes are strictly increasing
    sort_idx = np.argsort(best_nodes)
    best_nodes = best_nodes[sort_idx]
    best_weights = best_weights[sort_idx]
    
    # Final check for strict increase
    if np.any(np.diff(best_nodes) <= 0):
        # Force strict increase by adding tiny epsilon if needed
        # This should rarely happen with the checks above
        best_nodes = np.linspace(0.01, 0.99, 8)
        best_weights = np.ones(8) / 8

    return {"nodes": best_nodes, "weights": best_weights}

# EVOLVE-BLOCK-END
