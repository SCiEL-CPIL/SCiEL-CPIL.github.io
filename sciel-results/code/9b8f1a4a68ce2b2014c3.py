"""
Strategy: Multi-Resolution Grid Search + Direct Minimax Weight Optimization.

Key Structural Differences:
1. Node Search: Instead of random perturbations or a static coarse grid, we use a 
   multi-resolution grid search. We start with a coarse grid of node configurations 
   (e.g., Gauss-Legendre, Chebyshev, uniform, and shifted variants) and then 
   perform a local refinement using Nelder-Mead on the nodes, where the objective 
   is the worst-case relative error over a dense validation set.
2. Weight Solver: For a fixed set of nodes, the weights that minimize the worst-case 
   relative error are found by solving a Linear Program (LP). This is exact for the 
   fixed nodes and avoids the convergence issues of iterative methods like L-BFGS-B.
3. Validation Set: We use a dense, stratified set of (z, w0, wa) configurations to 
   ensure the rule is robust across the entire parameter space. The set is chosen to 
   cover the full range of z (0 to 10) and typical CPL parameters.
4. Efficiency: The LP is solved using scipy.optimize.linprog with the HiGHS solver, 
   which is fast and reliable. The node search is guided by the LP solution, ensuring 
   that we always have the best possible weights for the current nodes.
"""

import time
import numpy as np
from scipy.integrate import quad
from scipy.optimize import linprog, minimize
from scipy.special import roots_legendre, roots_jacobi

_START_TIME = time.time()
_DEADLINE = _START_TIME + 85.0  # Safety margin before 90s kill

def _E_of_z(z, w0, wa, Omega_m=0.3):
    """Compute E(z) = H(z)/H0 for CPL dark energy model."""
    Omega_de = 1.0 - Omega_m
    one_plus_z = 1.0 + z
    term_m = Omega_m * one_plus_z**3
    exponent = 3.0 * (1.0 + w0 + wa * z)
    term_de = Omega_de * one_plus_z**exponent
    return np.sqrt(term_m + term_de)

def _true_integral(z, w0, wa, Omega_m=0.3):
    """High-accuracy reference integral."""
    if z < 1e-10:
        return 1.0
    def integrand(t):
        return 1.0 / _E_of_z(z * t, w0, wa, Omega_m)
    val, err = quad(integrand, 0, 1, limit=200, epsabs=1e-12, epsrel=1e-12)
    return val

def _build_dense_test_set():
    """
    Build a dense, stratified set of (z, w0, wa) configurations.
    This set defines the 'worst-case' for the robust optimization.
    """
    # z range: logarithmic spacing for better coverage of small z
    z_vals = np.geomspace(0.01, 10.0, 30)
    # w0, wa range: cover typical CPL space
    w0_vals = np.linspace(-1.5, 0.5, 10)
    wa_vals = np.linspace(-1.0, 1.0, 10)
    
    test_set = []
    for z in z_vals:
        for w0 in w0_vals:
            for wa in wa_vals:
                test_set.append((z, w0, wa))
    
    return test_set

def _solve_minimax_weights(nodes, test_set):
    """
    Solve for weights that minimize the worst-case relative error using LP.
    
    Problem:
    min epsilon
    s.t.  -epsilon * ref_i <= sum_j A[i,j] w_j - ref_i <= epsilon * ref_i  for all i
          sum_j w_j = 1
          w_j >= 0
    
    This is a linear program in variables w and epsilon.
    """
    N = len(test_set)
    n = 8
    
    # Pre-compute A and refs for these nodes
    A = np.zeros((N, n))
    refs = np.zeros(N)
    
    for idx, (z, w0, wa) in enumerate(test_set):
        refs[idx] = _true_integral(z, w0, wa)
        if z < 1e-10:
            A[idx, :] = 1.0
        else:
            zt = z * nodes
            E = _E_of_z(zt, w0, wa)
            A[idx, :] = 1.0 / E
    
    # LP Formulation:
    # Variables: x = [w_0, ..., w_7, epsilon]
    # c = [0, ..., 0, 1]
    
    c = np.zeros(n + 1)
    c[-1] = 1.0
    
    # Equality constraints: sum w_j = 1
    A_eq = np.zeros((1, n + 1))
    A_eq[0, :n] = 1.0
    b_eq = np.array([1.0])
    
    # Inequality constraints:
    # For each i:
    # sum_j A[i,j] w_j - ref_i * epsilon <= ref_i
    # -sum_j A[i,j] w_j + ref_i * epsilon <= ref_i
    
    A_ub = np.zeros((2 * N, n + 1))
    b_ub = np.zeros(2 * N)
    
    for i in range(N):
        # Constraint 1: sum A w - ref_i * eps <= ref_i
        A_ub[2*i, :n] = A[i, :]
        A_ub[2*i, -1] = -refs[i]
        b_ub[2*i] = refs[i]
        
        # Constraint 2: -sum A w + ref_i * eps <= ref_i
        A_ub[2*i+1, :n] = -A[i, :]
        A_ub[2*i+1, -1] = refs[i]
        b_ub[2*i+1] = refs[i]
    
    # Bounds: w_j >= 0, epsilon >= 0
    bounds = [(0, None)] * n + [(0, None)]
    
    # Solve LP
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    if not res.success:
        return None, float('inf')
    
    weights = res.x[:n]
    epsilon = res.x[-1]
    
    # Check if weights are valid (positive, sum to 1)
    if np.any(weights < -1e-10):
        return None, float('inf')
        
    # Clip small negatives due to numerical noise
    weights = np.maximum(weights, 0.0)
    weights = weights / np.sum(weights)
    
    # Calculate actual max relative error with these weights
    approx = A.dot(weights)
    rel_errs = np.abs(approx - refs) / np.maximum(refs, 1e-15)
    max_err = np.max(rel_errs)
    
    return weights, max_err

def _generate_node_candidates():
    """Generate diverse node configurations."""
    candidates = []
    
    # 1. Gauss-Legendre
    nodes, _ = roots_legendre(8)
    candidates.append((nodes + 1.0) / 2.0)
    
    # 2. Chebyshev
    k = np.arange(8)
    nodes = np.cos((2 * k + 1) * np.pi / 16.0)
    candidates.append((nodes + 1.0) / 2.0)
    
    # 3. Gauss-Jacobi
    for alpha, beta in [(0.5, 0.5), (1.0, 0.0), (0.0, 1.0), (2.0, 0.0), (0.0, 2.0)]:
        try:
            nodes, _ = roots_jacobi(8, alpha, beta)
            nodes = (nodes + 1.0) / 2.0
            if np.all(np.diff(nodes) > 1e-10):
                candidates.append(nodes)
        except:
            pass
            
    # 4. Uniform
    candidates.append(np.linspace(1/17, 16/17, 8))
    
    # 5. Log-spaced (concentrated near 0)
    t = np.linspace(0, 1, 8)
    for base in [2.0, 3.0, 5.0]:
        nodes = 1.0 - (1.0 - t) ** base
        nodes = np.clip(nodes, 1e-4, 1 - 1e-4)
        if np.all(np.diff(nodes) > 1e-6):
            candidates.append(nodes)
            
    # 6. Power-law (concentrated near 1)
    for base in [0.5, 0.3]:
        nodes = t ** (1.0 / base)
        nodes = np.clip(nodes, 1e-4, 1 - 1e-4)
        if np.all(np.diff(nodes) > 1e-6):
            candidates.append(nodes)
            
    return candidates

def _objective(nodes, test_set):
    """Objective function for Nelder-Mead: worst-case relative error."""
    weights, err = _solve_minimax_weights(nodes, test_set)
    if weights is None:
        return 1e10
    return err

def construct():
    if time.time() > _DEADLINE - 2:
        nodes, weights = roots_legendre(8)
        return {"nodes": (nodes + 1) / 2, "weights": weights / 2}
    
    # Build dense test set
    test_set = _build_dense_test_set()
    
    # Coarse search over node candidates
    candidates = _generate_node_candidates()
    
    best_nodes = None
    best_weights = None
    best_err = float('inf')
    
    for cand in candidates:
        if time.time() > _DEADLINE - 10:
            break
            
        w, err = _solve_minimax_weights(cand, test_set)
        if w is not None and err < best_err:
            best_err = err
            best_nodes = cand.copy()
            best_weights = w.copy()
            
    if best_nodes is None:
        nodes, weights = roots_legendre(8)
        return {"nodes": (nodes + 1) / 2, "weights": weights / 2}
        
    # Refine best node configuration using Nelder-Mead
    # We optimize the nodes, and for each node configuration, we solve the LP for weights.
    
    n = 8
    current_nodes = best_nodes.copy()
    current_weights = best_weights.copy()
    current_err = best_err
    
    # Define bounds for nodes: strictly increasing in [0, 1]
    # We use a transformation to ensure this: nodes = sigmoid(x)
    # But Nelder-Mead doesn't handle constraints well, so we use a penalty or clipping.
    
    def _transform_nodes(x):
        # x are the raw parameters, we map them to [0, 1] and sort
        # We use a simple clipping and sorting strategy
        nodes = np.clip(x, 1e-4, 1 - 1e-4)
        nodes = np.sort(nodes)
        # Ensure strict increase by adding small epsilon if needed
        for i in range(1, n):
            if nodes[i] - nodes[i-1] < 1e-6:
                nodes[i] = nodes[i-1] + 1e-6
        # Re-clip to [0, 1]
        nodes = np.clip(nodes, 1e-4, 1 - 1e-4)
        return nodes
    
    # Initial point for Nelder-Mead
    x0 = current_nodes.copy()
    
    # Nelder-Mead options
    options = {
        'maxiter': 500,
        'xatol': 1e-8,
        'fatol': 1e-10,
        'adaptive': True
    }
    
    # Objective function for Nelder-Mead
    def _nm_objective(x):
        nodes = _transform_nodes(x)
        if not np.all(np.diff(nodes) > 1e-6):
            return 1e10
        weights, err = _solve_minimax_weights(nodes, test_set)
        if weights is None:
            return 1e10
        return err
    
    # Run Nelder-Mead
    try:
        res = minimize(_nm_objective, x0, method='Nelder-Mead', options=options)
        if res.success:
            new_nodes = _transform_nodes(res.x)
            new_weights, new_err = _solve_minimax_weights(new_nodes, test_set)
            if new_weights is not None and new_err < current_err:
                current_nodes = new_nodes
                current_weights = new_weights
                current_err = new_err
    except Exception as e:
        # If Nelder-Mead fails, keep the best from coarse search
        pass
    
    # Final cleanup
    best_nodes = current_nodes
    best_weights = current_weights
    
    # Ensure weights sum to 1 and are positive
    best_weights = np.maximum(best_weights, 0.0)
    best_weights = best_weights / np.sum(best_weights)
    
    # Ensure nodes are strictly increasing
    if not np.all(np.diff(best_nodes) > 0):
        nodes, weights = roots_legendre(8)
        return {"nodes": (nodes + 1) / 2, "weights": weights / 2}
        
    return {"nodes": best_nodes, "weights": best_weights}

# EVOLVE-BLOCK-END
