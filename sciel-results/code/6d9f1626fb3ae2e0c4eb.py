"""
Strategy: Universal 8-point quadrature for luminosity distance.

Key structural change: Instead of optimizing nodes and weights jointly with
slow adaptive integration in the objective, we:
1. Pre-compute high-accuracy reference integrals for a large, diverse set
   of (z, w0, wa) configurations using scipy.integrate.quad.
2. For a candidate set of nodes, solve the linear least-squares problem for
   weights that best approximate these references.
3. Use a nested search: coarse grid over node configurations, with each
   configuration evaluated via fast matrix-vector products.
4. Refine the best node configuration with a local optimizer that uses
   pre-computed references (no adaptive integration in the loop).

This is fundamentally different from the previous approach:
- References are computed ONCE upfront (expensive but bounded cost).
- Node search uses fast linear algebra (no O(n) quad calls per eval).
- The optimization landscape is smoother because we're fitting to fixed targets.
"""

import time
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.special import roots_legendre

_START_TIME = time.time()
_DEADLINE = _START_TIME + 82.0  # Safety margin

def _E_of_z(z, w0, wa, Omega_m=0.3):
    """Compute E(z) = H(z)/H0 for CPL dark energy model."""
    Omega_de = 1.0 - Omega_m
    one_plus_z = 1.0 + z
    term_m = Omega_m * one_plus_z**3
    exponent = 3.0 * (1.0 + w0 + wa * z)
    term_de = Omega_de * one_plus_z**exponent
    return np.sqrt(term_m + term_de)

def _reference_integral(z, w0, wa, Omega_m=0.3):
    """High-accuracy reference: \int_0^1 dt / E(z*t)."""
    if z < 1e-10:
        return 1.0
    def integrand(t):
        return 1.0 / _E_of_z(z * t, w0, wa, Omega_m)
    val, err = quad(integrand, 0, 1, limit=200, epsabs=1e-12, epsrel=1e-12)
    return val

def _quadrature_value(nodes, weights, z, w0, wa, Omega_m=0.3):
    """Compute \sum w_i / E(z * t_i)."""
    if z < 1e-10:
        return np.sum(weights)
    zt = z * nodes
    E = _E_of_z(zt, w0, wa, Omega_m)
    return np.dot(weights, 1.0 / E)

def _build_reference_set():
    """
    Build a diverse set of (z, w0, wa) configurations and pre-compute
    reference integrals. This is the expensive part done only once.
    """
    # Cover the validation range broadly
    w0_vals = np.array([-1.8, -1.5, -1.2, -0.9, -0.6, -0.3, 0.0, 0.3, 0.6])
    wa_vals = np.array([-1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0])
    # z range: from small to large
    z_vals = np.array([0.01, 0.03, 0.07, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0,
                        1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0])
    
    test_set = []
    for w0 in w0_vals:
        for wa in wa_vals:
            for z in z_vals:
                test_set.append((z, w0, wa))
    
    return test_set

def _precompute_references(test_set):
    """Pre-compute reference integrals for all test points."""
    refs = np.zeros(len(test_set))
    for idx, (z, w0, wa) in enumerate(test_set):
        refs[idx] = _reference_integral(z, w0, wa)
    return refs

def _eval_node_config(nodes, refs, test_set):
    """
    Given fixed nodes, solve for optimal weights via least squares.
    Returns (weights, max_rel_err).
    """
    n = len(nodes)
    N = len(test_set)
    
    # Build A matrix: A[i, j] = 1/E(z_i * t_j)
    A = np.zeros((N, n))
    for idx, (z, w0, wa) in enumerate(test_set):
        if z < 1e-10:
            A[idx, :] = 1.0
        else:
            zt = z * nodes
            E = _E_of_z(zt, w0, wa)
            A[idx, :] = 1.0 / E
    
    # Solve least squares: A w = refs
    w, residuals, rank, s = np.linalg.lstsq(A, refs, rcond=None)
    
    # Check positivity
    if np.any(w < -1e-8):
        # Try to project: clip negatives and renormalize
        w_clip = np.maximum(w, 0.0)
        if np.sum(w_clip) < 1e-10:
            return None, float('inf')
        w_clip = w_clip / np.sum(w_clip)
        # Evaluate error with clipped weights
        approx = A.dot(w_clip)
        rel_errs = np.abs(approx - refs) / np.maximum(refs, 1e-15)
        max_err = np.max(rel_errs)
        return w_clip, max_err
    
    # Normalize to sum to 1
    w_sum = np.sum(w)
    if w_sum < 1e-10:
        return None, float('inf')
    w = w / w_sum
    
    # Evaluate max relative error
    approx = A.dot(w)
    rel_errs = np.abs(approx - refs) / np.maximum(refs, 1e-15)
    max_err = np.max(rel_errs)
    
    return w, max_err

def _gauss_legendre_nodes(n=8):
    """Gauss-Legendre nodes mapped to [0,1]."""
    nodes, _ = roots_legendre(n)
    return (nodes + 1.0) / 2.0

def _gauss_jacobi_nodes(alpha, beta, n=8):
    """
    Gauss-Jacobi nodes mapped to [0,1].
    Useful when integrand has endpoint behavior.
    """
    # Use scipy for Gauss-Jacobi
    from scipy.special import roots_jacobi
    nodes, _ = roots_jacobi(n, alpha, beta)
    return (nodes + 1.0) / 2.0

def _chebyshev_nodes(n=8):
    """Chebyshev nodes of the first kind mapped to [0,1]."""
    k = np.arange(n)
    nodes = np.cos((2 * k + 1) * np.pi / (2 * n))
    return (nodes + 1.0) / 2.0

def _generate_node_candidates():
    """
    Generate diverse node configurations for the coarse search.
    Each is an array of 8 strictly increasing points in [0,1].
    """
    candidates = []
    
    # 1. Gauss-Legendre
    candidates.append(_gauss_legendre_nodes(8))
    
    # 2. Chebyshev
    candidates.append(_chebyshev_nodes(8))
    
    # 3. Gauss-Jacobi with various alpha, beta (endpoint concentration)
    for alpha, beta in [(0.5, 0.5), (1.0, 0.0), (0.0, 1.0), 
                         (0.5, 1.0), (1.0, 0.5), (2.0, 0.0), (0.0, 2.0)]:
        try:
            nodes = _gauss_jacobi_nodes(alpha, beta, 8)
            if np.all(np.diff(nodes) > 1e-10):
                candidates.append(nodes)
        except:
            pass
    
    # 4. Uniform
    candidates.append(np.linspace(1/17, 16/17, 8))
    
    # 5. Slightly perturbed GL nodes (to escape local structure)
    gl = _gauss_legendre_nodes(8)
    rng = np.random.RandomState(42)
    for trial in range(10):
        perturb = rng.uniform(-0.02, 0.02, 8)
        perturbed = gl + perturb
        # Project to [epsilon, 1-epsilon]
        eps = 1e-4
        perturbed = np.clip(perturbed, eps, 1 - eps)
        # Sort
        perturbed = np.sort(perturbed)
        if np.all(np.diff(perturbed) > 1e-6):
            candidates.append(perturbed)
    
    # 6. Log-spaced nodes (concentrated near 0)
    for base in [2.0, 3.0, 5.0, 10.0]:
        log_nodes = np.linspace(0, 1, 8)
        # Create log-concentrated nodes
        t = np.linspace(0, 1, 8)
        log_nodes = 1.0 - (1.0 - t) ** base
        log_nodes = np.clip(log_nodes, 1e-4, 1 - 1e-4)
        if np.all(np.diff(log_nodes) > 1e-6):
            candidates.append(log_nodes)
    
    # 7. Power-law concentrated near 1
    for base in [0.5, 0.3, 0.2]:
        t = np.linspace(0, 1, 8)
        pow_nodes = t ** (1.0 / base)
        pow_nodes = np.clip(pow_nodes, 1e-4, 1 - 1e-4)
        if np.all(np.diff(pow_nodes) > 1e-6):
            candidates.append(pow_nodes)
    
    return candidates

def construct():
    # Deadline check at start
    if time.time() > _DEADLINE - 2:
        # Absolute fallback
        nodes, weights = roots_legendre(8)
        nodes = (nodes + 1) / 2
        weights = weights / 2
        return {"nodes": nodes, "weights": weights}
    
    # Step 1: Build reference set
    test_set = _build_reference_set()
    N = len(test_set)
    
    # Step 2: Pre-compute references
    refs = _precompute_references(test_set)
    
    if time.time() > _DEADLINE - 3:
        nodes, weights = roots_legendre(8)
        nodes = (nodes + 1) / 2
        weights = weights / 2
        return {"nodes": nodes, "weights": weights}
    
    # Step 3: Coarse search over node candidates
    candidates = _generate_node_candidates()
    
    best_nodes = None
    best_weights = None
    best_err = float('inf')
    
    for cand in candidates:
        if time.time() > _DEADLINE - 3:
            break
        w, err = _eval_node_config(cand, refs, test_set)
        if w is not None and err < best_err:
            best_err = err
            best_nodes = cand.copy()
            best_weights = w.copy()
    
    if best_nodes is None:
        # Fallback
        nodes, weights = roots_legendre(8)
        nodes = (nodes + 1) / 2
        weights = weights / 2
        return {"nodes": nodes, "weights": weights}
    
    # Step 4: Refine best node configuration with local optimization
    # Objective: max relative error (minimize)
    # We optimize nodes only; weights are re-solved via LS at each eval.
    
    n = 8
    
    def _objective_nodes(node_params):
        """
        node_params: 16-dim vector representing 8 nodes.
        We use a parameterization that ensures strict increase:
        node_i = base + cumsum(gaps), where gaps > 0.
        
        Simpler: just use 8 bounded values and check monotonicity.
        """
        nodes_p = node_params.copy()
        
        # Ensure in (0, 1)
        if np.any(nodes_p <= 0) or np.any(nodes_p >= 1):
            return 1e10
        # Ensure strict increase
        diffs = np.diff(nodes_p)
        if np.any(diffs <= 1e-8):
            return 1e10
        
        w, err = _eval_node_config(nodes_p, refs, test_set)
        if w is None:
            return 1e10
        return err
    
    # Start from best_nodes
    x0 = best_nodes.copy()
    
    # Bounds: each node in [1e-4, 1 - 1e-4]
    bounds = [(1e-4, 1 - 1e-4)] * n
    
    try:
        result = minimize(
            _objective_nodes, x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if result.fun < best_err:
            refined_nodes = result.x
            # Validate
            if np.all(refined_nodes > 0) and np.all(refined_nodes < 1):
                if np.all(np.diff(refined_nodes) > 1e-8):
                    w_ref, err_ref = _eval_node_config(refined_nodes, refs, test_set)
                    if w_ref is not None and err_ref < best_err:
                        best_nodes = refined_nodes
                        best_weights = w_ref
                        best_err = err_ref
    except Exception:
        pass
    
    # Step 5: Final verification and cleanup
    # Ensure weights sum to 1
    w_sum = np.sum(best_weights)
    if w_sum > 1e-12:
        best_weights = best_weights / w_sum
    
    # Ensure nodes are strictly increasing
    sort_idx = np.argsort(best_nodes)
    best_nodes = best_nodes[sort_idx]
    best_weights = best_weights[sort_idx]
    
    # Final strict increase check
    diffs = np.diff(best_nodes)
    if np.any(diffs <= 0):
        # Fallback to Gauss-Legendre
        nodes, weights = roots_legendre(8)
        nodes = (nodes + 1) / 2
        weights = weights / 2
        return {"nodes": nodes, "weights": weights}
    
    # Ensure weights are positive
    if np.any(best_weights <= 0):
        # Fallback
        nodes, weights = roots_legendre(8)
        nodes = (nodes + 1) / 2
        weights = weights / 2
        return {"nodes": nodes, "weights": weights}
    
    # Final check: sum of weights = 1 within tolerance
    if abs(np.sum(best_weights) - 1.0) > 1e-12:
        best_weights = best_weights / np.sum(best_weights)
    
    return {"nodes": best_nodes, "weights": best_weights}

# EVOLVE-BLOCK-END
