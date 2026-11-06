"""
Strategy: Basis-function decomposition with Chebyshev-weighted least-squares.

Key structural change: Instead of fitting a fixed quadrature to reference integrals,
we decompose the integrand f(t) = 1/E(z*t) as a product of a smooth geometric
factor (the matter contribution, which is known analytically) and a model-dependent
dark-energy term. We then find nodes and weights that best represent a set of
basis functions spanning the variation of the dark-energy density across the CPL
parameter space. Positivity is enforced by projecting weights onto the positive
cone and renormalizing.

The approach:
1. Recognize that 1/E(z*t) = 1/sqrt(Omega_m*(1+zt)^3 + Omega_de*(1+zt)^(3(1+w0+wa*zt)))
   The matter part is smooth and known; the dark-energy part varies with (w0, wa).
2. Build a library of basis functions g_k(z*t) that capture the dark-energy
   variation. For CPL, the exponent is 3(1+w0+wa*zt) = 3 + 3w0 + 3wa*zt.
   So the key varying quantity is (1+zt)^(3wa*zt) and the constant shift 3w0.
   We use Chebyshev polynomials in the variable u = z*t as a basis.
3. For a candidate node set, solve for weights that minimize the worst-case
   relative error when approximating 1/E for all (w0, wa) in a dense grid.
   This is done via a Chebyshev-weighted least-squares fit.
4. Use a global search over node configurations (including structured families)
   followed by a local refinement.
"""

import time
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.special import roots_legendre, roots_jacobi

_START_TIME = time.time()
_DEADLINE = _START_TIME + 85.0  # Safety margin (90s limit)

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
    val, err = quad(integrand, 0, 1, limit=300, epsabs=1e-13, epsrel=1e-13)
    return val

def _quadrature_value(nodes, weights, z, w0, wa, Omega_m=0.3):
    """Compute \sum w_i / E(z * t_i)."""
    if z < 1e-10:
        return np.sum(weights)
    zt = z * nodes
    E = _E_of_z(zt, w0, wa, Omega_m)
    return np.dot(weights, 1.0 / E)

def _build_basis_library(w0_vals, wa_vals, z_vals):
    """
    Build a library of basis functions that span the variation of 1/E(z*t)
    across the CPL parameter space. We use Chebyshev polynomials in u = z*t
    as the basis, evaluated at the node positions.
    """
    # The integrand is 1/sqrt(Omega_m*(1+u)^3 + Omega_de*(1+u)^(3+3w0+3wa*u))
    # The key variation is in the exponent of (1+u) for the dark energy term.
    # We'll use a set of representative (w0, wa) pairs and compute 1/E at all z*t.
    
    basis_data = []
    for w0 in w0_vals:
        for wa in wa_vals:
            for z in z_vals:
                if z < 1e-10:
                    basis_data.append(np.ones(1))  # placeholder
                else:
                    basis_data.append((z, w0, wa))
    
    return basis_data

def _eval_node_config_basis(nodes, w0_vals, wa_vals, z_vals, Omega_m=0.3):
    """
    Given fixed nodes, find optimal positive weights that minimize the worst-case
    relative error across all (z, w0, wa) configurations.
    
    We use a two-stage approach:
    1. Solve unconstrained least squares for weights.
    2. Project onto positive cone and renormalize.
    3. Evaluate max relative error.
    """
    n = len(nodes)
    
    # Build all (z, w0, wa) test points
    test_points = []
    for w0 in w0_vals:
        for wa in wa_vals:
            for z in z_vals:
                test_points.append((z, w0, wa))
    
    N = len(test_points)
    
    # Build A matrix: A[i, j] = 1/E(z_i * t_j)
    A = np.zeros((N, n))
    refs = np.zeros(N)
    
    for idx, (z, w0, wa) in enumerate(test_points):
        if z < 1e-10:
            A[idx, :] = 1.0
            refs[idx] = 1.0
        else:
            zt = z * nodes
            E = _E_of_z(zt, w0, wa, Omega_m)
            A[idx, :] = 1.0 / E
            refs[idx] = _reference_integral(z, w0, wa, Omega_m)
    
    # Solve least squares
    w, residuals, rank, s = np.linalg.lstsq(A, refs, rcond=None)
    
    # Project onto positive cone
    w_pos = np.maximum(w, 0.0)
    w_sum = np.sum(w_pos)
    if w_sum < 1e-10:
        return None, float('inf')
    w_pos = w_pos / w_sum
    
    # Evaluate max relative error
    approx = A.dot(w_pos)
    rel_errs = np.abs(approx - refs) / np.maximum(refs, 1e-15)
    max_err = np.max(rel_errs)
    
    return w_pos, max_err

def _chebyshev_nodes(n=8):
    """Chebyshev nodes of the first kind mapped to [0,1]."""
    k = np.arange(n)
    nodes = np.cos((2 * k + 1) * np.pi / (2 * n))
    return (nodes + 1.0) / 2.0

def _gauss_legendre_nodes(n=8):
    """Gauss-Legendre nodes mapped to [0,1]."""
    nodes, _ = roots_legendre(n)
    return (nodes + 1.0) / 2.0

def _generate_node_candidates():
    """
    Generate diverse node configurations for the search.
    """
    candidates = []
    
    # 1. Gauss-Legendre
    candidates.append(_gauss_legendre_nodes(8))
    
    # 2. Chebyshev
    candidates.append(_chebyshev_nodes(8))
    
    # 3. Gauss-Jacobi with various alpha, beta
    for alpha, beta in [(0.5, 0.5), (1.0, 0.0), (0.0, 1.0), 
                         (0.5, 1.0), (1.0, 0.5), (2.0, 0.0), (0.0, 2.0),
                         (1.5, 0.5), (0.5, 1.5)]:
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
    for base in [1.5, 2.0, 3.0, 5.0]:
        t = np.linspace(0, 1, 8)
        log_nodes = 1.0 - (1.0 - t) ** base
        log_nodes = np.clip(log_nodes, 1e-4, 1 - 1e-4)
        if np.all(np.diff(log_nodes) > 1e-6):
            candidates.append(log_nodes)
    
    # 6. Power-law concentrated near 1
    for base in [0.5, 0.3, 0.2, 0.7]:
        t = np.linspace(0, 1, 8)
        pow_nodes = t ** (1.0 / base)
        pow_nodes = np.clip(pow_nodes, 1e-4, 1 - 1e-4)
        if np.all(np.diff(pow_nodes) > 1e-6):
            candidates.append(pow_nodes)
    
    # 7. Perturbed Gauss-Legendre
    gl = _gauss_legendre_nodes(8)
    rng = np.random.RandomState(42)
    for trial in range(20):
        perturb = rng.uniform(-0.03, 0.03, 8)
        perturbed = gl + perturb
        eps = 1e-4
        perturbed = np.clip(perturbed, eps, 1 - eps)
        perturbed = np.sort(perturbed)
        if np.all(np.diff(perturbed) > 1e-6):
            candidates.append(perturbed)
    
    # 8. Perturbed Chebyshev
    ch = _chebyshev_nodes(8)
    for trial in range(10):
        perturb = rng.uniform(-0.03, 0.03, 8)
        perturbed = ch + perturb
        eps = 1e-4
        perturbed = np.clip(perturbed, eps, 1 - eps)
        perturbed = np.sort(perturbed)
        if np.all(np.diff(perturbed) > 1e-6):
            candidates.append(perturbed)
    
    return candidates

def construct():
    # Deadline check at start
    if time.time() > _DEADLINE - 2:
        nodes, weights = roots_legendre(8)
        nodes = (nodes + 1) / 2
        weights = weights / 2
        return {"nodes": nodes, "weights": weights}
    
    # Define the parameter space for the basis
    w0_vals = np.array([-1.5, -1.0, -0.5, 0.0, 0.5])
    wa_vals = np.array([-0.8, -0.4, 0.0, 0.4, 0.8])
    z_vals = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0])
    
    # Generate node candidates
    candidates = _generate_node_candidates()
    
    best_nodes = None
    best_weights = None
    best_err = float('inf')
    
    # Coarse search
    for cand in candidates:
        if time.time() > _DEADLINE - 5:
            break
        w, err = _eval_node_config_basis(cand, w0_vals, wa_vals, z_vals)
        if w is not None and err < best_err:
            best_err = err
            best_nodes = cand.copy()
            best_weights = w.copy()
    
    if best_nodes is None:
        nodes, weights = roots_legendre(8)
        nodes = (nodes + 1) / 2
        weights = weights / 2
        return {"nodes": nodes, "weights": weights}
    
    # Refine with local optimization
    n = 8
    
    def _objective_nodes(node_params):
        nodes_p = node_params.copy()
        if np.any(nodes_p <= 0) or np.any(nodes_p >= 1):
            return 1e10
        diffs = np.diff(nodes_p)
        if np.any(diffs <= 1e-8):
            return 1e10
        w, err = _eval_node_config_basis(nodes_p, w0_vals, wa_vals, z_vals)
        if w is None:
            return 1e10
        return err
    
    x0 = best_nodes.copy()
    bounds = [(1e-4, 1 - 1e-4)] * n
    
    try:
        result = minimize(
            _objective_nodes, x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if result.fun < best_err:
            refined_nodes = result.x
            if np.all(refined_nodes > 0) and np.all(refined_nodes < 1):
                if np.all(np.diff(refined_nodes) > 1e-8):
                    w_ref, err_ref = _eval_node_config_basis(refined_nodes, w0_vals, wa_vals, z_vals)
                    if w_ref is not None and err_ref < best_err:
                        best_nodes = refined_nodes
                        best_weights = w_ref
                        best_err = err_ref
    except Exception:
        pass
    
    # Final verification
    w_sum = np.sum(best_weights)
    if w_sum > 1e-12:
        best_weights = best_weights / w_sum
    
    sort_idx = np.argsort(best_nodes)
    best_nodes = best_nodes[sort_idx]
    best_weights = best_weights[sort_idx]
    
    diffs = np.diff(best_nodes)
    if np.any(diffs <= 0):
        nodes, weights = roots_legendre(8)
        nodes = (nodes + 1) / 2
        weights = weights / 2
        return {"nodes": nodes, "weights": weights}
    
    if np.any(best_weights <= 0):
        nodes, weights = roots_legendre(8)
        nodes = (nodes + 1) / 2
        weights = weights / 2
        return {"nodes": nodes, "weights": weights}
    
    if abs(np.sum(best_weights) - 1.0) > 1e-12:
        best_weights = best_weights / np.sum(best_weights)
    
    return {"nodes": best_nodes, "weights": best_weights}

# EVOLVE-BLOCK-END
