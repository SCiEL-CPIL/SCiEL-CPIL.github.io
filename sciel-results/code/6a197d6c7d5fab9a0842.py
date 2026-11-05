"""Cosmology: universal luminosity-distance quadrature.
Hybrid global-local search for an 8-node positive quadrature rule.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy import optimize
from scipy.integrate import quad

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin

    # 1. Define the model library
    # We use a broad set of CPL (Chevallier-Polarski-Linder) models.
    # E(a) = sqrt(Om/a^3 + Ol * a^(-3(1+w0+wa)) * exp(3*wa*(1-a)/(1+w0)))
    # For w0=-1, wa=0, it reduces to LCDM.
    
    def E_cpl(a, w0, wa):
        Om = 0.3
        Ol = 1.0 - Om
        # Handle w0 = -1 case separately to avoid division by zero in exponent
        if abs(w0 + 1.0) < 1e-10:
            if abs(wa) < 1e-10:
                return np.sqrt(Om / a**3 + Ol)
            else:
                return np.sqrt(Om / a**3 + Ol * a**(-3.0) * np.exp(3.0 * wa * (1.0 - a)))
        else:
            return np.sqrt(Om / a**3 + Ol * a**(-3.0*(1.0+w0+wa)) * np.exp(3.0*wa*(1.0-a)/(1.0+w0)))

    # Broad grid of models to ensure universality
    w0_vals = [-1.5, -1.2, -1.0, -0.8, -0.5, -0.2, 0.0]
    wa_vals = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
    
    test_models = []
    for w0 in w0_vals:
        for wa in wa_vals:
            # Skip duplicates or invalid if any, but CPL is generally valid
            test_models.append((w0, wa))

    # Representative redshifts
    # Include low z for precision, high z for stability
    test_zs = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0]

    n_models = len(test_models)
    n_zs = len(test_zs)

    # 2. Pre-compute reference integrals
    # This is the expensive part, but it only happens once.
    ref_values = np.zeros((n_models, n_zs))
    
    for i, (w0, wa) in enumerate(test_models):
        for j, z in enumerate(test_zs):
            def integrand(x):
                a = 1.0 / (1.0 + z * x)
                return 1.0 / E_cpl(a, w0, wa)
            
            # High precision adaptive quadrature
            try:
                result, _ = quad(integrand, 0.0, 1.0, limit=500, epsabs=1e-14, epsrel=1e-14)
                ref_values[i, j] = result
            except Exception:
                ref_values[i, j] = 0.0 # Should not happen

    # 3. Vectorized Error Calculation
    # We want to compute the quadrature for all (model, z) pairs given nodes and weights.
    # quad_val[i, j] = sum_k weights[k] / E(a_k) where a_k = 1/(1 + z_j * nodes[k])
    
    def compute_errors(nodes, weights):
        """
        nodes: shape (8,)
        weights: shape (8,)
        Returns: max relative error
        """
        # a_vals: (n_models, n_zs, 8)
        # z is (n_zs,), nodes is (8,)
        # z[:, None, :] * nodes[None, None, :] -> (1, n_zs, 8) broadcasted
        z_col = np.array(test_zs) # (n_zs,)
        
        # Calculate a for all z and nodes
        # Shape: (n_zs, 8)
        a_z_nodes = 1.0 / (1.0 + z_col[:, None] * nodes[None, :])
        
        # We need E(a) for each model.
        # E_cpl is vectorized over a.
        # We can loop over models (small n_models ~ 42) and vectorize over z and nodes.
        
        max_err = 0.0
        for i, (w0, wa) in enumerate(test_models):
            # E_vals: (n_zs, 8)
            E_vals = E_cpl(a_z_nodes, w0, wa)
            inv_E = 1.0 / E_vals
            
            # Quadrature sum: weights dot inv_E for each z
            # weights: (8,), inv_E: (n_zs, 8) -> quad_vals: (n_zs,)
            quad_vals = np.sum(inv_E * weights[None, :], axis=1)
            
            # Relative error
            ref_row = ref_values[i, :]
            # Avoid division by zero
            mask = ref_row > 1e-15
            if np.any(mask):
                rel_errs = np.abs(quad_vals[mask] - ref_row[mask]) / ref_row[mask]
                current_max = np.max(rel_errs)
                if current_max > max_err:
                    max_err = current_max
        return max_err

    def soft_max_error(nodes, weights, beta=50.0):
        """
        Smooth approximation of max error for better gradients in local optimization.
        """
        # We need the vector of errors to do log-sum-exp properly
        # Re-computing the full error vector is expensive, but necessary for soft-max.
        # Alternatively, just use max error for DE and soft-max for NM?
        # Let's stick to max error for simplicity and speed, but use a slightly smoothed version if needed.
        # For now, let's just return max error, as NM can handle it with adaptive=True.
        return compute_errors(nodes, weights)

    # 4. Optimization Strategy
    # Start with Gauss-Legendre
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(8)
    nodes0 = (gl_nodes + 1.0) / 2.0
    weights0 = gl_weights / 2.0
    weights0 = weights0 / np.sum(weights0)
    
    best_nodes = nodes0.copy()
    best_weights = weights0.copy()
    best_err = compute_errors(best_nodes, best_weights)

    # Objective function for optimizers
    def objective(params):
        nodes = params[:8]
        weights = params[8:16]
        
        # Constraints: nodes in (0,1), strictly increasing, weights > 0
        if np.any(nodes <= 0.001) or np.any(nodes >= 0.999):
            return 1e10
        if not np.all(np.diff(nodes) > 1e-5):
            return 1e10
        if np.any(weights <= 0):
            return 1e10
            
        weights = weights / np.sum(weights)
        return compute_errors(nodes, weights)

    # Phase 1: Global Optimization with Differential Evolution
    # Use a moderate population to balance exploration and speed
    if time.time() < deadline - 20:
        bounds = [(0.001, 0.999) for _ in range(8)] + [(1e-6, 1.0) for _ in range(8)]
        
        try:
            de_result = optimize.differential_evolution(
                objective,
                bounds,
                seed=42,
                maxiter=100, # Reduced iterations to save time
                popsize=15,
                tol=1e-8,
                polish=False
            )
            if de_result.fun < 1e-10:
                cand_nodes = np.sort(de_result.x[:8])
                cand_weights = de_result.x[8:]
                if np.all(np.diff(cand_nodes) > 1e-5) and np.all(cand_weights > 0):
                    cand_weights = cand_weights / np.sum(cand_weights)
                    cand_err = compute_errors(cand_nodes, cand_weights)
                    if cand_err < best_err:
                        best_nodes = cand_nodes
                        best_weights = cand_weights
                        best_err = cand_err
        except Exception:
            pass

    # Phase 2: Local Refinement with Nelder-Mead
    # Try refining from the best DE result and from Gauss-Legendre
    start_points = [
        np.concatenate([best_nodes, best_weights]),
        np.concatenate([nodes0, weights0])
    ]
    
    # Add a few random perturbations of the best solution
    rng = np.random.RandomState(42)
    for _ in range(3):
        pert_nodes = best_nodes + rng.normal(0, 0.02, size=8)
        pert_weights = best_weights + rng.normal(0, 0.02, size=8)
        pert_nodes = np.clip(pert_nodes, 0.002, 0.998)
        pert_weights = np.abs(pert_weights)
        pert_weights = pert_weights / np.sum(pert_weights)
        pert_nodes = np.sort(pert_nodes)
        if np.all(np.diff(pert_nodes) > 1e-5):
            start_points.append(np.concatenate([pert_nodes, pert_weights]))

    for sp in start_points:
        if time.time() >= deadline - 5:
            break
        try:
            nm_result = optimize.minimize(
                objective,
                sp,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            if nm_result.fun < 1e-10:
                cand_nodes = np.sort(nm_result.x[:8])
                cand_weights = nm_result.x[8:]
                if np.all(np.diff(cand_nodes) > 1e-5) and np.all(cand_weights > 0):
                    cand_weights = cand_weights / np.sum(cand_weights)
                    cand_err = compute_errors(cand_nodes, cand_weights)
                    if cand_err < best_err:
                        best_nodes = cand_nodes
                        best_weights = cand_weights
                        best_err = cand_err
        except Exception:
            pass

    # Final normalization and validation
    best_nodes = np.sort(best_nodes)
    best_weights = best_weights / np.sum(best_weights)

    # Ensure strict constraints
    assert len(best_nodes) == 8
    assert len(best_weights) == 8
    assert np.all(best_nodes > 0) and np.all(best_nodes < 1)
    assert np.all(np.diff(best_nodes) > 0)
    assert np.all(best_weights > 0)
    assert abs(np.sum(best_weights) - 1.0) < 1e-12

    return {"nodes": best_nodes, "weights": best_weights}

# EVOLVE-BLOCK-END
