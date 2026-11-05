"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy import optimize
from scipy.integrate import quad


def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s

    def E_cpl(a, w0, wa, Om=0.3):
        Ol = 1.0 - Om
        if abs(w0 + 1.0) < 1e-10:
            if abs(wa) < 1e-10:
                return np.sqrt(Om / a**3 + Ol)
            else:
                return np.sqrt(Om / a**3 + Ol * a**(-3) * np.exp(3.0 * wa * (1.0 - a)))
        else:
            return np.sqrt(Om / a**3 + Ol * a**(-3.0*(1.0+w0+wa)) * np.exp(3.0*wa*(1.0-a)/(1.0+w0)))

    # Diverse set of models for moment matching - broader coverage
    test_models = [
        (-1.0, 0.0), (-1.0, 0.5), (-1.0, 1.0), (-1.0, 1.5),
        (-0.8, 0.3), (-0.8, 0.8), (-0.5, 0.5), (-0.5, 1.0),
        (-0.3, 0.2), (-0.3, 0.5), (-1.2, 0.5), (-1.2, 1.0),
        (-0.7, 0.7), (-0.9, 0.4), (-0.6, 0.6), (-1.1, 0.3),
        (-0.4, 0.8), (-0.2, 0.4), (-1.3, 0.7), (-0.85, 0.55),
    ]
    z_samples = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]

    # Precompute reference integrals
    ref_cache = {}
    def get_ref(w0, wa, z):
        key = (w0, wa, z)
        if key in ref_cache:
            return ref_cache[key]
        def integrand(x):
            a = 1.0 / (1.0 + z * x)
            return 1.0 / E_cpl(a, w0, wa)
        result, _ = quad(integrand, 0.0, 1.0, limit=500, epsabs=1e-14, epsrel=1e-14)
        ref_cache[key] = result
        return result

    # Build test case list
    cases = []
    for w0, wa in test_models:
        for z in z_samples:
            ref = get_ref(w0, wa, z)
            cases.append((w0, wa, z, ref))

    def compute_worst_error(nodes, weights):
        worst = 0.0
        for w0, wa, z, ref in cases:
            a_vals = 1.0 / (1.0 + z * nodes)
            E_vals = E_cpl(a_vals, w0, wa)
            quad_val = np.sum(weights / E_vals)
            if abs(ref) > 1e-15:
                err = abs(quad_val - ref) / abs(ref)
                if err > worst:
                    worst = err
        return worst

    def soft_max_error(nodes, weights, beta=80.0):
        errors = []
        for w0, wa, z, ref in cases:
            a_vals = 1.0 / (1.0 + z * nodes)
            E_vals = E_cpl(a_vals, w0, wa)
            quad_val = np.sum(weights / E_vals)
            if abs(ref) > 1e-15:
                err = abs(quad_val - ref) / abs(ref)
                errors.append(err)
        if not errors:
            return 0.0
        errors = np.array(errors)
        max_err = np.max(errors)
        if max_err < 1e-15:
            return 0.0
        shifted = errors - max_err
        log_sum = max_err + np.log(np.sum(np.exp(beta * shifted / max_err))) / beta
        return log_sum

    # Start from Gauss-Legendre
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(8)
    nodes0 = (gl_nodes + 1.0) / 2.0
    nodes0 = np.sort(nodes0)
    weights0 = gl_weights / 2.0
    weights0 = weights0 / weights0.sum()

    best_nodes = nodes0.copy()
    best_weights = weights0.copy()
    best_err = compute_worst_error(best_nodes, best_weights)

    # Phase 1: Moment-matching node optimization with diverse integrand selection
    if time.time() < deadline:
        # Select 8 representative integrands based on complexity
        complexities = []
        for w0, wa, z, ref in cases:
            x_grid = np.linspace(0.0, 1.0, 200)
            a_vals = 1.0 / (1.0 + z * x_grid)
            g = 1.0 / E_cpl(a_vals, w0, wa)
            d2g = np.gradient(np.gradient(g, x_grid), x_grid)
            complexities.append(np.max(np.abs(d2g)))
        
        complexities = np.array(complexities)
        selected = np.argsort(complexities)[-8:]
        selected_cases = [cases[i] for i in selected]
        selected_refs = np.array([c[3] for c in selected_cases])

        def moment_node_obj(params):
            n_nodes = np.sort(params)
            if np.any(n_nodes <= 0.001) or np.any(n_nodes >= 0.999):
                return 1e10
            if not np.all(np.diff(n_nodes) > 1e-4):
                return 1e10
            
            # Build moment matrix
            A = np.zeros((8, 8))
            for j, (w0, wa, z, ref) in enumerate(selected_cases):
                a_vals = 1.0 / (1.0 + z * n_nodes)
                E_vals = E_cpl(a_vals, w0, wa)
                A[j, :] = 1.0 / E_vals
            
            # Solve for weights
            try:
                w, residuals, rank, sv = np.linalg.lstsq(A, selected_refs, rcond=None)
            except:
                return 1e10
            
            if np.any(w <= 0):
                return 1e10
            w = w / np.sum(w)
            
            # Check conditioning
            cond = np.linalg.cond(A)
            if cond > 1e12:
                return 1e10
            
            # Compute worst error on all cases
            return compute_worst_error(n_nodes, w)

        # Try multiple starting points for moment matching
        np.random.seed(123)
        for restart in range(3):
            if time.time() > deadline:
                break
            if restart == 0:
                x0 = best_nodes.copy()
            else:
                x0 = best_nodes + np.random.normal(0, 0.01, size=8)
                x0 = np.sort(x0)
            
            try:
                result = optimize.minimize(
                    moment_node_obj,
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-11, 'fatol': 1e-13, 'adaptive': True}
                )
                if result.fun < best_err:
                    best_nodes = np.sort(result.x)
                    # Solve weights for these nodes
                    A = np.zeros((8, 8))
                    for j, (w0, wa, z, ref) in enumerate(selected_cases):
                        a_vals = 1.0 / (1.0 + z * best_nodes)
                        E_vals = E_cpl(a_vals, w0, wa)
                        A[j, :] = 1.0 / E_vals
                    w, _, _, _ = np.linalg.lstsq(A, selected_refs, rcond=None)
                    if np.all(w > 0):
                        best_weights = w / np.sum(w)
                        best_err = compute_worst_error(best_nodes, best_weights)
            except:
                pass

    # Phase 2: Joint refinement with soft-max objective
    if time.time() < deadline:
        def joint_obj(params):
            n_nodes = params[:8]
            n_weights = params[8:]
            if np.any(n_nodes <= 0.001) or np.any(n_nodes >= 0.999):
                return 1e10
            if not np.all(np.diff(n_nodes) > 1e-4):
                return 1e10
            if np.any(n_weights <= 0):
                return 1e10
            w = n_weights / np.sum(n_weights)
            return soft_max_error(n_nodes, w, beta=80.0)

        x0 = np.concatenate([best_nodes, best_weights])
        try:
            result = optimize.minimize(
                joint_obj,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            if result.fun < 1e-10:
                opt_nodes = np.sort(result.x[:8])
                opt_weights = result.x[8:]
                if np.all(opt_nodes > 0.001) and np.all(opt_nodes < 0.999) and np.all(np.diff(opt_nodes) > 1e-4):
                    if np.all(opt_weights > 0):
                        opt_weights = opt_weights / np.sum(opt_weights)
                        opt_err = compute_worst_error(opt_nodes, opt_weights)
                        if opt_err < best_err:
                            best_nodes = opt_nodes
                            best_weights = opt_weights
                            best_err = opt_err
        except:
            pass

    # Phase 3: Intensive local search with adaptive scaling
    if time.time() < deadline:
        np.random.seed(42)
        remaining_time = deadline - time.time()
        n_trials = min(100, int(remaining_time * 20))  # ~20 trials per second
        for trial in range(n_trials):
            if time.time() > deadline:
                break
            # Adaptive perturbation scale
            scale = max(0.001, 0.005 * (1 - trial / n_trials))
            pert_nodes = best_nodes + np.random.normal(0, scale, size=8)
            pert_weights = best_weights + np.random.normal(0, scale, size=8)
            pert_nodes = np.clip(pert_nodes, 0.001, 0.999)
            pert_nodes = np.sort(pert_nodes)
            if not np.all(np.diff(pert_nodes) > 1e-4):
                continue
            pert_weights = np.clip(pert_weights, 1e-6, None)
            pert_weights = pert_weights / np.sum(pert_weights)
            err = compute_worst_error(pert_nodes, pert_weights)
            if err < best_err:
                best_nodes = pert_nodes
                best_weights = pert_weights
                best_err = err

    # Phase 4: Final fine-tuning with very small steps
    if time.time() < deadline:
        np.random.seed(99)
        remaining_time = deadline - time.time()
        n_trials = min(50, int(remaining_time * 30))
        for trial in range(n_trials):
            if time.time() > deadline:
                break
            scale = 0.0005
            pert_nodes = best_nodes + np.random.normal(0, scale, size=8)
            pert_weights = best_weights + np.random.normal(0, scale, size=8)
            pert_nodes = np.clip(pert_nodes, 0.001, 0.999)
            pert_nodes = np.sort(pert_nodes)
            if not np.all(np.diff(pert_nodes) > 1e-4):
                continue
            pert_weights = np.clip(pert_weights, 1e-6, None)
            pert_weights = pert_weights / np.sum(pert_weights)
            err = compute_worst_error(pert_nodes, pert_weights)
            if err < best_err:
                best_nodes = pert_nodes
                best_weights = pert_weights
                best_err = err

    # Final normalization and validation
    best_nodes = np.sort(best_nodes)
    best_weights = best_weights / np.sum(best_weights)
    
    # Ensure exact sum to 1
    correction = 1.0 - np.sum(best_weights)
    best_weights += correction / 8.0
    best_weights = best_weights / np.sum(best_weights)

    assert len(best_nodes) == 8
    assert len(best_weights) == 8
    assert np.all(best_nodes > 0) and np.all(best_nodes < 1)
    assert np.all(np.diff(best_nodes) > 0)
    assert np.all(best_weights > 0)
    assert abs(np.sum(best_weights) - 1.0) < 1e-12

    return {"nodes": best_nodes, "weights": best_weights}
# EVOLVE-BLOCK-END
