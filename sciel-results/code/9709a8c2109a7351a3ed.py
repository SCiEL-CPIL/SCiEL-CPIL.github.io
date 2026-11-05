"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy import optimize
from scipy.integrate import quad


def construct():
    start_time = time.time()
    deadline = start_time + 82.0

    def E_cpl(a, w0, wa, Om=0.3):
        Ol = 1.0 - Om
        if abs(w0 + 1.0) < 1e-10:
            if abs(wa) < 1e-10:
                return np.sqrt(Om / a**3 + Ol)
            else:
                return np.sqrt(Om / a**3 + Ol * a**(-3) * np.exp(3.0 * wa * (1.0 - a)))
        else:
            return np.sqrt(Om / a**3 + Ol * a**(-3.0*(1.0+w0+wa)) * np.exp(3.0*wa*(1.0-a)/(1.0+w0)))

    # Large diverse model set for training (w0, wa)
    model_list = [
        (-1.0, 0.0), (-1.0, 0.3), (-1.0, 0.6), (-1.0, 0.9), (-1.0, 1.2),
        (-0.9, 0.2), (-0.9, 0.5), (-0.9, 0.8), (-0.9, 1.1),
        (-0.8, 0.0), (-0.8, 0.4), (-0.8, 0.7), (-0.8, 1.0),
        (-0.7, 0.2), (-0.7, 0.5), (-0.7, 0.8),
        (-0.6, 0.1), (-0.6, 0.4), (-0.6, 0.7), (-0.6, 1.0),
        (-0.5, 0.0), (-0.5, 0.3), (-0.5, 0.6), (-0.5, 0.9),
        (-0.4, 0.2), (-0.4, 0.5), (-0.4, 0.8),
        (-0.3, 0.1), (-0.3, 0.4), (-0.3, 0.7),
        (-0.2, 0.3), (-0.2, 0.6),
        (-1.1, 0.3), (-1.1, 0.6), (-1.1, 0.9),
        (-1.2, 0.4), (-1.2, 0.8),
        (-1.3, 0.5),
    ]
    z_list = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

    # Precompute reference integrals with quad
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

    # Build cases: (w0, wa, z, ref)
    cases = []
    for w0, wa in model_list:
        for z in z_list:
            if time.time() > deadline - 30.0:
                break
            ref = get_ref(w0, wa, z)
            cases.append((w0, wa, z, ref))

    n_cases = len(cases)
    # Precompute arrays for vectorized evaluation
    case_w0 = np.array([c[0] for c in cases])
    case_wa = np.array([c[1] for c in cases])
    case_z = np.array([c[2] for c in cases])
    case_ref = np.array([c[3] for c in cases])

    def vectorized_eval(nodes, weights):
        """Compute worst relative error over all cases. nodes: (8,), weights: (8,)"""
        # a[i, j] = 1/(1 + z_j * nodes_i)  shape (8, n_cases)
        a_vals = 1.0 / (1.0 + np.outer(nodes, case_z))  # (8, n_cases)
        # E for each (node, case)
        Om = 0.3
        Ol = 1.0 - Om
        w0_b = case_w0  # (n_cases,)
        wa_b = case_wa
        # Broadcast: a_vals is (8, n), w0_b is (n,)
        # For each case j, compute E at 8 nodes
        # E(a) depends on w0_j, wa_j, a
        # Handle special case w0=-1 separately
        is_cdm = np.abs(w0_b + 1.0) < 1e-10
        is_lcdM = is_cdm & (np.abs(wa_b) < 1e-10)
        is_wa_zero = is_cdm & ~is_lcdM

        # Compute E for all (node, case) pairs
        # a_vals shape (8, n_cases)
        # For CDM: E = sqrt(Om/a^3 + Ol)
        # For w0=-1, wa!=0: E = sqrt(Om/a^3 + Ol * a^-3 * exp(3*wa*(1-a)))
        # General: E = sqrt(Om/a^3 + Ol * a^(-3(1+w0+wa)) * exp(3*wa*(1-a)/(1+w0)))

        # Vectorized general formula (handle w0=-1 as limit)
        # Use the general form with safe handling
        a3 = a_vals**3  # (8, n)
        Om_term = Om / a3  # (8, n)

        # For the dark energy term, handle by category
        # General: Ol * a^(-3(1+w0+wa)) * exp(3*wa*(1-a)/(1+w0))
        # For w0=-1: Ol * a^(-3) * exp(3*wa*(1-a))
        # For w0=-1, wa=0: Ol

        # Compute exponent and power for each (node, case)
        # power = -3*(1+w0+wa) per case -> shape (n,)
        power = -3.0 * (1.0 + w0_b + wa_b)  # (n,)
        # exp_arg: for w0 != -1: 3*wa*(1-a)/(1+w0); for w0=-1: 3*wa*(1-a)
        # We'll compute both and select
        one_plus_w0 = 1.0 + w0_b  # (n,)
        safe_one_plus_w0 = np.where(np.abs(one_plus_w0) < 1e-10, 1.0, one_plus_w0)
        exp_arg_general = 3.0 * wa_b * (1.0 - a_vals) / safe_one_plus_w0  # (8, n)
        exp_arg_cdm = 3.0 * wa_b[None, :] * (1.0 - a_vals)  # (8, n)
        exp_arg = np.where(is_cdm[None, :], exp_arg_cdm, exp_arg_general)  # (8, n)

        # DE term: Ol * a^power * exp(exp_arg)
        # a^power: shape (8, n), power shape (n,) -> broadcast
        a_power = a_vals ** power[None, :]  # (8, n)
        de_term = Ol * a_power * np.exp(exp_arg)  # (8, n)

        # For CDM with wa=0, de_term should just be Ol (since a^-3 * exp(0) = a^-3, but actually...)
        # Wait: for w0=-1, wa=0: E = sqrt(Om/a^3 + Ol). The formula gives Ol * a^-3 * exp(0) = Ol/a^3, which is wrong.
        # The CDM limit is special. Let me handle it:
        # For w0=-1, wa=0: DE term = Ol (constant)
        # For w0=-1, wa!=0: DE term = Ol * a^-3 * exp(3*wa*(1-a))
        # For general: DE term = Ol * a^(-3(1+w0+wa)) * exp(3*wa*(1-a)/(1+w0))

        # Fix: for is_lcdM cases, set de_term = Ol
        de_term = np.where(is_lcdM[None, :], Ol, de_term)

        E_vals = np.sqrt(Om_term + de_term)  # (8, n)

        # Quadrature: sum_i weights[i] / E_vals[i, j] for each case j
        # weights shape (8,), E_vals shape (8, n)
        quad_vals = np.sum(weights[:, None] / E_vals, axis=0)  # (n,)

        # Relative errors
        rel_errs = np.abs(quad_vals - case_ref) / np.abs(case_ref)
        return np.max(rel_errs)

    def soft_max_error(nodes, weights, beta=80.0):
        """Smooth approximation to max error for gradient-free optimization."""
        a_vals = 1.0 / (1.0 + np.outer(nodes, case_z))
        Om = 0.3
        Ol = 1.0 - Om
        w0_b = case_w0
        wa_b = case_wa
        is_cdm = np.abs(w0_b + 1.0) < 1e-10
        is_lcdM = is_cdm & (np.abs(wa_b) < 1e-10)

        a3 = a_vals**3
        Om_term = Om / a3
        power = -3.0 * (1.0 + w0_b + wa_b)
        one_plus_w0 = 1.0 + w0_b
        safe_one_plus_w0 = np.where(np.abs(one_plus_w0) < 1e-10, 1.0, one_plus_w0)
        exp_arg_general = 3.0 * wa_b * (1.0 - a_vals) / safe_one_plus_w0
        exp_arg_cdm = 3.0 * wa_b[None, :] * (1.0 - a_vals)
        exp_arg = np.where(is_cdm[None, :], exp_arg_cdm, exp_arg_general)
        a_power = a_vals ** power[None, :]
        de_term = Ol * a_power * np.exp(exp_arg)
        de_term = np.where(is_lcdM[None, :], Ol, de_term)
        E_vals = np.sqrt(Om_term + de_term)
        quad_vals = np.sum(weights[:, None] / E_vals, axis=0)
        rel_errs = np.abs(quad_vals - case_ref) / np.abs(case_ref)

        max_err = np.max(rel_errs)
        if max_err < 1e-15:
            return 0.0
        shifted = rel_errs - max_err
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
    best_err = vectorized_eval(best_nodes, best_weights)

    # Phase 1: Differential Evolution global search
    if time.time() < deadline - 15.0:
        def de_obj(params):
            n_nodes = params[:8]
            n_weights = params[8:]
            if np.any(n_nodes <= 0.005) or np.any(n_nodes >= 0.995):
                return 1e10
            if not np.all(np.diff(n_nodes) > 1e-4):
                return 1e10
            if np.any(n_weights <= 0):
                return 1e10
            w = n_weights / np.sum(n_weights)
            return soft_max_error(n_nodes, w, beta=60.0)

        # Bounds for DE
        # Nodes must be in (0,1), strictly increasing with min gap 1e-4
        # We'll use bounds on individual nodes and check constraints
        # Strategy: parameterize as base + cumulative gaps
        # Simpler: just bound each node and check
        bounds_nodes = [(0.01, 0.99)] * 8
        bounds_weights = [(0.001, 1.0)] * 8
        bounds = bounds_nodes + bounds_weights

        try:
            result = optimize.differential_evolution(
                de_obj,
                bounds,
                seed=42,
                maxiter=200,
                tol=1e-8,
                popsize=20,
                init='sobol',
                polish=False,
                updating='immediate',
                workers=1,
            )
            if result.fun < 1e-6:
                opt_nodes = np.sort(result.x[:8])
                opt_weights = result.x[8:]
                if np.all(opt_nodes > 0.005) and np.all(opt_nodes < 0.995) and np.all(np.diff(opt_nodes) > 1e-4):
                    if np.all(opt_weights > 0):
                        opt_weights = opt_weights / np.sum(opt_weights)
                        opt_err = vectorized_eval(opt_nodes, opt_weights)
                        if opt_err < best_err:
                            best_nodes = opt_nodes
                            best_weights = opt_weights
                            best_err = opt_err
        except Exception as e:
            pass

    # Phase 2: Local polish with Nelder-Mead on the vectorized soft-max
    if time.time() < deadline - 5.0:
        def local_obj(params):
            n_nodes = params[:8]
            n_weights = params[8:]
            if np.any(n_nodes <= 0.005) or np.any(n_nodes >= 0.995):
                return 1e10
            if not np.all(np.diff(n_nodes) > 1e-4):
                return 1e10
            if np.any(n_weights <= 0):
                return 1e10
            w = n_weights / np.sum(n_weights)
            return soft_max_error(n_nodes, w, beta=100.0)

        x0 = np.concatenate([best_nodes, best_weights])
        try:
            result = optimize.minimize(
                local_obj,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            if result.fun < 1e-8:
                opt_nodes = np.sort(result.x[:8])
                opt_weights = result.x[8:]
                if np.all(opt_nodes > 0.005) and np.all(opt_nodes < 0.995) and np.all(np.diff(opt_nodes) > 1e-4):
                    if np.all(opt_weights > 0):
                        opt_weights = opt_weights / np.sum(opt_weights)
                        opt_err = vectorized_eval(opt_nodes, opt_weights)
                        if opt_err < best_err:
                            best_nodes = opt_nodes
                            best_weights = opt_weights
                            best_err = opt_err
        except:
            pass

    # Phase 3: Fine local perturbations
    if time.time() < deadline - 2.0:
        np.random.seed(123)
        scales = [0.002, 0.001, 0.0005, 0.0002]
        for scale in scales:
            if time.time() > deadline - 2.0:
                break
            for trial in range(80):
                if time.time() > deadline - 2.0:
                    break
                pert_nodes = best_nodes + np.random.normal(0, scale, size=8)
                pert_weights = best_weights + np.random.normal(0, scale * 0.5, size=8)
                pert_nodes = np.clip(pert_nodes, 0.005, 0.995)
                pert_nodes = np.sort(pert_nodes)
                if not np.all(np.diff(pert_nodes) > 1e-4):
                    continue
                pert_weights = np.clip(pert_weights, 1e-5, None)
                pert_weights = pert_weights / np.sum(pert_weights)
                err = vectorized_eval(pert_nodes, pert_weights)
                if err < best_err:
                    best_nodes = pert_nodes
                    best_weights = pert_weights
                    best_err = err

    # Final validation and cleanup
    best_nodes = np.sort(best_nodes)
    best_weights = best_weights / np.sum(best_weights)
    assert len(best_nodes) == 8
    assert len(best_weights) == 8
    assert np.all(best_nodes > 0) and np.all(best_nodes < 1)
    assert np.all(np.diff(best_nodes) > 0)
    assert np.all(best_weights > 0)
    assert abs(np.sum(best_weights) - 1.0) < 1e-12

    return {"nodes": best_nodes, "weights": best_weights}
# EVOLVE-BLOCK-END
