"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import least_squares, minimize
import time
from scipy.integrate import solve_ivp

def construct():
    """
    Fit six coefficients for the growth-index formula using spectral pre-conditioning:
    1. Build orthonormal basis of ansatz terms over dense grid
    2. Project true gamma onto basis for initial coefficients
    3. Refine with min-max optimization on f and D errors
    """
    start_time = time.time()
    deadline = start_time + 78  # Safety margin before 90s kill

    # Dense grid for basis construction
    w0_vals = np.linspace(-1.5, -0.5, 25)
    wa_vals = np.linspace(-1.0, 1.0, 25)
    Om0_vals = np.array([0.20, 0.25, 0.27, 0.30, 0.35])
    a_vals = np.linspace(0.05, 1.0, 200)

    # Generate reference data on dense grid
    all_data = []
    for Om0 in Om0_vals:
        for w0 in w0_vals:
            for wa in wa_vals:
                try:
                    f_ref, D_ref = _solve_growth_ode_fast(Om0, w0, wa, a_vals)
                    if f_ref is None:
                        continue
                    for a, f_r, D_r in zip(a_vals, f_ref, D_ref):
                        Om_a = _compute_Om_a(Om0, w0, wa, a)
                        q = 1 - Om_a
                        u = 1 + w0 + wa * (1 - a)
                        if np.isfinite(f_r) and f_r > 0 and np.isfinite(D_r) and D_r > 0:
                            all_data.append((Om_a, q, u, wa, a, f_r, D_r))
                except Exception:
                    continue

    if len(all_data) == 0:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Convert to arrays
    Om_a_arr = np.array([d[0] for d in all_data])
    q_arr = np.array([d[1] for d in all_data])
    u_arr = np.array([d[2] for d in all_data])
    wa_arr = np.array([d[3] for d in all_data])
    a_arr = np.array([d[4] for d in all_data])
    f_ref_arr = np.array([d[5] for d in all_data])
    D_ref_arr = np.array([d[6] for d in all_data])

    # True growth index: gamma_true = ln(f) / ln(Om_a) where Om_a != 1
    valid_om = (Om_a_arr > 0.01) & (Om_a_arr < 0.99)
    gamma_true = np.zeros_like(f_ref_arr)
    gamma_true[valid_om] = np.log(f_ref_arr[valid_om]) / np.log(Om_a_arr[valid_om])
    # For points where Om_a is near 1, use constant gamma
    gamma_true[~valid_om] = 0.55

    # Build design matrix for the six ansatz terms
    # gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q
    X = np.column_stack([
        np.ones_like(q_arr),
        q_arr,
        q_arr**2,
        u_arr,
        wa_arr * a_arr,
        u_arr * q_arr
    ])

    # Spectral pre-conditioning: orthonormalize basis via QR decomposition
    # This gives us coefficients that minimize L2 error in gamma space
    Q, R = np.linalg.qr(X)
    # Project gamma_true onto orthonormal basis
    coeffs_q = Q.T @ gamma_true
    # Transform back to original coefficient space
    # X @ c = Q @ coeffs_q => c = R^{-1} @ coeffs_q (since X = Q @ R)
    try:
        c0_init = np.linalg.solve(R, coeffs_q)
        c0_init = np.clip(c0_init, -10, 10)
    except np.linalg.LinAlgError:
        c0_init = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])

    # Also try a few other initializations
    initial_points = [
        c0_init,
        np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.5, 0.1, 0.0, 0.1, 0.0, 0.0]),
        np.array([0.6, -0.1, 0.0, 0.0, 0.1, 0.0]),
        c0_init + 0.1 * np.random.randn(6),
        c0_init - 0.1 * np.random.randn(6),
    ]

    # Downsample for optimization (use subset for speed)
    n_total = len(Om_a_arr)
    if n_total > 5000:
        idx = np.random.choice(n_total, 5000, replace=False)
        Om_a_opt = Om_a_arr[idx]
        q_opt = q_arr[idx]
        u_opt = u_arr[idx]
        wa_opt = wa_arr[idx]
        a_opt = a_arr[idx]
        f_ref_opt = f_ref_arr[idx]
        D_ref_opt = D_ref_arr[idx]
    else:
        Om_a_opt = Om_a_arr
        q_opt = q_arr
        u_opt = u_arr
        wa_opt = wa_arr
        a_opt = a_arr
        f_ref_opt = f_ref_arr
        D_ref_opt = D_ref_arr

    def compute_max_error(params):
        """Compute maximum relative error of f and D."""
        c0, c1, c2, c3, c4, c5 = params
        gamma = c0 + c1 * q_opt + c2 * q_opt**2 + c3 * u_opt + c4 * wa_opt * a_opt + c5 * u_opt * q_opt

        # Check gamma bounds
        if np.any(gamma < 0.1) or np.any(gamma > 1.2):
            penalty = np.sum(np.maximum(0.1 - gamma, 0)) + np.sum(np.maximum(gamma - 1.2, 0))
            return 10.0 + penalty

        f_model = Om_a_opt ** gamma

        # Relative error for f
        rel_err_f = np.abs(f_model - f_ref_opt) / np.maximum(f_ref_opt, 1e-10)

        # Compute D from model
        D_model = _compute_D_from_f(f_model, a_opt)
        rel_err_D = np.abs(D_model - D_ref_opt) / np.maximum(D_ref_opt, 1e-10)

        return np.maximum(np.max(rel_err_f), np.max(rel_err_D))

    def compute_error_vector(params):
        """Return error vector for least_squares."""
        c0, c1, c2, c3, c4, c5 = params
        gamma = c0 + c1 * q_opt + c2 * q_opt**2 + c3 * u_opt + c4 * wa_opt * a_opt + c5 * u_opt * q_opt

        if np.any(gamma < 0.1) or np.any(gamma > 1.2):
            return np.full(20, 10.0)

        f_model = Om_a_opt ** gamma
        rel_err_f = np.abs(f_model - f_ref_opt) / np.maximum(f_ref_opt, 1e-10)

        D_model = _compute_D_from_f(f_model, a_opt)
        rel_err_D = np.abs(D_model - D_ref_opt) / np.maximum(D_ref_opt, 1e-10)

        # Subsample for least_squares
        n = len(rel_err_f)
        if n > 200:
            idx = np.random.choice(n, 200, replace=False)
            return np.concatenate([rel_err_f[idx], rel_err_D[idx]])
        return np.concatenate([rel_err_f, rel_err_D])

    best_params = c0_init
    best_score = compute_max_error(c0_init)

    # Phase 1: least_squares from multiple initial points
    bounds = ([-10, -10, -10, -10, -10, -10], [10, 10, 10, 10, 10, 10])

    for i, x_init in enumerate(initial_points):
        if time.time() > deadline:
            break
        try:
            result = least_squares(compute_error_vector, x_init, bounds=bounds,
                                   max_nfev=500, ftol=1e-10, xtol=1e-10)
            score = compute_max_error(result.x)
            if score < best_score:
                best_score = score
                best_params = result.x
        except Exception:
            continue

    # Phase 2: min-max refinement using SLSQP
    def max_error_obj(params):
        return compute_max_error(params)

    for i in range(3):
        if time.time() > deadline:
            break
        try:
            result = minimize(max_error_obj, best_params, method='SLSQP',
                             bounds=[(-10, 10)] * 6,
                             options={'maxiter': 200, 'ftol': 1e-12})
            if result.success:
                score = compute_max_error(result.x)
                if score < best_score:
                    best_score = score
                    best_params = result.x
        except Exception:
            continue

    # Phase 3: local refinement with small perturbations
    if time.time() < deadline - 10:
        for _ in range(20):
            if time.time() > deadline:
                break
            perturbation = 0.01 * np.random.randn(6)
            candidate = np.clip(best_params + perturbation, -10, 10)
            score = compute_max_error(candidate)
            if score < best_score:
                best_score = score
                best_params = candidate

    best_params = np.clip(best_params, -10, 10)
    return list(best_params)

def _compute_Om_a(Om0, w0, wa, a):
    """Compute matter density at scale factor a."""
    Om_m = Om0 * a**3
    w_eff = w0 + wa * (1 - a)
    Om_de = (1 - Om0) * a**(-3 * w_eff)
    return Om_m / (Om_m + Om_de)

def _solve_growth_ode_fast(Om0, w0, wa, a_values):
    """Solve the growth ODE efficiently."""
    a_min = min(a_values)
    a_max = max(a_values)
    x_min = np.log(a_min)
    x_max = np.log(a_max)

    def ode_system(x, y):
        D, Dp = y
        a = np.exp(x)
        Om_m = _compute_Om_a(Om0, w0, wa, a)
        coeff = (3/2) * (1 + Om_m / (1 - Om_m))
        return [Dp, -coeff * Dp + (3/2) * Om_m * D]

    x_eval = np.linspace(x_max, x_min, len(a_values))
    sol = solve_ivp(ode_system, [x_max, x_min], [1.0, 0.0], t_eval=x_eval,
                    rtol=1e-6, atol=1e-8, method='RK45')

    if not sol.success:
        return None, None

    D_vals = sol.y[0]
    Dp_vals = sol.y[1]

    # Check for invalid values
    if np.any(D_vals <= 0) or np.any(~np.isfinite(D_vals)):
        return None, None

    f_vals = Dp_vals / D_vals
    D_vals = D_vals / D_vals[0]  # Normalize D(1) = 1

    if np.any(~np.isfinite(f_vals)) or np.any(~np.isfinite(D_vals)):
        return None, None

    return f_vals, D_vals

def _compute_D_from_f(f_vals, a_vals):
    """Compute D(a)/D(1) from f values using numerical integration."""
    n = len(a_vals)
    D_vals = np.zeros(n)
    x_vals = np.log(a_vals)

    for i in range(n):
        if a_vals[i] >= 1.0:
            D_vals[i] = 1.0
            continue
        # Integrate from ln(a_i) to 0
        mask = x_vals >= x_vals[i]
        if np.sum(mask) < 2:
            D_vals[i] = 1.0
            continue
        x_int = x_vals[mask]
        f_int = f_vals[mask]
        # Ensure we reach x=0
        if x_int[-1] < 0:
            x_int = np.append(x_int, 0.0)
            f_int = np.append(f_int, f_vals[-1])
        integral = np.trapz(f_int, x_int)
        D_vals[i] = np.exp(-integral)

    return D_vals
# EVOLVE-BLOCK-END
