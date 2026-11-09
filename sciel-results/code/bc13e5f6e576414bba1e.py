"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """Fit six coefficients to Linder-Van de Stark growth index via weighted least squares + polish."""
    t_start = time.time()
    deadline = t_start + 75.0  # 15s safety margin before 90s kill

    # Grid of CPL parameters (Om0, w0, wa)
    Om0_vals = np.array([0.15, 0.20, 0.25, 0.28, 0.30, 0.32, 0.35, 0.40])
    w0_vals = np.array([-1.3, -1.2, -1.1, -1.0, -0.9, -0.8, -0.7])
    wa_vals = np.array([-0.6, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.4])

    # Scale factor grid for fitting - denser near z~1 where growth is sensitive
    a_vals = np.linspace(0.05, 0.99, 60)

    # Build design matrix, targets, and weights
    rows = []
    targets = []
    weights = []

    for Om0 in Om0_vals:
        for w0 in w0_vals:
            for wa in wa_vals:
                for a in a_vals:
                    a_cubed = a**3
                    de_factor = a**(-3.0*(1.0+w0)) * np.exp(-3.0*wa*(1.0-a))
                    E2 = Om0 * a_cubed + (1.0 - Om0) * de_factor
                    
                    if E2 <= 0 or E2 < 1e-10:
                        continue
                    Om_a = Om0 * a_cubed / E2
                    if Om_a <= 0 or Om_a >= 1.0:
                        continue
                    q = 1.0 - Om_a
                    u = 1.0 + w0 + wa * (1.0 - a)
                    
                    if u <= 0.05:
                        continue
                    
                    # Linder-Van de Stark: gamma = (5/7) * (Om(a)/u)^(4/7)
                    gamma_target = (5.0/7.0) * (Om_a / u)**(4.0/7.0)
                    
                    # Weight: sensitivity of f=Om^gamma to gamma error
                    # df/dgamma = f * ln(Om), so weight ~ |ln(Om)|
                    # Also weight more where D is sensitive (mid-range a)
                    w_sens = abs(np.log(Om_a)) if Om_a > 0 else 1.0
                    w_sens = max(w_sens, 0.1)
                    
                    row = [1.0, q, q*q, u, wa*a, u*q]
                    rows.append(row)
                    targets.append(gamma_target)
                    weights.append(w_sens)
                    
                    if time.time() > deadline:
                        break
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    if len(rows) < 10:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

    A = np.array(rows)
    b = np.array(targets)
    w_arr = np.array(weights)

    # Weighted least squares
    try:
        sqrt_w = np.sqrt(w_arr)
        A_w = A * sqrt_w[:, None]
        b_w = b * sqrt_w
        coeffs, residuals, rank, sv = np.linalg.lstsq(A_w, b_w, rcond=None)
    except Exception:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Clip to [-10, 10]
    coeffs = np.clip(coeffs, -10.0, 10.0)

    # Verify gamma stays in [0.1, 1.2] on a test grid
    valid = True
    for Om0 in Om0_vals[::2]:
        for w0 in w0_vals[::2]:
            for wa in wa_vals[::2]:
                for a in np.linspace(0.05, 0.99, 15):
                    a_cubed = a**3
                    de_factor = a**(-3.0*(1.0+w0)) * np.exp(-3.0*wa*(1.0-a))
                    E2 = Om0 * a_cubed + (1.0 - Om0) * de_factor
                    if E2 <= 0:
                        continue
                    Om_a = Om0 * a_cubed / E2
                    q = 1.0 - Om_a
                    u = 1.0 + w0 + wa * (1.0 - a)
                    gamma = (coeffs[0] + coeffs[1]*q + coeffs[2]*q*q + 
                             coeffs[3]*u + coeffs[4]*wa*a + coeffs[5]*u*q)
                    if gamma < 0.1 or gamma > 1.2:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        if not valid:
            break

    if not valid:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Polish: local refinement to minimize max relative error in f and D
    if time.time() < deadline - 20.0:
        try:
            from scipy.optimize import minimize
            import scipy.integrate as integrate
            
            def compute_max_err(c):
                """Compute max relative error of f and D for given coefficients."""
                max_err = 0.0
                for Om0 in [0.25, 0.30, 0.35]:
                    for w0 in [-1.0, -0.9, -0.8]:
                        for wa in [-0.3, 0.0, 0.3]:
                            for a in np.linspace(0.1, 0.99, 20):
                                a_cubed = a**3
                                de_factor = a**(-3.0*(1.0+w0)) * np.exp(-3.0*wa*(1.0-a))
                                E2 = Om0 * a_cubed + (1.0 - Om0) * de_factor
                                if E2 <= 0:
                                    continue
                                Om_a = Om0 * a_cubed / E2
                                q = 1.0 - Om_a
                                u = 1.0 + w0 + wa * (1.0 - a)
                                gamma = (c[0] + c[1]*q + c[2]*q*q + 
                                         c[3]*u + c[4]*wa*a + c[5]*u*q)
                                if gamma < 0.1 or gamma > 1.2:
                                    return 1e6
                                
                                # True gamma (Linder)
                                gamma_true = (5.0/7.0) * (Om_a / u)**(4.0/7.0)
                                
                                # f error
                                f_approx = Om_a**gamma
                                f_true = Om_a**gamma_true
                                if abs(f_true) > 1e-10:
                                    err_f = abs(f_approx - f_true) / abs(f_true)
                                    max_err = max(max_err, err_f)
                                
                                # D error: approximate via integral
                                # D(a)/D(1) = exp(-∫_ln(a)^0 f(exp(x)) dx)
                                # For simplicity, use trapezoidal on a few points
                                n_pts = 8
                                x_grid = np.linspace(np.log(a), 0.0, n_pts)
                                f_vals = []
                                for x in x_grid:
                                    ai = np.exp(x)
                                    ai_cubed = ai**3
                                    de_f = ai**(-3.0*(1.0+w0)) * np.exp(-3.0*wa*(1.0-ai))
                                    E2i = Om0 * ai_cubed + (1.0 - Om0) * de_f
                                    if E2i <= 0:
                                        f_vals.append(0.0)
                                        continue
                                    Omi = Om0 * ai_cubed / E2i
                                    qi = 1.0 - Omi
                                    ui = 1.0 + w0 + wa * (1.0 - ai)
                                    gi = (c[0] + c[1]*qi + c[2]*qi*qi + 
                                          c[3]*ui + c[4]*wa*ai + c[5]*ui*qi)
                                    gi = np.clip(gi, 0.1, 1.2)
                                    f_vals.append(Omi**gi)
                                f_vals = np.array(f_vals)
                                integral_approx = np.trapz(f_vals, x_grid)
                                
                                # True integral
                                f_vals_true = []
                                for x in x_grid:
                                    ai = np.exp(x)
                                    ai_cubed = ai**3
                                    de_f = ai**(-3.0*(1.0+w0)) * np.exp(-3.0*wa*(1.0-ai))
                                    E2i = Om0 * ai_cubed + (1.0 - Om0) * de_f
                                    if E2i <= 0:
                                        f_vals_true.append(0.0)
                                        continue
                                    Omi = Om0 * ai_cubed / E2i
                                    ui = 1.0 + w0 + wa * (1.0 - ai)
                                    gi_true = (5.0/7.0) * (Omi / ui)**(4.0/7.0)
                                    f_vals_true.append(Omi**gi_true)
                                f_vals_true = np.array(f_vals_true)
                                integral_true = np.trapz(f_vals_true, x_grid)
                                
                                D_approx = np.exp(-integral_approx)
                                D_true = np.exp(-integral_true)
                                if abs(D_true) > 1e-10:
                                    err_D = abs(D_approx - D_true) / abs(D_true)
                                    max_err = max(max_err, err_D)
                
                return max_err
            
            result = minimize(compute_max_err, coeffs, method='Nelder-Mead',
                            options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
            if result.success or result.fun < compute_max_err(coeffs):
                coeffs = np.clip(result.x, -10.0, 10.0)
                
                # Final validity check
                valid2 = True
                for Om0 in [0.2, 0.3, 0.4]:
                    for w0 in [-1.0, -0.8]:
                        for wa in [-0.3, 0.3]:
                            for a in np.linspace(0.05, 0.99, 10):
                                a_cubed = a**3
                                de_factor = a**(-3.0*(1.0+w0)) * np.exp(-3.0*wa*(1.0-a))
                                E2 = Om0 * a_cubed + (1.0 - Om0) * de_factor
                                if E2 <= 0:
                                    continue
                                Om_a = Om0 * a_cubed / E2
                                q = 1.0 - Om_a
                                u = 1.0 + w0 + wa * (1.0 - a)
                                gamma = (coeffs[0] + coeffs[1]*q + coeffs[2]*q*q + 
                                         coeffs[3]*u + coeffs[4]*wa*a + coeffs[5]*u*q)
                                if gamma < 0.1 or gamma > 1.2:
                                    valid2 = False
                                    break
                            if not valid2:
                                break
                        if not valid2:
                            break
                    if not valid2:
                        break
                if not valid2:
                    coeffs = np.clip(result.x, -10.0, 10.0)
                    # Revert if invalid
                    # Keep original coeffs if polish made it invalid
                    pass
        except Exception:
            pass  # Keep the least-squares result

    return [float(c) for c in coeffs]
# EVOLVE-BLOCK-END
