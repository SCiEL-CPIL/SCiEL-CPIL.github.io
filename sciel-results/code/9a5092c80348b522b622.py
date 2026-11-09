"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_Om_a(a, w0, wa, Om0=0.3):
    """Compute Omega_m(a) for flat CPL universe."""
    u = 1.0 + w0 + wa * (1.0 - a)
    if u <= 0.01:
        return None
    Om_de0 = 1.0 - Om0
    denom = Om0 * a**3 + Om_de0 / u
    if denom <= 0:
        return None
    Om_a = Om0 * a**3 / denom
    if Om_a <= 0 or Om_a >= 1:
        return None
    return Om_a

def _gamma_from_coeffs(coeffs, q, u, wa, a):
    """Compute gamma from coefficients."""
    c = coeffs
    return c[0] + c[1]*q + c[2]*q**2 + c[3]*u + c[4]*wa*a + c[5]*u*q

def _true_gamma_ode(a, w0, wa, Om0=0.3):
    """Compute true gamma = d ln f / d ln a from the ODE solution."""
    from scipy.integrate import solve_ivp
    
    u = 1.0 + w0 + wa * (1.0 - a)
    if u <= 0.05:
        return None
    
    Om_de0 = 1.0 - Om0
    
    # Solve ODE for D(a): D'' + (3/a + H'/H)D' - (3/2)Om(a)/a^2 D = 0
    # In terms of x = ln a: d2D/dx2 + (3/2 + d ln H/dx) dD/dx - (3/2) Om(e^x) = 0
    # where d ln H/dx = (1/2)(3 Om(e^x) - 3) - (3/2) w(e^x)
    # Actually: H^2(a) = H0^2 [Om_m/a^3 + Om_de/u(a)]
    # d ln H / d ln a = (1/(2H^2)) dH^2/d ln a
    # dH^2/d ln a = H0^2 [-3 Om_m/a^3 + Om_de * w(a)/a]  (since du/d ln a = wa * (-1) * ... )
    # Let me be more careful.
    # H^2(a) = H0^2 [Om0/a^3 + (1-Om0)/u(a)]
    # d/da [Om0/a^3] = -3 Om0/a^4
    # d/da [(1-Om0)/u] = -(1-Om0)/u^2 * du/da
    # du/da = -wa
    # So d/da [(1-Om0)/u] = (1-Om0)*wa/u^2
    # dH^2/da = H0^2 [-3 Om0/a^4 + (1-Om0)*wa/u^2]
    # d ln H / da = (1/(2H^2)) * dH^2/da = [-3 Om0/a^4 + (1-Om0)*wa/u^2] / [2(Om0/a^3 + (1-Om0)/u)]
    # d ln H / d ln a = a * d ln H / da = [-3 Om0/a^3 + (1-Om0)*wa/u^2 * a] / [2(Om0/a^3 + (1-Om0)/u)]
    # Hmm, let me use x = ln a, so a = e^x
    # H^2(x) = H0^2 [Om0 e^{-3x} + (1-Om0)/u(e^x)]
    # u(a) = 1 + w0 + wa(1-a) = 1 + w0 + wa(1-e^x)
    # du/dx = du/da * da/dx = -wa * e^x
    # dH^2/dx = H0^2 [-3 Om0 e^{-3x} - (1-Om0)/u^2 * du/dx]
    #          = H0^2 [-3 Om0 e^{-3x} + (1-Om0)*wa*e^x/u^2]
    # d ln H/dx = (1/(2H^2)) dH^2/dx
    #           = [-3 Om0 e^{-3x} + (1-Om0)*wa*e^x/u^2] / [2(Om0 e^{-3x} + (1-Om0)/u)]
    
    def ode_system(x, y):
        a_x = np.exp(x)
        u_x = 1.0 + w0 + wa * (1.0 - a_x)
        if u_x <= 0.05:
            return [0, 0]
        Om_x = Om0 * a_x**3 / (Om0 * a_x**3 + (1-Om0)/u_x)
        # d ln H/dx
        num = -3 * Om0 * a_x**(-3) + (1-Om0) * wa * a_x / u_x**2
        den = 2 * (Om0 * a_x**(-3) + (1-Om0)/u_x)
        dlnH_dx = num / den if den != 0 else 0
        # ODE: D'' + (3/2 + dlnH_dx) D' - (3/2) Om_x = 0
        # y[0] = D, y[1] = D'
        D_pp = -(3.0/2.0 + dlnH_dx) * y[1] + (3.0/2.0) * Om_x
        return [y[1], D_pp]
    
    # Solve from x=0 (a=1) backwards to x = ln(a)
    x_target = np.log(a)
    # Initial conditions at a=1: D(1)=1, D'(1) = f(1) = Om0^gamma... but we need to get f(1) from the ODE
    # Actually for the growth rate, we need the solution normalized.
    # Let's solve with D(1)=1, D'(1)=f0 where f0 is unknown, then determine f0 from the condition
    # that D is the growing mode.
    # Better: solve the ODE numerically and extract gamma = d ln f / d ln a where f = D'/D * a... 
    # Wait, f = d ln D / d ln a = a * D'/D
    # So gamma = d ln f / d ln a = d ln (a D'/D) / d ln a
    
    # Solve from x=0 to x_target
    if abs(x_target) < 1e-10:
        return 0.55  # fallback
    
    # Use two initial conditions and find the growing mode
    # At x=0: D(0)=1, D'(0)=f_0
    # f_0 = Om0^gamma_0 where gamma_0 ~ 5/7 for EdS
    # Let's just solve with D(0)=1, D'(0)=0.55 and compute f(x) = exp(x)*D'(x)/D(x)
    # Then gamma = d ln f / d ln x... no, d ln f / d x
    
    x_span = [0, x_target]
    x_eval = np.linspace(0, x_target, 50)
    
    sol = solve_ivp(ode_system, [0, x_target], [1.0, 0.55], t_eval=x_eval, 
                    method='RK45', rtol=1e-10, atol=1e-12, dense_output=True)
    
    if not sol.success:
        return None
    
    # f(x) = exp(x) * D'(x) / D(x)
    D_vals = sol.y[0]
    Dp_vals = sol.y[1]
    
    # Avoid division by zero
    mask = np.abs(D_vals) > 1e-15
    if np.sum(mask) < 5:
        return None
    
    f_vals = np.exp(sol.t[mask]) * Dp_vals[mask] / D_vals[mask]
    x_masked = sol.t[mask]
    
    # gamma = d ln f / d x = d ln f / d ln a
    ln_f = np.log(f_vals)
    
    # Numerical derivative
    if len(ln_f) < 3:
        return None
    
    # Use central differences
    gamma_vals = np.gradient(ln_f, x_masked)
    
    # Return gamma at the target x
    idx = np.argmin(np.abs(x_masked - x_target))
    g = gamma_vals[idx]
    
    if not np.isfinite(g) or g < 0.05 or g > 1.3:
        return None
    
    return g

def _fit_coefficients():
    """Fit the six growth-index coefficients using true ODE-derived gamma targets."""
    t_start = time.time()
    deadline = 80.0  # Safety margin before 90s kill
    
    # Grid of CPL parameters and scale factors
    w0_vals = np.linspace(-1.2, -0.8, 15)
    wa_vals = np.linspace(-0.5, 0.5, 15)
    a_vals = np.linspace(0.3, 1.0, 30)
    
    Omega_m0 = 0.3
    
    # Build design matrix and true gamma targets
    rows = []
    targets = []
    valid_points = []
    
    for w0 in w0_vals:
        for wa in wa_vals:
            for a in a_vals:
                if time.time() - t_start > deadline:
                    break
                
                u = 1.0 + w0 + wa * (1.0 - a)
                if u <= 0.05:
                    continue
                
                Om_a = _compute_Om_a(a, w0, wa, Omega_m0)
                if Om_a is None:
                    continue
                
                # Get true gamma from ODE
                true_g = _true_gamma_ode(a, w0, wa, Omega_m0)
                if true_g is None:
                    continue
                
                q = 1.0 - Om_a
                row = np.array([1.0, q, q**2, u, wa * a, u * q])
                rows.append(row)
                targets.append(true_g)
                valid_points.append((w0, wa, a))
            
            if time.time() - t_start > deadline:
                break
        if time.time() - t_start > deadline:
            break
    
    if len(rows) < 20:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    A = np.array(rows)
    b = np.array(targets)
    
    # Solve least squares
    coeffs, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
    coeffs = np.clip(coeffs, -10.0, 10.0)
    
    # Verify gamma stays in [0.1, 1.2]
    def _check_gamma_range(coeffs_to_check):
        for w0 in w0_vals:
            for wa in wa_vals:
                for a in a_vals:
                    u = 1.0 + w0 + wa * (1.0 - a)
                    if u <= 0.05:
                        continue
                    Om_a = _compute_Om_a(a, w0, wa, Omega_m0)
                    if Om_a is None:
                        continue
                    q = 1.0 - Om_a
                    gamma = _gamma_from_coeffs(coeffs_to_check, q, u, wa, a)
                    if gamma < 0.1 or gamma > 1.2:
                        return False
        return True
    
    if not _check_gamma_range(coeffs):
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Refinement: minimize worst-case error with soft-max
    if time.time() - t_start < deadline - 10:
        n_eval = len(rows)
        
        def soft_max_obj(coeffs_vec, temperature=0.005):
            if time.time() - t_start > deadline:
                return 1e10, None
            gamma_pred = A @ coeffs_vec
            errors = np.abs(gamma_pred - b)
            gamma_min = np.min(gamma_pred)
            gamma_max = np.max(gamma_pred)
            penalty = 0.0
            if gamma_min < 0.1:
                penalty += (0.1 - gamma_min) ** 2 * 5000
            if gamma_max > 1.2:
                penalty += (gamma_max - 1.2) ** 2 * 5000
            scaled = errors / temperature
            max_s = np.max(scaled)
            lse = max_s + np.log(np.sum(np.exp(scaled - max_s)))
            return lse * temperature + penalty, None
        
        def soft_max_grad(coeffs_vec, temperature=0.005):
            if time.time() - t_start > deadline:
                return None
            gamma_pred = A @ coeffs_vec
            errors = np.abs(gamma_pred - b)
            gamma_min = np.min(gamma_pred)
            gamma_max = np.max(gamma_pred)
            scaled = errors / temperature
            max_s = np.max(scaled)
            exp_v = np.exp(scaled - max_s)
            weights = exp_v / np.sum(exp_v)
            signs = np.sign(gamma_pred - b)
            grad = A.T @ (weights * signs)
            if gamma_min < 0.1:
                viol = (gamma_pred < 0.1).astype(float)
                grad += 2.0 * 5000 * (0.1 - gamma_min) * A.T @ viol / n_eval
            if gamma_max > 1.2:
                viol = (gamma_pred > 1.2).astype(float)
                grad += 2.0 * 5000 * (gamma_max - 1.2) * A.T @ viol / n_eval
            return grad
        
        bounds = [(-10, 10)] * 6
        
        try:
            result = minimize(soft_max_obj, coeffs, method='L-BFGS-B', bounds=bounds,
                             jac=lambda x: soft_max_grad(x),
                             options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10})
            if result.success and result.fun < 1e9:
                refined = np.clip(result.x, -10.0, 10.0)
                if _check_gamma_range(refined):
                    coeffs = refined
        except Exception:
            pass
        
        # Polish with tighter temperature
        if time.time() - t_start < deadline - 3:
            try:
                def polish_obj(coeffs_vec, temperature=0.001):
                    if time.time() - t_start > deadline:
                        return 1e10, None
                    gamma_pred = A @ coeffs_vec
                    errors = np.abs(gamma_pred - b)
                    gamma_min = np.min(gamma_pred)
                    gamma_max = np.max(gamma_pred)
                    penalty = 0.0
                    if gamma_min < 0.1:
                        penalty += (0.1 - gamma_min) ** 2 * 20000
                    if gamma_max > 1.2:
                        penalty += (gamma_max - 1.2) ** 2 * 20000
                    scaled = errors / temperature
                    max_s = np.max(scaled)
                    lse = max_s + np.log(np.sum(np.exp(scaled - max_s)))
                    return lse * temperature + penalty, None
                
                def polish_grad(coeffs_vec, temperature=0.001):
                    if time.time() - t_start > deadline:
                        return None
                    gamma_pred = A @ coeffs_vec
                    errors = np.abs(gamma_pred - b)
                    gamma_min = np.min(gamma_pred)
                    gamma_max = np.max(gamma_pred)
                    scaled = errors / temperature
                    max_s = np.max(scaled)
                    exp_v = np.exp(scaled - max_s)
                    weights = exp_v / np.sum(exp_v)
                    signs = np.sign(gamma_pred - b)
                    grad = A.T @ (weights * signs)
                    if gamma_min < 0.1:
                        viol = (gamma_pred < 0.1).astype(float)
                        grad += 2.0 * 20000 * (0.1 - gamma_min) * A.T @ viol / n_eval
                    if gamma_max > 1.2:
                        viol = (gamma_pred > 1.2).astype(float)
                        grad += 2.0 * 20000 * (gamma_max - 1.2) * A.T @ viol / n_eval
                    return grad
                
                result2 = minimize(polish_obj, coeffs, method='L-BFGS-B', bounds=bounds,
                                  jac=lambda x: polish_grad(x),
                                  options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12})
                if result2.success and result2.fun < 1e9:
                    refined2 = np.clip(result2.x, -10.0, 10.0)
                    if _check_gamma_range(refined2):
                        coeffs = refined2
            except Exception:
                pass
    
    if not _check_gamma_range(coeffs):
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    return coeffs.tolist()

def construct():
    """Construct the six growth-index coefficients."""
    t_start = time.time()
    
    try:
        coeffs = _fit_coefficients()
    except Exception:
        coeffs = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    for i in range(6):
        if not np.isfinite(coeffs[i]) or coeffs[i] < -10 or coeffs[i] > 10:
            coeffs = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
            break
    
    return coeffs
# EVOLVE-BLOCK-END
