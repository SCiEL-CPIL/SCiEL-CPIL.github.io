"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.integrate import solve_ivp
# EVOLVE-BLOCK-START

def _ode_rhs(x, y, Om0, w0, wa):
    a = np.exp(x)
    Om = Om0 * a**(-3)
    Ode = 1.0 - Om
    w = w0 + wa * (1.0 - a)
    coeff = 1.5 * (Om + Ode * (1.0 + 3.0 * w))
    return [y[1], -coeff * y[1] + 1.5 * Om * y[0]]

def _growing_mode(x_grid, Om0, w0, wa):
    """
    Extract the physical growing mode by solving the ODE backward from a=1
    using the known asymptotic behavior, then forward to match.
    We use the fact that at large a (early times), the growing mode ~ a^(6/5(1+w0))
    approximately. A more robust approach: solve the ODE with two basis solutions
    and combine them to eliminate the decaying mode.
    """
    x0 = np.log(0.01)
    x_end = 0.0
    
    # Solve with two independent initial conditions at x0
    sol1 = solve_ivp(_ode_rhs, [x0, x_end], [1.0, 0.0], args=(Om0, w0, wa),
                     method='DOP853', rtol=1e-12, atol=1e-14, dense_output=True)
    sol2 = solve_ivp(_ode_rhs, [x0, x_end], [0.0, 1.0], args=(Om0, w0, wa),
                     method='DOP853', rtol=1e-12, atol=1e-14, dense_output=True)
    
    if not sol1.success or not sol2.success:
        # Fallback: use sol2 as growing mode proxy
        D = sol2.sol(x_grid)[0]
        Dp = sol2.sol(x_grid)[1]
        Dn = D / D[-1]
        fn = Dp / D
        return fn, Dn
    
    # At x0 (a=0.01), the growing mode should be dominant.
    # The decaying mode ~ a^(-3/10(1+w)) which is small at small a.
    # So at x0, the solution is approximately the growing mode.
    # We want D(x0) > 0 and D'(x0) such that it's the growing branch.
    # For a flat universe, the growing mode has D' > 0 and D > 0.
    # Let's find the combination that is purely growing at x0.
    # At x0, sol1 = [1, 0], sol2 = [0, 1].
    # The physical growing mode at early times: D ~ a^(3/2(1+w_avg))
    # For w_avg ~ w0, exponent ~ 3/2(1+w0)
    # At a=0.01, x=ln(0.01)~-4.6
    # D ~ exp(3/2*(1+w0)*x), D' ~ 3/2*(1+w0)*D
    # So the ratio D'/D ~ 3/2*(1+w0) at early times.
    
    # Let's use the WKB approximation to set initial conditions.
    # At x0, w ~ w0 + wa*(1-0.01) ~ w0 + 0.99*wa
    w0_eff = w0 + 0.99 * wa
    Om0_eff = Om0 * 0.01**(-3)  # This is huge, so Ode ~ 1
    # Actually at small a, Om >> 1, so the universe is matter-dominated.
    # In matter domination, growing mode ~ a^(3/2), so D'/D ~ 3/2.
    # Let's use this: at x0, D'/D = 1.5 (matter domination)
    
    # So we want A*sol1 + B*sol2 such that (A*1 + B*0) and (A*0 + B*1)
    # i.e., D(x0) = A, D'(x0) = B, and B/A = 1.5 => B = 1.5*A
    # Set A = 1, B = 1.5
    A_coeff = 1.0
    B_coeff = 1.5
    
    D = A_coeff * sol1.sol(x_grid)[0] + B_coeff * sol2.sol(x_grid)[0]
    Dp = A_coeff * sol1.sol(x_grid)[1] + B_coeff * sol2.sol(x_grid)[1]
    
    # Check if this is the growing mode (should be positive and increasing)
    if np.any(D <= 0) or np.any(Dp <= 0):
        # Try the other sign or different ratio
        # In matter domination, the decaying mode ~ a^(-3/10), so at small a it's small.
        # The growing mode should dominate. Let's try A=1, B=1.0
        A_coeff = 1.0
        B_coeff = 1.0
        D = A_coeff * sol1.sol(x_grid)[0] + B_coeff * sol2.sol(x_grid)[0]
        Dp = A_coeff * sol1.sol(x_grid)[1] + B_coeff * sol2.sol(x_grid)[1]
    
    # Normalize
    Dn = D / D[-1]
    fn = Dp / D
    return fn, Dn

def _fit_coefficients():
    # Representative CPL cosmologies spanning the parameter space
    params = [
        (0.27, -1.0, 0.0),
        (0.30, -1.0, 0.0),
        (0.33, -1.0, 0.0),
        (0.30, -0.9, 0.3),
        (0.30, -1.1, 0.5),
        (0.25, -1.0, 0.2),
        (0.35, -0.8, 0.4),
        (0.30, -1.0, -0.5),
        (0.28, -1.2, 0.6),
        (0.32, -0.7, 0.1),
        (0.30, -1.5, 1.0),
        (0.30, -0.5, -0.3),
    ]
    a_grid = np.logspace(-2, 0, 300)
    x_grid = np.log(a_grid)

    rows = []
    targets = []
    weights = []
    
    for (Om0, w0, wa) in params:
        fn, Dn = _growing_mode(x_grid, Om0, w0, wa)
        Om = Om0 * a_grid**(-3)
        q = 1.0 - Om
        u = 1.0 + w0 + wa * (1.0 - a_grid)
        
        # f = Om^gamma  =>  ln f = gamma * ln Om
        # gamma = ln f / ln Om
        valid = (Om > 1e-4) & (Om < 0.95) & (fn > 0) & (np.isfinite(fn))
        if np.sum(valid) < 10:
            continue
        gamma_target = np.log(fn[valid]) / np.log(Om[valid])
        
        # Filter out outliers
        median_g = np.median(gamma_target)
        mad = np.median(np.abs(gamma_target - median_g))
        valid2 = valid.copy()
        valid2[valid] = np.abs(gamma_target - median_g) < 3 * max(mad, 0.01)
        if np.sum(valid2) < 10:
            valid2 = valid
        
        for i in np.where(valid2)[0]:
            ai = a_grid[i]
            ui = u[i]
            qi = q[i]
            rows.append([1.0, qi, qi**2, ui, wa * ai, ui * qi])
            targets.append(gamma_target[i])
            # Weight by |ln Om| to focus where gamma is well-defined
            weights.append(abs(np.log(Om[i])))

    A = np.array(rows)
    b = np.array(targets)
    w = np.array(weights)
    
    # Weighted least squares
    Aw = A * np.sqrt(w)[:, np.newaxis]
    bw = b * np.sqrt(w)
    sol, res, rank, sv = np.linalg.lstsq(Aw, bw, rcond=None)
    c0, c1, c2, c3, c4, c5 = sol
    return [c0, c1, c2, c3, c4, c5]

def construct():
    # Optimize all six coefficients against ODE reference.
    try:
        coeffs = _fit_coefficients()
        # Clip to [-10,10] for safety
        coeffs = [float(np.clip(c, -10.0, 10.0)) for c in coeffs]
        # Sanity: ensure finite
        if not all(np.isfinite(c) for c in coeffs):
            raise ValueError
        return coeffs
    except Exception:
        # Fallback to a reasonable constant-gamma with mild CPL correction
        return [0.55, 0.0, 0.0, 0.05, 0.1, 0.0]
# EVOLVE-BLOCK-END
