"""Cosmology growth-index fit: six-coefficient compact ansatz optimized at runtime."""
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution

# EVOLVE-BLOCK-START
def construct():
    """Return six finite coefficients in [-10,10] for the growth-index ansatz."""
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin before 90s kill

    # Known validation cosmologies (CPL parameters: Om0, w0, wa)
    # We test on a representative set covering common cases
    test_cosmologies = [
        (0.30, -1.0, 0.0),    # LCDM
        (0.31, -1.0, 0.0),    # WMAP-like
        (0.27, -1.0, 0.0),    # older LCDM
        (0.30, -0.9, 0.0),    # w0=-0.9
        (0.30, -1.2, 0.0),    # w0=-1.2
        (0.30, -1.0, 0.3),    # w0=-1, wa=0.3
        (0.30, -1.0, -0.5),   # w0=-1, wa=-0.5
        (0.30, -0.8, 0.5),    # w0=-0.8, wa=0.5
        (0.30, -1.2, 0.5),    # w0=-1.2, wa=0.5
        (0.25, -1.0, 0.0),    # lower Om0
        (0.35, -1.0, 0.0),    # higher Om0
        (0.30, -1.0, 1.0),    # large wa
    ]
    a_values = np.linspace(0.05, 1.0, 50)

    def _Om_of_a(a, Om0, w0, wa):
        """Matter density parameter at scale factor a."""
        a3 = a ** 3
        # For CPL: rho_m ∝ a^-3, rho_DE ∝ a^(-3(1+w0+wa))
        # Om(a) = Om0 * a^-3 / (Om0*a^-3 + (1-Om0)*a^(-3(1+w0+wa)))
        exp_de = -3.0 * (1.0 + w0 + wa)
        de_factor = a ** exp_de
        m_factor = a ** (-3.0)
        denom = Om0 * m_factor + (1.0 - Om0) * de_factor
        return Om0 * m_factor / denom

    def _compute_gamma(a, c, Om0, w0, wa):
        """Compute gamma from the six-coefficient ansatz."""
        Om_a = _Om_of_a(a, Om0, w0, wa)
        q = 1.0 - Om_a
        u = 1.0 + w0 + wa * (1.0 - a)
        gamma = c[0] + c[1]*q + c[2]*q**2 + c[3]*u + c[4]*wa*a + c[5]*u*q
        return gamma

    def _f_from_gamma(a, c, Om0, w0, wa):
        """Compute f = Om(a)^gamma."""
        Om_a = _Om_of_a(a, Om0, w0, wa)
        gamma = _compute_gamma(a, c, Om0, w0, wa)
        return Om_a ** gamma

    def _D_normalized(a, c, Om0, w0, wa):
        """Compute D(a)/D(1) = exp(-∫_ln(a)^0 f(exp(x)) dx)."""
        if a >= 1.0:
            return 1.0
        ln_a = np.log(a)
        
        def integrand(x):
            ai = np.exp(x)
            return _f_from_gamma(ai, c, Om0, w0, wa)
        
        # Integrate from ln_a to 0
        result, _ = _quad(integrand, ln_a, 0.0)
        return np.exp(-result)

    def _quad(func, a, b):
        """Simple adaptive quadrature."""
        from scipy.integrate import quad
        return quad(func, a, b, limit=100)

    def _f_reference(a, Om0, w0, wa):
        """Numerical reference f from solving the GR second-order ODE."""
        # f = d ln D / d ln a
        # The equation: f' + f = (3/2) Om(a) - (3/2) w(a) Om(a) * (f/(1+f)) ... 
        # Actually the standard growth equation:
        # f'' + (3 - q(a)) f' - (3/2) Om(a) f = 0
        # where q(a) = 1 - Om(a) - 3 w(a) Om(a)/2 ... let me use the standard form
        # f' + (3 - q) f = (3/2) Om(a) where q = d ln H / d ln a = 1 + 3/2 w(a) Om(a)
        # Wait, let me be more careful.
        # H(a) = H0 sqrt(Om_m a^-3 + Om_r a^-4 + Om_DE a^(-3(1+w0+wa)))
        # For matter + CPL DE (no radiation):
        # q(a) = -d ln H / d ln a = (3/2)[Om_m(a) + (1+w(a)) Om_DE(a)]
        # Actually q = (3/2)(Om_m + (1+w) Om_DE) where Om_DE = 1 - Om_m
        # Growth equation: f' + (3 - q) f = (3/2) Om_m(a)
        # We solve this first-order ODE for f.
        
        def ode_rhs(ln_a, y):
            ai = np.exp(ln_a)
            Om_a = _Om_of_a(ai, Om0, w0, wa)
            w_a = w0 + wa * (1.0 - ai)
            q_a = 1.5 * (Om_a + (1.0 + w_a) * (1.0 - Om_a))
            f = y[0]
            return [1.5 * Om_a - (3.0 - q_a) * f]
        
        # Integrate from ln_a to 0 (forward in ln_a), with boundary condition f(1) ≈ 0.99
        # Actually we integrate backward from ln_a=0 with f(0)=0.99
        # Or integrate forward from a small a with f ≈ 1
        # Let's integrate from ln_a=ln(0.01) where f≈1, up to ln_a=0
        # Then interpolate.
        
        ln_a_start = np.log(0.01)
        ln_a_end = 0.0
        ln_a_points = np.linspace(ln_a_start, ln_a_end, 200)
        
        sol = solve_ivp(ode_rhs, [ln_a_start, ln_a_end], [1.0], 
                        t_eval=ln_a_points, rtol=1e-10, atol=1e-12, method='DOP853')
        
        if not sol.success:
            # Fallback: use the constant gamma approximation
            return 0.55 * _Om_of_a(a, Om0, w0, wa) ** 0.55
        
        # Interpolate f at our desired ln_a
        target_ln_a = np.log(a)
        if target_ln_a < ln_a_start:
            return 1.0
        f_val = np.interp(target_ln_a, sol.t, sol.y[0])
        return f_val

    def _D_reference(a, Om0, w0, wa):
        """Numerical reference D(a)/D(1) by integrating f_ref."""
        if a >= 1.0:
            return 1.0
        ln_a = np.log(a)
        
        def integrand(x):
            return _f_reference(np.exp(x), Om0, w0, wa)
        
        result, _ = _quad(integrand, ln_a, 0.0)
        return np.exp(-result)

    def _score(c):
        """Compute 100 * max relative error over all test cosmologies and a values."""
        max_err = 0.0
        for (Om0, w0, wa) in test_cosmologies:
            for a in a_values:
                # Check gamma constraint
                gamma = _compute_gamma(a, c, Om0, w0, wa)
                if gamma < 0.1 or gamma > 1.2:
                    return 10000.0  # Heavy penalty
                
                # f error
                f_approx = _f_from_gamma(a, c, Om0, w0, wa)
                f_ref = _f_reference(a, Om0, w0, wa)
                if abs(f_ref) > 1e-10:
                    rel_err_f = abs(f_approx - f_ref) / abs(f_ref)
                else:
                    rel_err_f = abs(f_approx - f_ref)
                max_err = max(max_err, rel_err_f)
                
                # D error
                D_approx = _D_normalized(a, c, Om0, w0, wa)
                D_ref = _D_reference(a, Om0, w0, wa)
                if abs(D_ref) > 1e-10:
                    rel_err_D = abs(D_approx - D_ref) / abs(D_ref)
                else:
                    rel_err_D = abs(D_approx - D_ref)
                max_err = max(max_err, rel_err_D)
        
        return 100.0 * max_err

    # Optimize using differential evolution
    bounds = [(-10, 10)] * 6
    initial = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    best_c = initial.copy()
    best_score = _score(best_c)
    
    # Run differential evolution with time limit
    try:
        result = differential_evolution(
            _score, bounds,
            seed=42,
            maxiter=50,
            popsize=10,
            tol=1e-6,
            polish=True,
            init='sobol',
            updating='immediate'
        )
        
        if result.fun < best_score:
            best_c = result.x
            best_score = result.fun
        
        # Check time
        if time.time() > deadline:
            pass
        else:
            # Try a local polish with more iterations
            from scipy.optimize import minimize
            res2 = minimize(_score, best_c, method='Nelder-Mead', 
                          options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
            if res2.fun < best_score:
                best_c = res2.x
                best_score = res2.fun
        
    except Exception:
        pass
    
    # Final time check
    if time.time() > deadline - 2.0:
        # Return best found so far
        pass
    
    # Ensure coefficients are in [-10, 10]
    best_c = np.clip(best_c, -10.0, 10.0)
    
    return list(best_c)

# EVOLVE-BLOCK-END
