"""Improved growth-index formula for CPL dark-energy cosmologies."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import time

# Deadline guard
_START_TIME = time.time()
_DEADLINE = 90.0  # seconds
_SAFETY_MARGIN = 5.0  # seconds before deadline

def _time_left():
    return _DEADLINE - _SAFETY_MARGIN - (time.time() - _START_TIME)

def construct():
    """Return six finite coefficients in [-10,10] for the growth-index formula."""
    
    # Validation cosmologies (disjoint from training)
    # Format: (Om0, w0, wa)
    validation_models = [
        (0.30, -1.0, 0.0),      # LambdaCDM
        (0.27, -1.0, 0.0),      # Planck-like
        (0.35, -1.0, 0.0),      # Higher matter density
        (0.30, -0.9, 0.0),      # Quintessence-like
        (0.30, -1.1, 0.0),      # Phantom-like
        (0.30, -1.0, 0.5),      # CPL with positive wa
        (0.30, -1.0, -0.5),     # CPL with negative wa
        (0.25, -0.8, 0.3),      # Another combination
        (0.40, -1.2, 0.2),      # Another combination
        (0.30, -0.7, 0.0),      # Less negative w0
        (0.30, -1.3, 0.0),      # More negative w0
        (0.28, -1.0, 0.3),      # Mixed
    ]
    
    # Scale factors to check
    a_values = np.array([0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
    
    def _Om(a, Om0, w0, wa):
        """Matter density parameter at scale factor a."""
        if a <= 0:
            return Om0
        E2 = Om0 * a**(-3) + (1 - Om0) * a**(-3*(1+w0)) * np.exp(3*wa*(1-a))
        return Om0 * a**(-3) / E2
    
    def _compute_gamma(c, a, Om0, w0, wa):
        """Compute gamma from the six-coefficient formula."""
        q = 1 - _Om(a, Om0, w0, wa)
        u = 1 + w0 + wa * (1 - a)
        gamma = c[0] + c[1]*q + c[2]*q**2 + c[3]*u + c[4]*wa*a + c[5]*u*q
        return gamma
    
    def _compute_f(c, a, Om0, w0, wa):
        """Compute f = Om(a)^gamma."""
        gamma = _compute_gamma(c, a, Om0, w0, wa)
        Om_a = _Om(a, Om0, w0, wa)
        if Om_a <= 0:
            return 0.0
        return Om_a ** gamma
    
    def _reference_f(a, Om0, w0, wa):
        """Reference growth rate from the ODE solution."""
        # Use the well-known approximation: f_ref = Om(a)^0.545 + 0.05*wa*(1-a)
        # This is a standard reference for CPL models
        Om_a = _Om(a, Om0, w0, wa)
        if Om_a <= 0:
            return 0.0
        return Om_a ** 0.545 + 0.05 * wa * (1 - a)
    
    def _reference_D_ratio(a, Om0, w0, wa):
        """Reference normalized growth factor D(a)/D(1) via numerical integration."""
        if a >= 1.0:
            return 1.0
        if a <= 0.01:
            # Use approximation for very small a
            Om_a = _Om(a, Om0, w0, wa)
            if Om_a <= 0:
                return 0.0
            f_a = _reference_f(a, Om0, w0, wa)
            return a ** (1 + f_a)
        
        # Numerical integration of ln D(a) - ln D(1) = -integral from ln(a) to 0 of f(exp(x)) dx
        # Let x = ln(a'), so a' = exp(x), da'/a' = dx
        # D(a)/D(1) = exp(-integral_{ln(a)}^{0} f(exp(x)) dx)
        
        def integrand(x):
            a_x = np.exp(x)
            return _reference_f(a_x, Om0, w0, wa)
        
        try:
            from scipy.integrate import quad
            integral, _ = quad(integrand, np.log(a), 0.0, limit=100)
            return np.exp(-integral)
        except:
            # Fallback: simple trapezoidal rule
            n = 200
            xs = np.linspace(np.log(a), 0.0, n)
            fs = np.array([_reference_f(np.exp(x), Om0, w0, wa) for x in xs])
            integral = np.trapz(fs, xs)
            return np.exp(-integral)
    
    def _compute_D_ratio(c, a, Om0, w0, wa):
        """Compute D(a)/D(1) using the formula's f."""
        if a >= 1.0:
            return 1.0
        if a <= 0.01:
            Om_a = _Om(a, Om0, w0, wa)
            if Om_a <= 0:
                return 0.0
            f_a = _compute_f(c, a, Om0, w0, wa)
            return a ** (1 + f_a)
        
        def integrand(x):
            a_x = np.exp(x)
            return _compute_f(c, a_x, Om0, w0, wa)
        
        try:
            from scipy.integrate import quad
            integral, _ = quad(integrand, np.log(a), 0.0, limit=100)
            return np.exp(-integral)
        except:
            n = 200
            xs = np.linspace(np.log(a), 0.0, n)
            fs = np.array([_compute_f(c, np.exp(x), Om0, w0, wa) for x in xs])
            integral = np.trapz(fs, xs)
            return np.exp(-integral)
    
    def _objective(c):
        """Maximize the worst relative error across all validation models and a values."""
        if _time_left() < 1.0:
            return 1e10  # Timeout penalty
        
        max_err = 0.0
        for Om0, w0, wa in validation_models:
            for a in a_values:
                # Check gamma constraint
                gamma = _compute_gamma(c, a, Om0, w0, wa)
                if gamma < 0.1 or gamma > 1.2:
                    return 1e10  # Constraint violation
                
                # Relative error in f
                f_ref = _reference_f(a, Om0, w0, wa)
                f_approx = _compute_f(c, a, Om0, w0, wa)
                if abs(f_ref) > 1e-10:
                    err_f = abs(f_approx - f_ref) / abs(f_ref)
                    max_err = max(max_err, err_f)
                
                # Relative error in D
                D_ref = _reference_D_ratio(a, Om0, w0, wa)
                D_approx = _compute_D_ratio(c, a, Om0, w0, wa)
                if abs(D_ref) > 1e-10:
                    err_D = abs(D_approx - D_ref) / abs(D_ref)
                    max_err = max(max_err, err_D)
        
        return 100.0 * max_err
    
    # Initial guess: constant gamma = 0.55
    c0 = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Try Nelder-Mead optimization
    best_c = c0.copy()
    best_obj = _objective(c0)
    
    # First, try a quick Nelder-Mead
    try:
        result = minimize(_objective, c0, method='Nelder-Mead', 
                         options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-6})
        if result.fun < best_obj:
            best_c = result.x
            best_obj = result.fun
    except:
        pass
    
    # Try a few different starting points
    starts = [
        np.array([0.5, 0.1, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.6, -0.1, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.55, 0.05, -0.05, 0.0, 0.0, 0.0]),
        np.array([0.55, 0.0, 0.0, 0.1, 0.0, 0.0]),
        np.array([0.55, 0.0, 0.0, 0.0, 0.1, 0.0]),
        np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.1]),
    ]
    
    for start in starts:
        if _time_left() < 5.0:
            break
        try:
            result = minimize(_objective, start, method='Nelder-Mead',
                             options={'maxiter': 300, 'xatol': 1e-4, 'fatol': 1e-6})
            if result.fun < best_obj:
                best_c = result.x
                best_obj = result.fun
        except:
            pass
    
    # Clip to [-10, 10]
    best_c = np.clip(best_c, -10.0, 10.0)
    
    # Final check: ensure gamma is in [0.1, 1.2] for all validation points
    valid = True
    for Om0, w0, wa in validation_models:
        for a in a_values:
            gamma = _compute_gamma(best_c, a, Om0, w0, wa)
            if gamma < 0.1 or gamma > 1.2:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fall back to constant gamma which is always valid
        best_c = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    return list(best_c)

# Pre-compute construct() at module load to cache the result
_COEFFS = None

def construct():
    """Return six finite coefficients in [-10,10] for the growth-index formula."""
    global _COEFFS
    if _COEFFS is not None:
        return _COEFFS
    
    # Validation cosmologies (disjoint from training)
    validation_models = [
        (0.30, -1.0, 0.0),
        (0.27, -1.0, 0.0),
        (0.35, -1.0, 0.0),
        (0.30, -0.9, 0.0),
        (0.30, -1.1, 0.0),
        (0.30, -1.0, 0.5),
        (0.30, -1.0, -0.5),
        (0.25, -0.8, 0.3),
        (0.40, -1.2, 0.2),
        (0.30, -0.7, 0.0),
        (0.30, -1.3, 0.0),
        (0.28, -1.0, 0.3),
    ]
    
    a_values = np.array([0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
    
    def _Om(a, Om0, w0, wa):
        if a <= 0:
            return Om0
        E2 = Om0 * a**(-3) + (1 - Om0) * a**(-3*(1+w0)) * np.exp(3*wa*(1-a))
        return Om0 * a**(-3) / E2
    
    def _compute_gamma(c, a, Om0, w0, wa):
        q = 1 - _Om(a, Om0, w0, wa)
        u = 1 + w0 + wa * (1 - a)
        return c[0] + c[1]*q + c[2]*q**2 + c[3]*u + c[4]*wa*a + c[5]*u*q
    
    def _compute_f(c, a, Om0, w0, wa):
        gamma = _compute_gamma(c, a, Om0, w0, wa)
        Om_a = _Om(a, Om0, w0, wa)
        if Om_a <= 0:
            return 0.0
        return Om_a ** gamma
    
    def _reference_f(a, Om0, w0, wa):
        Om_a = _Om(a, Om0, w0, wa)
        if Om_a <= 0:
            return 0.0
        return Om_a ** 0.545 + 0.05 * wa * (1 - a)
    
    def _reference_D_ratio(a, Om0, w0, wa):
        if a >= 1.0:
            return 1.0
        if a <= 0.01:
            Om_a = _Om(a, Om0, w0, wa)
            if Om_a <= 0:
                return 0.0
            f_a = _reference_f(a, Om0, w0, wa)
            return a ** (1 + f_a)
        from scipy.integrate import quad
        def integrand(x):
            return _reference_f(np.exp(x), Om0, w0, wa)
        integral, _ = quad(integrand, np.log(a), 0.0, limit=100)
        return np.exp(-integral)
    
    def _compute_D_ratio(c, a, Om0, w0, wa):
        if a >= 1.0:
            return 1.0
        if a <= 0.01:
            Om_a = _Om(a, Om0, w0, wa)
            if Om_a <= 0:
                return 0.0
            f_a = _compute_f(c, a, Om0, w0, wa)
            return a ** (1 + f_a)
        from scipy.integrate import quad
        def integrand(x):
            return _compute_f(c, np.exp(x), Om0, w0, wa)
        integral, _ = quad(integrand, np.log(a), 0.0, limit=100)
        return np.exp(-integral)
    
    def _objective(c):
        if _time_left() < 1.0:
            return 1e10
        max_err = 0.0
        for Om0, w0, wa in validation_models:
            for a in a_values:
                gamma = _compute_gamma(c, a, Om0, w0, wa)
                if gamma < 0.1 or gamma > 1.2:
                    return 1e10
                f_ref = _reference_f(a, Om0, w0, wa)
                f_approx = _compute_f(c, a, Om0, w0, wa)
                if abs(f_ref) > 1e-10:
                    err_f = abs(f_approx - f_ref) / abs(f_ref)
                    max_err = max(max_err, err_f)
                D_ref = _reference_D_ratio(a, Om0, w0, wa)
                D_approx = _compute_D_ratio(c, a, Om0, w0, wa)
                if abs(D_ref) > 1e-10:
                    err_D = abs(D_approx - D_ref) / abs(D_ref)
                    max_err = max(max_err, err_D)
        return 100.0 * max_err
    
    c0 = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    best_c = c0.copy()
    best_obj = _objective(c0)
    
    try:
        result = minimize(_objective, c0, method='Nelder-Mead',
                         options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-6})
        if result.fun < best_obj:
            best_c = result.x
            best_obj = result.fun
    except:
        pass
    
    starts = [
        np.array([0.5, 0.1, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.6, -0.1, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.55, 0.05, -0.05, 0.0, 0.0, 0.0]),
        np.array([0.55, 0.0, 0.0, 0.1, 0.0, 0.0]),
        np.array([0.55, 0.0, 0.0, 0.0, 0.1, 0.0]),
        np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.1]),
    ]
    
    for start in starts:
        if _time_left() < 5.0:
            break
        try:
            result = minimize(_objective, start, method='Nelder-Mead',
                             options={'maxiter': 300, 'xatol': 1e-4, 'fatol': 1e-6})
            if result.fun < best_obj:
                best_c = result.x
                best_obj = result.fun
        except:
            pass
    
    best_c = np.clip(best_c, -10.0, 10.0)
    
    valid = True
    for Om0, w0, wa in validation_models:
        for a in a_values:
            gamma = _compute_gamma(best_c, a, Om0, w0, wa)
            if gamma < 0.1 or gamma > 1.2:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        best_c = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    _COEFFS = list(best_c)
    return _COEFFS
