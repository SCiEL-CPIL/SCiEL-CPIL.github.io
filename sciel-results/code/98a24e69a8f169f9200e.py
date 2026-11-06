"""Improved growth-index approximation for CPL dark energy."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import time

def construct():
    """
    Return six coefficients for the growth-index formula.
    gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q
    where q = 1-Om(a), u = 1+w0+wa*(1-a)
    """
    start_time = time.time()
    deadline = 85.0  # safety margin from 90s limit

    # Test cosmologies covering typical validation range
    test_models = [
        (0.3, 0.0, 0.0),    # LCDM
        (0.3, -0.5, 0.5),   # CPL
        (0.3, -1.0, 0.0),   # Cosmological constant
        (0.3, -0.8, 0.3),   # CPL
        (0.3, -0.6, -0.2),  # CPL
        (0.27, -0.5, 0.5),  # Lower Om
        (0.35, -0.8, 0.3),  # Higher Om
        (0.3, -0.3, 0.8),   # Strong CPL
        (0.3, -1.2, 0.5),   # Phantom-ish
        (0.3, -0.9, -0.3),  # CPL
    ]

    a_values = np.array([0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])

    def Om_a(Om0, w0, wa, a):
        """Matter density parameter at scale factor a for flat CPL universe."""
        E2 = Om0 * a**(-3) + (1 - Om0) * np.exp(3 * wa * (1 - a))
        return Om0 * a**(-3) / E2

    def Hprime_over_H(Om0, w0, wa, a):
        """Compute H'/H = (1/2) * a * E2' / E2 where E2 = H^2/H0^2."""
        E2 = Om0 * a**(-3) + (1 - Om0) * np.exp(3 * wa * (1 - a))
        E2_prime = -3 * Om0 * a**(-4) + 3 * wa * (1 - Om0) * np.exp(3 * wa * (1 - a))
        return 0.5 * a * E2_prime / E2

    def compute_reference_f(Om0, w0, wa, a_eval):
        """
        Compute the true f(a) = d ln D / d ln a by integrating the ODE
        D'' + (3 + H'/H) D' - (3/2) Om D = 0 in ln a variable.
        We integrate backward from a=1 where f(1) ≈ 1 (since Om(1)=1).
        Returns f at a_eval.
        """
        def ode_system(x, y):
            # x = ln(a), y = [D, f]
            a = np.exp(x)
            Om = Om_a(Om0, w0, wa, a)
            HpH = Hprime_over_H(Om0, w0, wa, a)
            D, f = y
            # D' = f * D (where ' is d/d ln a)
            # f' = -f^2 - (3 + HpH)*f + (3/2)*Om
            dD = f * D
            df = -f**2 - (3 + HpH) * f + 1.5 * Om
            return [dD, df]

        # Integrate from ln(a)=0 (a=1) to ln(a_eval)
        x_end = np.log(a_eval)
        if abs(x_end) < 1e-10:
            return 1.0  # f(1) ≈ 1

        try:
            sol = solve_ivp(
                ode_system,
                [0, x_end],
                [1.0, 1.0],  # D(1)=1, f(1)=1
                method='DOP853',
                rtol=1e-10,
                atol=1e-12,
                dense_output=False
            )
            if sol.success and len(sol.y[1]) > 0:
                return sol.y[1][-1]
        except Exception:
            pass

        # Fallback: use the approximation itself
        Om = Om_a(Om0, w0, wa, a_eval)
        return Om ** 0.55

    def compute_reference_D(Om0, w0, wa, a_eval):
        """
        Compute normalized D(a)/D(1) by integrating the ODE.
        D(a)/D(1) = exp(-integral from a to 1 of f(exp(x)) dx) where x = ln(a)
        Actually: D(a)/D(1) = exp(integral from 0 to ln(a) of f(x) dx)
        """
        if a_eval >= 0.999:
            return 1.0

        def ode_system(x, y):
            a = np.exp(x)
            Om = Om_a(Om0, w0, wa, a)
            HpH = Hprime_over_H(Om0, w0, wa, a)
            D, f = y
            dD = f * D
            df = -f**2 - (3 + HpH) * f + 1.5 * Om
            return [dD, df]

        x_end = np.log(a_eval)
        try:
            sol = solve_ivp(
                ode_system,
                [0, x_end],
                [1.0, 1.0],
                method='DOP853',
                rtol=1e-10,
                atol=1e-12,
                dense_output=False
            )
            if sol.success and len(sol.y[0]) > 0:
                return sol.y[0][-1]
        except Exception:
            pass

        # Fallback
        gamma = 0.55
        Om = Om_a(Om0, w0, wa, a_eval)
        return Om ** gamma

    def compute_error(coeffs):
        """Compute max relative error for f and D across test models."""
        c0, c1, c2, c3, c4, c5 = coeffs
        max_err = 0.0

        for Om0, w0, wa in test_models:
            for a in a_values:
                if a <= 0:
                    continue
                Om = Om_a(Om0, w0, wa, a)
                q = 1 - Om
                u = 1 + w0 + wa * (1 - a)

                gamma = c0 + c1 * q + c2 * q**2 + c3 * u + c4 * wa * a + c5 * u * q

                # Check gamma bounds
                if gamma < 0.1 or gamma > 1.2:
                    return 100.0  # Heavy penalty

                f_approx = Om ** gamma

                # Get reference values
                f_ref = compute_reference_f(Om0, w0, wa, a)
                D_ref = compute_reference_D(Om0, w0, wa, a)

                # Compute approximate D
                # D_approx/D(1) = exp(-integral from ln(a) to 0 of f(exp(x)) dx)
                # = exp(integral from 0 to ln(a) of f(exp(x)) dx)
                # We can approximate this using the same gamma formula
                # But for the error, we need to compare with reference D

                # For D, use the integral approximation
                # D_approx/D(1) = exp(integral_0^{ln(a)} Om(exp(x))^gamma(x) dx)
                # This is expensive, so let's just use f error for now
                # and assume D error is correlated

                if abs(f_ref) > 1e-6:
                    err_f = abs(f_approx - f_ref) / abs(f_ref)
                    max_err = max(max_err, err_f)

                if abs(D_ref) > 1e-6:
                    # Approximate D using the integral of f_approx
                    # For efficiency, use a simple quadrature
                    if a < 0.999:
                        x_vals = np.linspace(0, np.log(a), 20)
                        f_vals = np.array([
                            Om_a(Om0, w0, wa, np.exp(x)) ** (
                                c0 + c1 * (1 - Om_a(Om0, w0, wa, np.exp(x))) +
                                c2 * (1 - Om_a(Om0, w0, wa, np.exp(x)))**2 +
                                c3 * (1 + w0 + wa * (1 - np.exp(x))) +
                                c4 * wa * np.exp(x) +
                                c5 * (1 + w0 + wa * (1 - np.exp(x))) * (1 - Om_a(Om0, w0, wa, np.exp(x)))
                            )
                            for x in x_vals
                        ])
                        D_approx = np.exp(np.trapz(f_vals, x_vals))
                        err_D = abs(D_approx - D_ref) / abs(D_ref)
                        max_err = max(max_err, err_D)

        return max_err

    # Initial guess from literature
    best_coeffs = np.array([0.55, 0.05, 0.0, 0.02, 0.0, 0.0])
    best_score = compute_error(best_coeffs)

    # Try to improve with scipy optimization
    elapsed = time.time() - start_time
    if elapsed < deadline - 10.0:
        # Use Nelder-Mead for robustness
        try:
            result = minimize(
                compute_error,
                best_coeffs,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            if result.fun < best_score:
                best_coeffs = result.x
                best_score = result.fun
        except Exception:
            pass

    # Second round with tighter tolerance if time permits
    elapsed = time.time() - start_time
    if elapsed < deadline - 5.0:
        try:
            result = minimize(
                compute_error,
                best_coeffs,
                method='Nelder-Mead',
                options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if result.fun < best_score:
                best_coeffs = result.x
                best_score = result.fun
        except Exception:
            pass

    # Ensure coefficients are in [-10, 10]
    best_coeffs = np.clip(best_coeffs, -10, 10)

    # Final validation: check gamma bounds
    valid = True
    for Om0, w0, wa in test_models:
        for a in a_values:
            Om = Om_a(Om0, w0, wa, a)
            q = 1 - Om
            u = 1 + w0 + wa * (1 - a)
            gamma = best_coeffs[0] + best_coeffs[1] * q + best_coeffs[2] * q**2 + \
                    best_coeffs[3] * u + best_coeffs[4] * wa * a + best_coeffs[5] * u * q
            if gamma < 0.1 or gamma > 1.2:
                valid = False
                break
        if not valid:
            break

    if not valid:
        # Fall back to literature values
        best_coeffs = np.array([0.55, 0.05, 0.0, 0.02, 0.0, 0.0])

    return list(best_coeffs)
# EVOLVE-BLOCK-END
