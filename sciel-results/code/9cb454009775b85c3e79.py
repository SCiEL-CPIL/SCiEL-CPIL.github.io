"""Growth-index fit for CPL dark energy; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
import warnings
warnings.filterwarnings("ignore")

_DEADLINE = time.time() + 85.0  # safety margin under 90s wall-clock

def _H2(a, Om, w0, wa):
    u = 1.0 + w0 + wa * (1.0 - a)
    de = 1.0 - Om
    if de <= 0.0:
        return 1.0 + Om
    return Om * a ** (-3.0) + de * a ** (-3.0 * (1.0 + u))

def _Om_a(a, Om, w0, wa):
    h2 = _H2(a, Om, w0, wa)
    return Om * a ** (-3.0) / h2

def _solve_D(a_min, Om, w0, wa):
    # D(1)=1, D'(1)=f(1)=Om^gamma. Integrate backward.
    def rhs(y, x):
        a = math.exp(x)
        O = _Om_a(a, Om, w0, wa)
        g = 0.55  # only needed for initial slope
        f = O ** g
        dD = y[1]
        ddD = 2.0 * (1.0 - 0.5 * O) * dD - f * y[0]
        return [dD, ddD]
    
    f1 = Om ** 0.55
    sol = solve_ivp(rhs, [0.0, math.log(a_min)], [1.0, f1],
                    method="RK45", rtol=1e-9, atol=1e-12, max_step=0.1)
    if not sol.success:
        return None
    return sol

def _ref_model(a_grid, Om, w0, wa):
    a_min = a_grid[0]
    sol = _solve_D(a_min, Om, w0, wa)
    if sol is None:
        return None
    xs = np.log(a_grid)
    D = sol.sol(xs)
    Dn = D[0] / D[0][-1]
    f = np.array([_Om_a(a, Om, w0, wa) ** 0.55 for a in a_grid])
    return f, Dn

def _cosmos():
    out = []
    for Om in [0.22, 0.25, 0.27, 0.30, 0.33, 0.36]:
        for w0 in [-1.2, -1.1, -1.05, -1.0, -0.95, -0.9]:
            for wa in [-0.2, -0.1, 0.0, 0.1, 0.2]:
                out.append((Om, w0, wa))
    return out

def _a_grid():
    return np.linspace(0.05, 1.0, 40)

def _eval(c, cosmos, a_grid):
    c0, c1, c2, c3, c4, c5 = c
    worst = 0.0
    for (Om, w0, wa) in cosmos:
        ref = _ref_model(a_grid, Om, w0, wa)
        if ref is None:
            continue
        f_ref, Dn_ref = ref
        # Compute Dn for candidate
        # Dn(a) = exp(-int_ln(a)^0 f(exp(x)) dx)
        # We need to compute the integral for the candidate f
        # Since f = Om(a)^gamma, and gamma varies, we can't just use the reference f.
        # We must integrate the candidate f.
        # To save time, we can approximate the integral using the grid.
        # Let's use simple trapezoidal rule on the log-a grid for the integral part.
        # Actually, the evaluator computes D(a)/D(1) = exp(-integral_ln(a)^0 f(exp(x)) dx).
        # We can compute this integral numerically for the candidate.
        
        for i, a in enumerate(a_grid):
            q = 1.0 - _Om_a(a, Om, w0, wa)
            u = 1.0 + w0 + wa * (1.0 - a)
            g = c0 + c1 * q + c2 * q * q + c3 * u + c4 * wa * a + c5 * u * q
            
            # Check gamma bounds
            if g < 0.1 or g > 1.2:
                # Penalty for out-of-bounds gamma
                return 1.0 # Return high error to discourage
                
            f = _Om_a(a, Om, w0, wa) ** g
            err_f = abs(f - f_ref[i]) / abs(f_ref[i])
            if err_f > worst:
                worst = err_f
        
        # Compute Dn error
        # We need the integral of f(exp(x)) from x=ln(a) to 0.
        # Let's vectorize this for speed.
        # We already have f values on the grid.
        # Let's re-calculate f for all a in this cosmology to get the full vector.
        f_vec = np.zeros_like(a_grid)
        for i, a in enumerate(a_grid):
            q = 1.0 - _Om_a(a, Om, w0, wa)
            u = 1.0 + w0 + wa * (1.0 - a)
            g = c0 + c1 * q + c2 * q * q + c3 * u + c4 * wa * a + c5 * u * q
            f_vec[i] = _Om_a(a, Om, w0, wa) ** g
        
        # Integrate f(exp(x)) dx from ln(a) to 0.
        # x = ln(a). dx = da/a.
        # Integral from a to 1 of f(a')/a' da'
        # We have f_vec and a_grid.
        # Use cumulative trapezoidal rule.
        # We want I(a_i) = int_{a_i}^{1} f(a')/a' da'
        # Let g(a) = f(a)/a.
        g_vec = f_vec / a_grid
        # Cumulative integral from 1 down to a_i
        # np.cumsum works forward. We can reverse, cumsum, reverse back.
        # Or just loop for simplicity if grid is small (40 points).
        # 40 points is small.
        Dn_cand = np.zeros_like(a_grid)
        Dn_cand[-1] = 1.0 # At a=1, D=1
        # Integrate backward
        for i in range(len(a_grid) - 2, -1, -1):
            # Trapezoid rule for integral from a_i to a_{i+1}
            # int_{a_i}^{a_{i+1}} g(a) da approx 0.5 * (g_i + g_{i+1}) * (a_{i+1} - a_i)
            # But we need integral of f(a)/a da.
            # Let's use log-space integration for better accuracy.
            # x_i = ln(a_i). dx = a_{i+1}-a_i / a_i approx.
            # Actually, just use the a-grid trapezoid on g(a)=f(a)/a.
            da = a_grid[i+1] - a_grid[i]
            int_seg = 0.5 * (g_vec[i] + g_vec[i+1]) * da
            # D(a_i) = D(a_{i+1}) * exp(int_{a_i}^{a_{i+1}} f/a da)
            # Wait, D' = f D. dD/D = f da. 
            # The task says: D(a)/D(1) = exp(-integral_ln(a)^0 f(exp(x)) dx)
            # Let x = ln a. dx = da/a.
            # Integral is int_{ln a}^0 f(e^x) dx.
            # This is equivalent to int_a^1 f(a')/a' da'.
            # So D(a) = D(1) * exp( - int_a^1 f(a')/a' da' )?
            # Let's check the sign.
            # dD/dlna = f D.
            # D(1) = 1.
            # D(a) = exp( int_1^lna f(e^x) dx ) = exp( - int_lna^0 f(e^x) dx ).
            # So D(a) = exp( - I(a) ) where I(a) = int_lna^0 f(e^x) dx.
            # My loop calculates I(a_i) = int_{a_i}^{a_{i+1}} ... + I(a_{i+1}).
            # So I(a_i) = int_seg + I(a_{i+1}).
            I_i = int_seg + (-math.log(Dn_cand[i+1])) # Since Dn_cand[i+1] = exp(-I(i+1))
            Dn_cand[i] = math.exp(-I_i)
            
            err_D = abs(Dn_cand[i] - Dn_ref[i]) / abs(Dn_ref[i])
            if err_D > worst:
                worst = err_D

    return worst

def _objective(c, cosmos, a_grid):
    if time.time() > _DEADLINE:
        return 1e6
    # gamma bound penalty
    bad = 0.0
    for (Om, w0, wa) in cosmos:
        for a in a_grid:
            q = 1.0 - _Om_a(a, Om, w0, wa)
            u = 1.0 + w0 + wa * (1.0 - a)
            g = c[0] + c[1]*q + c[2]*q*q + c[3]*u + c[4]*wa*a + c[5]*u*q
            if g < 0.1:
                bad += (0.1 - g)
            elif g > 1.2:
                bad += (g - 1.2)
    base = _eval(c, cosmos, a_grid)
    if bad > 0:
        return base + 10.0 * bad
    return base

# EVOLVE-BLOCK-END

def construct():
    global _DEADLINE
    try:
        # 1. Fast Least-Squares Fit to LvS Approximation
        # Grid for fitting
        Om0_vals = np.array([0.20, 0.25, 0.28, 0.30, 0.32, 0.35])
        w0_vals = np.array([-1.2, -1.1, -1.0, -0.9, -0.8])
        wa_vals = np.array([-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3])
        a_vals = np.linspace(0.1, 0.99, 50)

        rows = []
        targets = []
        
        for Om0 in Om0_vals:
            for w0 in w0_vals:
                for wa in wa_vals:
                    for a in a_vals:
                        a_cubed = a**3
                        de_factor = a**(-3.0*(1.0+w0)) * np.exp(-3.0*wa*(1.0-a))
                        E2 = Om0 * a_cubed + (1.0 - Om0) * de_factor
                        
                        if E2 <= 0:
                            continue
                        Om_a = Om0 * a_cubed / E2
                        q = 1.0 - Om_a
                        u = 1.0 + w0 + wa * (1.0 - a)
                        
                        if u <= 0:
                            continue
                        
                        # LvS: gamma = (5/7) * (Om(a)/u)^(4/7)
                        gamma_target = (5.0/7.0) * (Om_a / u)**(4.0/7.0)
                        
                        # Design matrix row: [1, q, q^2, u, wa*a, u*q]
                        row = [1.0, q, q*q, u, wa*a, u*q]
                        rows.append(row)
                        targets.append(gamma_target)
                        
                        if time.time() > _DEADLINE - 20.0:
                            break
                    if time.time() > _DEADLINE - 20.0:
                        break
                if time.time() > _DEADLINE - 20.0:
                    break
            if time.time() > _DEADLINE - 20.0:
                break
        if time.time() > _DEADLINE - 20.0:
            break

        if len(rows) < 50:
            return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

        A = np.array(rows)
        b = np.array(targets)
        
        # Iterative reweighted least squares to approximate min-max
        coeffs = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        for iteration in range(3):
            pred = A @ coeffs
            errors = np.abs(pred - b)
            # Weight points with larger errors more heavily
            w = 1.0 / (errors + 0.01)
            Aw = A * w[:, None]
            bw = b * w
            try:
                coeffs, residuals, rank, sv = np.linalg.lstsq(Aw, bw, rcond=None)
            except Exception:
                break
        
        # Clip to reasonable range
        coeffs = np.clip(coeffs, -5.0, 5.0)
        
        # 2. Local Polish with Exact ODE Objective
        # Use the LSQ result as initial guess
        x0 = [float(c) for c in coeffs]
        
        # Prepare validation grid
        cosmos = _cosmos()
        a_grid = _a_grid()
        
        # Quick check if x0 is valid
        val = _objective(x0, cosmos, a_grid)
        if time.time() > _DEADLINE - 10.0:
            return x0
            
        # Nelder-Mead polish
        best = x0
        best_val = val
        for _ in range(2):
            if time.time() > _DEADLINE - 5.0:
                break
            try:
                res = minimize(_objective, best, args=(cosmos, a_grid),
                               method="Nelder-Mead",
                               options={"maxiter": 100, "xatol": 1e-6,
                                        "fatol": 1e-7, "adaptive": True})
                if res.x is not None and res.fun < best_val:
                    best = list(res.x)
                    best_val = res.fun
            except Exception:
                break
        
        # Clamp to [-10,10]
        best = [min(10.0, max(-10.0, float(v))) for v in best]
        return best
    except Exception:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
