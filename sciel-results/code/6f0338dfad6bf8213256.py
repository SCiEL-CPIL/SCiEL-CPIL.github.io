"""Compact dark-energy growth-index fit (CPL cosmologies).
SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
def construct():
    return list(_best_coeffs())

def _best_coeffs():
    t0 = time.time()
    deadline = t0 + 82.0
    c = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    try:
        c = _fit(deadline)
    except Exception:
        pass
    return c

def _fit(deadline):
    # Dense reference grid
    om0_vals = np.array([0.20, 0.24, 0.26, 0.27, 0.28, 0.30, 0.32, 0.35, 0.40])
    w0_vals = np.array([-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2])
    wa_vals = np.array([-0.4, -0.2, 0.0, 0.2, 0.4])
    a_vals = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                       0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00])
    A = a_vals[1:]  # a > 0
    N = len(A)
    M = len(om0_vals) * len(w0_vals) * len(wa_vals)

    # Build meshgrid
    Om0_g, W0_g, Wa_g, A_g = np.meshgrid(om0_vals, w0_vals, wa_vals, A, indexing='ij')
    Om0_g = Om0_g.ravel(); W0_g = W0_g.ravel()
    Wa_g = Wa_g.ravel(); A_g = A_g.ravel()
    Npts = len(A_g)

    # Compute Om(a) for all points
    Om_a = Om0_g * A_g**3 / (Om0_g * A_g**3 + (1.0 - Om0_g) * A_g**(3.0*(1.0+W0_g+Wa_g*(1.0-A_g))))
    u_g = 1.0 + W0_g + Wa_g * (1.0 - A_g)
    q_g = 1.0 - Om0_g

    # Solve ODE using scipy for accuracy: D'' + (3/a - f'/f)D' - (3/2)(1+w)Om(a)/a * D = 0
    # Actually the standard growth equation is:
    # d^2D/dlna^2 + [3/2 + f'/f] dD/dlna - (3/2)(1+w)Om(a) D = 0
    # Let y = D, then y'' + P(a) y' + Q(a) y = 0 where derivatives are w.r.t. ln a
    # P(a) = 3/2 + f'/f, Q(a) = -(3/2)(1+w)Om(a)
    # But we don't know f a priori. Use the standard form:
    # d^2D/dlna^2 + [3/2 + (d f/d ln a)/f] dD/dlna - (3/2)(1+w)Om(a) D = 0
    # This is implicit in f. Instead, use the original second-order ODE in a:
    # D'' + (3/a + f'/f * 1/a) D' - (3/2)(1+w)Om(a)/a^2 D = 0
    # Simpler: use the well-known form d^2D/dlna^2 + (3/2 + f'/f) dD/dlna - (3/2)(1+w)Om(a) D = 0
    # We'll solve this iteratively or use a known reference.
    # For now, use a high-accuracy RK4 integration of the ODE in ln a.
    
    def _solve_growth_ode(Om0, w0, wa, a_grid):
        """Solve growth ODE using RK4 in ln a coordinate."""
        n = len(a_grid)
        D = np.zeros(n)
        Dprime = np.zeros(n)  # dD/dlna
        D[0] = 1.0
        # Initial condition: at early times D ~ a, so dD/dlna ~ 1
        Dprime[0] = 1.0
        
        ln_a = np.log(a_grid)
        h = ln_a[1] - ln_a[0]
        
        for i in range(n-1):
            a_i = a_grid[i]
            Da_i = Om0 * a_i**3 / (Om0 * a_i**3 + (1-Om0) * a_i**(3*(1+w0+wa*(1-a_i))))
            u_i = 1 + w0 + wa*(1-a_i)
            
            # f = Da^gamma, use Linder-Waugh for f'/f
            gamma_i = (5/7) * (Da_i / u_i)**0.38
            f_i = Da_i ** gamma_i
            # Approximate f'/f = dlnf/dlna = gamma * dlnDa/dlna
            # dlnDa/dlna = 3Da(1-Da) - 3(1+w)Da = 3Da(1-Da - (1+w)Da) = 3Da(1 - (2+w)Da)
            dlnDa_dlna = 3.0 * Da_i * (1.0 - (2.0 + w0 + wa*(1-a_i)) * Da_i)
            df_dlna_over_f = gamma_i * dlnDa_dlna
            
            P = 1.5 + df_dlna_over_f
            Q = -1.5 * (1 + w0 + wa*(1-a_i)) * Da_i
            
            # RK4 step
            k1D = Dprime[i]
            k1Dp = -P * Dprime[i] - Q * D[i]
            
            a_half = a_grid[i] * np.exp(h/2)
            Da_half = Om0 * a_half**3 / (Om0 * a_half**3 + (1-Om0) * a_half**(3*(1+w0+wa*(1-a_half))))
            u_half = 1 + w0 + wa*(1-a_half)
            gamma_half = (5/7) * (Da_half / u_half)**0.38
            f_half = Da_half ** gamma_half
            dlnDa_dlna_half = 3.0 * Da_half * (1.0 - (2.0 + w0 + wa*(1-a_half)) * Da_half)
            df_dlna_over_f_half = gamma_half * dlnDa_dlna_half
            P_half = 1.5 + df_dlna_over_f_half
            Q_half = -1.5 * (1 + w0 + wa*(1-a_half)) * Da_half
            
            k2D = Dprime[i] + h/2 * k1Dp
            k2Dp = -P_half * (Dprime[i] + h/2 * k1Dp) - Q_half * (D[i] + h/2 * k1D)
            
            k3D = Dprime[i] + h/2 * k2Dp
            k3Dp = -P_half * (Dprime[i] + h/2 * k2Dp) - Q_half * (D[i] + h/2 * k2D)
            
            a_full = a_grid[i+1]
            Da_full = Om0 * a_full**3 / (Om0 * a_full**3 + (1-Om0) * a_full**(3*(1+w0+wa*(1-a_full))))
            u_full = 1 + w0 + wa*(1-a_full)
            gamma_full = (5/7) * (Da_full / u_full)**0.38
            f_full = Da_full ** gamma_full
            dlnDa_dlna_full = 3.0 * Da_full * (1.0 - (2.0 + w0 + wa*(1-a_full)) * Da_full)
            df_dlna_over_f_full = gamma_full * dlnDa_dlna_full
            P_full = 1.5 + df_dlna_over_f_full
            Q_full = -1.5 * (1 + w0 + wa*(1-a_full)) * Da_full
            
            k4D = Dprime[i] + h * k3Dp
            k4Dp = -P_full * (Dprime[i] + h * k3Dp) - Q_full * (D[i] + h * k3D)
            
            D[i+1] = D[i] + h/6 * (k1D + 2*k2D + 2*k3D + k4D)
            Dprime[i+1] = Dprime[i] + h/6 * (k1Dp + 2*k2Dp + 2*k3Dp + k4Dp)
        
        return D, Dprime

    # Solve ODE for all combinations
    refF = np.zeros((M, N))
    refDprime = np.zeros((M, N))
    idx = 0
    for i_om in range(len(om0_vals)):
        for i_w0 in range(len(w0_vals)):
            for i_wa in range(len(wa_vals)):
                D, Dp = _solve_growth_ode(om0_vals[i_om], w0_vals[i_w0], wa_vals[i_wa], A)
                refF[idx, :] = D
                refDprime[idx, :] = Dp
                idx += 1

    # Reference f = D' (since D' = dD/dlna = a*dD/da = f*D, so f = D'/D)
    f_ref = refDprime / refF

    # Least-squares initial guess
    def _lsq_initial():
        a_sub = np.linspace(0.10, 0.95, 15)
        o_sub = np.linspace(0.20, 0.40, 6)
        w_sub = np.linspace(-0.8, 0.2, 6)
        wa_sub = np.linspace(-0.4, 0.4, 6)
        rows = []
        for o in o_sub:
            for w in w_sub:
                for ww in wa_sub:
                    for ai in a_sub:
                        u = 1.0 + w + ww * (1.0 - ai)
                        if u <= 1e-8:
                            continue
                        om_a = o * ai**3 / (o * ai**3 + (1.0 - o) * ai**(3.0*(1.0+w+ww)))
                        if om_a <= 0 or om_a >= 1:
                            continue
                        g_ref = (5.0/7.0) * (om_a / u) ** 0.38
                        q = 1.0 - o
                        rows.append([1.0, q, q*q, u, ww*ai, u*q, g_ref])
        X = np.array(rows)
        Xf = X[:, :6]
        y = X[:, 6]
        try:
            sol, *_ = np.linalg.lstsq(Xf, y, rcond=None)
        except Exception:
            sol = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
        sol = np.clip(sol, -10.0, 10.0)
        return sol

    x0 = _lsq_initial()

    def _gamma(c):
        g = c[0] + c[1]*q_g + c[2]*q_g*q_g + c[3]*u_g + c[4]*Wa_g*A_g + c[5]*u_g*q_g
        return np.clip(g, 0.1, 1.2)

    def _f(c):
        g = _gamma(c)
        return Om0_g ** g

    log_refF = np.log(np.maximum(refF, 1e-30))

    def _D_err(c):
        """Compute relative error in normalized growth factor D(a)/D(1)."""
        f = _f(c).reshape(M, N)
        h = np.diff(A)
        # Trapezoidal integration of f from a_j to a_i
        cum = np.zeros((M, N))
        for i in range(1, N):
            cum[:, i] = cum[:, i-1] + 0.5 * (f[:, i-1] + f[:, i]) * h[i-1]
        
        # D(a_i)/D(1) = exp(-integral from ln(a_i) to 0 f dx) = exp(-(cum[:, i] - cum[:, N-1]))
        # But we need D(a)/D(1) where a < 1, so integral from a to 1
        # log(D(a)/D(1)) = -integral_a^1 f(a') da'/a' = -integral_{ln a}^0 f dx
        # = -(cum[:, N-1] - cum[:, i]) if cum is integral from A[0] to A[i]
        # Actually cum[:, i] = integral from A[0] to A[i]
        # integral from a_i to 1 = cum[:, N-1] - cum[:, i]
        log_ratio_ref = log_refF - log_refF[:, N-1]  # log(D_ref(a_i)/D_ref(1))
        log_ratio_approx = -(cum[:, N-1] - cum)  # log(D_approx(a_i)/D_approx(1))
        
        # Relative error: |D_approx/D_ref - 1| = |exp(log_ratio_approx - log_ratio_ref) - 1|
        diff = log_ratio_approx - log_ratio_ref
        # Only consider a < 1 (i.e., all except last point)
        diff = diff[:, :N-1]
        e = np.where(np.abs(diff) < 0.5, np.expm1(diff), np.exp(diff) - 1.0)
        return np.abs(e)

    def _f_err(c):
        f = _f(c).reshape(M, N)
        return np.abs(f - f_ref) / f_ref

    def objective(c):
        fe = _f_err(c).ravel()
        de = _D_err(c).ravel()
        vals = np.concatenate([fe, de])
        # Use exact max for Nelder-Mead (it handles non-smooth ok with adaptive)
        return float(np.max(vals))

    best = np.clip(x0, -10.0, 10.0)
    best_f = objective(best)
    x0 = best.copy()
    rng = np.random.default_rng(42)
    
    while time.time() < deadline - 6.0:
        res = minimize(objective, x0, method="Nelder-Mead",
                       options=dict(xatol=1e-6, fatol=1e-9, maxiter=300, adaptive=True))
        if res.fun < best_f:
            best_f = res.fun
            best = np.clip(res.x, -10.0, 10.0)
            x0 = best.copy()
        else:
            improved = False
            for _ in range(5):
                cand = best + 0.05 * rng.standard_normal(6)
                cand = np.clip(cand, -10.0, 10.0)
                fc = objective(cand)
                if fc < best_f:
                    best_f = fc
                    best = cand
                    x0 = cand.copy()
                    improved = True
            if not improved:
                break

    if time.time() < deadline - 2.0:
        res = minimize(objective, best, method="Nelder-Mead",
                       options=dict(xatol=1e-9, fatol=1e-12, maxiter=500, adaptive=True))
        if res.fun < best_f:
            best = np.clip(res.x, -10.0, 10.0)

    best = np.clip(best, -10.0, 10.0)
    return list(best)
# EVOLVE-BLOCK-END
