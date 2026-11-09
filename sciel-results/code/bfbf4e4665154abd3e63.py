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

    # ODE coefficients: y'' = P(a) y' + Q(om0, a) y
    P = np.empty(N)
    Qm = np.empty((M, N))
    for i, ai in enumerate(A):
        P[i] = 1.5 / ai
        Qm[:, i] = 0.5 / ai * (3.0 * om0_vals * ai**(-3.0) + 1.0)

    def ref_y_vec():
        """Solve y'' = P y' + Q y with y(A[0])=1, y(A[1])=y(A[0])+h*(P[0]+Q[0]/P[0])*h... """
        Y = np.empty((M, N))
        Y[:, 0] = 1.0
        h0 = A[1] - A[0]
        Y[:, 1] = 1.0 + h0 * (P[0] * 1.0 + Qm[:, 0] * 1.0) / P[0] * P[0] / 1.0
        # Actually: y' ≈ (y[1]-y[0])/h. Let's use central difference more carefully.
        # y[1] = y[0] + h*y'[0] + 0.5*h^2*y''[0]
        # y''[0] = P[0]*y'[0] + Q[0]*y[0]
        # y'[0] ≈ (y[1] - y[0]) / h  (first order)
        # This is circular. Use: y[1] = y[0] + h*y'[0] + 0.5*h^2*(P[0]*y'[0]+Q[0]*y[0])
        # We need y'[0]. Assume y'[0] ≈ 1.0 (growth rate ~1 at early times? No, f~Om^gamma~0.3)
        # Better: use the analytic approximation for initial slope.
        # f(a) = Om(a)^gamma ~ Om0^(5/7) at small a. Let's just use a reasonable guess.
        # Actually, for the ODE, we need two initial conditions. Let y[0]=1, and estimate y'[0].
        # y'[0] = d ln D / d ln a * D / a... this is getting complex.
        # Simpler: use the fact that at early times, D ∝ a, so y'[0] ≈ 1/a[0] * y[0] = 1/A[0].
        # Wait, D ∝ a means dD/da = D/a, so y'[0] = y[0]/A[0] = 1/A[0].
        y0 = 1.0
        y1_prime = 1.0 / A[0]  # dD/da at a=A[0]
        Y[:, 0] = y0
        Y[:, 1] = y0 + (A[1]-A[0]) * y1_prime + 0.5*(A[1]-A[0])**2 * (P[0]*y1_prime + Qm[:,0]*y0)
        for i in range(1, N - 1):
            h = A[i + 1] - A[i]
            # Central difference: y[i+1] = 2*y[i] - y[i-1] + h^2*(P[i]*y'_i + Q[i]*y_i)
            # y'_i ≈ (y[i+1] - y[i-1]) / (2h)
            # Substitute: y[i+1] = 2y[i] - y[i-1] + h^2*P[i]*(y[i+1]-y[i-1])/(2h) + h^2*Q[i]*y[i]
            # y[i+1]*(1 - h*P[i]/2) = 2y[i] - y[i-1] + h*P[i]/2*y[i-1] + h^2*Q[i]*y[i]
            denom = 1.0 - h * P[i] / 2.0
            Y[:, i+1] = (2.0*Y[:, i] - Y[:, i-1] + h*P[i]/2.0*Y[:, i-1] + h**2*Qm[:, i]*Y[:, i]) / denom
        return Y

    refF = ref_y_vec()

    # Compute reference f from ODE solution: f_ref = a * d ln D / d a = a * D'/D
    # D' ≈ (D[i+1] - D[i-1]) / (2*A[i]) for interior points
    f_ref = np.empty((M, N))
    for i in range(1, N-1):
        Dprime = (refF[:, i+1] - refF[:, i-1]) / (2.0 * (A[i+1]-A[i-1]) / 2.0)
        f_ref[:, i] = A[i] * Dprime / refF[:, i]
    # Edge points: one-sided
    f_ref[:, 0] = A[0] * (refF[:, 1] - refF[:, 0]) / (A[1]-A[0]) / refF[:, 0]
    f_ref[:, N-1] = A[N-1] * (refF[:, N-1] - refF[:, N-2]) / (A[N-1]-A[N-2]) / refF[:, N-1]

    # Least-squares initial guess using Linder-Waugh
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
        u = 1.0 + W0_g + Wa_g * (1.0 - A_g)
        q = 1.0 - Om0_g
        g = c[0] + c[1]*q + c[2]*q*q + c[3]*u + c[4]*Wa_g*A_g + c[5]*u*q
        return np.clip(g, 0.1, 1.2)

    def _f(c):
        g = _gamma(c)
        return Om0_g ** g

    log_refF = np.log(np.maximum(refF, 1e-30))

    def _D_err(c):
        f = _f(c).reshape(M, N)
        h = np.diff(A)
        trap = 0.5 * (f[:, :-1] + f[:, 1:]) * h
        cum = np.cumsum(trap, axis=1)
        errs = []
        for j in range(N):
            I_row = np.zeros((M, N))
            for i in range(j+1, N):
                if j == 0:
                    I_row[:, i] = cum[:, i-1]
                else:
                    I_row[:, i] = cum[:, i-1] - cum[:, j-1]
            log_R = log_refF - log_refF[:, j:j+1]
            mask = np.zeros((M, N), dtype=bool)
            mask[:, j:] = True
            x_val = -(I_row + log_R)
            e = np.where(np.abs(x_val) < 0.5, np.expm1(x_val), np.exp(x_val) - 1.0)
            err = np.abs(e)
            errs.append(err[mask])
        return np.concatenate(errs)

    def _f_err(c):
        f = _f(c).reshape(M, N)
        return np.abs(f - f_ref) / f_ref

    def objective(c):
        fe = _f_err(c).ravel()
        de = _D_err(c)
        vals = np.concatenate([fe, de])
        m = vals.max()
        if m < 1e-15:
            return 0.0
        # Use a smooth max with temperature for better gradient behavior
        temp = 0.01
        s = temp * np.log(np.sum(np.exp(vals / temp)))
        return float(s)

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
